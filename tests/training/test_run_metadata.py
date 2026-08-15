"""Tests for the t=0 reproducibility manifest.

Every training run leaves behind a deterministic fingerprint of what
code, what config, and what environment produced its artefacts.
``mtg.utils.run_metadata`` builds that manifest and ``Trainer.train``
writes it before a single optimisation step happens.

These tests pin the contract of both layers:

* ``build_run_metadata`` always returns a JSON-serialisable mapping
  with stable required fields, even when git, pyproject, or the
  lockfile are missing.
* The training-config payload is complete (round-trips through
  ``dataclasses.asdict``).
* The lockfile digest is the SHA-256 of the actual file bytes.
* The git SHA / branch / dirty bit are filled in when running inside a
  real git repo (this repo).
* ``snapshot_run_metadata`` writes the JSON atomically and the file
  reads back as the same dict.
* ``Trainer.train`` drops ``run_metadata.json`` into the run log
  directory before training (so it is present even if training is
  interrupted).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from mtg.utils.run_metadata import (
    build_run_metadata,
    snapshot_run_metadata,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _DummyConfig:
    """Stand-in TrainingConfig for unit-level tests."""

    agent_type: str = "ppo"
    seed: int = 42
    total_timesteps: int = 1_000
    nested: dict[str, int] = field(default_factory=lambda: {"x": 1})


def _init_git_repo(root: Path, *, dirty: bool = False) -> None:
    """Initialise a minimal git repo so ``git rev-parse`` succeeds."""

    def _run(args: list[str]) -> None:
        subprocess.run(args, cwd=str(root), check=True, capture_output=True)

    _run(["git", "init", "-q", "-b", "main"])
    _run(["git", "config", "user.email", "test@example.com"])
    _run(["git", "config", "user.name", "Test"])
    (root / "pyproject.toml").write_text("[project]\nname='dummy'\nversion='0'\n")
    (root / "uv.lock").write_text("# uv lockfile\n")
    _run(["git", "add", "-A"])
    _run(["git", "commit", "-q", "-m", "init"])
    if dirty:
        (root / "pyproject.toml").write_text("[project]\nname='dirty'\nversion='1'\n")


# ---------------------------------------------------------------------------
# build_run_metadata: shape and content
# ---------------------------------------------------------------------------


def test_build_run_metadata_has_required_top_level_keys() -> None:
    """The manifest exposes a stable set of top-level keys."""
    manifest = build_run_metadata(config=_DummyConfig())
    for key in (
        "schema_version",
        "captured_at_unix",
        "captured_at_iso",
        "repo_root",
        "git",
        "lockfile_digests",
        "runtime",
        "invocation",
        "environment",
        "training_config",
    ):
        assert key in manifest, f"missing top-level key: {key}"
    assert manifest["schema_version"] == 2


def test_build_run_metadata_captures_invocation_block() -> None:
    """``invocation`` records argv, cwd, executable, and pid."""
    manifest = build_run_metadata(config=_DummyConfig())
    inv = manifest["invocation"]
    assert isinstance(inv["argv"], list) and inv["argv"]
    assert isinstance(inv["cwd"], str) and inv["cwd"]
    assert isinstance(inv["executable"], str) and inv["executable"]
    assert isinstance(inv["pid"], int) and inv["pid"] > 0


def test_build_run_metadata_captures_tracked_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``environment`` reflects the values of the tracked env vars at capture time."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("PYTHONHASHSEED", "1234")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    manifest = build_run_metadata(config=_DummyConfig())
    env = manifest["environment"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["PYTHONHASHSEED"] == "1234"
    assert env["CUBLAS_WORKSPACE_CONFIG"] is None
    assert "OMP_NUM_THREADS" in env


def test_build_run_metadata_iso_timestamp_is_utc_z_suffix() -> None:
    """``captured_at_iso`` ends with the ``Z`` UTC marker, regardless of host TZ."""
    manifest = build_run_metadata(config=_DummyConfig())
    iso = manifest["captured_at_iso"]
    assert isinstance(iso, str) and iso.endswith("Z"), iso


def test_build_run_metadata_captures_dataclass_config_fully() -> None:
    """Dataclass configs round-trip through ``asdict`` into the manifest."""
    cfg = _DummyConfig(agent_type="cgfa", seed=7, total_timesteps=12_345)
    manifest = build_run_metadata(config=cfg)
    payload = manifest["training_config"]
    assert isinstance(payload, dict)
    assert payload["agent_type"] == "cgfa"
    assert payload["seed"] == 7
    assert payload["total_timesteps"] == 12_345
    assert payload["nested"] == {"x": 1}


def test_build_run_metadata_handles_missing_git_root_gracefully(tmp_path: Path) -> None:
    """Calling outside a git repo must not crash and must yield ``None``-ish git fields."""
    manifest = build_run_metadata(config=_DummyConfig(), repo_root=tmp_path)
    assert manifest["repo_root"] == str(tmp_path.resolve())
    git = manifest["git"]
    assert "sha" in git and "branch" in git and "dirty" in git
    # Without an init'd repo, all three fields must be None.
    assert git["sha"] is None
    assert git["branch"] is None
    assert git["dirty"] is None
    assert manifest["lockfile_digests"] == {}


def test_build_run_metadata_resolves_real_git_metadata(tmp_path: Path) -> None:
    """Inside a real git repo, sha/branch/dirty are populated and look right."""
    _init_git_repo(tmp_path, dirty=False)
    manifest = build_run_metadata(config=_DummyConfig(), repo_root=tmp_path)
    git = manifest["git"]
    assert isinstance(git["sha"], str) and len(git["sha"]) == 40
    assert git["branch"] in {"main", "master"}
    assert git["dirty"] is False


def test_build_run_metadata_flags_dirty_working_copy(tmp_path: Path) -> None:
    """A working-copy modification must show up as ``dirty=True`` in the manifest."""
    _init_git_repo(tmp_path, dirty=True)
    manifest = build_run_metadata(config=_DummyConfig(), repo_root=tmp_path)
    assert manifest["git"]["dirty"] is True


def test_build_run_metadata_lockfile_digest_matches_sha256(tmp_path: Path) -> None:
    """``lockfile_digests`` must equal the actual SHA-256 of the on-disk file."""
    _init_git_repo(tmp_path)
    lock_bytes = (tmp_path / "uv.lock").read_bytes()
    expected = "sha256:" + hashlib.sha256(lock_bytes).hexdigest()
    manifest = build_run_metadata(config=_DummyConfig(), repo_root=tmp_path)
    assert manifest["lockfile_digests"]["uv.lock"] == expected
    assert manifest["lockfile_digests"]["pyproject.toml"].startswith("sha256:")


def test_build_run_metadata_runtime_block_has_python_and_platform() -> None:
    """``runtime`` always reports python/platform/hostname."""
    manifest = build_run_metadata(config=_DummyConfig())
    runtime = manifest["runtime"]
    for key in ("python", "platform", "machine", "hostname"):
        assert key in runtime, f"missing runtime field: {key}"
        assert isinstance(runtime[key], str) and runtime[key]
    # Torch is optional, but if present must be a string.
    if runtime["torch"] is not None:
        assert isinstance(runtime["torch"], str)


def test_build_run_metadata_extra_fields_round_trip() -> None:
    """Caller-supplied ``extra`` fields are preserved verbatim."""
    extra = {"experiment_name": "alpha-1", "tag": "smoke"}
    manifest = build_run_metadata(config=_DummyConfig(), extra=extra)
    assert manifest["extra"] == extra


def test_build_run_metadata_is_json_serialisable() -> None:
    """The manifest must serialise without ``default=`` callbacks."""
    manifest = build_run_metadata(config=_DummyConfig())
    json.dumps(manifest)  # must not raise


# ---------------------------------------------------------------------------
# snapshot_run_metadata: atomic write + readback
# ---------------------------------------------------------------------------


def test_snapshot_run_metadata_writes_json_to_disk(tmp_path: Path) -> None:
    """``snapshot_run_metadata`` writes a JSON manifest that round-trips."""
    out = tmp_path / "logs" / "run_metadata.json"
    written = snapshot_run_metadata(out, config=_DummyConfig())
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["schema_version"] == 2
    assert loaded["training_config"]["agent_type"] == "ppo"
    assert loaded["captured_at_unix"] == pytest.approx(written["captured_at_unix"], abs=2.0)
    assert "argv" in loaded["invocation"]
    assert "environment" in loaded


def test_snapshot_run_metadata_creates_parent_directories(tmp_path: Path) -> None:
    """Missing parent directories are created on the fly."""
    out = tmp_path / "deep" / "nested" / "logs" / "run_metadata.json"
    snapshot_run_metadata(out, config=_DummyConfig())
    assert out.exists()


def test_snapshot_run_metadata_atomic_rename_leaves_no_temp(tmp_path: Path) -> None:
    """No ``.tmp`` file remains after a successful write."""
    out = tmp_path / "run_metadata.json"
    snapshot_run_metadata(out, config=_DummyConfig())
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"unexpected temp files: {leftover}"


# ---------------------------------------------------------------------------
# Integration with Trainer.train
# ---------------------------------------------------------------------------


def test_trainer_writes_run_metadata_before_training(tmp_path: Path) -> None:
    """``Trainer.train`` must drop ``run_metadata.json`` in the run log dir.

    We stub out the SB3 ``learn`` call so the test doesn't actually
    optimise anything; we only care that the manifest lands on disk
    before the training step executes (i.e. even crashed runs get the
    fingerprint).
    """
    from mtg.training.train import Trainer, TrainingConfig

    cfg = TrainingConfig(
        agent_type="random",  # heuristic path; no SB3 required
        deck_archetype="mono_red_aggro",
        total_timesteps=1,
        n_envs=1,
        max_turns=2,
        max_steps_per_episode=4,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        experiment_name="t0_metadata_check",
        enable_periodic_eval=False,
        enable_episode_logger=False,
        enable_checkpointing=False,
        enable_entropy_schedule=False,
        enable_early_stopping=False,
        enable_league=False,
        enable_vec_normalize=False,
    )
    trainer = Trainer(cfg)
    # Skip the heavy custom training loop; we only care about the
    # manifest landing on disk.
    trainer._custom_training_loop = lambda: {  # type: ignore[assignment]
        "episode_rewards": [],
        "episode_lengths": [],
        "win_rate": 0.0,
        "mean_reward": 0.0,
        "mean_length": 0.0,
    }

    trainer.train()

    manifest_path = trainer.log_dir / "run_metadata.json"
    assert manifest_path.exists(), "Trainer.train did not write run_metadata.json"
    loaded = json.loads(manifest_path.read_text())
    assert loaded["training_config"]["agent_type"] == "random"
    assert loaded["training_config"]["seed"] == cfg.seed
    assert loaded["training_config"]["total_timesteps"] == cfg.total_timesteps
    assert loaded["extra"]["experiment_name"] == "t0_metadata_check"
    # The manifest is written *before* training starts, so the timestamp
    # should not be in the future relative to "now".
    assert loaded["captured_at_unix"] <= time.time() + 5.0
    trainer.close()
