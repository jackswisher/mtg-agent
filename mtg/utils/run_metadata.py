"""Reproducibility manifest written at the start of every training run.

Every artefact that lands on disk is tied back to a deterministic
snapshot of the environment that produced it. Anyone re-running
``mtg-research`` later can:

* check out the exact git SHA recorded in ``run_metadata.json``,
* recreate the Python env from the lockfile digest (or read the
  captured lockfile bytes), and
* hand the captured ``training_config`` straight back into
  ``Trainer(TrainingConfig(...))`` to rerun the experiment.

The manifest is written at ``t=0`` (before training starts) so that
crashed or interrupted runs still leave behind a clear record of what
they were trying to do.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import typing as tp
from dataclasses import asdict, is_dataclass
from pathlib import Path

# Environment variables worth capturing for reproducibility. ``CUDA_*``
# and ``CUBLAS_*`` settings affect numeric determinism; ``PYTHONHASHSEED``
# pins dict iteration order; ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``
# pin BLAS thread counts.
_TRACKED_ENV_VARS: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_USE_CUDA_DSA",
    "PYTORCH_CUDA_ALLOC_CONF",
)

__all__ = [
    "RunMetadata",
    "build_run_metadata",
    "snapshot_run_metadata",
]


def _safe_git_command(args: list[str], cwd: Path) -> str | None:
    """Run a ``git`` command and return its raw stdout (stripped), or ``None``.

    ``None`` means the command itself failed (git is unavailable, the
    cwd is not inside a repo, etc.). An empty string means the command
    succeeded but produced no output (for example, ``git status
    --porcelain`` on a clean tree). Callers MUST distinguish the two.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    return result.stdout.strip()


def _safe_git_command_nonempty(args: list[str], cwd: Path) -> str | None:
    """Like :func:`_safe_git_command` but converts empty stdout to ``None``."""
    out = _safe_git_command(args, cwd)
    if out is None:
        return None
    return out or None


def _git_sha(repo_root: Path) -> str | None:
    return _safe_git_command_nonempty(["rev-parse", "HEAD"], repo_root)


def _git_branch(repo_root: Path) -> str | None:
    return _safe_git_command_nonempty(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)


def _git_dirty(repo_root: Path) -> bool | None:
    """Return True if the working copy has uncommitted changes.

    ``None`` means git is unavailable or the cwd is not inside a repo.
    Callers must NOT fall back to ``False`` in that case; a missing
    reading is not the same as a clean tree.
    """
    out = _safe_git_command(["status", "--porcelain"], repo_root)
    if out is None:
        return None
    return out != ""


def _file_digest(path: Path) -> str | None:
    """Return ``sha256:<hex>`` of ``path``, or ``None`` if it doesn't exist."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
    except OSError:
        return None
    return f"sha256:{h.hexdigest()}"


def _resolve_repo_root(explicit: Path | None) -> Path:
    """Pick a sensible repo root: explicit -> git -> ``cwd``."""
    if explicit is not None:
        return Path(explicit).resolve()
    cwd = Path.cwd()
    git_root = _safe_git_command_nonempty(["rev-parse", "--show-toplevel"], cwd)
    if git_root:
        return Path(git_root).resolve()
    return cwd


def _torch_version() -> str | None:
    try:
        import torch  # noqa: PLC0415

        return str(torch.__version__)
    except ImportError:
        return None


def _cuda_summary() -> dict[str, tp.Any] | None:
    """Capture the active CUDA / device summary, if PyTorch is installed."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    info: dict[str, tp.Any] = {
        "available": bool(torch.cuda.is_available()),
        "version": getattr(torch.version, "cuda", None),
    }
    if info["available"]:
        try:
            info["device_count"] = int(torch.cuda.device_count())
            info["device_name_0"] = torch.cuda.get_device_name(0)
        except RuntimeError:
            pass
    return info


def _config_to_jsonable(config: tp.Any) -> tp.Any:
    """Best-effort conversion of a config object into a JSON-serialisable dict."""
    if config is None:
        return None
    if is_dataclass(config) and not isinstance(config, type):
        try:
            return asdict(config)
        except TypeError:
            pass
    if hasattr(config, "to_dict"):
        try:
            return config.to_dict()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(config, dict):
        return config
    # Fall back to the repr so we still record *something*.
    return {"__repr__": repr(config)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RunMetadata(dict):
    """Just a dict subclass; lets callers treat it as a struct."""


def build_run_metadata(
    *,
    config: tp.Any,
    repo_root: Path | None = None,
    extra: dict[str, tp.Any] | None = None,
) -> RunMetadata:
    """Assemble (but do not write) a run metadata manifest.

    Args:
        config: Training configuration to embed.  ``dataclass`` instances
            are converted via :func:`dataclasses.asdict`; objects with a
            ``to_dict()`` method use that; otherwise the repr is captured.
        repo_root: Override for the git repo root.  If ``None``, auto-
            detected via ``git rev-parse --show-toplevel`` falling back
            to ``Path.cwd()``.
        extra: Optional additional fields to merge into the manifest
            (caller's hostname/seed seed list/etc.).

    Returns:
        :class:`RunMetadata` with the captured fields.  Always JSON-
        serialisable.
    """
    root = _resolve_repo_root(repo_root)

    git_section: dict[str, tp.Any] = {
        "sha": _git_sha(root),
        "branch": _git_branch(root),
        "dirty": _git_dirty(root),
    }

    lockfile_section: dict[str, tp.Any] = {}
    for name in ("uv.lock", "poetry.lock", "requirements.txt"):
        digest = _file_digest(root / name)
        if digest is not None:
            lockfile_section[name] = digest
    pyproject_digest = _file_digest(root / "pyproject.toml")
    if pyproject_digest is not None:
        lockfile_section["pyproject.toml"] = pyproject_digest

    runtime_section: dict[str, tp.Any] = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "torch": _torch_version(),
        "cuda": _cuda_summary(),
    }

    invocation_section: dict[str, tp.Any] = {
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "executable": sys.executable,
        "pid": os.getpid(),
    }

    env_section: dict[str, str | None] = {name: os.environ.get(name) for name in _TRACKED_ENV_VARS}

    manifest: RunMetadata = RunMetadata(
        {
            "schema_version": 2,
            "captured_at_unix": float(time.time()),
            "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repo_root": str(root),
            "git": git_section,
            "lockfile_digests": lockfile_section,
            "runtime": runtime_section,
            "invocation": invocation_section,
            "environment": env_section,
            "training_config": _config_to_jsonable(config),
        }
    )
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def snapshot_run_metadata(
    out_path: Path,
    *,
    config: tp.Any,
    repo_root: Path | None = None,
    extra: dict[str, tp.Any] | None = None,
) -> RunMetadata:
    """Build the manifest and atomically write it to ``out_path`` as JSON.

    If ``out_path``'s parent directory does not exist it will be created.
    """
    manifest = build_run_metadata(config=config, repo_root=repo_root, extra=extra)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(out_path)
    return manifest
