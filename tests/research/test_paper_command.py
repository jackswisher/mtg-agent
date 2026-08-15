"""Tests for the ``mtg-research paper`` end-to-end orchestrator.

The ``paper`` subcommand chains five stages (headline pipeline,
6-point ablation, held-out transfer, calibration plot, qualitative
case study) into a single command.  Stage execution is expensive
(hours of training), so these tests only exercise the orchestration
boundary: stage selection, argument forwarding, dry-run safety, and
the path-chaining that links the calibration / case-study stages to
the ablation outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts.research import cli as cli_mod

# ---------------------------------------------------------------------------
# _resolve_paper_stages: selection + validation
# ---------------------------------------------------------------------------


def test_resolve_paper_stages_default_order() -> None:
    """No --only / --skip => every stage in canonical order."""
    assert cli_mod._resolve_paper_stages(None, None) == list(cli_mod.PAPER_STAGES)


def test_resolve_paper_stages_only_preserves_canonical_order() -> None:
    """--only filters but never reorders the stage list."""
    out = cli_mod._resolve_paper_stages(["calibration", "headline"], None)
    assert out == ["headline", "calibration"]


def test_resolve_paper_stages_skip_preserves_canonical_order() -> None:
    """--skip removes the listed stages and keeps everything else in order."""
    out = cli_mod._resolve_paper_stages(None, ["transfer", "case-study"])
    assert out == ["headline", "ablation", "calibration", "cross-source"]


def test_resolve_paper_stages_includes_cross_source_by_default() -> None:
    """Cross-source aggregate is always the final stage when nothing is skipped."""
    out = cli_mod._resolve_paper_stages(None, None)
    assert (
        out[-1] == "cross-source"
    ), "cross-source must be the last stage so it sees every prior eval_results.json"


def test_resolve_paper_stages_only_and_skip_are_mutually_exclusive() -> None:
    """Reject commands that combine mutually exclusive stage filters."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli_mod._resolve_paper_stages(["headline"], ["transfer"])


def test_resolve_paper_stages_rejects_unknown_only() -> None:
    """Reject unknown stage names passed through --only."""
    with pytest.raises(ValueError, match="Unknown stage"):
        cli_mod._resolve_paper_stages(["headline", "bogus"], None)


def test_resolve_paper_stages_rejects_unknown_skip() -> None:
    """Reject unknown stage names passed through --skip."""
    with pytest.raises(ValueError, match="Unknown stage"):
        cli_mod._resolve_paper_stages(None, ["bogus"])


# ---------------------------------------------------------------------------
# _find_cgfa_full_artifacts: seed-to-run mapping via sweep_manifest.yaml
# ---------------------------------------------------------------------------


def test_find_cgfa_full_artifacts_returns_empty_without_manifest(tmp_path: Path) -> None:
    """No manifest on disk => no artefacts; never raise."""
    out = cli_mod._find_cgfa_full_artifacts(tmp_path, "mono_red_aggro")
    assert out == []


def _write_fake_ablation_tree(
    base_dir: Path,
    *,
    deck: str,
    seeds: list[int],
) -> Path:
    """Materialise a minimal ablation tree for cgfa_full under ``base_dir``.

    Layout::

        base_dir/
            cgfa_full/
                sweep_manifest.yaml
                cgfa_<deck>_vs_multi_<seed>/
                    cgfa_<deck>.zip
                    cgfa/cgfa_calibration.csv

    Returns ``base_dir`` so callers can assert on the produced layout.
    """
    import yaml

    variant_dir = base_dir / "cgfa_full"
    variant_dir.mkdir(parents=True)
    runs = []
    for s in seeds:
        run_subdir = f"cgfa_{deck}_vs_multi_seed{s}"
        rdir = variant_dir / run_subdir
        (rdir / "cgfa").mkdir(parents=True)
        (rdir / f"cgfa_{deck}.zip").write_bytes(b"fake-checkpoint")
        (rdir / "cgfa" / "cgfa_calibration.csv").write_text(
            "step,n_updates,cgfa/pearson_card_advantage\n0,0,0.0\n"
        )
        runs.append(
            {
                "agent": "cgfa",
                "player_deck": deck,
                "seed": s,
                "opponents": ["azorius_control"],
                "output_dir": run_subdir,
                "status": "completed",
                "error": None,
                "training_time_seconds": 0,
                "timesteps_per_opponent": 1,
                "total_timesteps": 1,
                "agent_kwargs": {},
            }
        )
    manifest = {
        "experiment_name": "cgfa_full",
        "created": "2026-01-01T00:00:00",
        "config": {},
        "runs": runs,
    }
    (variant_dir / "sweep_manifest.yaml").write_text(yaml.safe_dump(manifest))
    return base_dir


def test_find_cgfa_full_artifacts_maps_seeds_to_runs(tmp_path: Path) -> None:
    """One artefact triple per completed seed, with model + CSV both present."""
    base = _write_fake_ablation_tree(tmp_path, deck="mono_red_aggro", seeds=[42, 7])
    found = cli_mod._find_cgfa_full_artifacts(base, "mono_red_aggro")

    assert len(found) == 2
    seeds = {seed for (_m, _csv, seed) in found}
    assert seeds == {42, 7}
    for model, csv, _seed in found:
        assert model.exists() and model.suffix == ".zip"
        assert csv.exists() and csv.name == "cgfa_calibration.csv"


def test_find_cgfa_full_artifacts_filters_other_decks(tmp_path: Path) -> None:
    """Only artefacts for the requested player deck are returned."""
    base = _write_fake_ablation_tree(tmp_path, deck="mono_red_aggro", seeds=[42])
    assert cli_mod._find_cgfa_full_artifacts(base, "azorius_control") == []


def test_find_cgfa_full_artifacts_skips_failed_runs(tmp_path: Path) -> None:
    """Runs with status='failed' are excluded even if files exist."""
    import yaml

    base = _write_fake_ablation_tree(tmp_path, deck="mono_red_aggro", seeds=[42])
    manifest_path = base / "cgfa_full" / "sweep_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["runs"][0]["status"] = "failed"
    manifest_path.write_text(yaml.safe_dump(manifest))

    assert cli_mod._find_cgfa_full_artifacts(base, "mono_red_aggro") == []


# ---------------------------------------------------------------------------
# _run_paper: dry-run + stage selection
# ---------------------------------------------------------------------------


def _spy_paper_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch every heavy stage with a MagicMock and return the spy registry."""
    spies: dict[str, MagicMock] = {
        "pipeline": MagicMock(return_value=0),
        "ablation": MagicMock(),
        "transfer": MagicMock(),
        "calibration": MagicMock(),
        "case_study": MagicMock(return_value=0),
    }
    monkeypatch.setattr(cli_mod, "_run_pipeline", spies["pipeline"])
    monkeypatch.setattr("scripts.research.ablation_sweep.run_ablation", spies["ablation"])
    monkeypatch.setattr("scripts.research.transfer_sweep.run_transfer", spies["transfer"])
    monkeypatch.setattr("scripts.research.calibration_plot.render", spies["calibration"])
    monkeypatch.setattr(cli_mod, "_dispatch_case_study", spies["case_study"])
    return spies


def _paper_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default kwargs for _run_paper that satisfy every required field."""
    base: dict[str, Any] = {
        "experiment_name": "x",
        "agents": ["ppo", "cgfa"],
        "player_decks": ["mono_red_aggro"],
        "ablation_decks": ["mono_red_aggro"],
        "transfer_decks": ["mono_red_aggro"],
        "opponents": ["mono_red_aggro", "azorius_control"],
        "heldout_opponents": ["domain_ramp"],
        "seeds": [42, 123],
        "timesteps_per_opponent": 1000,
        "eval_episodes": 10,
        "training_mode": "round-robin",
        "agency_mode": "auto",
        "n_envs": 1,
        "reward_type": "shaped",
        "max_turns": 20,
        "output_root": "results/research",
        "case_study_player_deck": "mono_red_aggro",
        "case_study_opponent_deck": "azorius_control",
        "case_study_seed": 7,
        "include_baselines": True,
        "transfer_mode": "fixed",
        "only": None,
        "skip": None,
        "force": False,
        "dry_run": False,
    }
    base.update(overrides)
    return base


def test_paper_dry_run_runs_no_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """--dry-run prints the plan without invoking any heavy stage."""
    spies = _spy_paper_stages(monkeypatch)
    rc = cli_mod._run_paper(**_paper_kwargs(dry_run=True))

    assert rc == 0
    for name, spy in spies.items():
        assert spy.call_count == 0, f"{name} was invoked despite --dry-run"


def test_paper_only_calibration_skips_training_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--only calibration must not trigger training/eval/transfer/case-study.

    It also must locate cgfa_full artefacts on disk and pass each seed's
    CSV (with a 'seedN' label) to the calibration renderer.
    """
    _write_fake_ablation_tree(tmp_path / "x" / "ablation", deck="mono_red_aggro", seeds=[42, 123])
    spies = _spy_paper_stages(monkeypatch)

    rc = cli_mod._run_paper(
        **_paper_kwargs(
            only=["calibration"],
            output_root=str(tmp_path),
            experiment_name="x",
        ),
    )

    assert rc == 0
    assert spies["pipeline"].call_count == 0
    assert spies["ablation"].call_count == 0
    assert spies["transfer"].call_count == 0
    assert spies["case_study"].call_count == 0
    assert spies["calibration"].call_count == 1

    csv_paths, out_path = spies["calibration"].call_args[0]
    labels = spies["calibration"].call_args[1]["labels"]
    assert len(csv_paths) == 2
    assert sorted(labels) == ["seed123", "seed42"]
    assert isinstance(out_path, Path)
    assert out_path.name == "cgfa_calibration.png"


def test_paper_only_case_study_picks_lowest_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--only case-study dispatches with the lowest-seed checkpoint.

    Determinism matters: re-running the same flag must always pick the
    same checkpoint regardless of filesystem iteration order.
    """
    _write_fake_ablation_tree(
        tmp_path / "x" / "ablation", deck="mono_red_aggro", seeds=[123, 42, 456]
    )
    spies = _spy_paper_stages(monkeypatch)

    rc = cli_mod._run_paper(
        **_paper_kwargs(
            only=["case-study"],
            output_root=str(tmp_path),
            experiment_name="x",
        ),
    )

    assert rc == 0
    assert spies["case_study"].call_count == 1
    cs_argv = spies["case_study"].call_args[0][0]
    assert "--model-path" in cs_argv
    model_path = Path(cs_argv[cs_argv.index("--model-path") + 1])
    assert "seed42" in model_path.parent.name, (
        f"Case study picked {model_path.parent.name}; "
        "expected the lowest-seed run for determinism."
    )
    assert cs_argv[cs_argv.index("--episode-seed") + 1] == "7"
    assert cs_argv[cs_argv.index("--player-deck") + 1] == "mono_red_aggro"
    assert cs_argv[cs_argv.index("--opponent-deck") + 1] == "azorius_control"


def test_paper_skips_calibration_and_case_study_when_no_cgfa_artefacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the ablation produced no cgfa_full artefacts, post-hoc stages bail.

    This is the realistic outcome when the user passes
    ``--skip ablation`` without a pre-existing run on disk.
    """
    spies = _spy_paper_stages(monkeypatch)

    rc = cli_mod._run_paper(
        **_paper_kwargs(
            only=["calibration", "case-study"],
            output_root=str(tmp_path),
        ),
    )

    assert rc == 0
    assert spies["calibration"].call_count == 0
    assert spies["case_study"].call_count == 0


# ---------------------------------------------------------------------------
# Leave-one-out transfer mode
# ---------------------------------------------------------------------------


def test_loo_folds_produces_n_disjoint_train_heldout_pairs() -> None:
    """For an N-deck pool, _loo_folds returns N folds with disjoint splits."""
    pool = ["a", "b", "c", "d", "e"]
    folds = cli_mod._loo_folds(pool)

    assert len(folds) == 5
    for heldout, train_set in folds:
        assert (
            heldout not in train_set
        ), f"held-out deck {heldout!r} leaked into train set {train_set!r}"
        assert len(train_set) == 4
        assert set(train_set) | {heldout} == set(pool)
    # Held-out decks across folds cover the whole pool exactly once.
    assert sorted(h for h, _ in folds) == sorted(pool)


def test_loo_folds_preserves_input_order() -> None:
    """Re-running LOO with the same pool must produce folds in the same order."""
    pool = ["mono_red_aggro", "azorius_control", "dimir_midrange"]
    folds_a = cli_mod._loo_folds(pool)
    folds_b = cli_mod._loo_folds(pool)

    assert folds_a == folds_b
    assert [h for h, _ in folds_a] == pool


def test_loo_folds_rejects_pool_too_small() -> None:
    """LOO requires at least 2 opponents (1 train + 1 held-out)."""
    with pytest.raises(ValueError, match="at least 2 opponents"):
        cli_mod._loo_folds(["only_one"])


def test_paper_loo_transfer_runs_one_fold_per_opponent_in_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--transfer-mode leave-one-out invokes run_transfer once per pool deck.

    Each fold must have a disjoint (train, held-out) split derived from the
    union of --opponents and --heldout-opponents.
    """
    spies = _spy_paper_stages(monkeypatch)

    rc = cli_mod._run_paper(
        **_paper_kwargs(
            opponents=["mono_red_aggro", "azorius_control", "dimir_midrange"],
            heldout_opponents=["domain_ramp", "boros_convoke"],
            transfer_mode="leave-one-out",
            only=["transfer"],
            output_root=str(tmp_path),
            experiment_name="loo_exp",
        ),
    )

    assert rc == 0
    assert spies["transfer"].call_count == 5, (
        f"Expected 5 LOO folds (one per deck in the 5-deck pool); "
        f"got {spies['transfer'].call_count} run_transfer invocations."
    )

    seen_heldout: set[str] = set()
    for call in spies["transfer"].call_args_list:
        cfg = call.args[0]
        assert (
            len(cfg.heldout_opponents) == 1
        ), f"LOO must hold out exactly one deck per fold; got {cfg.heldout_opponents}"
        heldout = cfg.heldout_opponents[0]
        assert heldout not in cfg.train_opponents, (
            f"LOO fold leaked: heldout={heldout} appears in train_opponents="
            f"{cfg.train_opponents}"
        )
        assert len(cfg.train_opponents) == 4
        assert cfg.experiment_name == f"transfer/loo_{heldout}", (
            f"LOO fold output must be nested under transfer/loo_<deck>; "
            f"got experiment_name={cfg.experiment_name!r}"
        )
        seen_heldout.add(heldout)

    assert seen_heldout == {
        "mono_red_aggro",
        "azorius_control",
        "dimir_midrange",
        "domain_ramp",
        "boros_convoke",
    }, "LOO must hold out every deck in the pool exactly once; " f"got {seen_heldout}"


def test_paper_fixed_transfer_unchanged_by_loo_flag_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default transfer_mode='fixed' produces a single TransferConfig.

    Locks in backward-compat: omitting --transfer-mode (or passing 'fixed')
    must not alter the existing single-split behavior.
    """
    spies = _spy_paper_stages(monkeypatch)

    rc = cli_mod._run_paper(
        **_paper_kwargs(
            opponents=["mono_red_aggro", "azorius_control"],
            heldout_opponents=["domain_ramp"],
            transfer_mode="fixed",
            only=["transfer"],
            output_root=str(tmp_path),
            experiment_name="fixed_exp",
        ),
    )

    assert rc == 0
    assert spies["transfer"].call_count == 1
    cfg = spies["transfer"].call_args.args[0]
    assert cfg.experiment_name == "transfer"
    assert cfg.train_opponents == ["mono_red_aggro", "azorius_control"]
    assert cfg.heldout_opponents == ["domain_ramp"]


def test_paper_loo_transfer_rejects_invalid_mode() -> None:
    """An unknown transfer_mode raises ValueError before any stage runs."""
    with pytest.raises(ValueError, match="--transfer-mode must be one of"):
        cli_mod._run_paper(**_paper_kwargs(transfer_mode="bogus"))


# ---------------------------------------------------------------------------
# Cross-source aggregate stage
# ---------------------------------------------------------------------------


def _write_fake_eval_results(path: Path, *, agent: str = "ppo") -> None:
    """Materialise a minimal eval_results.json the cross-source stage can ingest."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained": [
            {
                "agent": agent,
                "player_deck": "mono_red_aggro",
                "seed": 42,
                "per_opponent": {
                    "azorius_control": {
                        "win_rate": 0.5,
                        "win_rate_ci_lo": 0.4,
                        "win_rate_ci_hi": 0.6,
                        "draw_rate": 0.0,
                        "avg_reward": 0.0,
                        "n": 100,
                    }
                },
            }
        ],
        "baselines": [],
    }
    path.write_text(json.dumps(payload))


def test_discover_eval_results_finds_headline_ablation_and_transfer(
    tmp_path: Path,
) -> None:
    """All three stage families must surface in the discovered tuples.

    Order must be deterministic so cross-source aggregates stay reproducible.
    """
    paper_root = tmp_path / "exp"
    _write_fake_eval_results(
        paper_root / "headline" / "ppo_mono_red_aggro_seed42" / "eval" / "eval_results.json"
    )
    _write_fake_eval_results(
        paper_root / "ablation" / "cgfa_full" / "eval" / "eval_results.json",
        agent="cgfa",
    )
    _write_fake_eval_results(paper_root / "ablation" / "ppo" / "eval" / "eval_results.json")
    _write_fake_eval_results(paper_root / "transfer" / "eval" / "eval_results.json", agent="cgfa")
    _write_fake_eval_results(
        paper_root / "transfer" / "eval_heldout" / "eval_results.json", agent="cgfa"
    )

    found = cli_mod._discover_eval_results(paper_root)
    labels = [lbl for (lbl, _path) in found]

    assert "headline" in labels
    assert "ablation_cgfa_full" in labels
    assert "ablation_ppo" in labels
    assert "transfer_indist" in labels
    assert "transfer_heldout" in labels
    # Determinism: ablation labels are alphabetically sorted by variant name.
    abl_labels = [label for label in labels if label.startswith("ablation_")]
    assert abl_labels == sorted(abl_labels)


def test_discover_eval_results_picks_up_loo_folds(tmp_path: Path) -> None:
    """LOO transfer folds (transfer/loo_<deck>/) must be discovered too."""
    paper_root = tmp_path / "exp"
    for deck in ("mono_red_aggro", "azorius_control", "dimir_midrange"):
        _write_fake_eval_results(
            paper_root / "transfer" / f"loo_{deck}" / "eval" / "eval_results.json"
        )
        _write_fake_eval_results(
            paper_root / "transfer" / f"loo_{deck}" / "eval_heldout" / "eval_results.json"
        )

    found = cli_mod._discover_eval_results(paper_root)
    labels = sorted(lbl for (lbl, _path) in found)

    expected = sorted(
        f"transfer_loo_{deck}_{split}"
        for deck in ("mono_red_aggro", "azorius_control", "dimir_midrange")
        for split in ("indist", "heldout")
    )
    assert labels == expected


def test_discover_eval_results_returns_empty_for_fresh_dir(tmp_path: Path) -> None:
    """An empty paper root must return [] (never raise) so dry runs work."""
    assert cli_mod._discover_eval_results(tmp_path / "nothing_yet") == []


def test_paper_cross_source_stage_runs_aggregate_with_every_eval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--only cross-source dispatches aggregate() with every discovered file.

    Pins the contract: every eval_results.json under paper_root is fed to
    aggregate, with stable source labels matching the on-disk hierarchy.
    """
    spies = _spy_paper_stages(monkeypatch)

    paper_root = tmp_path / "exp"
    _write_fake_eval_results(paper_root / "headline" / "r" / "eval" / "eval_results.json")
    _write_fake_eval_results(paper_root / "ablation" / "ppo" / "eval" / "eval_results.json")
    _write_fake_eval_results(paper_root / "ablation" / "cgfa_full" / "eval" / "eval_results.json")

    agg_spy = MagicMock()
    monkeypatch.setattr("scripts.research.aggregate.aggregate", agg_spy)

    rc = cli_mod._run_paper(
        **_paper_kwargs(
            only=["cross-source"],
            output_root=str(tmp_path),
            experiment_name="exp",
        ),
    )

    assert rc == 0
    assert agg_spy.call_count == 1, "Cross-source must invoke aggregate exactly once per paper run"
    kwargs = agg_spy.call_args.kwargs
    assert len(kwargs["eval_paths"]) == 3
    assert kwargs["source_labels"] == [
        "headline",
        "ablation_cgfa_full",
        "ablation_ppo",
    ]
    assert kwargs["output_dir"] == paper_root / "cross_source"
    # baseline_agent defaults to first --agents (ppo for the canonical setup)
    assert kwargs["baseline_agent"] == "ppo"

    # No heavy stages should have run.
    assert spies["pipeline"].call_count == 0
    assert spies["ablation"].call_count == 0
    assert spies["transfer"].call_count == 0


def test_paper_cross_source_stage_skips_when_no_eval_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty paper root must not raise --- cross-source bails gracefully."""
    _spy_paper_stages(monkeypatch)
    agg_spy = MagicMock()
    monkeypatch.setattr("scripts.research.aggregate.aggregate", agg_spy)

    rc = cli_mod._run_paper(
        **_paper_kwargs(
            only=["cross-source"],
            output_root=str(tmp_path),
            experiment_name="empty_exp",
        ),
    )

    assert rc == 0
    assert (
        agg_spy.call_count == 0
    ), "Cross-source must not invoke aggregate when no eval_results.json exists"


def test_paper_nests_every_stage_under_master_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each stage runs under ``<output_root>/<experiment_name>/<stage>/``.

    Locks in the master-directory contract: regressing to the legacy
    flat ``<experiment_name>_<stage>`` sibling layout (which polluted
    ``results/research/`` with three top-level dirs per paper run)
    must fail this test.
    """
    spies = _spy_paper_stages(monkeypatch)
    _write_fake_ablation_tree(tmp_path / "exp" / "ablation", deck="mono_red_aggro", seeds=[42])

    rc = cli_mod._run_paper(
        **_paper_kwargs(
            output_root=str(tmp_path),
            experiment_name="exp",
        ),
    )

    assert rc == 0
    master_root = str(tmp_path / "exp")
    assert (tmp_path / "exp").is_dir(), "_run_paper must create the master dir"

    pipe_kwargs = spies["pipeline"].call_args.kwargs
    assert pipe_kwargs["experiment_name"] == "headline"
    assert pipe_kwargs["output_root"] == master_root

    abl_kwargs = spies["ablation"].call_args.kwargs
    assert abl_kwargs["experiment_name"] == "ablation"
    assert abl_kwargs["output_root"] == master_root

    transfer_cfg = spies["transfer"].call_args.args[0]
    assert transfer_cfg.experiment_name == "transfer"
    assert transfer_cfg.output_root == master_root

    cal_csv_paths, cal_out = spies["calibration"].call_args.args
    for csv in cal_csv_paths:
        assert str(tmp_path / "exp" / "ablation") in str(csv)
    assert str(tmp_path / "exp" / "ablation") in str(cal_out)

    cs_argv = spies["case_study"].call_args.args[0]
    cs_model = cs_argv[cs_argv.index("--model-path") + 1]
    cs_out_dir = cs_argv[cs_argv.index("--output-dir") + 1]
    assert str(tmp_path / "exp" / "ablation") in cs_model
    assert cs_out_dir == str(tmp_path / "exp" / "ablation" / "case_study")
