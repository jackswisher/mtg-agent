"""SB3 callback that snapshots CGFA-specific logger keys to CSV.

The CGFA-PPO trainer logs a rich set of per-factor diagnostics
(``cgfa/factor_corr/<name>``, ``cgfa/factor_share/<name>``,
``cgfa/gate/mean`` etc.) via the SB3 ``Logger`` after each PPO update.
Those values are visible on TensorBoard but are awkward to load for
publication-quality figures.

This callback reads :pyattr:`stable_baselines3.common.logger.Logger.name_to_value`,
filters for the ``cgfa/...`` namespace, and appends one row to a CSV per
PPO update.  The CSV is consumed by :mod:`scripts.research.calibration_plot`.

Implementation notes:

*   Snapshots are gated on the SB3 ``_n_updates`` counter so the writer
    fires exactly once per :meth:`PPO.train` call.  This is important
    because the more obvious ``on_rollout_end`` hook fires *before* the
    first ``train()`` call (so the first row would be empty) and SB3's
    ``logger.dump`` clears ``name_to_value`` between rollouts.
*   A snapshot is also flushed on ``on_training_end`` so the final
    ``train()`` invocation (which has no follow-up rollout) still
    lands in the CSV. This is critical for short budgets where only a
    single full rollout fits in the timestep budget; otherwise the
    CSV would be header-only.
"""

from __future__ import annotations

import csv
import typing as tp
from pathlib import Path

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover (SB3 is a hard dep for training)
    BaseCallback = object  # type: ignore[misc, assignment]


# Keys we always emit (defaults so the CSV header is stable across runs).
_BASE_COLUMNS: tuple[str, ...] = (
    "step",
    "n_updates",
)


class CGFACalibrationCallback(BaseCallback):  # type: ignore[misc]
    """Snapshot ``cgfa/*`` logger keys to a CSV after every PPO update.

    Args:
        log_dir: Directory in which the ``cgfa_calibration.csv`` file
            is created.  The parent directory is created if missing.
        prefix: Logger-key prefix to capture.  Defaults to ``"cgfa/"``.
        flush_every: Force a flush after every ``flush_every`` rows so
            partial runs are still readable mid-training.
    """

    def __init__(
        self,
        log_dir: str | Path,
        *,
        prefix: str = "cgfa/",
        flush_every: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_dir = Path(log_dir)
        self.prefix = prefix
        self.flush_every = max(1, int(flush_every))
        self.csv_path = self.log_dir / "cgfa_calibration.csv"
        self._fp: tp.Any = None
        self._writer: csv.DictWriter | None = None
        self._fieldnames: list[str] | None = None
        self._row_count = 0
        # Track the last seen SB3 PPO update counter so we snapshot
        # exactly once per train() call.
        self._last_seen_updates: int = -1

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # The header is materialised lazily after the first rollout
        # because we don't know which CGFA keys are present until then.
        self._last_seen_updates = int(getattr(self.model, "_n_updates", 0))

    def _on_step(self) -> bool:
        # Detect a completed PPO update (train() has fired since the
        # last step) and snapshot ``cgfa/*`` while name_to_value is
        # still populated; the next logger.dump() would clear it.
        self._maybe_write_snapshot()
        return True

    def _on_rollout_end(self) -> None:
        # Belt-and-braces snapshot in case the loop breaks before the
        # next rollout's first _on_step (e.g. early stopping).
        self._maybe_write_snapshot()

    def _on_training_end(self) -> None:
        # Capture the metrics from the FINAL train() invocation, which
        # has no follow-up rollout and would otherwise be lost (this is
        # the common case for budget=N*n_steps*n_envs runs that fit
        # exactly one PPO iteration).
        self._maybe_write_snapshot()
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None
            self._writer = None

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _maybe_write_snapshot(self) -> None:
        """Snapshot one CSV row iff a new PPO update has completed."""
        if self.model is None:
            return
        n_updates = int(getattr(self.model, "_n_updates", 0))
        if n_updates <= self._last_seen_updates:
            return
        snapshot = self._snapshot()
        # Only commit the snapshot if it actually contains cgfa/* data;
        # otherwise we'd write a row of (step, n_updates, ...empty...)
        # before the trainer has populated the logger.
        has_cgfa = any(k.startswith(self.prefix) for k in snapshot)
        if not has_cgfa:
            return
        self._last_seen_updates = n_updates
        self._ensure_writer(list(snapshot.keys()))
        assert self._writer is not None
        row = {k: snapshot.get(k, float("nan")) for k in self._fieldnames or []}
        self._writer.writerow(row)
        self._row_count += 1
        if self._row_count % self.flush_every == 0 and self._fp is not None:
            self._fp.flush()

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict[str, float]:
        """Read the latest ``cgfa/*`` values from the SB3 logger."""
        if self.model is None or not hasattr(self.model, "logger"):
            return {}
        name_to_value = getattr(self.model.logger, "name_to_value", None)
        if not name_to_value:
            return {}
        out: dict[str, float] = {
            "step": float(getattr(self.model, "num_timesteps", 0)),
            "n_updates": float(getattr(self.model, "_n_updates", 0)),
        }
        for key, value in name_to_value.items():
            if not isinstance(key, str) or not key.startswith(self.prefix):
                continue
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def _ensure_writer(self, sample_keys: tp.Iterable[str]) -> None:
        if self._writer is not None:
            return
        # Stable header order: base columns first, then the cgfa keys
        # alphabetised so plots come out in a deterministic order.
        cgfa_keys = sorted(k for k in sample_keys if k.startswith(self.prefix))
        fieldnames = list(_BASE_COLUMNS) + cgfa_keys
        self._fieldnames = fieldnames
        self._fp = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=fieldnames)
        self._writer.writeheader()
