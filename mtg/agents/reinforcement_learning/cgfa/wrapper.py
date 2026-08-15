"""Gymnasium wrapper that surfaces per-factor signals to CGFA-PPO.

CGFA-PPO needs three extra arrays on every ``info`` dict that pass
through the rollout buffer:

* ``factor_values`` ``(K,)``: current values of the K causal factors.
* ``factor_rewards`` ``(K,)``: per-factor "reward" used for per-factor
  GAE; defined as the change in factor values between consecutive
  steps, optionally normalised by ``FactorSpec.scale``.
* ``factor_eps`` ``(K,)``: the SCM's predicted per-factor change
  under the structural equations ``do(state_t -> state_{t+1})``. Used
  by the intervention-calibration auxiliary loss in
  :class:`CGFAMaskablePPO`.

The wrapper is a thin shim over the canonical :class:`MTGEnv` that
reads ``info["causal_variables"]`` produced by
:meth:`mtg.env.reward.RewardCalculator.get_causal_variable_values`
and converts it into the three arrays above.
"""

from __future__ import annotations

import logging
import typing as tp

import numpy as np

from mtg.agents.reinforcement_learning.cgfa.factor_spec import (
    FactorSpec,
    extract_factor_values,
)

logger = logging.getLogger(__name__)

try:
    import gymnasium as _gym

    _WrapperBase = _gym.Wrapper
except ImportError:  # pragma: no cover - gymnasium is a hard dep for training

    class _WrapperBase:  # type: ignore[no-redef]
        """Fallback wrapper base used when gymnasium is unavailable."""

        def __init__(self, env: tp.Any):
            self.env = env


class CGFAEnvWrapper(_WrapperBase):
    """Adds per-factor reward signals to ``info`` for CGFA-PPO.

    Args:
        env: The underlying environment (typically an :class:`MTGEnv`
            optionally already wrapped by an ``ActionMasker``).
        factor_spec: Spec describing which factors to extract and how
            to normalise their changes.
        scm: Optional :class:`StructuralCausalModel`. When provided the
            wrapper computes ``factor_eps`` by running the SCM's
            structural equations on the prior causal-variable snapshot
            and comparing to the new snapshot. When omitted,
            ``factor_eps`` defaults to the same value as
            ``factor_rewards`` (so calibration loss reduces to a
            sanity check).
    """

    def __init__(
        self,
        env: tp.Any,
        factor_spec: FactorSpec,
        scm: tp.Any = None,
        calibration_mode: str = "factual",
    ) -> None:
        super().__init__(env)
        if calibration_mode not in {"factual", "interventional"}:
            raise ValueError(
                "CGFAEnvWrapper calibration_mode must be 'factual' or "
                f"'interventional', got {calibration_mode!r}."
            )
        self.factor_spec = factor_spec
        self.scm = scm
        self.calibration_mode = calibration_mode
        self._prev_factor_values: np.ndarray | None = None
        self._prev_causal_vars: dict[str, float] | None = None
        self._prev_info: dict[str, tp.Any] | None = None
        self._scm_eval_failures: int = 0
        self._scm_warned: bool = False

    def reset(self, **kwargs: tp.Any) -> tp.Any:
        """Reset the inner env and seed the per-factor cache from info."""
        obs, info = self.env.reset(**kwargs)
        cv = info.get("causal_variables") if info else None
        cur = extract_factor_values(cv, self.factor_spec)
        self._prev_factor_values = cur.copy()
        self._prev_causal_vars = dict(cv) if cv else None
        self._prev_info = dict(info) if info else None
        info["factor_values"] = cur.copy()
        info["factor_rewards"] = np.zeros(self.factor_spec.n_factors, dtype=np.float32)
        info["factor_eps"] = np.zeros(self.factor_spec.n_factors, dtype=np.float32)
        return obs, info

    def step(self, action: tp.Any) -> tp.Any:
        """Step the inner env and attach per-factor signals to ``info``."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        cv = info.get("causal_variables") if info else None
        cur = extract_factor_values(cv, self.factor_spec)
        prev = self._prev_factor_values if self._prev_factor_values is not None else cur.copy()

        delta = cur - prev
        normalised = self.factor_spec.normalised_rewards(delta)
        info["factor_values"] = cur.copy()
        info["factor_rewards"] = normalised.astype(np.float32)
        info["factor_eps"] = self._compute_scm_eps(prev, cur, cv, action).astype(np.float32)

        self._prev_factor_values = cur.copy()
        self._prev_causal_vars = dict(cv) if cv else None
        self._prev_info = dict(info) if info else None
        return obs, reward, terminated, truncated, info

    def _compute_scm_eps(
        self,
        prev: np.ndarray,
        cur: np.ndarray,
        causal_vars: dict[str, float] | None,
        action: tp.Any,
    ) -> np.ndarray:
        """Return the SCM-predicted per-factor change for calibration.

        When the SCM is wired in, the prior causal-variable snapshot
        is re-evaluated via the structural equations
        (``force_recompute=True``) and so is the current snapshot. The
        difference is the SCM's structural prediction of the
        per-factor change, independent of the actual transition (which
        may be noisier or include un-modelled effects).

        When no SCM is provided this simply returns the raw delta so
        the calibration loss is well-defined; it then reduces to a
        consistency check between two views of the same change.
        """
        if self.scm is None or causal_vars is None or self._prev_causal_vars is None:
            return cur - prev
        try:
            prev_eval = self.scm.evaluate(self._prev_causal_vars, force_recompute=True)
            if self.calibration_mode == "interventional":
                cur_eval = self._interventional_eval(action) or self.scm.evaluate(
                    causal_vars, force_recompute=True
                )
            else:
                cur_eval = self.scm.evaluate(causal_vars, force_recompute=True)
        except (KeyError, ValueError, ZeroDivisionError, TypeError) as exc:
            # Catch the narrow set of exceptions actually expected from
            # SCM evaluation; broader catches would hide genuine bugs
            # like import errors. Log once at warning level so any
            # degradation is visible in training stdout/stderr.
            self._scm_eval_failures += 1
            if not self._scm_warned:
                logger.warning(
                    "CGFAEnvWrapper: SCM.evaluate() raised %s; falling back "
                    "to raw factor deltas for this transition. Further "
                    "occurrences will be counted but not re-logged.",
                    type(exc).__name__,
                )
                self._scm_warned = True
            return cur - prev
        prev_arr = extract_factor_values(prev_eval, self.factor_spec)
        cur_arr = extract_factor_values(cur_eval, self.factor_spec)
        return self.factor_spec.normalised_rewards(cur_arr - prev_arr)

    def _interventional_eval(self, action: tp.Any) -> dict[str, float] | None:
        """Approximate the factor effect of the selected action from the prior state.

        This is an experimental, deliberately conservative target: it uses
        action metadata to perturb only root SCM variables that the action
        directly changes (land drop, mana spent, board commitment, removal
        availability). If an action cannot be mapped cleanly, callers fall
        back to the factual SCM delta rather than inventing a causal target.
        """
        if self.scm is None or self._prev_causal_vars is None or self._prev_info is None:
            return None
        metadata = self._prev_info.get("action_metadata") or {}
        try:
            action_idx = int(action)
        except (TypeError, ValueError):
            return None
        action_meta = metadata.get(action_idx)
        if not isinstance(action_meta, dict):
            return None

        kind = str(action_meta.get("kind") or action_meta.get("action_type") or "")
        card_type = str(action_meta.get("card_type") or "")
        intervened = dict(self._prev_causal_vars)

        if kind == "play_land":
            intervened["land_drop"] = 1.0
            intervened["mana_t"] = float(intervened.get("mana_t", 0.0)) + 1.0
        elif kind in {"cast_sorcery", "cast_instant"}:
            intervened["mana_spent"] = float(intervened.get("mana_spent", 0.0)) + 1.0
            if card_type in {"creature", "planeswalker"}:
                intervened["board_presence"] = float(intervened.get("board_presence", 0.0)) + 1.0
                power = float(action_meta.get("power", 0.0) or 0.0)
                intervened["own_power"] = float(intervened.get("own_power", 0.0)) + power
                if power >= 3.0:
                    intervened["threat_count"] = float(intervened.get("threat_count", 0.0)) + 1.0
            if action_meta.get("is_removal") or float(action_meta.get("deals_damage", 0) or 0) > 0:
                # Casting removal consumes the available answer in the short-run
                # causal snapshot. Future draws will restore this via factual state.
                intervened["has_removal"] = 0.0
        elif kind == "attack_toggle":
            power = float(action_meta.get("power", 0.0) or 0.0)
            intervened["mana_spent"] = float(intervened.get("mana_spent", 0.0)) + 0.25
            intervened["own_power"] = float(intervened.get("own_power", 0.0)) + 0.25 * power
        else:
            return None

        return self.scm.evaluate(intervened, force_recompute=True)
