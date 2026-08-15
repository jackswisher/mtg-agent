"""PPO agent using Stable-Baselines3.

This agent wraps the PPO implementation from SB3 for the MTG environment,
handling action masking and providing a consistent interface.
"""

from __future__ import annotations

import typing as tp
from pathlib import Path

import numpy as np
import torch

from mtg.agents.base.base import BaseAgent

# Disable PyTorch distribution validation globally.
# PyTorch >= 2.7 defaults _validate_args = __debug__ (True), which causes
# MaskablePPO to crash with a Simplex constraint violation.  The stale
# cached `self.probs` inside MaskableCategorical.apply_masking() can drift
# beyond the 1e-6 tolerance after many gradient updates in float32.
# Disabling validation is standard practice in RL libraries.
torch.distributions.Distribution.set_default_validate_args(False)

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def _mask_fn(env: tp.Any) -> np.ndarray:
    """Extract action mask from environment.

    Args:
        env: The MTG environment.

    Returns:
        Boolean action mask array.

    """
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

    # Build proper action mask from the environment's action builder
    if (
        hasattr(unwrapped, "action_builder")
        and hasattr(unwrapped, "state")
        and unwrapped.state is not None
    ):
        mask = unwrapped.action_builder.build_action_mask(unwrapped.state, player_id=0).astype(bool)
    else:
        mask = np.ones(unwrapped.action_space.n, dtype=bool)

    return mask.astype(bool)


class PPOAgent(BaseAgent):
    """PPO agent with action masking support.

    Uses Stable-Baselines3 MaskablePPO for training with legal action
    constraints. Falls back to random if SB3 not installed.

    Attributes:
        model: The SB3 PPO model.
        learning_rate: Learning rate for training.
        batch_size: Batch size for updates.

    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        lr_end: float = 1e-5,
        batch_size: int = 256,
        n_steps: int = 2048,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_end: float | None = None,
        clip_range_vf: float | None = 0.2,
        ent_coef: float = 0.05,
        ent_coef_end: float = 0.005,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        target_kl: float | None = 0.03,
        vf_coef: float = 0.5,
        use_set_encoder: bool = False,
        features_dim: int = 512,
        seed: int | None = None,
    ) -> None:
        """Initialize the PPO agent.

        Args:
            observation_dim: Dimension of observation space.
            action_dim: Number of discrete actions.
            learning_rate: Initial (peak) learning rate.
            lr_end: Final learning rate after linear annealing.  Set equal
                to ``learning_rate`` to disable annealing.
            batch_size: Minibatch size.
            n_steps: Steps per rollout buffer. Should cover 10-20+ full episodes
                for proper credit assignment. MTG episodes are ~80-120 steps
                with auto_resolve.
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            clip_range: PPO clip range (initial value).
            clip_range_end: Final clip range after linear annealing.  When
                set, the clip range interpolates linearly from
                ``clip_range`` to ``clip_range_end`` over the training
                budget.  ``None`` disables scheduling.
            clip_range_vf: Value function clipping range.  When set, the
                value function loss is computed with PPO-style clipping
                (``L_vf = max((V - target)^2, (V_clipped - target)^2)``)
                instead of plain MSE, which reduces value-function
                divergence when returns have high variance.  Set to
                ``None`` to disable (falls back to MSE).
            ent_coef: Initial entropy coefficient.  Higher values drive
                exploration early in training.
            ent_coef_end: Final entropy coefficient after linear annealing.
            max_grad_norm: Max gradient norm for clipping (stability).
            n_epochs: Number of PPO epochs per rollout.
            target_kl: If set, early-stop PPO epochs when the approximate KL
                divergence exceeds this threshold. Prevents destructive updates.
            vf_coef: Value function loss coefficient.
            use_set_encoder: If True, use the permutation-invariant
                ``MTGFeaturesExtractor`` (attention over card slots)
                instead of the default MLP features extractor.  This
                respects the unordered nature of the hand / battlefield
                / graveyard zones and gives a large sample-efficiency
                boost on card-game observations.
            features_dim: Output dimensionality of the set encoder when
                ``use_set_encoder=True``.
            seed: Random seed.

        """
        super().__init__(name="PPOAgent", deterministic=False)

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.lr_end = lr_end
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_end = clip_range_end
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.ent_coef_end = ent_coef_end
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.target_kl = target_kl
        self.vf_coef = vf_coef
        self.use_set_encoder = use_set_encoder
        self.features_dim = features_dim
        self.seed = seed

        self.model: tp.Any = None
        self._rng = np.random.default_rng(seed)

        if not HAS_SB3:
            import warnings

            warnings.warn(
                "stable-baselines3 not installed. PPOAgent will use random actions.",
                stacklevel=2,
            )

    def _make_lr_schedule(self) -> tp.Callable[[float], float]:
        """Build a linear annealing schedule for the learning rate.

        SB3 calls this with ``progress_remaining`` which goes from 1.0 → 0.0
        over the course of training.
        """
        lr_start = self.learning_rate
        lr_end = self.lr_end

        def _schedule(progress_remaining: float) -> float:
            return lr_end + (lr_start - lr_end) * progress_remaining

        return _schedule

    def _make_clip_range_schedule(self) -> tp.Any:
        """Build the clip-range schedule passed to ``MaskablePPO``.

        If ``clip_range_end`` is ``None`` we pass the constant float
        through (SB3 accepts either).  Otherwise we return a callable
        that interpolates linearly.
        """
        if self.clip_range_end is None:
            return self.clip_range

        start = self.clip_range
        end = self.clip_range_end

        def _schedule(progress_remaining: float) -> float:
            return end + (start - end) * progress_remaining

        return _schedule

    def _make_clip_range_vf_schedule(self) -> tp.Any:
        """Return the clip_range_vf schedule (constant or ``None``)."""
        if self.clip_range_vf is None:
            return None
        return float(self.clip_range_vf)

    def get_ent_coef_for_progress(self, progress_remaining: float) -> float:
        """Compute annealed entropy coefficient for a given progress.

        Meant to be called from a training callback on every step.
        """
        return self.ent_coef_end + (self.ent_coef - self.ent_coef_end) * progress_remaining

    def initialize_model(self, env: tp.Any) -> None:
        """Initialize the PPO model with an environment.

        Accepts either a raw gymnasium env (wrapped with ActionMasker
        automatically) or a pre-wrapped VecEnv (used as-is).

        Args:
            env: Gymnasium environment for training.

        """
        if not HAS_SB3:
            return

        from stable_baselines3.common.vec_env import VecEnv

        wrapped_env = env if isinstance(env, VecEnv) else ActionMasker(env, _mask_fn)

        policy_kwargs: dict[str, tp.Any] = {
            "net_arch": {"pi": [512, 256], "vf": [512, 256]},
            "activation_fn": torch.nn.ReLU,
        }
        if self.use_set_encoder:
            from mtg.agents.reinforcement_learning.features import MTGFeaturesExtractor

            policy_kwargs["features_extractor_class"] = MTGFeaturesExtractor
            policy_kwargs["features_extractor_kwargs"] = {
                "features_dim": self.features_dim,
            }
            # When using the set encoder the extractor already produces a
            # rich ``features_dim`` representation; the pi/vf heads can
            # be slimmer MLPs on top of it.
            policy_kwargs["net_arch"] = {"pi": [256, 128], "vf": [256, 128]}

        ppo_kwargs: dict[str, tp.Any] = {
            "learning_rate": self._make_lr_schedule(),
            "batch_size": self.batch_size,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self._make_clip_range_schedule(),
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "n_epochs": self.n_epochs,
            "target_kl": self.target_kl,
            "normalize_advantage": True,
            "seed": self.seed,
            "verbose": 0,
            "policy_kwargs": policy_kwargs,
        }
        clip_vf = self._make_clip_range_vf_schedule()
        if clip_vf is not None:
            ppo_kwargs["clip_range_vf"] = clip_vf

        self.model = MaskablePPO("MlpPolicy", wrapped_env, **ppo_kwargs)
        self.model._ppo_agent_ref = self

    def set_env(self, env: tp.Any) -> None:
        """Swap the training environment while preserving learned weights.

        Accepts either a raw gymnasium env (wrapped with ActionMasker
        automatically) or a pre-wrapped VecEnv (used as-is).

        Args:
            env: New gymnasium environment (e.g. different opponent).

        """
        if not HAS_SB3 or self.model is None:
            return
        # VecEnv subclasses are already wrapped; only raw envs need masking
        from stable_baselines3.common.vec_env import VecEnv

        if isinstance(env, VecEnv):
            self.model.set_env(env)
        else:
            wrapped_env = ActionMasker(env, _mask_fn)
            self.model.set_env(wrapped_env)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict[str, tp.Any] | None = None,
    ) -> int:
        """Select action using the learned policy.

        Args:
            observation: Current state.
            action_mask: Legal action mask.
            info: Optional additional info.

        Returns:
            Selected action index.

        """
        if self.model is None or not HAS_SB3:
            legal = np.where(action_mask > 0)[0]
            if len(legal) == 0:
                return 0
            return int(self._rng.choice(legal))

        action, _ = self.model.predict(
            observation,
            deterministic=self.deterministic,
            action_masks=action_mask.astype(bool),
        )
        return int(action)

    def train(
        self,
        total_timesteps: int,
        callback: tp.Any | None = None,
        progress_bar: bool = True,
        reset_num_timesteps: bool = True,
    ) -> None:
        """Train the agent.

        Args:
            total_timesteps: Total training steps.
            callback: Optional training callback.
            progress_bar: Whether to show progress.
            reset_num_timesteps: If False, continue from the current
                timestep count (preserves LR schedule across calls).

        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model first.")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
        )

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Temporarily detaches the environment so SubprocVecEnv worker
        processes (which contain unpicklable AuthenticationString objects)
        are not serialized.

        Args:
            path: Save path.

        """
        if self.model is not None:
            env_backup = self.model.env
            self.model.env = None
            try:
                self.model.save(str(path))
            finally:
                self.model.env = env_backup

    def load(self, path: str | Path) -> None:
        """Load the model from disk.

        Args:
            path: Load path.

        """
        if HAS_SB3:
            self.model = MaskablePPO.load(str(path))
            self.model._ppo_agent_ref = self

    def get_action_probabilities(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Get action probability distribution.

        Args:
            observation: Current state.
            action_mask: Legal action mask.

        Returns:
            Probability distribution over actions.

        """
        if self.model is None or not HAS_SB3:
            return super().get_action_probabilities(observation, action_mask)

        import torch

        obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
        # Use the masked distribution for consistency with MaskablePPO's
        # action selection; this avoids divergence between policy probs
        # here and the actual sampling distribution used during training.
        mask_tensor = torch.tensor(
            action_mask.reshape(1, -1), dtype=torch.bool, device=obs_tensor.device
        )
        distribution = self.model.policy.get_distribution(obs_tensor, mask_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy().flatten()

        return probs
