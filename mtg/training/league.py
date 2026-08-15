"""League-style opponent pool with Elo ratings and PFSP sampling.

Motivation
----------
Training against a *single* hand-coded heuristic opponent is enough to
get started but it is a fundamentally limited evaluation signal:

1. Win rate against a fixed baseline saturates (at which point the
   learning signal dies), yet we cannot tell whether the policy is
   still improving on harder match-ups.
2. A single opponent induces opponent-specific exploitation: the
   policy learns tactics that are uniquely effective against that
   opponent's biases and the deck's linear play pattern, and that
   doesn't transfer.
3. Without multi-opponent statistics we cannot report Elo / TrueSkill
   deltas that are comparable across runs and across the literature.

The ``League`` here solves all three problems by maintaining a pool
of opponents (heuristic agents plus historical snapshots of the
learning policy), assigning each a live Elo rating, and sampling
from the pool with Prioritised Fictitious Self-Play (PFSP), the same
mechanism AlphaStar (Vinyals et al. 2019) used to escape
single-opponent traps. Match results are fed back into the rating
update, so "who is currently hard for the agent" drifts over time
and the sampler automatically focuses training on those opponents.

The league is training-time infrastructure; it does **not** replace
the canonical held-out evaluation pipeline, which still runs against
a frozen fixed suite so cross-run numbers are comparable.

Public API
----------
``League``
    The central object.  Holds the opponent pool, the Elo table, and
    exposes ``sample_opponent(scheme)`` + ``record_match(name, win)``.
``elo_update``
    Functional Elo update used by ``League`` and by tests.
``PFSPSampler``
    Picks the next opponent proportional to the inverse of the
    expected win rate so the agent trains on "almost-winnable"
    opponents.
``snapshot_policy``
    Helper that serialises a live PPO/Causal agent as a pool entry
    that can be replayed later (lazy-loaded on first match).
"""

from __future__ import annotations

import math
import shutil
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DEFAULT_ELO",
    "League",
    "LeagueConfig",
    "Match",
    "OpponentEntry",
    "PFSPSampler",
    "elo_update",
    "expected_score",
    "snapshot_policy",
]


DEFAULT_ELO = 1500.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Classic Elo expected-score formula ``E_a = 1/(1+10^((r_b-r_a)/400))``."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 16.0,
) -> tuple[float, float]:
    """Return new ``(rating_a, rating_b)`` after one game.

    ``score_a`` is ``1.0`` for win, ``0.0`` for loss, ``0.5`` for draw.
    """
    exp_a = expected_score(rating_a, rating_b)
    delta = k * (score_a - exp_a)
    return rating_a + delta, rating_b - delta


# ---------------------------------------------------------------------------
# Pool entries
# ---------------------------------------------------------------------------


@dataclass
class OpponentEntry:
    """A single opponent in the league pool.

    The opponent is either:

    * A live ``agent`` object plus a deck name (used for heuristic
      opponents which have no training state).
    * A ``snapshot_path`` on disk plus an ``agent_factory`` that will
      be lazily loaded the first time the opponent is sampled (used
      for historical policy snapshots so they do not sit in GPU
      memory until needed).
    """

    name: str
    deck: str
    rating: float = DEFAULT_ELO
    n_games: int = 0
    n_wins: int = 0
    agent: tp.Any | None = None
    snapshot_path: Path | None = None
    agent_factory: tp.Callable[[Path], tp.Any] | None = None
    is_historical: bool = False

    @property
    def win_rate(self) -> float:
        """Historical win rate (from the agent-under-training's perspective)."""
        return (self.n_wins / self.n_games) if self.n_games else 0.0

    def resolve_agent(self) -> tp.Any:
        """Return the underlying agent, loading from disk if necessary."""
        if self.agent is not None:
            return self.agent
        if self.snapshot_path is None or self.agent_factory is None:
            raise RuntimeError(
                f"Opponent {self.name!r} has neither a live agent nor a "
                "snapshot path + factory; cannot resolve."
            )
        self.agent = self.agent_factory(self.snapshot_path)
        return self.agent


@dataclass
class Match:
    """Single match record used for history and rating reconstruction."""

    opponent: str
    win: bool
    learner_rating_before: float
    opponent_rating_before: float
    learner_rating_after: float
    opponent_rating_after: float


@dataclass
class LeagueConfig:
    """Hyper-parameters for league bookkeeping."""

    elo_k: float = 16.0
    sampling: str = "pfsp"  # "pfsp" or "uniform"
    pfsp_p: float = 2.0
    pfsp_eps: float = 0.02
    snapshot_dir: Path = Path("results/league_snapshots")
    max_historical: int = 6
    keep_snapshots_on_disk: bool = True


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


class PFSPSampler:
    """Prioritised Fictitious Self-Play sampler.

    Given the current learner rating and a pool of opponents, sample
    opponent ``i`` with probability proportional to ``(1 - p_i)^p +
    eps`` where ``p_i`` is the learner's expected score against
    opponent ``i`` and ``p`` concentrates the distribution on
    "almost-winnable" opponents.

    This is the sampler used in AlphaStar for their main learner
    training loop (Vinyals et al. 2019).  The ``eps`` term keeps
    dominated opponents from going completely unseen so rating
    estimates don't rot.
    """

    def __init__(self, p: float = 2.0, eps: float = 0.02, rng: np.random.Generator | None = None):
        self.p = float(p)
        self.eps = float(eps)
        self.rng = rng or np.random.default_rng()

    def sample_index(self, learner_rating: float, ratings: list[float]) -> int:
        """Return an index into ``ratings`` sampled via PFSP."""
        if not ratings:
            raise ValueError("Cannot sample from an empty opponent pool.")
        weights = np.array(
            [(1.0 - expected_score(learner_rating, r)) ** self.p + self.eps for r in ratings],
            dtype=np.float64,
        )
        total = float(weights.sum())
        if total <= 0.0 or not math.isfinite(total):
            # Degenerate case: fall back to uniform
            return int(self.rng.integers(0, len(ratings)))
        probs = weights / total
        return int(self.rng.choice(len(ratings), p=probs))


# ---------------------------------------------------------------------------
# League
# ---------------------------------------------------------------------------


class League:
    """Opponent pool with Elo ratings and PFSP sampling.

    The league does NOT orchestrate the actual training loop or the
    environment; it is a pure bookkeeping object that callers consult
    to decide who to play next and to record the outcome afterwards.

    Usage
    -----
    ::

        league = League(LeagueConfig())
        league.add_heuristic("greedy", "mono_red_aggro", agent=GreedyAggroAgent())
        opp = league.sample_opponent()
        ...play one game against opp.resolve_agent()...
        league.record_match(opp.name, learner_won)

    Periodically dump the current learner into the pool with
    ``league.add_snapshot(name, agent, deck)`` so the learner starts
    playing against its own past selves.
    """

    def __init__(
        self,
        config: LeagueConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.config = config or LeagueConfig()
        self.rng = rng or np.random.default_rng()
        self.pool: list[OpponentEntry] = []
        self.learner_rating: float = DEFAULT_ELO
        self.match_history: list[Match] = []
        self._pfsp = PFSPSampler(p=self.config.pfsp_p, eps=self.config.pfsp_eps, rng=self.rng)

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def add_heuristic(self, name: str, deck: str, agent: tp.Any) -> OpponentEntry:
        """Add a live heuristic opponent to the pool."""
        entry = OpponentEntry(name=name, deck=deck, agent=agent)
        self._add(entry)
        return entry

    def add_snapshot(
        self,
        name: str,
        agent: tp.Any,
        deck: str,
        agent_factory: tp.Callable[[Path], tp.Any] | None = None,
    ) -> OpponentEntry:
        """Persist the current ``agent`` and add it as a historical opponent.

        ``agent_factory(path) -> agent`` is used to rebuild the snapshot
        on demand; it defaults to ``lambda p: type(agent).load(p)`` when
        ``agent`` exposes a ``load`` classmethod.  Snapshots are capped
        at ``config.max_historical`` entries; the oldest historical
        snapshot is evicted FIFO when the cap is exceeded.
        """
        snap_dir = Path(self.config.snapshot_dir)
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"{name}.zip"

        if hasattr(agent, "save"):
            try:
                agent.save(str(snap_path.with_suffix("")))
            except Exception:  # noqa: BLE001 - best-effort snapshot
                snap_path = None

        factory = agent_factory or _default_factory(type(agent))

        entry = OpponentEntry(
            name=name,
            deck=deck,
            snapshot_path=snap_path,
            agent_factory=factory,
            rating=self.learner_rating,  # seed with learner rating
            is_historical=True,
        )
        self._add(entry)
        self._evict_old_snapshots()
        return entry

    def _add(self, entry: OpponentEntry) -> None:
        for existing in self.pool:
            if existing.name == entry.name:
                raise ValueError(f"Opponent {entry.name!r} already in pool")
        self.pool.append(entry)

    def _evict_old_snapshots(self) -> None:
        historicals = [e for e in self.pool if e.is_historical]
        if len(historicals) <= self.config.max_historical:
            return
        to_evict = historicals[: len(historicals) - self.config.max_historical]
        for entry in to_evict:
            self.pool.remove(entry)
            if (
                not self.config.keep_snapshots_on_disk
                and entry.snapshot_path
                and entry.snapshot_path.exists()
            ):
                try:
                    if entry.snapshot_path.is_dir():
                        shutil.rmtree(entry.snapshot_path)
                    else:
                        entry.snapshot_path.unlink(missing_ok=True)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Sampling + updates
    # ------------------------------------------------------------------

    def sample_opponent(self, scheme: str | None = None) -> OpponentEntry:
        """Sample the next opponent using either PFSP or uniform."""
        if not self.pool:
            raise RuntimeError("League pool is empty")
        scheme = scheme or self.config.sampling
        if scheme == "uniform":
            return self.pool[int(self.rng.integers(0, len(self.pool)))]
        idx = self._pfsp.sample_index(self.learner_rating, [entry.rating for entry in self.pool])
        return self.pool[idx]

    def record_match(self, opponent_name: str, win: bool) -> Match:
        """Update Elo ratings and match stats from a single completed game."""
        entry = self.get(opponent_name)
        lr_before = self.learner_rating
        or_before = entry.rating
        score_a = 1.0 if win else 0.0
        lr_after, or_after = elo_update(lr_before, or_before, score_a, k=self.config.elo_k)
        self.learner_rating = lr_after
        entry.rating = or_after
        entry.n_games += 1
        if win:
            entry.n_wins += 1

        match = Match(
            opponent=opponent_name,
            win=win,
            learner_rating_before=lr_before,
            opponent_rating_before=or_before,
            learner_rating_after=lr_after,
            opponent_rating_after=or_after,
        )
        self.match_history.append(match)
        return match

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, name: str) -> OpponentEntry:
        """Return the opponent entry with the given name."""
        for entry in self.pool:
            if entry.name == name:
                return entry
        raise KeyError(f"No opponent named {name!r} in league")

    def standings(self) -> list[dict[str, tp.Any]]:
        """Return a sorted list of opponent summaries for logging."""
        rows: list[dict[str, tp.Any]] = []
        for entry in sorted(self.pool, key=lambda e: e.rating, reverse=True):
            rows.append(
                {
                    "name": entry.name,
                    "deck": entry.deck,
                    "rating": round(entry.rating, 1),
                    "games": entry.n_games,
                    "win_rate": round(entry.win_rate, 3),
                    "historical": entry.is_historical,
                }
            )
        rows.insert(
            0,
            {
                "name": "[learner]",
                "deck": "-",
                "rating": round(self.learner_rating, 1),
                "games": len(self.match_history),
                "win_rate": (
                    sum(1 for m in self.match_history if m.win) / max(len(self.match_history), 1)
                ),
                "historical": False,
            },
        )
        return rows


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def snapshot_policy(
    league: League,
    agent: tp.Any,
    deck: str,
    name: str | None = None,
) -> OpponentEntry:
    """Convenience wrapper that names snapshots ``snapshot_{N}`` by default."""
    if name is None:
        n = sum(1 for e in league.pool if e.is_historical)
        name = f"snapshot_{n:03d}"
    return league.add_snapshot(name=name, agent=agent, deck=deck)


def _default_factory(agent_cls: type) -> tp.Callable[[Path], tp.Any]:
    """Return a factory that re-hydrates ``agent_cls`` from a snapshot path."""

    def _factory(path: Path) -> tp.Any:
        if hasattr(agent_cls, "load") and callable(agent_cls.load):
            return agent_cls.load(str(path))
        instance = agent_cls()  # type: ignore[call-arg]
        if hasattr(instance, "load"):
            instance.load(str(path))
        return instance

    return _factory
