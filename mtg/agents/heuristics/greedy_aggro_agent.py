"""Greedy Aggro agent implementing domain knowledge."""

from __future__ import annotations

import typing as tp

from mtg.agents.heuristics.heuristic_base_agent import (
    HeuristicBaseAgent,
    HeuristicStyle,
    MulliganProfile,
)
from mtg.env.card_definitions import Card, CardType, Keyword, TriggerEffect


class GreedyAggroAgent(HeuristicBaseAgent):
    """Aggressive heuristic agent tuned for low-curve decks."""

    def __init__(
        self,
        aggression: float = 0.9,
        defensive_threshold: int = 5,
        seed: int | None = None,
    ) -> None:
        style = HeuristicStyle(
            aggression=aggression,
            defense=0.35,
            value=0.3,
            development=0.75,
            hold_up=0.15,
            curve=0.65,
            land_priority=0.7,
            risk_tolerance=0.85,
        )
        # Aggro wants low CMC openers: 2+ early plays and no more than 1
        # expensive card stranded in hand.
        mulligan = MulliganProfile(
            min_lands=2,
            max_lands=4,
            keep_one_land_with_early=True,
            min_early_spells=2,
            min_interaction=0,
            max_top_end_cards=1,
            max_hand_cmc_avg=3.0,
        )
        super().__init__(
            name="Greedy Aggro",
            seed=seed,
            style=style,
            mulligan_profile=mulligan,
            low_life_threshold=defensive_threshold,
        )

    def _score_card(self, card: Card, info: dict[str, tp.Any]) -> float:
        score = 0.0
        cmc = card.mana_cost.cmc

        if card.card_type == CardType.CREATURE:
            score += 4.0 + 1.1 * card.power + 0.5 * card.toughness
            if Keyword.HASTE in card.keywords:
                score += 2.0
            if Keyword.FLYING in card.keywords:
                score += 1.2
            if Keyword.TRAMPLE in card.keywords:
                score += 0.8

        if card.is_pump_spell:
            score += 2.5

        if card.deals_damage > 0:
            if card.can_target_any:
                score += 3.0 + 0.8 * card.deals_damage
            else:
                score += 2.0 + 0.6 * card.deals_damage

        if card.is_removal:
            score += 2.0

        if card.draws_cards > 0:
            score += 0.6 * card.draws_cards

        token_triggers = sum(1 for t in card.triggers if t.effect == TriggerEffect.CREATE_TOKEN)
        if token_triggers:
            score += 1.5 * token_triggers

        score -= 0.45 * cmc
        return score

    def _score_target(
        self,
        slot: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        """Greedy Aggro targeting: use base pro-play heuristics.

        The base _score_target now handles board-state-aware targeting.
        Aggro agents should still slightly prefer face damage when even,
        but respect the board-state logic.
        """
        score = super()._score_target(slot, info, context)
        candidates = info.get("pending_target_candidates", [])
        if slot < 0 or slot >= len(candidates):
            return score
        target = candidates[slot]

        # Aggro bonus: slight preference for face when not behind
        if target.get("kind") == "player" and target.get("name") == "Opponent":
            my_creatures = len(context.get("player_creatures", []))
            opp_creatures = len(context.get("opponent_creatures", []))
            # If we have more creatures, we're racing - slight face bonus
            if my_creatures > opp_creatures:
                score += 0.5 * self.style.aggression

        return score
