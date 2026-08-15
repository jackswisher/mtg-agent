"""Ramp heuristic agent."""

from __future__ import annotations

import typing as tp

from mtg.agents.heuristics.heuristic_base_agent import (
    HeuristicBaseAgent,
    HeuristicStyle,
    MulliganProfile,
)
from mtg.env.card_definitions import Card, CardType, Keyword


class RampAgent(HeuristicBaseAgent):
    """Ramp-focused heuristic agent."""

    def __init__(self, seed: int | None = None) -> None:
        style = HeuristicStyle(
            aggression=0.45,
            defense=0.75,
            value=0.65,
            development=0.9,
            hold_up=0.35,
            curve=0.8,
            land_priority=1.0,
            risk_tolerance=0.5,
        )
        # Ramp needs some interaction to survive aggro early. Banning hands
        # that are all-finisher was a blind spot before.
        mulligan = MulliganProfile(
            min_lands=3,
            max_lands=5,
            keep_one_land_with_early=False,
            min_early_spells=1,
            min_interaction=0,
            max_top_end_cards=3,
            max_hand_cmc_avg=5.0,
        )
        super().__init__(
            name="RampAgent",
            seed=seed,
            style=style,
            mulligan_profile=mulligan,
            low_life_threshold=8,
        )

    def _score_card(self, card: Card, info: dict[str, tp.Any]) -> float:
        score = 0.0
        cmc = card.mana_cost.cmc
        # Turn 1-3: early blockers keep us alive until payoffs come online.
        turn = self._current_turn(info)
        early_game = turn <= 4

        if card.card_type == CardType.CREATURE:
            score += 2.3 + 0.5 * card.power + 0.6 * card.toughness
            if cmc >= 5:
                score += 2.5
            if Keyword.TRAMPLE in card.keywords:
                score += 1.0
            # Early defensive bodies: big toughness for cheap is priceless.
            if early_game and cmc <= 3 and card.toughness >= 3:
                score += 1.5
            # Reach / flyers for defense against aggressive flyers.
            if early_game and (Keyword.REACH in card.keywords or Keyword.FLYING in card.keywords):
                score += 0.7

        if card.draws_cards:
            score += 1.6 + 0.6 * card.draws_cards

        if card.is_removal:
            # Removal matters more to ramp than it looks; it is how
            # the deck survives turns 3-5 against aggro.
            score += 2.5 if early_game else 1.5

        if cmc >= 6:
            score += 2.0

        score -= 0.08 * cmc
        return score

    def _current_aggression(self, info: dict[str, tp.Any]) -> float:
        aggression = super()._current_aggression(info)
        mana_available = info.get("mana_available", 0)
        if mana_available < 4:
            aggression *= 0.6
        return aggression

    def _should_prioritize_creature(self, card: Card, context: dict[str, tp.Any]) -> bool:
        """Prioritize early bodies for defense and late-game payoffs."""
        # Turn-1/2 cheap defenders or ramp creatures.
        if card.mana_cost.cmc <= 2:
            return True
        # Early defensive bodies (big toughness) are worth tapping out for.
        if card.mana_cost.cmc <= 3 and card.toughness >= 3:
            return True
        # Late game: finishers.
        return context.get("mana_available", 0) >= 5 and card.mana_cost.cmc >= 5
