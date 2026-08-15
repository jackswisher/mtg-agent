"""Midrange heuristic agent."""

from __future__ import annotations

import typing as tp

from mtg.agents.heuristics.heuristic_base_agent import (
    HeuristicBaseAgent,
    HeuristicStyle,
    MulliganProfile,
)
from mtg.env.card_definitions import Card, CardType, Keyword


class MidrangeAgent(HeuristicBaseAgent):
    """Balanced midrange heuristic agent."""

    def __init__(self, seed: int | None = None) -> None:
        style = HeuristicStyle(
            aggression=0.65,
            defense=0.6,
            value=0.7,
            development=0.75,
            hold_up=0.5,
            curve=0.6,
            land_priority=0.75,
            risk_tolerance=0.6,
        )
        mulligan = MulliganProfile(
            min_lands=2,
            max_lands=5,
            keep_one_land_with_early=False,
            min_early_spells=1,
            min_interaction=1,
            max_top_end_cards=3,
            max_hand_cmc_avg=4.5,
        )
        super().__init__(
            name="MidrangeAgent",
            seed=seed,
            style=style,
            mulligan_profile=mulligan,
            low_life_threshold=8,
        )

    def _score_card(self, card: Card, info: dict[str, tp.Any]) -> float:
        score = 0.0
        cmc = card.mana_cost.cmc
        turn = self._current_turn(info)

        if card.card_type == CardType.CREATURE:
            score += 3.2 + 0.6 * card.power + 0.6 * card.toughness
            if Keyword.TRAMPLE in card.keywords:
                score += 0.6
            if Keyword.FLYING in card.keywords:
                score += 0.8
            # Prefer 2-4 drops early (midrange curve).
            if turn <= 4 and 2 <= cmc <= 4:
                score += 0.6

        if card.is_removal:
            score += 3.5

        if card.draws_cards:
            score += 2.2 + 0.7 * card.draws_cards

        if card.deals_damage > 0:
            score += 1.1 + 0.3 * card.deals_damage

        if card.is_pump_spell:
            score += 0.7

        score -= 0.2 * cmc
        return score

    def _current_aggression(self, info: dict[str, tp.Any]) -> float:
        aggression = super()._current_aggression(info)
        # Adjust role based on board power advantage.
        player_power = sum(
            c[1] if isinstance(c, list | tuple) and len(c) > 1 else c.get("power", 0)
            for c in info.get("player_creatures", [])
        )
        opponent_power = sum(
            c[1] if isinstance(c, list | tuple) and len(c) > 1 else c.get("power", 0)
            for c in info.get("opponent_creatures", [])
        )
        if player_power >= opponent_power + 2:
            aggression += 0.2
        elif opponent_power >= player_power + 2:
            aggression -= 0.2
        return max(0.1, min(1.2, aggression))
