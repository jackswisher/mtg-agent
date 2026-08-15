"""Convoke aggro heuristic agent."""

from __future__ import annotations

import typing as tp

from mtg.agents.heuristics.heuristic_base_agent import (
    HeuristicBaseAgent,
    HeuristicStyle,
    MulliganProfile,
)
from mtg.env.card_definitions import Card, CardType, Keyword, TriggerEffect


class ConvokeAggroAgent(HeuristicBaseAgent):
    """Aggro agent that values tokens and cheap creatures."""

    def __init__(self, seed: int | None = None) -> None:
        style = HeuristicStyle(
            aggression=0.85,
            defense=0.45,
            value=0.45,
            development=0.85,
            hold_up=0.2,
            curve=0.65,
            land_priority=0.7,
            risk_tolerance=0.75,
        )
        mulligan = MulliganProfile(
            min_lands=2,
            max_lands=4,
            keep_one_land_with_early=True,
            min_early_spells=2,
            min_interaction=0,
            max_top_end_cards=1,
            max_hand_cmc_avg=3.2,
        )
        super().__init__(
            name="ConvokeAggroAgent",
            seed=seed,
            style=style,
            mulligan_profile=mulligan,
            low_life_threshold=6,
        )

    def _score_card(self, card: Card, info: dict[str, tp.Any]) -> float:
        score = 0.0
        cmc = card.mana_cost.cmc

        if card.card_type == CardType.CREATURE:
            score += 4.2 + 0.7 * card.power + 0.6 * card.toughness
            if Keyword.HASTE in card.keywords:
                score += 1.5

        token_triggers = sum(1 for t in card.triggers if t.effect == TriggerEffect.CREATE_TOKEN)
        if token_triggers:
            score += 3.2 * token_triggers

        if card.is_pump_spell:
            score += 2.0
        if card.deals_damage:
            score += 1.2 + 0.3 * card.deals_damage

        if cmc <= 2:
            score += 0.6

        score -= 0.3 * cmc
        return score

    def _current_aggression(self, info: dict[str, tp.Any]) -> float:
        aggression = super()._current_aggression(info)
        if len(info.get("player_creatures", [])) >= 3:
            aggression += 0.2
        return min(1.2, aggression)

    def _should_prioritize_creature(self, card: Card, context: dict[str, tp.Any]) -> bool:
        token_triggers = sum(1 for t in card.triggers if t.effect == TriggerEffect.CREATE_TOKEN)
        if token_triggers:
            return True
        return card.mana_cost.cmc <= 2

    def _score_attack_confirm(self, info: dict[str, tp.Any], context: dict[str, tp.Any]) -> float:
        score = super()._score_attack_confirm(info, context)
        attackers = info.get("attack_candidates", [])
        blockers = info.get("opponent_blockers_available", 0)
        if len(attackers) >= 3 and blockers < len(attackers):
            score += 0.8 * self.style.aggression
        return score
