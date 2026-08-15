"""Control heuristic agent."""

from __future__ import annotations

import typing as tp

from mtg.agents.heuristics.heuristic_base_agent import (
    HeuristicBaseAgent,
    HeuristicStyle,
    MulliganProfile,
)
from mtg.env.action_mask import ActionKind
from mtg.env.card_definitions import Card, CardType, Keyword


class ControlAgent(HeuristicBaseAgent):
    """Control-oriented heuristic agent."""

    def __init__(self, seed: int | None = None) -> None:
        style = HeuristicStyle(
            aggression=0.35,
            defense=0.95,
            value=0.9,
            development=0.7,
            hold_up=0.75,
            curve=0.55,
            land_priority=0.8,
            risk_tolerance=0.4,
        )
        # Control needs: playable early game + actual sorcery-speed removal
        # (counterspells can't save you in the first 3 turns vs aggro).
        mulligan = MulliganProfile(
            min_lands=2,
            max_lands=4,
            keep_one_land_with_early=False,
            min_early_spells=1,
            min_interaction=1,
            max_top_end_cards=2,
            max_hand_cmc_avg=4.0,
        )
        super().__init__(
            name="ControlAgent",
            seed=seed,
            style=style,
            mulligan_profile=mulligan,
            low_life_threshold=8,
        )

    def _mulligan_decision(
        self,
        hand_size: int,
        legal,
        info: dict[str, tp.Any],
    ) -> int:
        """Control-specific mulligan: require real (non-counter) interaction.

        Counterspells look like "interaction" to the base agent, but in this
        environment they are often unusable (opponent creatures resolve without
        a priority window, priority-window mana is scarce). Control wins only
        when it can actually kill or slow early threats.
        """
        hand_cards = self._get_hand_cards(info)
        lands = info.get("lands", 0)
        non_counter_interaction = sum(
            1 for c in hand_cards if (c.is_removal or c.deals_damage > 0) and not c.is_counterspell
        )
        mull_action = self._first_action(legal, info, ActionKind.MULLIGAN.value)

        # Demand at least ONE sorcery-speed / instant-speed-but-not-counter
        # answer in opening hands of 6 or 7. Only relax for 5-or-fewer.
        if (
            hand_size >= 6
            and lands >= 2
            and non_counter_interaction == 0
            and mull_action is not None
        ):
            return mull_action

        # Otherwise fall through to the base heuristic.
        return super()._mulligan_decision(hand_size, legal, info)

    def _score_card(self, card: Card, info: dict[str, tp.Any]) -> float:
        score = 0.0
        cmc = card.mana_cost.cmc

        if card.is_counterspell:
            score += 7.0
        if card.is_removal:
            score += 6.0
        if card.draws_cards:
            score += 4.0 + 1.4 * card.draws_cards
        if card.gains_life > 0:
            score += 2.0 + 0.6 * card.gains_life
        if card.card_type == CardType.CREATURE:
            score += 1.5 + 0.5 * card.toughness + 0.2 * card.power
            if Keyword.FLASH in card.keywords:
                score += 1.0
            if Keyword.LIFELINK in card.keywords:
                score += 1.2
        if card.deals_damage and card.can_target_any:
            score += 0.4 * card.deals_damage

        if cmc >= 4 and card.card_type == CardType.CREATURE and card.power >= 3:
            score += 1.5  # finishers

        score -= 0.18 * cmc
        return score

    def _has_counterspell(self, context: dict[str, tp.Any]) -> bool:
        """Check if we have a counterspell in hand."""
        return any(c.is_counterspell for c in context.get("hand_cards", []))

    def _has_instant_draw(self, context: dict[str, tp.Any]) -> bool:
        """Check if we have instant-speed card draw in hand."""
        return any(
            c.draws_cards > 0 and c.card_type == CardType.INSTANT
            for c in context.get("hand_cards", [])
        )

    def _cheapest_counterspell_cmc(self, context: dict[str, tp.Any]) -> int:
        """Return the CMC of the cheapest counterspell in hand, or 99 if none."""
        counterspells = [c for c in context.get("hand_cards", []) if c.is_counterspell]
        if not counterspells:
            return 99
        return min(c.mana_cost.cmc for c in counterspells)

    def _cheapest_interaction_cmc(self, context: dict[str, tp.Any]) -> int:
        """Return the CMC of the cheapest interaction (counterspell/removal) in hand."""
        interaction = [
            c for c in context.get("hand_cards", []) if c.is_counterspell or c.is_removal
        ]
        if not interaction:
            return 99
        return min(c.mana_cost.cmc for c in interaction)

    def _can_hold_up_interaction_after(self, card: Card, context: dict[str, tp.Any]) -> bool:
        """Check if we can still hold up interaction after casting this card."""
        mana_available = context.get("mana_available", 0)
        mana_after_cast = mana_available - card.mana_cost.cmc
        cheapest_interaction = self._cheapest_interaction_cmc(context)
        return mana_after_cast >= cheapest_interaction

    def _score_cast_spell(
        self,
        action: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        """Score spell casting with Control-specific timing logic.

        Key pro-play principles for control:

        1. Hold up interaction, but not at the cost of never winning.
           Control decks lose games they never close. Commit to threats
           ("slam") when one is available and the window is safe.

        2. Commit a threat (creature / finisher) when ANY of:
           - opponent has >=2 creatures on board (race is real)
           - our hand is at/over discard threshold (cards rotting)
           - turn >= 7 (late game, clock is ticking vs the 20-turn timeout)
           - opponent is tapped out (they can't protect / counter)
           - opponent has no creatures and we have an empty board (develop)

        3. Hold up interaction in main phase only when we have it AND none
           of the "commit" triggers fire. This prevents infinite stalling.

        4. Card-draw instants strongly prefer end step, as before.
        """
        score = super()._score_cast_spell(action, info, context)
        card = self._get_card_from_action(action, info)
        if card is None:
            return score

        phase = info.get("phase_enum", "")
        is_active = self._is_active_player(info)
        stack_empty = info.get("stack_size", 0) == 0
        turn = self._current_turn(info)

        in_main_phase = is_active and phase in {"MAIN_PRECOMBAT", "MAIN_POSTCOMBAT"} and stack_empty

        if in_main_phase:
            has_counterspell = self._has_counterspell(context)
            can_hold_up_after = self._can_hold_up_interaction_after(card, context)

            # "Must commit" triggers: deploy the threat NOW. Control
            # decks that never close lose to the turn cap.
            opp_creatures = len(context.get("opponent_creatures", []))
            own_creatures = len(context.get("player_creatures", []))
            hand_size = context.get("hand_size", 0)
            opp_mana = info.get("opponent_mana_available", 0)
            must_commit = (
                opp_creatures >= 1  # opponent has any board presence
                or hand_size >= 6  # hand is full, cards rotting
                or turn >= 6  # start closing by turn 6
                or opp_mana <= 1  # opponent can't respond
                or (
                    own_creatures == 0 and card.card_type == CardType.CREATURE and card.power >= 2
                )  # we have no board and a threat to deploy
            )

            # Non-instant spells (creatures, sorceries, planeswalkers).
            if card.card_type != CardType.INSTANT:
                if must_commit:
                    # Large bonus to deploy the win condition. Counterspells
                    # score ~7; without this, creatures score ~4 and lose.
                    score += 3.0
                    if (
                        card.card_type == CardType.CREATURE
                        and card.mana_cost.cmc >= 4
                        and card.power >= 3
                    ):
                        score += 2.0
                    # Planeswalkers are premier win conditions for control.
                    if card.card_type == CardType.PLANESWALKER:
                        score += 2.5
                elif has_counterspell and not can_hold_up_after:
                    # Tapping out leaves the deck vulnerable to 1 counter:
                    # mild penalty, but do not block the play.
                    score -= 1.0
                elif can_hold_up_after:
                    score += 0.5

            # Non-reactive instants (e.g. card draw).
            elif not card.is_counterspell and not card.is_removal:
                if can_hold_up_after:
                    score -= 0.5  # Wait for end step when we can.
                else:
                    player_life = info.get("player_life", 20)
                    opp_power = context.get("opponent_power", 0)
                    is_desperate = player_life <= 8 or opp_power >= player_life or must_commit
                    if is_desperate:
                        score -= 0.3
                    else:
                        return float("-inf")

        # Prefer end-step instant draw to maximize information.
        if (
            is_active
            and phase == "END_STEP"
            and card.card_type == CardType.INSTANT
            and card.draws_cards > 0
        ):
            score += 1.5

        score = self._score_priority_removal_cast(score, card, info, context)
        return score

    def _score_pass(self, info: dict[str, tp.Any], context: dict[str, tp.Any]) -> float:
        """Score passing with Control-specific hold-up logic.

        Holding up counterspells is good strategy, but only when:
        - Early in the game (turns 1-6) where opportunity cost is low.
        - Opponent has gas in hand / mana to deploy threats worth countering.

        Late game (turn >= 7), stalling means losing to the turn cap.
        """
        score = super()._score_pass(info, context)
        turn = self._current_turn(info)

        is_active_main = (
            self._is_active_player(info)
            and info.get("phase_enum") in {"MAIN_PRECOMBAT", "MAIN_POSTCOMBAT"}
            and info.get("stack_size", 0) == 0
        )
        if is_active_main and self._has_counterspell(context):
            # Reward hold-up early; penalise it once turns are scarce
            # because threats in hand have to land eventually.
            if turn <= 5:
                score += 1.0
            elif turn <= 8:
                score += 0.3
            else:
                # Past turn 8, holding up interaction costs us the game.
                score -= 0.6
        # End step: prefer casting instant draw instead of passing.
        if (
            self._is_active_player(info)
            and info.get("phase_enum") == "END_STEP"
            and self._has_instant_draw(context)
        ):
            score -= 0.8
        return score

    def _score_target(
        self,
        slot: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        """Score targeting with Control-specific threat prioritisation.

        Control wins by trading 1-for-1 with the opponent's biggest threats.
        Wasting a removal spell on a 1/1 utility creature when an opposing
        4/4 is on the board is a textbook control mistake. This scoring
        function biases hard toward the priority target on the board:

        * Planeswalkers: massive bonus. A planeswalker left alone
          frequently wins the game by itself ("must answer" cards in
          MTG terminology).
        * Evasive damage (flying / haste): boosted because they bypass
          the ground board state.
        * Lifelink: boosted because it inverts the race math.
        * Power tiers: scale the bonus by how big the threat is so a
          5/5 ranks strictly above a 3/3 ranks above a 2/2.
        * Single-removal conservation: when only one piece of removal
          is in hand, the opportunity cost of mis-targeting is much
          higher, so the priority-tier delta is amplified further.
        """
        score = super()._score_target(slot, info, context)
        candidates = info.get("pending_target_candidates", [])
        if slot < 0 or slot >= len(candidates):
            return score
        target = candidates[slot]
        spell_name = info.get("pending_spell_name")
        if not spell_name:
            return score
        try:
            card = self.card_registry.get(spell_name)
        except KeyError:
            return score

        target_kind = target.get("kind")
        is_removal_like = card.is_removal or card.deals_damage > 0

        # Planeswalker prioritisation. Untouched planeswalkers are
        # treated as a 5-alarm fire; the base agent already gives some
        # value to PW removal and this stacks on top so PWs always
        # outrank creatures in the same target prompt.
        if target_kind == "permanent" and card.is_removal:
            try:
                target_card = self.card_registry.get(target.get("name", ""))
            except KeyError:
                target_card = None
            if target_card and target_card.card_type == CardType.PLANESWALKER:
                score += 5.0

        # Creature prioritisation.
        if target_kind == "creature" and is_removal_like:
            try:
                target_card = self.card_registry.get(target.get("name", ""))
            except KeyError:
                target_card = None
            if target_card is not None:
                # When few removal spells remain in hand, mis-targeting
                # hurts more, so the priority gap is amplified.
                interaction_in_hand = sum(
                    1
                    for c in context.get("hand_cards", [])
                    if c.is_removal or (c.deals_damage > 0 and c.can_target_any)
                )
                scarcity_mult = 1.6 if interaction_in_hand <= 1 else 1.0

                # Evasive bodies dominate the air, lifelink inverts the
                # race, haste lets the threat connect immediately.
                if Keyword.FLYING in target_card.keywords:
                    score += 1.6 * scarcity_mult
                if Keyword.LIFELINK in target_card.keywords:
                    score += 1.8 * scarcity_mult
                if Keyword.HASTE in target_card.keywords:
                    score += 1.2 * scarcity_mult
                if Keyword.TRAMPLE in target_card.keywords:
                    score += 0.8 * scarcity_mult

                # Finer-grained power tiers than the base agent's flat
                # "power >= 3 -> +0.8" so bigger threats strictly outrank
                # smaller threats in the same prompt.
                power = int(getattr(target_card, "power", 0) or 0)
                if power >= 5:
                    score += 2.0 * scarcity_mult
                elif power >= 4:
                    score += 1.4 * scarcity_mult
                elif power >= 3:
                    score += 0.9 * scarcity_mult
                elif power >= 2:
                    score += 0.4 * scarcity_mult

                # Penalise wasting removal on a 1/X utility creature
                # when the opponent has any larger creature on the
                # board (the canonical "do not respond to bait" play).
                if power <= 1:
                    opp_creatures = context.get("opponent_creatures", []) or []
                    bigger = max(
                        (int(c.get("power", 0) or 0) for c in opp_creatures),
                        default=0,
                    )
                    if bigger >= 3:
                        score -= 1.5 * scarcity_mult

        return score

    def _score_priority_removal_cast(
        self,
        score: float,
        card: Card,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        """Add a "kill the priority threat" bonus to removal casts.

        The bonus fires when:

        * the agent is casting a removal or burn spell, AND
        * the opponent controls at least one creature with power >= 3 OR
          any planeswalker.

        This prevents the agent from passing priority instead of casting
        sorcery-speed removal in its main phase (and letting an opposing
        4/4 connect another turn). The bonus is intentionally moderate
        (+1.5 in main, +2.5 if a planeswalker exists) so it does not
        override clearly bad casts such as tapping out into a known
        counter-window.
        """
        if score == float("-inf"):
            return score
        if not (card.is_removal or card.deals_damage > 0):
            return score

        phase = info.get("phase_enum", "")
        is_active = self._is_active_player(info)
        if not (is_active and phase in {"MAIN_PRECOMBAT", "MAIN_POSTCOMBAT"}):
            return score
        if info.get("stack_size", 0) > 0:
            return score

        opp_creatures = context.get("opponent_creatures", []) or []
        biggest_opp_power = max(
            (int(c.get("power", 0) or 0) for c in opp_creatures),
            default=0,
        )
        opp_perms = (
            info.get("opponent_permanents", []) or info.get("opponent_battlefield", []) or []
        )
        has_opp_planeswalker = False
        for entry in opp_perms:
            type_str: str | None = None
            if isinstance(entry, dict):
                type_str = str(entry.get("card_type") or entry.get("type") or "")
            elif isinstance(entry, list | tuple) and len(entry) >= 3:
                type_str = str(entry[2])
            if type_str and "planeswalker" in type_str.lower():
                has_opp_planeswalker = True
                break

        if has_opp_planeswalker and card.is_removal:
            # Planeswalkers ignore the combat board entirely; always answer.
            score += 2.5
        elif biggest_opp_power >= 3:
            # Cast removal now rather than letting the threat untap.
            score += 1.5
        elif biggest_opp_power >= 2:
            score += 0.5
        return score
