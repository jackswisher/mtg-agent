"""Causal variable definitions for the MTG SCM.

This module defines the high-level causal variables that capture
strategic aspects of MTG gameplay, as described in the paper:
- Mana: Available mana production
- CardAdvantage: Relative card advantage
- BoardPressure: Board threat level
- Tempo: Initiative and timing
- LifeBuffer: Damage tolerance
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VariableType(Enum):
    """Types of causal variables."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BINARY = "binary"


@dataclass
class CausalVariable:
    """A causal variable in the SCM.

    Attributes:
        name: Variable name (identifier).
        display_name: Human-readable name.
        var_type: Type of variable.
        domain: Valid value range (min, max) for continuous, values for discrete.
        description: Description of what this variable represents.
        parents: Names of parent variables in the causal graph.
        compute_fn: Function to compute value from game state.
    """

    name: str
    display_name: str
    var_type: VariableType
    domain: tuple[float, float] | list[Any]
    description: str = ""
    parents: list[str] = field(default_factory=list)
    compute_fn: Callable[..., float] | None = None

    def __hash__(self) -> int:
        """Return hash based on variable name."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Check equality based on variable name."""
        if not isinstance(other, CausalVariable):
            return NotImplemented
        return self.name == other.name


@dataclass
class CausalVariableSet:
    """Collection of causal variables for the MTG SCM."""

    variables: dict[str, CausalVariable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize with default MTG causal variables."""
        if not self.variables:
            self._initialize_default_variables()

    def _initialize_default_variables(self) -> None:
        """Create the default set of MTG causal variables."""
        # Opening Hand variables
        self.add_variable(
            CausalVariable(
                name="opening_hand",
                display_name="Opening Hand",
                var_type=VariableType.DISCRETE,
                domain=list(range(8)),  # 0-7 cards
                description="Initial hand after mulligan decisions.",
                parents=[],
            )
        )

        self.add_variable(
            CausalVariable(
                name="mulligan_decision",
                display_name="Mulligan Decision",
                var_type=VariableType.BINARY,
                domain=(0, 1),
                description="Whether to keep (1) or mulligan (0).",
                parents=["opening_hand"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="hand_quality",
                display_name="Hand Quality",
                var_type=VariableType.CONTINUOUS,
                domain=(0.0, 1.0),
                description="Quality score of current hand.",
                parents=["mulligan_decision"],
            )
        )

        # Mana variables
        self.add_variable(
            CausalVariable(
                name="mana_t",
                display_name="Mana (Current Turn)",
                var_type=VariableType.DISCRETE,
                domain=list(range(11)),  # 0-10 mana
                description="Available mana on current turn.",
                parents=["land_drop", "mana_creatures"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="mana_t_plus_1",
                display_name="Mana (Next Turn)",
                var_type=VariableType.DISCRETE,
                domain=list(range(11)),
                description="Expected available mana next turn.",
                parents=["mana_t", "land_in_hand"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="land_drop",
                display_name="Land Drop",
                var_type=VariableType.BINARY,
                domain=(0, 1),
                description="Whether a land was played this turn.",
                parents=["land_in_hand"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="land_in_hand",
                display_name="Lands in Hand",
                var_type=VariableType.DISCRETE,
                domain=list(range(8)),
                description="Number of lands in hand.",
                parents=["opening_hand", "cards_drawn"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="mana_creatures",
                display_name="Mana Creatures",
                var_type=VariableType.DISCRETE,
                domain=list(range(5)),
                description="Mana-producing creatures in play.",
                parents=["spells_cast"],
            )
        )

        # Card advantage variables
        self.add_variable(
            CausalVariable(
                name="card_advantage",
                display_name="Card Advantage",
                var_type=VariableType.CONTINUOUS,
                domain=(-10.0, 10.0),
                description="Relative card advantage over opponent.",
                parents=["cards_drawn", "cards_played", "opponent_cards"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="cards_drawn",
                display_name="Cards Drawn",
                var_type=VariableType.DISCRETE,
                domain=list(range(20)),
                description="Total cards drawn this game.",
                parents=["draw_spells", "turn_draws"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="cards_played",
                display_name="Cards Played",
                var_type=VariableType.DISCRETE,
                domain=list(range(20)),
                description="Total cards played this game.",
                parents=["mana_t", "spells_in_hand"],
            )
        )

        # Board pressure variables
        self.add_variable(
            CausalVariable(
                name="board_pressure",
                display_name="Board Pressure",
                var_type=VariableType.CONTINUOUS,
                domain=(-20.0, 20.0),
                description="Net board presence (power difference).",
                parents=["own_creatures", "opponent_creatures"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="own_creatures",
                display_name="Own Creatures",
                var_type=VariableType.DISCRETE,
                domain=list(range(10)),
                description="Number of creatures controlled.",
                parents=["spells_cast", "creatures_died"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="opponent_creatures",
                display_name="Opponent Creatures",
                var_type=VariableType.DISCRETE,
                domain=list(range(10)),
                description="Number of opponent creatures.",
                parents=[],  # Partially observable
            )
        )

        # Tempo variables
        self.add_variable(
            CausalVariable(
                name="tempo",
                display_name="Tempo",
                var_type=VariableType.CONTINUOUS,
                domain=(-1.0, 1.0),
                description="Initiative and timing advantage.",
                parents=["mana_efficiency", "board_pressure", "threats_deployed"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="mana_efficiency",
                display_name="Mana Efficiency",
                var_type=VariableType.CONTINUOUS,
                domain=(0.0, 1.0),
                description="Mana spent vs mana available ratio.",
                parents=["mana_spent", "mana_t"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="threats_deployed",
                display_name="Threats Deployed",
                var_type=VariableType.DISCRETE,
                domain=list(range(10)),
                description="Number of threat cards played.",
                parents=["spells_cast"],
            )
        )

        # Life buffer variables
        self.add_variable(
            CausalVariable(
                name="life_buffer",
                display_name="Life Buffer",
                var_type=VariableType.CONTINUOUS,
                domain=(-20.0, 40.0),
                description="Life total safety margin.",
                parents=["own_life", "opponent_board_damage"],
            )
        )

        self.add_variable(
            CausalVariable(
                name="own_life",
                display_name="Own Life",
                var_type=VariableType.DISCRETE,
                domain=list(range(41)),  # 0-40 life
                description="Current life total.",
                parents=["damage_taken", "life_gained"],
            )
        )

        # Outcome variable
        self.add_variable(
            CausalVariable(
                name="win_probability",
                display_name="Win Probability",
                var_type=VariableType.CONTINUOUS,
                domain=(0.0, 1.0),
                description="Estimated probability of winning.",
                parents=["card_advantage", "board_pressure", "tempo", "life_buffer"],
            )
        )

    def add_variable(self, var: CausalVariable) -> None:
        """Add a variable to the set."""
        self.variables[var.name] = var

    def get_variable(self, name: str) -> CausalVariable:
        """Get a variable by name."""
        if name not in self.variables:
            raise KeyError(f"Unknown causal variable: {name}")
        return self.variables[name]

    def get_parents(self, name: str) -> list[str]:
        """Get parent variable names for a given variable."""
        return self.get_variable(name).parents

    def get_children(self, name: str) -> list[str]:
        """Get child variable names for a given variable."""
        children = []
        for var_name, var in self.variables.items():
            if name in var.parents:
                children.append(var_name)
        return children

    def topological_order(self) -> list[str]:
        """Return variables in topological order (parents before children)."""
        # Kahn's algorithm
        in_degree = {name: len(var.parents) for name, var in self.variables.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order

    def get_all_names(self) -> list[str]:
        """Get all variable names."""
        return list(self.variables.keys())

    def __len__(self) -> int:
        """Return the number of variables in the set."""
        return len(self.variables)

    def __iter__(self):
        """Iterate over the variables in the set."""
        return iter(self.variables.values())


def compute_variable_from_state(
    var_name: str,
    game_state: Any,
    player_id: int = 0,
) -> float:
    """Compute a causal variable value from game state.

    Args:
        var_name: Name of the causal variable.
        game_state: Current game state.
        player_id: Player perspective.

    Returns:
        Variable value.
    """
    from mtg.env.reward import RewardCalculator

    # Use reward calculator's causal variable extraction
    calculator = RewardCalculator()
    cv = calculator.get_causal_variable_values(game_state, player_id)

    # Map internal names to SCM names
    mapping = {
        "mana_t": "mana",
        "card_advantage": "card_advantage",
        "board_pressure": "board_pressure",
        "tempo": "tempo",
        "life_buffer": "life_buffer",
    }

    if var_name in mapping and mapping[var_name] in cv:
        return cv[mapping[var_name]]

    # Handle other variables with defaults
    player = game_state.players[player_id]

    if var_name == "own_life":
        return float(player.life)
    elif var_name == "land_in_hand":
        from mtg.env.card_definitions import CardType

        return float(sum(1 for c in player.hand if c.card_type == CardType.LAND))
    elif var_name == "own_creatures":
        from mtg.env.card_definitions import CardType

        return float(sum(1 for c in player.battlefield if c.card_type == CardType.CREATURE))

    return 0.0
