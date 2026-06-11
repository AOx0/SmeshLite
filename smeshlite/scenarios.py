"""
Scenario System -- named game-state presets for targeted training.

Scenarios are Match.get_state()-compatible dicts with metadata (tags,
difficulty). They can be loaded via env.reset(options={"scenario": "name"})
to start the match from a specific state instead of default spawn.

Built-in scenarios cover the most important fighting game situations:
high damage, edge guarding, recovery, combo follow-ups, etc.

Curriculum logic and scenario sampling strategies are training concerns
(Layer 8), not part of this module.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """A named game-state preset.

    Attributes:
        name:        unique identifier (used as dict key in registry)
        description: human-readable summary
        state:       dict compatible with Match.get_state() / Match.set_state()
        difficulty:  "easy" | "medium" | "hard" (for curriculum filtering)
        tags:        category tags for filtering (e.g., "recovery", "edge_guard")
    """
    name: str
    description: str
    state: dict
    difficulty: str = "medium"
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, Scenario] = {}


def register_scenario(scenario: Scenario) -> None:
    """Add a scenario to the global registry (overwrites if name exists)."""
    SCENARIOS[scenario.name] = scenario


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name. Raises KeyError if not found."""
    if name not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS)) or "(none)"
        raise KeyError(f"Unknown scenario '{name}'. Available: {available}")
    return SCENARIOS[name]


def list_scenarios() -> list[str]:
    """Return sorted list of all registered scenario names."""
    return sorted(SCENARIOS.keys())


def sample_scenario(
    tags: tuple[str, ...] | None = None,
    difficulty: str | None = None,
) -> Scenario:
    """Return a random scenario, optionally filtered by tags and/or difficulty.

    Tags match if the scenario has ANY tag in the requested set (OR logic).
    """
    candidates = list(SCENARIOS.values())
    if tags:
        candidates = [s for s in candidates if any(t in s.tags for t in tags)]
    if difficulty:
        candidates = [s for s in candidates if s.difficulty == difficulty]
    if not candidates:
        raise ValueError("No scenarios match the given filters")
    return random.choice(candidates)


# ---------------------------------------------------------------------------
# Scenario builder helper
# ---------------------------------------------------------------------------

def scenario_from_match(
    name: str,
    description: str,
    match,
    difficulty: str = "medium",
    tags: tuple[str, ...] = (),
) -> Scenario:
    """Capture the current match state as a named scenario."""
    return Scenario(
        name=name,
        description=description,
        state=match.get_state(),
        difficulty=difficulty,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

def _char_state(
    player_id: int = 0,
    x: float = 0.0,
    y: float = 68.1,
    vx: float = 0.0,
    vy: float = 0.0,
    facing: float = 1.0,
    action: int = 0,       # Action.NONE
    action_frame: int = 0,
    in_air: bool = False,
    recover_used: bool = False,
    falling_through: bool = False,
    damage_pct: float = 0.0,
    stocks: int = 3,
    attack_num: int = 0,
    charge_amount: float = 0.0,
    has_hit: bool = False,
    invincibility: int = 0,
    anim_frame: float = 0.0,
    pending_damage: float = 0.0,
    pending_knockback_vx: float = 0.0,
    pending_knockback_vy: float = 0.0,
    spawn_x: float = -245.0,
    spawn_y: float = 70.0,
) -> dict:
    """Helper to build a character state dict with sensible defaults.

    Default state: standing on the ground (y=68.1, action=NONE, in_air=False).
    """
    return {
        "player_id": player_id,
        "x": x, "y": y, "vx": vx, "vy": vy,
        "facing": facing,
        "action": action, "action_frame": action_frame,
        "in_air": in_air, "recover_used": recover_used,
        "falling_through": falling_through,
        "damage_pct": damage_pct, "stocks": stocks,
        "attack_num": attack_num, "charge_amount": charge_amount,
        "has_hit": has_hit, "invincibility": invincibility,
        "anim_frame": anim_frame,
        "pending_damage": pending_damage,
        "pending_knockback_vx": pending_knockback_vx,
        "pending_knockback_vy": pending_knockback_vy,
        "spawn_x": spawn_x, "spawn_y": spawn_y,
    }


def _match_state(
    chars: list[dict],
    frame: int = 1,
    done: bool = False,
    winner=None,
    stocks: int = 3,
    time_limit: int = 7200,
    gravity_scale: float = 1.0,
    knockback_scale: float = 1.0,
) -> dict:
    """Helper to build a match state dict from character state dicts."""
    return {
        "frame": frame,
        "done": done,
        "winner": winner,
        "characters": chars,
        "config": {
            "stocks": stocks,
            "time_limit": time_limit,
            "gravity_scale": gravity_scale,
            "knockback_scale": knockback_scale,
        },
    }


# -- Scenario 1: default (standard match start) --
register_scenario(Scenario(
    name="default",
    description="Standard match start. Both players at spawn, 0% damage, 3 stocks.",
    state=_match_state([
        _char_state(player_id=0, x=-245.0, y=70.0, facing=1.0,
                    action=3, action_frame=0, in_air=True,
                    invincibility=0, spawn_x=-245.0, spawn_y=70.0),
        _char_state(player_id=1, x=245.0, y=70.0, facing=-1.0,
                    action=3, action_frame=0, in_air=True,
                    invincibility=0, spawn_x=245.0, spawn_y=70.0),
    ]),
    difficulty="easy",
    tags=(),
))

# -- Scenario 2: high_damage --
register_scenario(Scenario(
    name="high_damage",
    description="Both players at 120% damage. Small hit = huge knockback.",
    state=_match_state([
        _char_state(player_id=0, x=-100.0, y=68.1, facing=1.0,
                    damage_pct=120.0, spawn_x=-245.0, spawn_y=70.0),
        _char_state(player_id=1, x=100.0, y=68.1, facing=-1.0,
                    damage_pct=120.0, spawn_x=245.0, spawn_y=70.0),
    ]),
    difficulty="medium",
    tags=("high_damage",),
))

# -- Scenario 3: one_stock_left --
register_scenario(Scenario(
    name="one_stock_left",
    description="P0 has 1 stock at 80% damage, P1 has 3 stocks at 80%. P0 is under pressure.",
    state=_match_state([
        _char_state(player_id=0, x=-100.0, y=68.1, facing=1.0,
                    damage_pct=80.0, stocks=1, spawn_x=-245.0, spawn_y=70.0),
        _char_state(player_id=1, x=100.0, y=68.1, facing=-1.0,
                    damage_pct=80.0, stocks=3, spawn_x=245.0, spawn_y=70.0),
    ]),
    difficulty="hard",
    tags=("pressure",),
))

# -- Scenario 4: off_stage_recovery --
register_scenario(Scenario(
    name="off_stage_recovery",
    description="P0 is below the stage, falling. Must recover or die.",
    state=_match_state([
        _char_state(player_id=0, x=350.0, y=-50.0, vy=-8.0, facing=1.0,
                    in_air=True, damage_pct=60.0,
                    spawn_x=-245.0, spawn_y=70.0),
        _char_state(player_id=1, x=0.0, y=68.1, facing=-1.0,
                    spawn_x=245.0, spawn_y=70.0),
    ]),
    difficulty="hard",
    tags=("recovery",),
))

# -- Scenario 5: edge_guard --
register_scenario(Scenario(
    name="edge_guard",
    description="P0 on stage edge, P1 off-stage trying to recover.",
    state=_match_state([
        _char_state(player_id=0, x=300.0, y=68.1, facing=1.0,
                    spawn_x=-245.0, spawn_y=70.0),
        _char_state(player_id=1, x=380.0, y=-30.0, vy=-5.0, facing=-1.0,
                    in_air=True, damage_pct=90.0,
                    spawn_x=245.0, spawn_y=70.0),
    ]),
    difficulty="medium",
    tags=("edge_guard", "recovery"),
))

# -- Scenario 6: charging_smash --
register_scenario(Scenario(
    name="charging_smash",
    description="P0 is charging a smash, P1 is approaching. Timing decision.",
    state=_match_state([
        _char_state(player_id=0, x=-20.0, y=68.1, facing=1.0,
                    action=6, attack_num=4, charge_amount=0.5,
                    action_frame=10, spawn_x=-245.0, spawn_y=70.0),
        _char_state(player_id=1, x=80.0, y=68.1, facing=-1.0,
                    action=1, spawn_x=245.0, spawn_y=70.0),
    ]),
    difficulty="medium",
    tags=("timing",),
))

# -- Scenario 7: combo_practice --
register_scenario(Scenario(
    name="combo_practice",
    description="P1 in hitstun, P0 free to act. Follow-up combo opportunity.",
    state=_match_state([
        _char_state(player_id=0, x=20.0, y=68.1, facing=1.0,
                    spawn_x=-245.0, spawn_y=70.0),
        _char_state(player_id=1, x=40.0, y=72.0, facing=-1.0,
                    action=7, action_frame=15, in_air=True,
                    damage_pct=45.0, vy=3.0,
                    spawn_x=245.0, spawn_y=70.0),
    ]),
    difficulty="easy",
    tags=("combo",),
))

# -- Scenario 8: center_stage --
register_scenario(Scenario(
    name="center_stage",
    description="Both characters standing on center platform, close together.",
    state=_match_state([
        _char_state(player_id=0, x=-20.0, y=68.1, facing=1.0,
                    spawn_x=-245.0, spawn_y=70.0),
        _char_state(player_id=1, x=20.0, y=68.1, facing=-1.0,
                    spawn_x=245.0, spawn_y=70.0),
    ]),
    difficulty="easy",
    tags=("neutral",),
))

# -- Scenario 9: platform_above --
register_scenario(Scenario(
    name="platform_above",
    description="P0 on left side platform (y=85), P1 on main stage.",
    state=_match_state([
        _char_state(player_id=0, x=-130.0, y=153.1, facing=1.0,
                    spawn_x=-245.0, spawn_y=70.0),
        _char_state(player_id=1, x=0.0, y=68.1, facing=-1.0,
                    spawn_x=245.0, spawn_y=70.0),
    ]),
    difficulty="medium",
    tags=("positioning",),
))

# -- Scenario 10: high_knockback --
register_scenario(Scenario(
    name="high_knockback",
    description="P1 at 150% flying at high velocity. Demonstrate knockback physics.",
    state=_match_state([
        _char_state(player_id=0, x=-50.0, y=68.1, facing=1.0,
                    spawn_x=-245.0, spawn_y=70.0),
        _char_state(player_id=1, x=50.0, y=100.0, facing=-1.0,
                    action=7, action_frame=30, in_air=True,
                    damage_pct=150.0, vx=20.0, vy=10.0,
                    spawn_x=245.0, spawn_y=70.0),
    ]),
    difficulty="medium",
    tags=("knockback",),
))
