"""
CharacterBrain — pluggable input provider for Character instances.

Usage:
    char.set_brain(PlayerInput())          # keyboard
    char.set_brain(ExternalBrain())        # RL env injection
    char.set_brain(MyMLAgent())            # custom subclass

Each tick, Match calls char.gather_input(opponents, stage), which calls
brain.think(context, self._input).  The brain writes directly into the
character-owned InputState buffer — no allocation per frame.

BrainContext is a lightweight snapshot; ML brains read it to make decisions.
PlayerInput ignores it entirely and reads hardware state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from smeshlite.core.sensors import RaycastSensor, RaycastResult


# ---------------------------------------------------------------------------
# InputState — the 4-button input buffer owned by each Character
# ---------------------------------------------------------------------------

@dataclass
class InputState:
    """
    4-directional input, matching the original Scratch 4-button scheme:
      left   — A / left-arrow    (move left; also attack direction)
      right  — D / right-arrow   (move right; also attack direction)
      up     — W / up-arrow      (jump; also reduces gravity when held)
      attack — S / down-arrow    (all attacks and smash charging)
    """
    left:   bool = False
    right:  bool = False
    up:     bool = False
    attack: bool = False

    def clear(self) -> None:
        self.left = self.right = self.up = self.attack = False


# ---------------------------------------------------------------------------
# Context types passed to brains
# ---------------------------------------------------------------------------

@dataclass
class OpponentContext:
    """Simplified view of one opponent, exposed to the brain."""
    x: float
    y: float
    vx: float
    vy: float
    damage_pct: float
    stocks: int
    facing: float
    in_air: bool
    action: int
    attack_num: int
    action_frame: int
    charge_amount: float
    invincibility: int = 0


@dataclass
class BrainContext:
    """
    Lightweight game-state snapshot passed to brain.think() each tick.
    Contains only what a real-world controller could observe.
    """
    # Self
    x: float
    y: float
    vx: float
    vy: float
    in_air: bool
    facing: float
    damage_pct: float
    stocks: int
    action: int
    action_frame: int
    attack_num: int
    charge_amount: float
    invincibility: int
    # Others
    opponents: list[OpponentContext] = field(default_factory=list)
    # Stage
    stage_x1: float = -700.0
    stage_x2: float = 700.0
    stage_y_floor: float = 0.0
    # Combined X-extent of all platforms (i.e. the actual walkable footprint,
    # as opposed to stage_x1/x2 which are the far-away kill-zone walls).
    stage_platform_x1: float = -700.0
    stage_platform_x2: float = 700.0
    # Sensors (only populated if the brain declares CharacterBrain.SENSORS)
    sensors: dict[str, RaycastResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CharacterBrain — abstract base
# ---------------------------------------------------------------------------

class CharacterBrain:
    """
    Base class for all character input providers.

    Override think() to write desired inputs into `out` each frame.
    The default implementation is a no-op (all inputs remain False).

    ML agents should subclass this and use BrainContext features;
    rule-based or keyboard agents may ignore context entirely.

    Subclasses placed in /agents are auto-discovered by
    smeshlite.core.brain_registry.  Set BRAIN_NAME to control the display
    name used by the registry; if left as None, the class name is used.

    Set SENSORS to a tuple of RaycastSensor specs to have Character evaluate them
    each tick and populate BrainContext.sensors (and the renderer's debug overlay).
    Empty by default — zero cost for brains that don't use sensors.
    """

    BRAIN_NAME: str | None = None
    SENSORS: tuple[RaycastSensor, ...] = ()

    def think(self, context: BrainContext, out: InputState) -> None:
        """Write desired inputs for this frame into `out`."""
        pass


# ---------------------------------------------------------------------------
# PlayerInput — reads live pygame keyboard state
# ---------------------------------------------------------------------------

class PlayerInput(CharacterBrain):
    """
    Keyboard-controlled brain.  Reads pygame key state each frame.

    Default bindings: A/D/W/S  (P1).
    Pass alternate keys for P2: PlayerInput(pygame.K_LEFT, ...).
    """

    def __init__(
        self,
        key_left: int = None,
        key_right: int = None,
        key_up: int = None,
        key_attack: int = None,
    ) -> None:
        # Defer pygame import so this module is importable without pygame
        try:
            import pygame
            self._keys = (
                key_left   if key_left   is not None else pygame.K_a,
                key_right  if key_right  is not None else pygame.K_d,
                key_up     if key_up     is not None else pygame.K_w,
                key_attack if key_attack is not None else pygame.K_s,
            )
        except ImportError:
            self._keys = (0, 0, 0, 0)

    def think(self, context: BrainContext, out: InputState) -> None:
        try:
            import pygame
            pressed = pygame.key.get_pressed()
            out.left   = bool(pressed[self._keys[0]])
            out.right  = bool(pressed[self._keys[1]])
            out.up     = bool(pressed[self._keys[2]])
            out.attack = bool(pressed[self._keys[3]])
        except Exception:
            out.clear()


# ---------------------------------------------------------------------------
# ExternalBrain — RL / scripted injection
# ---------------------------------------------------------------------------

class ExternalBrain(CharacterBrain):
    """
    Brain whose inputs are written from outside each step.

    The RL environment (or any external driver) sets `pending` before
    calling match.tick(); the brain copies it into the character buffer.

    Example::

        brain = ExternalBrain()
        char.set_brain(brain)
        brain.pending = InputState(left=True)   # set before each tick
        match.tick()
    """

    def __init__(self) -> None:
        self.pending: InputState = InputState()

    def think(self, context: BrainContext, out: InputState) -> None:
        out.left   = self.pending.left
        out.right  = self.pending.right
        out.up     = self.pending.up
        out.attack = self.pending.attack


# ---------------------------------------------------------------------------
# Observation helpers — for ML brains that need the flat vector
# Match.get_obs() / Character.obs_vector() would produce, reconstructed
# purely from a BrainContext snapshot at inference time.
# ---------------------------------------------------------------------------

def _entity_obs(
    x: float, y: float, vx: float, vy: float,
    damage_pct: float, stocks: int, action: int, action_frame: int,
    facing: float, in_air: bool, charge_amount: float,
) -> list[float]:
    return [
        x, y, vx, vy,
        damage_pct, float(stocks),
        float(action), float(action_frame),
        facing, float(in_air), charge_amount,
    ]


def brain_context_to_obs(ctx: BrainContext) -> list[float]:
    """
    Build the same flat observation vector as Match.get_obs(perspective) /
    Character.obs_vector(), reconstructed from a BrainContext snapshot.

    Order: [self: 11 floats] + [opponent_i: 11 floats for each opponent],
    with per-entity field order matching Character.obs_vector():
        [x, y, vx, vy, damage_pct, stocks, action, action_frame,
         facing, in_air, charge_amount]

    ML brains can call this in think() to feed a model trained against
    SmeshLiteEnv's observation space (Box(22,) for 1v1).
    """
    obs = _entity_obs(
        ctx.x, ctx.y, ctx.vx, ctx.vy,
        ctx.damage_pct, ctx.stocks, ctx.action, ctx.action_frame,
        ctx.facing, ctx.in_air, ctx.charge_amount,
    )
    for o in ctx.opponents:
        obs.extend(_entity_obs(
            o.x, o.y, o.vx, o.vy,
            o.damage_pct, o.stocks, o.action, o.action_frame,
            o.facing, o.in_air, o.charge_amount,
        ))
    return obs


# ---------------------------------------------------------------------------
# Full observation helpers — extended BrainContext serialization
# ---------------------------------------------------------------------------

_MAX_SENSORS: int = 8


def _full_entity_obs(
    x: float, y: float, vx: float, vy: float,
    damage_pct: float, stocks: int, action: int, action_frame: int,
    facing: float, in_air: bool, charge_amount: float, invincibility: float,
) -> list[float]:
    return [
        x, y, vx, vy,
        damage_pct, float(stocks),
        float(action), float(action_frame),
        facing, float(in_air), charge_amount, invincibility,
    ]


def full_obs_size(n_players: int = 2, max_sensors: int = _MAX_SENSORS) -> int:
    """Return the flat observation size for the 'full' obs mode."""
    return 12 * n_players + 5 + 3 * max_sensors


def brain_context_to_full_obs(
    ctx: BrainContext, max_sensors: int = _MAX_SENSORS
) -> list[float]:
    """
    Build an extended observation vector from a BrainContext snapshot.

    Layout (for 1v1, max_sensors=8):
      [self: 12 floats] + [opponent: 12 floats] + [stage: 5 floats]
      + [sensors: 3 * max_sensors floats]
      = 53 floats

    Per-entity fields (12 each):
      [x, y, vx, vy, damage_pct, stocks, action, action_frame,
       facing, in_air, charge_amount, invincibility]

    Stage fields (5):
      [stage_x1, stage_x2, stage_y_floor,
       stage_platform_x1, stage_platform_x2]

    Sensor fields (3 per slot, max_sensors slots):
      [distance, hit (0/1), value]
      Padded with zeros for unused slots.
    """
    obs = _full_entity_obs(
        ctx.x, ctx.y, ctx.vx, ctx.vy,
        ctx.damage_pct, ctx.stocks, ctx.action, ctx.action_frame,
        ctx.facing, ctx.in_air, ctx.charge_amount,
        float(ctx.invincibility),
    )
    for o in ctx.opponents:
        obs.extend(_full_entity_obs(
            o.x, o.y, o.vx, o.vy,
            o.damage_pct, o.stocks, o.action, o.action_frame,
            o.facing, o.in_air, o.charge_amount,
            float(o.invincibility),
        ))
    # Stage
    obs.extend([
        ctx.stage_x1, ctx.stage_x2, ctx.stage_y_floor,
        ctx.stage_platform_x1, ctx.stage_platform_x2,
    ])
    # Sensors (fixed-size, padded)
    sensor_list = list(ctx.sensors.values())[:max_sensors]
    for s in sensor_list:
        obs.extend([s.distance, float(s.hit), s.value])
    for _ in range(max_sensors - len(sensor_list)):
        obs.extend([0.0, 0.0, 0.0])
    return obs
