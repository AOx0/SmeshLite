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
    """

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
