"""
CharacterDef — loads a character's complete data from a self-contained folder.

A character folder contains:
    <name>.json     — stats, physics, attacks, costume metadata
    sprites/        — SVG files referenced by the JSON

═══════════════════════════════════════════════════════════════
  SVG → GAME COORDINATE CONVERSION
═══════════════════════════════════════════════════════════════
  SVG_TO_GAME = 0.5  (game units per SVG pixel, confirmed from body hitbox).
  SVG Y-axis is down; game Y-axis is up.
  rc_x / rc_y is the SVG pixel that aligns with the character's game origin.

  HitboxRect corners in game coordinates:
    gx1 = (0       − rc_x) × SVG_TO_GAME
    gx2 = (width   − rc_x) × SVG_TO_GAME
    gy1 = −(height − rc_y) × SVG_TO_GAME   (SVG bottom → game −Y)
    gy2 = −(0      − rc_y) × SVG_TO_GAME   (SVG top    → game +Y)

═══════════════════════════════════════════════════════════════
  COSTUME INDEX LAYOUT  (74 SVG costumes, 0-indexed)
═══════════════════════════════════════════════════════════════
  0         — body collision hitbox SVG
  1         — semisolid platform hitbox SVG
  2         — smash attack hitbox SVG (unused in Python collision)
  3         — idle
  4–11      — run cycle (8 frames)
  12        — jump
  13        — fall / hitstun
  For attack N (N = 1..5):
    base = 14 + (N−1) × 10
    base+0 … base+4   — visual frames 1–5
    base+5 … base+9   — hitbox SVG frames 1–5
                         Stored as dict key (N, frame_i+1)
                         frame_i = 0 → key 1 (always empty)
  64–73     — entrance animation (10 frames)

═══════════════════════════════════════════════════════════════
  HITBOX KEY MAPPING
═══════════════════════════════════════════════════════════════
  get_attack_hitbox(attack_num, key) where key = phase + 1:
    phase 1 → key 2   (visual base+1; hitbox base+6)
    phase 2 → key 3
    phase 3 → key 4
    phase 4 → key 5   (recovery; usually empty)

  Key 1 (base+5) is never queried during ATTACK/SMASH.
  It corresponds to the charging visual (base+0) which has no hitbox.

═══════════════════════════════════════════════════════════════
  ATTACK FRAME DATA
═══════════════════════════════════════════════════════════════
  frame_data = [t1, t2, t3, t4]:
    phase 1 while action_frame < t1  (startup / charging release)
    phase 2 while action_frame < t2
    phase 3 while action_frame < t3
    phase 4 while action_frame < t4  (recovery)
    attack ends when action_frame > t4

  Smash attacks (attack 2 = nsmash, attack 4 = fsmash):
    hitbox keys 2 and 3 are active (phases 1 and 2 after release).
    key 1 is empty — the charging visual has no active hitbox.

  Air attacks (attack 1 = nair, attack 3 = fair, attack 5 = uair):
    active keys vary by attack; determined purely by SVG content.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


SVG_TO_GAME: float = 0.5   # game units per SVG pixel (confirmed from body hitbox)


# ---------------------------------------------------------------------------
# Sub-dataclasses (mirrors of JSON fields)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CharacterStats:
    size: float          # collision half-radius (used in renderer fallback box)
    weight: float        # inverse knockback scale
    charge_speed: float  # charge amount added per frame while holding attack
    run_anim_delay: int  # run cycle animation sub-frame divisor


@dataclass(frozen=True)
class PhysicsConstants:
    gravity: float           # SpeedY decrement per frame when holding up (lighter fall)
    fast_fall_gravity: float # SpeedY decrement per frame when not holding up (heavier)
    jump_speed: float        # initial SpeedY on jump / uair
    wall_bounce_x: float     # |SpeedX| after wall bounce (see Stage.resolve_walls TODO)
    run_accel: float         # SpeedX delta per frame when pressing left/right
    speed_decay: float       # SpeedX multiplier per frame (friction)
    max_speed: float         # SpeedX clamp
    max_fall_speed: float    # maximum downward velocity magnitude


@dataclass(frozen=True)
class AttackDef:
    """
    Data for one attack move.

    frame_data = [t1, t2, t3, t4]:
      Phase boundary counts.  Phase P is active while action_frame < t_P.
      Attack ends when action_frame > t4.  See module docstring for detail.
    """
    name: str
    frame_data: tuple[int, int, int, int]
    damage: float
    knockback: float
    multihit: bool = False

    @property
    def total_frames(self) -> int:
        return self.frame_data[3]


@dataclass(frozen=True)
class CostumeMeta:
    name: str
    file: str    # filename relative to sprites/ subfolder
    rc_x: float  # rotation center X in SVG pixels
    rc_y: float  # rotation center Y in SVG pixels (SVG Y-down)


# ---------------------------------------------------------------------------
# Hitbox geometry
# ---------------------------------------------------------------------------

@dataclass
class HitboxRect:
    """Axis-aligned bounding box in game units, Y-up, origin = character center."""
    x1: float
    y1: float
    x2: float
    y2: float
    is_empty: bool = False

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def half_w(self) -> float:
        return abs(self.x2 - self.x1) / 2.0

    @property
    def half_h(self) -> float:
        return abs(self.y2 - self.y1) / 2.0


_EMPTY_HB = HitboxRect(0.0, 0.0, 0.0, 0.0, is_empty=True)


def _load_hitbox_rect(svg_path: str, rc_x: float, rc_y: float) -> HitboxRect:
    """Parse an SVG hitbox file and return bounds in game units (Y-up)."""
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_text = f.read()
    except OSError:
        return _EMPTY_HB

    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return _EMPTY_HB

    try:
        w = float(root.get("width", "0"))
        h = float(root.get("height", "0"))
    except ValueError:
        return _EMPTY_HB

    if w <= 0 or h <= 0:
        return _EMPTY_HB

    gx1 = (0.0 - rc_x) * SVG_TO_GAME
    gx2 = (w   - rc_x) * SVG_TO_GAME
    gy1 = -(h   - rc_y) * SVG_TO_GAME
    gy2 = -(0.0 - rc_y) * SVG_TO_GAME

    return HitboxRect(gx1, gy1, gx2, gy2)


# ---------------------------------------------------------------------------
# Costume index constants
# ---------------------------------------------------------------------------
IDX_BODY_HB   = 0
IDX_IDLE      = 3
IDX_RUN_START = 4
IDX_RUN_END   = 11
IDX_JUMP      = 12
IDX_FALL      = 13
# Attack visuals: attack N (1-based) → base = 14 + (N-1)*10
# Attack hitboxes: +5 from visual base


# ---------------------------------------------------------------------------
# CharacterDef
# ---------------------------------------------------------------------------

class CharacterDef:
    """
    Complete definition of a playable character loaded from a JSON folder.

    One CharacterDef can be shared by multiple Character instances (same fighter,
    different players). It is read-only after construction.
    """

    def __init__(
        self,
        name: str,
        stats: CharacterStats,
        physics: PhysicsConstants,
        attacks: list[AttackDef],
        costumes: list[CostumeMeta],
        folder: Path,
    ) -> None:
        self.name = name
        self.stats = stats
        self.physics = physics
        self.attacks = attacks
        self.costumes = costumes
        self.folder = folder
        self._sprites_dir = folder / "sprites"

        self._body_hitbox: HitboxRect = _load_hitbox_rect(
            str(self._sprites_dir / costumes[IDX_BODY_HB].file),
            costumes[IDX_BODY_HB].rc_x,
            costumes[IDX_BODY_HB].rc_y,
        )
        self._attack_hitboxes: dict[tuple[int, int], HitboxRect] = (
            self._build_attack_hitboxes()
        )

    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "CharacterDef":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        stats = CharacterStats(
            size=data["stats"]["size"],
            weight=data["stats"]["weight"],
            charge_speed=data["stats"]["charge_speed"],
            run_anim_delay=int(data["stats"]["run_anim_delay"]),
        )
        physics = PhysicsConstants(
            gravity=data["physics"]["gravity"],
            fast_fall_gravity=data["physics"]["fast_fall_gravity"],
            jump_speed=data["physics"]["jump_speed"],
            wall_bounce_x=data["physics"]["wall_bounce_x"],
            run_accel=data["physics"]["run_accel"],
            speed_decay=data["physics"]["speed_decay"],
            max_speed=data["physics"]["max_speed"],
            max_fall_speed=data["physics"]["max_fall_speed"],
        )
        attacks = [
            AttackDef(
                name=a["name"],
                frame_data=tuple(a["frame_data"]),
                damage=a["damage"],
                knockback=a["knockback"],
                multihit=a.get("multihit", False),
            )
            for a in data["attacks"]
        ]
        costumes = [
            CostumeMeta(
                name=c["name"],
                file=c["file"],
                rc_x=c["rc_x"],
                rc_y=c["rc_y"],
            )
            for c in data["costumes"]
        ]
        return cls(
            name=data["name"],
            stats=stats,
            physics=physics,
            attacks=attacks,
            costumes=costumes,
            folder=path.parent,
        )

    # ------------------------------------------------------------------

    def get_body_hitbox(self) -> HitboxRect:
        return self._body_hitbox

    def get_attack_hitbox(self, attack_num: int, phase: int) -> HitboxRect:
        return self._attack_hitboxes.get(
            (attack_num, max(1, min(5, phase))), _EMPTY_HB
        )

    def sprite_path(self, costume_idx: int) -> str:
        return str(self._sprites_dir / self.costumes[costume_idx].file)

    # ------------------------------------------------------------------

    def _build_attack_hitboxes(self) -> dict[tuple[int, int], HitboxRect]:
        result: dict[tuple[int, int], HitboxRect] = {}
        n_attacks = len(self.attacks)
        for atk_i in range(n_attacks):
            hb_base = 14 + atk_i * 10 + 5   # costume index of hitbox frame 1
            for frame_i in range(5):
                costume_idx = hb_base + frame_i
                if costume_idx >= len(self.costumes):
                    break
                meta = self.costumes[costume_idx]
                rect = _load_hitbox_rect(
                    str(self._sprites_dir / meta.file),
                    meta.rc_x,
                    meta.rc_y,
                )
                result[(atk_i + 1, frame_i + 1)] = rect
        return result


# ---------------------------------------------------------------------------
# Singleton — MINIUM loaded from the bundled JSON
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).parent
MINIUM_DEF: CharacterDef = CharacterDef.from_json(
    _THIS_DIR / "characters" / "Minium" / "minium.json"
)
