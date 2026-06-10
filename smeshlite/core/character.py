"""
Character: physics engine + fighting state machine.

Ported from the Character sprite in ULTRA MEGA SMESH (Scratch 3.0).

    Scratch procedures → Python methods
    ─────────────────────────────────────
    PhysicsBlock        → physics_step()
    Fighting engine     → action_step()
    Be damaged          → receive_damage()
    HitCode / Post atk  → get_active_hitbox() + has_hit flag

═══════════════════════════════════════════════════════════════
  COORDINATE SYSTEM
═══════════════════════════════════════════════════════════════
  +X right, +Y up.  Origin = character rotation center.
  Game units: 1 game unit = 2 SVG pixels (SVG_TO_GAME = 0.5).

═══════════════════════════════════════════════════════════════
  4-BUTTON INPUT  (matches original Scratch scheme)
═══════════════════════════════════════════════════════════════
  left   — A / left-arrow   (move + attack direction)
  right  — D / right-arrow  (move + attack direction)
  up     — W / up-arrow     (jump; holding reduces gravity = "floating")
  attack — S / down-arrow   (all attacks and smash charging)

  Input lives in a Character-owned InputState buffer.
  Supply a CharacterBrain via set_brain() to fill it each tick.

═══════════════════════════════════════════════════════════════
  ATTACK ROUTING  (mirrors Scratch Fighting engine procedure)
═══════════════════════════════════════════════════════════════
  In air + attack + up       → uair  (attack 5); propels up; sets recover_used
  In air + attack + forward  → fair  (attack 3)
  In air + attack            → nair  (attack 1)
  On ground + attack + fwd   → charge → fsmash (attack 4)
  On ground + attack         → charge → nsmash (attack 2)

  "forward" = exactly one of left/right pressed (exclusive).

═══════════════════════════════════════════════════════════════
  FRAME DATA FORMAT  (frame_data = [t1, t2, t3, t4])
═══════════════════════════════════════════════════════════════
  Defines 4 phase boundaries using accumulated frame counts:

    phase 1: action_frame < t1  (startup)
    phase 2: action_frame < t2  (usually active)
    phase 3: action_frame < t3  (usually active or late)
    phase 4: action_frame < t4  (recovery)
    attack ends: action_frame > t4

  Confirmed from Scratch source: "when X frames in, next phase starts."
  The strict < boundary is correct — frame = tN is the FIRST frame of phase N+1.

═══════════════════════════════════════════════════════════════
  COSTUME LAYOUT  (74 SVG costumes, 0-indexed)
═══════════════════════════════════════════════════════════════
  0         — body collision hitbox SVG
  1         — semisolid platform hitbox SVG
  2         — smash attack hitbox SVG (unused)
  3         — idle
  4–11      — run cycle (8 frames)
  12        — jump
  13        — fall / hitstun
  For attack N (N = 1..5):
    base = 14 + (N−1) × 10
    base+0 … base+4   — visual frames 1–5
    base+5 … base+9   — hitbox SVG frames 1–5  (dict key = frame_i+1)
  64–73     — entrance animation (10 frames, 2 ticks each)

═══════════════════════════════════════════════════════════════
  HITBOX ALGORITHM
═══════════════════════════════════════════════════════════════
  During ATTACK/SMASH at phase P (1–4):
    visual shown    = _IDX_ATK_VIS[attack_num-1][P]   (visual frame P+1)
    hitbox queried  = get_attack_hitbox(attack_num, P+1)

  Key insight from Scratch (HitCode procedure):
    The hitbox sprite is switched every frame for every phase,
    and collision is tested unconditionally.  Whether a phase
    deals damage is determined by SVG content — an empty SVG
    (zero width/height) produces no overlap → no damage.
    This port reproduces that by returning None when is_empty.

  Key 1 (base+5) is never queried during ATTACK/SMASH since the
  minimum phase is 1 → key 2.  Key 1 corresponds to the charging
  visual (base+0), whose hitbox is intentionally empty.

  During CHARGING state, visual frame 1 (base+0) is shown.
  On release, action becomes SMASH and phase computation begins.

═══════════════════════════════════════════════════════════════
  RECOVER / UAIR
═══════════════════════════════════════════════════════════════
  recover_used tracks whether uair (attack 5) has been used
  since the character last touched the ground.  There is NO
  double jump — jumping is ground-only.  "Recovering" refers to
  the uair in the air, identical to the original Scratch game.
  recover_used is reset to False on landing (ground_update).
"""
from __future__ import annotations
import math
from enum import Enum

from smeshlite.data.character_def import CharacterDef, MINIUM_DEF
from smeshlite.core.brain import InputState, BrainContext, OpponentContext, CharacterBrain
from smeshlite.core.sensors import RaycastResult, evaluate_sensors


class Action(Enum):
    NONE     = 0
    RUN      = 1
    JUMP     = 2
    AERIAL   = 3
    ATTACK   = 4
    SMASH    = 5
    CHARGING = 6
    HITSTUN  = 7
    RESPAWN  = 8
    DEAD     = 9


RESPAWN_INVINCIBILITY = 90
HITSTUN_BASE          = 20
_ENTRANCE_FRAMES      = 20   # 10 entrance costumes × 2 ticks each @ 30 fps


class Character:
    """
    Stateful character instance.  One instance = one player.
    All coordinates are in game-unit space (+Y up).
    """

    def __init__(
        self,
        player_id: int,
        spawn_x: float,
        spawn_y: float,
        character_def: CharacterDef | None = None,
        facing: float = 1.0,
        stocks: int = 3,
    ) -> None:
        self.id = player_id
        self._def: CharacterDef = character_def if character_def is not None else MINIUM_DEF

        # Physics collision dimensions from body hitbox SVG
        bh = self._def.get_body_hitbox()
        if not bh.is_empty:
            self.phys_half_w: float = bh.half_w
            self.phys_half_h: float = -bh.y1   # feet extend this far below origin
        else:
            self.phys_half_w = 38.5
            self.phys_half_h = 68.1

        # Position / velocity
        self.x = spawn_x
        self.y = spawn_y
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.facing: float = facing   # +1 = right, −1 = left

        # State
        self.action = Action.RESPAWN
        self.action_frame: int = 0
        self.in_air: bool = True
        self.recover_used: bool = False
        self.falling_through: bool = False

        # Combat
        self.damage_pct: float = 0.0
        self.stocks: int = stocks
        self.attack_num: int = 0
        self.charge_amount: float = 0.0
        self.has_hit: bool = False
        self.invincibility: int = 0

        # Animation
        self.anim_frame: float = 0.0

        # Damage buffer (flushed each tick by apply_pending_damage)
        self._pending_damage: float = 0.0
        self._pending_knockback_vx: float = 0.0
        self._pending_knockback_vy: float = 0.0

        # Respawn origin
        self._spawn_x = spawn_x
        self._spawn_y = spawn_y

        # Brain + input buffer
        self._brain: CharacterBrain | None = None
        self._input: InputState = InputState()

        # Sprite loader (lazy, set on first render call)
        self._sprite_loader: object | None = None

        # Last sensor readings (for the brain's SENSORS, if any); cached for the
        # renderer's debug overlay
        self._sensor_results: dict[str, RaycastResult] = {}

    # ------------------------------------------------------------------
    # Brain / input API
    # ------------------------------------------------------------------

    def set_brain(self, brain: CharacterBrain) -> None:
        self._brain = brain

    def gather_input(self, opponents: list[Character], stage) -> None:
        """Ask brain to write into self._input.  Called by Match each tick."""
        if self._brain is not None:
            self._update_sensors(stage)
            ctx = self._build_context(opponents, stage)
            self._brain.think(ctx, self._input)

    def _build_context(self, opponents: list[Character], stage) -> BrainContext:
        return BrainContext(
            x=self.x, y=self.y, vx=self.vx, vy=self.vy,
            in_air=self.in_air, facing=self.facing,
            damage_pct=self.damage_pct, stocks=self.stocks,
            action=self.action.value, action_frame=self.action_frame,
            attack_num=self.attack_num, charge_amount=self.charge_amount,
            invincibility=self.invincibility,
            opponents=[
                OpponentContext(
                    x=o.x, y=o.y, vx=o.vx, vy=o.vy,
                    damage_pct=o.damage_pct, stocks=o.stocks,
                    facing=o.facing, in_air=o.in_air,
                    action=o.action.value, attack_num=o.attack_num,
                    action_frame=o.action_frame, charge_amount=o.charge_amount,
                )
                for o in opponents
            ],
            stage_x1=-stage.kill_x,
            stage_x2=stage.kill_x,
            stage_y_floor=stage.platforms[0].y if stage.platforms else 0.0,
            sensors=self._sensor_results,
        )

    def _update_sensors(self, stage) -> None:
        """Evaluate this character's brain's SENSORS (if any) against the stage."""
        if self._brain is not None and self._brain.SENSORS:
            self._sensor_results = evaluate_sensors(self._brain.SENSORS, self, stage)
        elif self._sensor_results:
            self._sensor_results = {}

    @property
    def sprite_loader(self):
        """Lazy-init SpriteLoader for this character's assets."""
        if self._sprite_loader is None:
            from smeshlite.render.sprite_loader import SpriteLoader
            self._sprite_loader = SpriteLoader(self._def)
        return self._sprite_loader

    # ------------------------------------------------------------------
    # Main updates — called once per tick by Match
    # ------------------------------------------------------------------

    def physics_step(self, gravity_scale: float = 1.0) -> None:
        """Apply gravity, friction, and integrate position."""
        if self.action in (Action.DEAD, Action.RESPAWN):
            return

        phys = self._def.physics

        # Variable gravity: holding up = lighter fall (original Scratch behavior)
        if self.in_air:
            g = phys.gravity if self._input.up else phys.fast_fall_gravity
            self.vy -= g * gravity_scale
            self.vy = max(self.vy, -phys.max_fall_speed)

        self.vx *= phys.speed_decay
        self.vx = max(-phys.max_speed, min(phys.max_speed, self.vx))

        self.x += self.vx
        self.y += self.vy

    def action_step(self) -> None:
        """Process self._input and advance the action state machine."""
        if self.action == Action.DEAD:
            return

        if self.action == Action.RESPAWN:
            self._tick_respawn()
            return

        if self.invincibility > 0:
            self.invincibility -= 1

        self.action_frame += 1

        if self.action == Action.HITSTUN:
            self._tick_hitstun()
            return

        if self.action in (Action.ATTACK, Action.SMASH):
            self._tick_attack()
            return

        if self.action == Action.CHARGING:
            self._tick_charging()
            return

        self._tick_movement()

    def ground_update(self, stage) -> None:
        """Resolve platform collisions.  Called after physics_step()."""
        if self.action in (Action.DEAD, Action.RESPAWN):
            return

        new_y, new_vy, on_ground = stage.resolve_ground(
            self.x, self.y, self.vy,
            half_w=self.phys_half_w, half_h=self.phys_half_h,
            falling_through=self.falling_through,
        )
        if on_ground and not self.in_air:
            self.y = new_y
            self.vy = new_vy
        elif on_ground and self.in_air:
            self.y = new_y
            self.vy = 0.0
            self.in_air = False
            self.recover_used = False
            self.falling_through = False
            if self.action in (Action.AERIAL, Action.JUMP, Action.HITSTUN):
                self.action = Action.NONE
                self.action_frame = 0
        elif not on_ground:
            self.in_air = True

        if stage.is_dead(self.x, self.y):
            self._die()

    def apply_pending_damage(self) -> None:
        """Flush the damage buffer accumulated this tick."""
        if self._pending_damage > 0:
            self.damage_pct += self._pending_damage
            self.vx += self._pending_knockback_vx
            self.vy += self._pending_knockback_vy
            hitstun = int(
                HITSTUN_BASE
                + abs(self._pending_knockback_vx + self._pending_knockback_vy) * 2
            )
            self.action = Action.HITSTUN
            self.action_frame = hitstun
            self.in_air = True
        self._pending_damage = 0.0
        self._pending_knockback_vx = 0.0
        self._pending_knockback_vy = 0.0

    # ------------------------------------------------------------------
    # Combat API — called by Match after hitbox resolution
    # ------------------------------------------------------------------

    def receive_damage(
        self,
        dmg: float,
        knockback: float,
        x_origin: float,
        y_origin: float,
    ) -> None:
        if self.invincibility > 0:
            return

        damage_multiplier = 1.0 + self.damage_pct * 0.01
        dx = self.x - x_origin
        dy = self.y - y_origin
        magnitude = math.hypot(dx, dy) or 1.0

        kb_scale = (
            damage_multiplier
            * knockback
            * (1.0 / self._def.stats.weight)
        )
        self._pending_damage += dmg
        self._pending_knockback_vx += kb_scale * (dx / magnitude) * 20.0
        self._pending_knockback_vy += kb_scale * (dy / magnitude) * 20.0

    def get_active_hitbox(self) -> tuple[float, float, float, float] | None:
        """
        Return (hx, hy, hw, hh) for the current attack's active hitbox,
        or None if no active hitbox this frame.

        Hitbox key = phase + 1.  An empty SVG (is_empty=True) returns None.
        See module docstring for the full hitbox algorithm.
        """
        if self.action not in (Action.ATTACK, Action.SMASH):
            return None
        n = len(self._def.attacks)
        if self.attack_num < 1 or self.attack_num > n:
            return None

        atk = self._def.attacks[self.attack_num - 1]
        fd = atk.frame_data

        phase = 4
        for i, t in enumerate(fd):
            if self.action_frame < t:
                phase = i + 1
                break
        phase = max(1, min(4, phase))

        if self.has_hit and not atk.multihit:
            return None

        hb = self._def.get_attack_hitbox(self.attack_num, phase + 1)
        if hb.is_empty:
            return None

        hx = self.x + hb.center_x * self.facing
        hy = self.y + hb.center_y
        return hx, hy, hb.half_w, hb.half_h

    # ------------------------------------------------------------------
    # Internal state machine helpers
    # ------------------------------------------------------------------

    def _tick_movement(self) -> None:
        inp = self._input
        phys = self._def.physics

        pressing_left    = inp.left
        pressing_right   = inp.right
        pressing_up      = inp.up
        pressing_attack  = inp.attack
        pressing_forward = (pressing_right and not pressing_left) or \
                           (pressing_left  and not pressing_right)

        # Horizontal acceleration
        if pressing_right and not pressing_left:
            self.vx += phys.run_accel
            self.facing = 1.0
        elif pressing_left and not pressing_right:
            self.vx -= phys.run_accel
            self.facing = -1.0

        # Up aerial: attack + up while in air (consumes recover)
        if self.in_air and pressing_attack and pressing_up:
            if not (self.action == Action.AERIAL
                    and self.attack_num == 5
                    and self.action_frame > 6):
                self.recover_used = True
                self.vy = phys.jump_speed
                self._start_attack(5)
                return

        # Jump (ground only — no double jump in original game)
        if pressing_up and not pressing_attack:
            if not self.in_air:
                self.vy = phys.jump_speed
                self.in_air = True
                self.action = Action.JUMP
                self.action_frame = 0
                self.anim_frame = 0.0
                return

        # Attack (no up key)
        if pressing_attack and not pressing_up:
            if not self.in_air:
                # Ground: begin smash charge
                self.attack_num = 4 if pressing_forward else 2
                self.action = Action.CHARGING
                self.action_frame = 0
                self.charge_amount = 0.2
                return
            else:
                # Air: fair or nair
                atk_num = 3 if pressing_forward else 1
                self._start_attack(atk_num)
                return

        # Update action label and run animation
        if not self.in_air:
            if abs(self.vx) > 1.0:
                self.action = Action.RUN
                self.anim_frame = (
                    self.anim_frame + abs(self.vx) / 20.0
                ) % 8.0
            else:
                self.action = Action.NONE
                self.anim_frame = 0.0
        else:
            self.action = Action.AERIAL

    def _start_attack(self, attack_num: int) -> None:
        self.attack_num = attack_num
        self.action = Action.ATTACK
        self.action_frame = 0
        self.has_hit = False

    def _tick_attack(self) -> None:
        atk = self._def.attacks[self.attack_num - 1]
        if self.action_frame > atk.total_frames:
            self.action = Action.NONE if not self.in_air else Action.AERIAL
            self.action_frame = 0
            self.attack_num = 0
        elif self.action == Action.ATTACK:
            # Aerial attacks allow horizontal drift; SMASH is ground-only
            inp = self._input
            phys = self._def.physics
            if inp.right and not inp.left:
                self.vx += phys.run_accel
                self.facing = 1.0
            elif inp.left and not inp.right:
                self.vx -= phys.run_accel
                self.facing = -1.0

    def _tick_charging(self) -> None:
        inp = self._input
        if inp.attack:
            self.charge_amount = min(
                1.0, self.charge_amount + self._def.stats.charge_speed
            )
        else:
            # attack_num was locked in when charging started — direction is committed
            self.action = Action.SMASH
            self.action_frame = 0
            self.has_hit = False

    def _tick_hitstun(self) -> None:
        self.action_frame -= 1
        if self.action_frame <= 0:
            self.action = Action.NONE if not self.in_air else Action.AERIAL
            self.action_frame = 0

    def _tick_respawn(self) -> None:
        if self.action_frame == 0:
            # First tick: freeze character at spawn and reset combat state
            self.damage_pct = 0.0
            self.vx = 0.0
            self.vy = 0.0
            self.x = self._spawn_x
            self.y = self._spawn_y
            self.in_air = True
            self.anim_frame = 0.0
            self.attack_num = 0
            self.charge_amount = 0.0
            self.recover_used = False
            self.invincibility = RESPAWN_INVINCIBILITY

        self.action_frame += 1
        if self.action_frame > _ENTRANCE_FRAMES:
            self.action = Action.AERIAL
            self.action_frame = 0
            self.anim_frame = 0.0

    def _die(self) -> None:
        self.stocks -= 1
        if self.stocks > 0:
            self.action = Action.RESPAWN
            self.action_frame = 0
        else:
            self.action = Action.DEAD

    # ------------------------------------------------------------------
    # Renderer query — costume index for current animation state
    # ------------------------------------------------------------------

    _IDX_IDLE      = 3
    _IDX_RUN_START = 4
    _IDX_JUMP      = 12
    _IDX_FALL      = 13
    # _IDX_ATK_VIS[N][P] = costume index for attack N+1, visual frame P
    # Index 0 = charging pose (frame 1); indices 1-4 = phase 1-4 visuals
    _IDX_ATK_VIS   = [
        [14, 15, 16, 17, 18],
        [24, 25, 26, 27, 28],
        [34, 35, 36, 37, 38],
        [44, 45, 46, 47, 48],
        [54, 55, 56, 57, 58],
    ]

    def get_current_costume_index(self) -> int:
        act = self.action
        n_attacks = len(self._def.attacks)

        if act == Action.DEAD:
            return self._IDX_IDLE

        if act == Action.RESPAWN:
            return 64 + min(9, self.action_frame // 2)

        if act == Action.NONE:
            return self._IDX_IDLE

        if act == Action.RUN:
            return self._IDX_RUN_START + int(self.anim_frame) % 8

        if act == Action.JUMP:
            return self._IDX_JUMP

        if act == Action.AERIAL:
            return self._IDX_JUMP if self.vy > 0 else self._IDX_FALL

        if act == Action.HITSTUN:
            return self._IDX_FALL

        if act == Action.CHARGING and 1 <= self.attack_num <= n_attacks:
            return self._IDX_ATK_VIS[self.attack_num - 1][0]

        if act in (Action.ATTACK, Action.SMASH) and 1 <= self.attack_num <= n_attacks:
            vis_frames = self._IDX_ATK_VIS[self.attack_num - 1]
            atk = self._def.attacks[self.attack_num - 1]
            fd = atk.frame_data
            phase = 4
            for i, t in enumerate(fd):
                if self.action_frame < t:
                    phase = i + 1
                    break
            phase = max(1, min(4, phase))
            return vis_frames[phase]

        return self._IDX_IDLE

    # ------------------------------------------------------------------
    # Observation vector
    # ------------------------------------------------------------------

    def obs_vector(self) -> list[float]:
        return [
            self.x, self.y, self.vx, self.vy,
            self.damage_pct, float(self.stocks),
            float(self.action.value), float(self.action_frame),
            self.facing, float(self.in_air), self.charge_amount,
        ]

    @staticmethod
    def obs_size() -> int:
        return 11


# Module-level constants retained for use in tests.
PHYS_HALF_W: float = MINIUM_DEF.get_body_hitbox().half_w
PHYS_HALF_H: float = -MINIUM_DEF.get_body_hitbox().y1
