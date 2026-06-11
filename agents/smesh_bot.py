"""
SmeshBOT — port of the original ULTRA MEGA SMESH CPU opponent.

Source: `(AI Behavior) Master Follow` / `React` / `Attack` / `Charge Attack` procedures
in ULTRA MEGA SMESH/project.json (Character sprite). The original AI is a blocking-loop
state machine driven by a live "Level Ground Records" map built by walking the stage;
this port keeps its DECISION structure but replaces that map-building navigation with
direct opponent-tracking + the RaycastSensor system (ground_ahead/ground_below), since
SmeshLite is a fixed-stage, step-based env. Kept from the original:
  - pursue the opponent, jumping over gaps / onto platforms
  - roll Attack Frequency when within Aggresiveness Distance: quick Attack while
    airborne, Charge Attack while grounded
  - Charge Attack: pick a charge target (lower if opponent's damage > KO Hunger,
    "Likes Charging Up To" otherwise), keep charging while the opponent stays roughly
    in the original aim direction (dot product >= 0.5), bail on timeout/launch/falling,
    and self-tune a per-attack "sweetspot" distance toward whatever range actually hit
  - retreat from an invulnerable (respawning) opponent
  - fall-recovery: if airborne, falling, and no ground within reach, hold up+attack
    and steer back toward center
  - Groundpoint Exclusion Zone: how far the ground_ahead sensor probes for gaps
  - Max Target X Offset: a "predictive aim" jitter added ahead of the opponent (in
    their facing direction), re-rolled every Patience-in-secs window, so the bot
    walks toward where the opponent is heading rather than straight at them
  - Barrier Reaction Threshold: refuse to walk within this distance of the
    platform's x1/x2 edges while pursuing, and treat falling below -threshold
    as an extra fall-recovery trigger alongside the ground sensor
Not ported: multi-target selection, the Level Ground Records pathfinding, the
exact original recovery-move sequencing, and Allowed Player Offset (confirmed
unused/dead in the original AI logic).
"""
from __future__ import annotations

import math
import random

from smeshlite.core.brain import CharacterBrain, BrainContext, InputState
from smeshlite.core.character import Action
from smeshlite.core.sensors import RaycastSensor


class SmeshBot(CharacterBrain):
    """Pursue, jump, attack/charge in range, retreat from invulnerable foes, recover falls."""

    BRAIN_NAME = "Smesh Bot"

    def __init__(
        self,
        aggressiveness_distance: float = 150.0,
        attack_frequency: float = 0.80,
        jump_frequency: float = 0.50,
        ko_hunger: float = 40.0,
        likes_charging_up_to: float = 50.0,
        patience_frames: int = 90,
        attack_cooldown_frames: int = 25,
        groundpoint_exclusion_zone: float = 60.0,
        max_target_x_offset: float = 250.0,
        barrier_reaction_threshold: float = 200.0,
    ) -> None:
        self.aggressiveness_distance = aggressiveness_distance
        self.attack_frequency = attack_frequency
        self.jump_frequency = jump_frequency
        self.ko_hunger = ko_hunger
        self.likes_charging_up_to = likes_charging_up_to
        self.patience_frames = patience_frames
        self.attack_cooldown_frames = attack_cooldown_frames
        self.groundpoint_exclusion_zone = groundpoint_exclusion_zone
        self.max_target_x_offset = max_target_x_offset
        self.barrier_reaction_threshold = barrier_reaction_threshold

        # Per-instance so groundpoint_exclusion_zone tunes how far ahead it checks for gaps.
        self.SENSORS = (
            RaycastSensor(name="ground_ahead", origin=(40.0, 0.0), direction=(0.0, -1.0), max_dist=groundpoint_exclusion_zone),
            RaycastSensor(name="ground_below", origin=(0.0, 0.0), direction=(0.0, -1.0), max_dist=150.0),
        )

        self._frame = 0
        self._cooldown = 0
        self._sweetspots: dict[int, float] = {}

        self._target_offset = 0.0
        self._target_offset_frame = 0

        self._charging = False
        self._charge_dir = (0.0, 0.0)
        self._charge_target = 0.0
        self._charge_start_frame = 0

        self._watch_hit_frames = 0
        self._watch_distance = 0.0
        self._watch_attack_num = 0
        self._watch_prev_damage = 0.0

        self._recovery_used = False

    # ------------------------------------------------------------------

    def think(self, context: BrainContext, out: InputState) -> None:
        out.clear()
        self._frame += 1
        if self._cooldown > 0:
            self._cooldown -= 1
        if not context.in_air:
            self._recovery_used = False

        if self._watch_hit_frames > 0:
            self._watch_hit_frames -= 1
            if context.opponents and context.opponents[0].damage_pct > self._watch_prev_damage:
                prev = self._sweetspots.get(self._watch_attack_num, self._watch_distance)
                self._sweetspots[self._watch_attack_num] = (prev + self._watch_distance) / 2.0
                self._watch_hit_frames = 0

        if not context.opponents:
            return
        target = context.opponents[0]

        if self._react(context, target, out):
            return

        if self._charging:
            self._continue_charge(context, target, out)
            return

        dx = target.x - context.x
        dy = target.y - context.y

        if self._frame - self._target_offset_frame >= self.patience_frames or self._target_offset_frame == 0:
            self._target_offset = random.uniform(0.0, self.max_target_x_offset)
            self._target_offset_frame = self._frame
        facing_sign = 1.0 if target.facing >= 0 else -1.0
        pursue_dx = (target.x + self._target_offset * facing_sign) - context.x

        self._pursue(context, pursue_dx, dy, out)

        if self._cooldown == 0 and abs(dx) < self.aggressiveness_distance and abs(dy) < self.aggressiveness_distance:
            if random.random() < self.attack_frequency:
                if context.in_air:
                    if abs(dx) < self.aggressiveness_distance * 0.6:
                        self._attack(context, target, out)
                        self._cooldown = self.attack_cooldown_frames
                else:
                    self._start_charge(context, target)
                    out.clear()
                    out.attack = True
                    self._cooldown = self.attack_cooldown_frames

    # ------------------------------------------------------------------

    def _react(self, context: BrainContext, target, out: InputState) -> bool:
        """Safety net: retreat from an invulnerable foe, or recover a falling drop."""
        if target.action == Action.RESPAWN.value:
            out.clear()
            out.left = context.x > 0
            out.right = context.x <= 0
            self._charging = False
            return True

        ground = context.sensors.get("ground_below")
        falling_near_bottom = context.y < -self.barrier_reaction_threshold
        no_ground_below = ground is not None and not ground.hit
        if not self._recovery_used and context.in_air and context.vy < -2.0 and (no_ground_below or falling_near_bottom):
            self._recovery_used = True
            out.clear()
            out.up = True
            out.attack = True
            out.left = context.x > 0
            out.right = context.x <= 0
            self._charging = False
            return True

        return False

    def _pursue(self, context: BrainContext, dx: float, dy: float, out: InputState) -> None:
        out.right = dx > 0
        out.left = dx <= 0

        # Don't walk ourselves off the platform chasing the target (or a
        # predictive offset that happens to point off-stage).
        if out.left and context.x <= context.stage_platform_x1 + self.barrier_reaction_threshold:
            out.left, out.right = False, True
        elif out.right and context.x >= context.stage_platform_x2 - self.barrier_reaction_threshold:
            out.right, out.left = False, True

        ahead = context.sensors.get("ground_ahead")
        if dy > 30.0:
            out.up = True
        elif ahead is not None and not ahead.hit and not context.in_air:
            out.up = True
        elif random.random() < self.jump_frequency:
            out.up = True

    def _attack(self, context: BrainContext, target, out: InputState) -> None:
        out.up = False
        out.attack = True
        if abs(target.x - context.x) > abs(target.y - context.y):
            out.right = target.x > context.x
            out.left = not out.right

    def _start_charge(self, context: BrainContext, target) -> None:
        self._charging = True
        self._charge_start_frame = self._frame
        self._charge_target = 75.0 if target.damage_pct > self.ko_hunger else self.likes_charging_up_to

        dx, dy = target.x - context.x, target.y - context.y
        mag = math.hypot(dx, dy) or 1.0
        self._charge_dir = (dx / mag, dy / mag)

    def _continue_charge(self, context: BrainContext, target, out: InputState) -> None:
        out.clear()

        dx, dy = target.x - context.x, target.y - context.y
        distance = math.hypot(dx, dy) or 1.0
        cur_dir = (dx / distance, dy / distance)
        dot = cur_dir[0] * self._charge_dir[0] + cur_dir[1] * self._charge_dir[1]

        sweetspot = self._sweetspots.get(context.attack_num, 80.0)
        timed_out = (self._frame - self._charge_start_frame) > self.patience_frames
        in_range_and_charged = distance < sweetspot and context.charge_amount * 100 >= self._charge_target - 1

        if dot < 0.5 or context.action == Action.AERIAL.value or in_range_and_charged or timed_out or context.vy < -3.0:
            self._charging = False
            self._watch_hit_frames = 10
            self._watch_distance = distance
            self._watch_attack_num = context.attack_num
            self._watch_prev_damage = target.damage_pct
            return

        out.attack = True
        out.right = cur_dir[0] > 0
        out.left = not out.right
