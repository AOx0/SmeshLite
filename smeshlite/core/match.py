"""
Match: tick coordinator.

Tick order per frame:
  1. gather_input  — each Character polls its CharacterBrain
  2. action_step   — advance each Character's state machine (reads _input buffer)
  3. physics_step  — gravity + velocity integration
  4. ground_update — platform collision, kill zone
  5. hitbox step   — resolve attack collisions, schedule damage
  6. damage_flush  — apply accumulated damage / knockback
  7. win_check     — determine if match is over
"""
from __future__ import annotations
from dataclasses import dataclass, field

from smeshlite.core.character import Character, Action
from smeshlite.core.stage import Stage
from smeshlite.data.stage_data import DEFAULT_STAGE, StageData
from smeshlite.data.character_def import CharacterDef, MINIUM_DEF


@dataclass
class MatchConfig:
    stocks: int = 3
    time_limit: int = 7200
    stage_data: StageData = field(default_factory=lambda: DEFAULT_STAGE)
    gravity_scale: float = 1.0
    knockback_scale: float = 1.0
    character_def: CharacterDef | None = None  # None = MINIUM_DEF for all players


class Match:
    """Single 1vN match simulation."""

    def __init__(self, config: MatchConfig | None = None) -> None:
        self.config = config or MatchConfig()
        self.stage = Stage(self.config.stage_data)
        self.characters: list[Character] = []
        self.frame: int = 0
        self.done: bool = False
        self.winner: int | None = None

    def reset(self, n_players: int = 2) -> None:
        char_def = self.config.character_def or MINIUM_DEF
        spawns = [self.stage.spawn_p1, self.stage.spawn_p2]
        facings = [1.0, -1.0]
        self.characters = [
            Character(
                player_id=i,
                spawn_x=spawns[i % len(spawns)][0],
                spawn_y=spawns[i % len(spawns)][1],
                character_def=char_def,
                facing=facings[i % len(facings)],
                stocks=self.config.stocks,
            )
            for i in range(n_players)
        ]
        self.frame = 0
        self.done = False
        self.winner = None

    # ------------------------------------------------------------------
    # Main tick — no external actions needed; each Character uses its brain
    # ------------------------------------------------------------------

    def tick(self) -> dict:
        if self.done:
            return {}

        self.frame += 1
        info: dict = {"damage_events": []}

        # 1. Poll brains → fill each character's _input buffer
        for char in self.characters:
            opponents = [c for c in self.characters if c is not char]
            char.gather_input(opponents, self.stage)

        # 2. Action state machines
        for char in self.characters:
            char.action_step()

        # 3. Physics
        for char in self.characters:
            char.physics_step(self.config.gravity_scale)

        # 4. Ground collision + kill zone
        for char in self.characters:
            char.ground_update(self.stage)

        # 5. Melee hitbox resolution
        for attacker in self.characters:
            hb = attacker.get_active_hitbox()
            if hb is None:
                continue
            hx, hy, hw, hh = hb
            atk_data = attacker._def.attacks[attacker.attack_num - 1]
            for target in self.characters:
                if target.id == attacker.id:
                    continue
                if target.action == Action.DEAD:
                    continue
                if (
                    abs(hx - target.x) < hw + target.phys_half_w
                    and abs(hy - target.y) < hh + target.phys_half_h
                ):
                    charge = attacker.charge_amount if attacker.action == Action.SMASH else 1.0
                    target.receive_damage(
                        atk_data.damage   * charge,
                        atk_data.knockback * self.config.knockback_scale * charge,
                        attacker.x,
                        attacker.y,
                    )
                    attacker.has_hit = True
                    info["damage_events"].append({
                        "attacker": attacker.id,
                        "target":   target.id,
                        "damage":   atk_data.damage,
                    })

        # 6. Flush damage
        for char in self.characters:
            char.apply_pending_damage()

        # 7. Win check
        self._check_win()

        return info

    def _check_win(self) -> None:
        alive = [c for c in self.characters if c.action != Action.DEAD]
        if len(self.characters) > 1 and len(alive) <= 1:
            self.done = True
            self.winner = alive[0].id if alive else None
            return
        if self.config.time_limit > 0 and self.frame >= self.config.time_limit:
            self.done = True
            best = max(self.characters, key=lambda c: (c.stocks, -c.damage_pct))
            self.winner = best.id

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_obs(self, perspective: int = 0) -> list[float]:
        obs: list[float] = []
        obs.extend(self.characters[perspective].obs_vector())
        for char in self.characters:
            if char.id != perspective:
                obs.extend(char.obs_vector())
        return obs

    @staticmethod
    def obs_size(n_players: int = 2) -> int:
        return Character.obs_size() * n_players
