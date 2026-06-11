"""
SmeshLiteEnv -- Gymnasium environment wrapping the SmeshLite match engine.

Observation modes:
  obs_mode="minimal" (default): 22-float vector (11 per player), same as
      Character.obs_vector(): [x, y, vx, vy, damage_pct, stocks, action,
      action_frame, facing, in_air, charge_amount]
  obs_mode="full": 53-float vector (1v1, max_sensors=8). Includes invincibility,
      stage geometry, and sensor raycast results per slot.
      See brain.brain_context_to_full_obs() docstring for layout.

Action space (MultiBinary(4)):
  [left, right, up, attack]
  Matches the original 4-button scheme: A/D/W/S (or arrow keys).

Reward (default, configurable via reward_config):
  +damage_dealt / 100   per step
  +1.0                  on KO
  -1.0                  on being KO'd

render_mode:
  "none"      -- headless (fastest, for training)
  "human"     -- opens a Pygame window
  "rgb_array" -- returns RGB array (for recording)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from smeshlite.core.match import Match, MatchConfig
from smeshlite.core.brain import (
    ExternalBrain, InputState, brain_context_to_full_obs,
)
from smeshlite.core.brain import full_obs_size as _full_obs_size
from smeshlite.data.stage_data import DEFAULT_STAGE, StageData


_OBS_LOW  = -2000.0
_OBS_HIGH =  2000.0
_OBS_SIZE = 22   # 11 floats x 2 players (minimal mode)


class SmeshLiteEnv(gym.Env):
    """
    1v1 SmeshLite environment.

    Example::

        env = SmeshLiteEnv()
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

    For self-play, set brains on both characters via match.characters[i].set_brain(...),
    or match.set_player_brain(i, "BrainName") to pick from /agents. Note this must be
    re-applied after each reset(), since reset() rewires ExternalBrain to every character
    (required so step() can drive player 0).
    """

    metadata = {"render_modes": ["none", "human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = "none",
        obs_mode: str = "minimal",
        n_players: int = 2,
        max_sensors: int = 8,
        stocks: int = 3,
        time_limit: int = 7200,
        stage_data: StageData | None = None,
        gravity_scale: float = 1.0,
        knockback_scale: float = 1.0,
        reward_config: dict | None = None,
    ) -> None:
        super().__init__()

        assert render_mode in self.metadata["render_modes"], \
            f"render_mode must be one of {self.metadata['render_modes']}"
        assert n_players >= 1, "n_players must be >= 1"
        assert obs_mode in ("minimal", "full"), \
            f"obs_mode must be 'minimal' or 'full', got {obs_mode!r}"

        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.n_players = n_players
        self.max_sensors = max_sensors
        self._reward_cfg = reward_config or {}

        config = MatchConfig(
            stocks=stocks,
            time_limit=time_limit,
            stage_data=stage_data or DEFAULT_STAGE,
            gravity_scale=gravity_scale,
            knockback_scale=knockback_scale,
        )
        self.match = Match(config)

        if obs_mode == "minimal":
            obs_size = Match.obs_size(n_players)
        elif obs_mode == "full":
            obs_size = _full_obs_size(n_players, max_sensors)
        self.observation_space = spaces.Box(
            low=_OBS_LOW, high=_OBS_HIGH,
            shape=(obs_size,), dtype=np.float32,
        )
        # 4-button scheme: [left, right, up, attack]
        self.action_space = spaces.MultiBinary(4)

        self._brains: list[ExternalBrain] = []
        self._renderer = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.match.reset(n_players=self.n_players)

        # Wire ExternalBrain to every character so the env drives all inputs
        self._brains = []
        for char in self.match.characters:
            brain = ExternalBrain()
            char.set_brain(brain)
            self._brains.append(brain)

        # Restore from state dict if provided via options
        if options and "state" in options:
            self.match.set_state(options["state"])

        # One idle tick to settle spawn
        self.match.tick()

        obs = self._get_obs()
        info = {"frame": self.match.frame}
        return obs, info

    def step(
        self,
        action,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Control player 0 with `action` ([left, right, up, attack]).
        All other players receive a no-op idle input.
        For self-play or multi-agent control, set brains directly on characters.
        """
        prev_dmg    = [c.damage_pct for c in self.match.characters]
        prev_stocks = [c.stocks     for c in self.match.characters]

        # Inject P0 action; others stay idle (pending = InputState())
        self._brains[0].pending = InputState(
            left=bool(action[0]),
            right=bool(action[1]),
            up=bool(action[2]),
            attack=bool(action[3]),
        )
        for brain in self._brains[1:]:
            brain.pending = InputState()

        info = self.match.tick()

        obs = self._get_obs()
        reward = self._compute_reward(prev_dmg, prev_stocks)
        terminated = self.match.done
        truncated = False

        info.update({
            "frame":     self.match.frame,
            "winner":    self.match.winner,
            "p0_damage": self.match.characters[0].damage_pct,
            "p0_stocks": self.match.characters[0].stocks,
        })
        if self.n_players > 1:
            info["p1_damage"] = self.match.characters[1].damage_pct
            info["p1_stocks"] = self.match.characters[1].stocks

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "none":
            return None
        renderer = self._get_renderer()
        return renderer.render(self.match)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == "minimal":
            return np.array(
                self.match.get_obs(perspective=0),
                dtype=np.float32,
            )
        else:  # "full"
            char = self.match.characters[0]
            opponents = [c for c in self.match.characters if c.id != 0]
            ctx = char._build_context(opponents, self.match.stage)
            return np.array(
                brain_context_to_full_obs(ctx, self.max_sensors),
                dtype=np.float32,
            )

    def _compute_reward(
        self,
        prev_dmg: list[float],
        prev_stocks: list[float],
    ) -> float:
        reward = 0.0
        chars = self.match.characters

        if len(chars) > 1:
            dmg_dealt = max(0.0, chars[1].damage_pct - prev_dmg[1])
            reward += dmg_dealt * self._reward_cfg.get("dmg_scale", 0.01)
            if chars[1].stocks < prev_stocks[1]:
                reward += self._reward_cfg.get("ko_reward", 1.0)

        if chars[0].stocks < prev_stocks[0]:
            reward -= self._reward_cfg.get("death_penalty", 1.0)

        return float(reward)

    def _get_renderer(self):
        if self._renderer is None:
            from smeshlite.render.renderer import Renderer
            self._renderer = Renderer(render_mode=self.render_mode)
        return self._renderer
