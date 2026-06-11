"""
SmeshLiteEnv -- Gymnasium environment wrapping the SmeshLite match engine.

Observation mode (default: full):
  obs_mode="full": 53-float vector (1v1, max_sensors=8). Includes invincibility,
      stage geometry, and sensor raycast results per slot.
      See brain.brain_context_to_full_obs() docstring for layout.
  obs_mode="minimal": 22-float vector (11 per player), same as
      Character.obs_vector(): [x, y, vx, vy, damage_pct, stocks, action,
      action_frame, facing, in_air, charge_amount]

Action mode (default: multi):
  action_mode="multi": step(actions) takes an (n_players, 4) array, one row
      per player. For self-play and multi-agent training.
  action_mode="single": step(action) takes a single 4-element array
      for player 0. Other players idle (or use opponent_brain).

Action space (MultiBinary((n_players, 4)) by default, or MultiBinary(4)):
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
    CharacterBrain, ExternalBrain, InputState, brain_context_to_full_obs,
)
from smeshlite.core.brain import full_obs_size as _full_obs_size
from smeshlite.data.stage_data import DEFAULT_STAGE, StageData


_OBS_LOW  = -2000.0
_OBS_HIGH =  2000.0
_OBS_SIZE = 22   # 11 floats x 2 players (minimal mode)


def _resolve_opponent_brain(
    opponent_brain: CharacterBrain | type[CharacterBrain] | str | None,
) -> CharacterBrain | None:
    """Resolve an opponent_brain spec to a CharacterBrain instance, or None."""
    if opponent_brain is None:
        return None
    if isinstance(opponent_brain, CharacterBrain):
        return opponent_brain
    if isinstance(opponent_brain, type) and issubclass(opponent_brain, CharacterBrain):
        return opponent_brain()
    if isinstance(opponent_brain, str):
        from smeshlite.core import brain_registry
        return brain_registry.create_brain(opponent_brain)
    raise TypeError(
        f"opponent_brain must be a CharacterBrain instance, subclass, "
        f"registry name string, or None; got {type(opponent_brain)!r}"
    )


class SmeshLiteEnv(gym.Env):
    """
    SmeshLite fighting game environment.

    Example (default -- multi-agent, full obs)::

        env = SmeshLiteEnv()
        obs, info = env.reset()
        for _ in range(1000):
            actions = env.action_space.sample()  # shape (2, 4)
            obs, reward, terminated, truncated, info = env.step(actions)

    Example (single-agent vs SmeshBot)::

        env = SmeshLiteEnv(action_mode="single", opponent_brain="Smesh Bot")
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

    Info dict keys (returned by step):
        frame:           match frame number
        winner:          winner player_id or None
        actions:         list of [left,right,up,attack] per player (env-injected)
        applied_actions: list of [left,right,up,attack] per player (what brains chose)
        damage_events:   list of {attacker, target, damage} dicts from this tick
        action_masks:    list of [left,right,up,attack] booleans per player
        obs:             list of obs arrays, one per player (each player's perspective)
        rewards:         list of float, one per player (symmetric zero-sum)
        damage:          list of damage_pct per player
        stocks:          list of stocks per player
        prev_obs:        obs array from the previous step (None after reset)
    """

    metadata = {"render_modes": ["none", "human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = "none",
        obs_mode: str = "full",
        action_mode: str = "multi",
        n_players: int = 2,
        max_sensors: int = 8,
        stocks: int = 3,
        time_limit: int = 7200,
        stage_data: StageData | None = None,
        gravity_scale: float = 1.0,
        knockback_scale: float = 1.0,
        reward_config: dict | None = None,
        opponent_brain: CharacterBrain | type[CharacterBrain] | str | None = None,
    ) -> None:
        super().__init__()

        assert render_mode in self.metadata["render_modes"], \
            f"render_mode must be one of {self.metadata['render_modes']}"
        assert n_players >= 1, "n_players must be >= 1"
        assert obs_mode in ("minimal", "full"), \
            f"obs_mode must be 'minimal' or 'full', got {obs_mode!r}"
        assert action_mode in ("single", "multi"), \
            f"action_mode must be 'single' or 'multi', got {action_mode!r}"

        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.n_players = n_players
        self.max_sensors = max_sensors
        self._reward_cfg = reward_config or {}

        # Resolve opponent_brain at init time (for instances/classes)
        # String names are resolved at each reset() so new agents are picked up.
        self._opponent_brain_spec = opponent_brain
        self._opponent_brain: CharacterBrain | None = None
        if not isinstance(opponent_brain, str):
            self._opponent_brain = _resolve_opponent_brain(opponent_brain)

        config = MatchConfig(
            stocks=stocks,
            time_limit=time_limit,
            stage_data=stage_data or DEFAULT_STAGE,
            gravity_scale=gravity_scale,
            knockback_scale=knockback_scale,
        )
        self.match = Match(config)

        # Observation space
        if obs_mode == "minimal":
            obs_size = Match.obs_size(n_players)
        elif obs_mode == "full":
            obs_size = _full_obs_size(n_players, max_sensors)
        self.observation_space = spaces.Box(
            low=_OBS_LOW, high=_OBS_HIGH,
            shape=(obs_size,), dtype=np.float32,
        )

        # Action space
        if action_mode == "single":
            self.action_space = spaces.MultiBinary(4)
        elif action_mode == "multi":
            self.action_space = spaces.MultiBinary((n_players, 4))

        self._brains: list[ExternalBrain] = []
        self._renderer = None
        self._last_obs: np.ndarray | None = None

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

        # Resolve string opponent_brain at reset time
        if isinstance(self._opponent_brain_spec, str):
            self._opponent_brain = _resolve_opponent_brain(self._opponent_brain_spec)

        # Wire brains to characters
        self._brains = []
        for i, char in enumerate(self.match.characters):
            if self.action_mode == "multi" or i == 0:
                # Env-driven player: ExternalBrain (env sets pending each step)
                brain = ExternalBrain()
                char.set_brain(brain)
                self._brains.append(brain)
            elif self._opponent_brain is not None:
                # Single mode + opponent brain: set the bot directly
                char.set_brain(self._opponent_brain)
                self._brains.append(ExternalBrain())  # placeholder (not connected)
            else:
                # Single mode, no opponent brain: idle ExternalBrain
                brain = ExternalBrain()
                char.set_brain(brain)
                self._brains.append(brain)

        # Restore from scenario or state dict if provided via options
        if options:
            if "scenario" in options:
                from smeshlite.scenarios import get_scenario
                scenario = get_scenario(options["scenario"])
                self.match.set_state(scenario.state)
            elif "state" in options:
                self.match.set_state(options["state"])

        # One idle tick to settle spawn
        self.match.tick()

        obs = self._get_obs()
        self._last_obs = obs
        info = {
            "frame": self.match.frame,
            "prev_obs": None,  # no previous obs on first step
        }
        return obs, info

    def step(
        self,
        action,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the environment by one tick.

        In action_mode="single": control player 0 with a 4-element array
            [left, right, up, attack]. Other players idle or use opponent_brain.

        In action_mode="multi": control all players with an (n_players, 4) array.
        """
        prev_dmg    = [c.damage_pct for c in self.match.characters]
        prev_stocks = [c.stocks     for c in self.match.characters]

        # Inject actions into ExternalBrains
        if self.action_mode == "single":
            self._brains[0].pending = InputState(
                left=bool(action[0]),
                right=bool(action[1]),
                up=bool(action[2]),
                attack=bool(action[3]),
            )
            for brain in self._brains[1:]:
                brain.pending = InputState()
        else:  # "multi"
            for i, brain in enumerate(self._brains):
                if i < len(action):
                    brain.pending = InputState(
                        left=bool(action[i][0]),
                        right=bool(action[i][1]),
                        up=bool(action[i][2]),
                        attack=bool(action[i][3]),
                    )
                else:
                    brain.pending = InputState()

        info = self.match.tick()

        # Record requested actions (what the env injected)
        info["actions"] = [
            [b.pending.left, b.pending.right, b.pending.up, b.pending.attack]
            for b in self._brains
        ]

        # Record applied actions (what brains actually chose, after tick)
        info["applied_actions"] = [
            [c._input.left, c._input.right, c._input.up, c._input.attack]
            for c in self.match.characters
        ]

        obs = self._get_obs()
        rewards = self._compute_rewards(prev_dmg, prev_stocks)
        terminated = self.match.done
        truncated = False

        # Primary reward: P0's reward (matches existing API)
        reward = rewards[0]

        info.update({
            # Existing
            "frame":     self.match.frame,
            "winner":    self.match.winner,
            "actions":        info["actions"],
            "applied_actions": info["applied_actions"],
            "damage_events":  info.get("damage_events", []),

            # NEW: action masks (game logic)
            "action_masks": [
                self.match.characters[i].action_mask()
                for i in range(self.n_players)
            ],

            # NEW: per-agent observations
            "obs": [
                self._get_obs(i) for i in range(self.n_players)
            ],

            # NEW: per-agent rewards (symmetric zero-sum)
            "rewards": rewards,

            # NEW: per-player stats (generalized)
            "damage": [c.damage_pct for c in self.match.characters],
            "stocks": [c.stocks for c in self.match.characters],

            # NEW: prev obs for (s_t, a_t, s_{t+1}) recording
            "prev_obs": self._last_obs,
        })

        self._last_obs = obs

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

    def _get_obs(self, player_id: int = 0) -> np.ndarray:
        """Return observation from the given player's perspective."""
        if self.obs_mode == "minimal":
            return np.array(
                self.match.get_obs(perspective=player_id),
                dtype=np.float32,
            )
        else:  # "full"
            char = self.match.characters[player_id]
            opponents = [c for c in self.match.characters if c.id != player_id]
            ctx = char._build_context(opponents, self.match.stage)
            return np.array(
                brain_context_to_full_obs(ctx, self.max_sensors),
                dtype=np.float32,
            )

    def _compute_rewards(
        self,
        prev_dmg: list[float],
        prev_stocks: list[float],
    ) -> list[float]:
        """Compute reward for each player. Returns list of length n_players.

        For 1v1 the rewards are symmetric zero-sum: P0's gain is P1's loss.
        The engine provides raw computation; shaping is a training concern.
        """
        chars = self.match.characters
        rewards = []

        for i in range(self.n_players):
            r = 0.0
            # Damage dealt to opponents (sum across all opponents)
            for j in range(self.n_players):
                if j == i:
                    continue
                dmg_dealt = max(0.0, chars[j].damage_pct - prev_dmg[j])
                r += dmg_dealt * self._reward_cfg.get("dmg_scale", 0.01)
                if chars[j].stocks < prev_stocks[j]:
                    r += self._reward_cfg.get("ko_reward", 1.0)

            # Damage received / death penalty
            dmg_received = max(0.0, chars[i].damage_pct - prev_dmg[i])
            r -= dmg_received * self._reward_cfg.get("dmg_received_scale", 0.01)
            if chars[i].stocks < prev_stocks[i]:
                r -= self._reward_cfg.get("death_penalty", 1.0)

            rewards.append(float(r))

        # For 2 players, enforce strict zero-sum: r[1] = -r[0]
        # This makes self-play mathematically consistent.
        # The individual terms above are kept for clarity / n_players > 2 future use.
        if self.n_players == 2:
            rewards[1] = -rewards[0]

        return rewards

    def _get_renderer(self):
        if self._renderer is None:
            from smeshlite.render.renderer import Renderer
            self._renderer = Renderer(render_mode=self.render_mode)
        return self._renderer
