"""
QTableBot — a from-scratch tabular Q-learning brain.

A deliberately simple baseline, meant as a "can a learner beat the hand-tuned
heuristic?" experiment against `agents.smesh_bot.SmeshBot`. Unlike SmeshBot, it
declares no raycast SENSORS and has no hand-coded pursuit/charge/recovery logic —
everything it does is learned online, one tick at a time.

State (504 discrete states, all derived from BrainContext / the nearest opponent):
  - dx bin (7): horizontal offset to the opponent, bucketed
  - dy bin (3): vertical offset to the opponent, bucketed
  - self in_air (2)
  - opponent "threat" (2): opponent is ATTACK / SMASH / CHARGING
  - self edge zone (3): near the platform's left edge / center / near the right edge,
    using BrainContext.stage_platform_x1/x2
  - self hitstun (2)

Action space (8): every InputState combo of {left, right, up} crossed with attack,
excluding left+right and left+up/right+up combos:
  NONE, LEFT, RIGHT, UP, ATTACK, LEFT+ATTACK, RIGHT+ATTACK, UP+ATTACK.

Reward, computed each tick from the diff between consecutive ticks' (own/opponent
damage_pct, stocks) snapshots — mirrors SmeshLiteEnv._compute_reward:
  + dmg_scale * damage dealt to the opponent this tick
  + ko_reward if the opponent lost a stock
  - death_penalty if we lost a stock
  - time_penalty per tick (default 0.0)

Q-learning is plain tabular Q-learning (Q[s][a] += alpha * (r + gamma * max(Q[s']) -
Q[s][a])), updated every tick using the *previous* tick's (state, action) and the
reward/next-state observed now. `epsilon`/`alpha`/`gamma` are public mutable
attributes so a training loop can drive an exploration schedule. `new_episode()`
must be called between matches (clears the pending transition and resets
`episode_reward`), since the same instance is meant to be reused across many
matches so its q_table persists and grows.
"""
from __future__ import annotations

import json
import os
import random

from smeshlite.core.brain import CharacterBrain, BrainContext, InputState
from smeshlite.core.character import Action


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

_DX_BREAKS = (-250.0, -100.0, -25.0, 25.0, 100.0, 250.0)   # -> 7 bins
_DY_BREAKS = (-30.0, 30.0)                                 # -> 3 bins
_EDGE_MARGIN = 80.0

_THREAT_ACTIONS = (Action.ATTACK.value, Action.SMASH.value, Action.CHARGING.value)


def _bin(value: float, breaks: tuple[float, ...]) -> int:
    for i, b in enumerate(breaks):
        if value < b:
            return i
    return len(breaks)


def _edge_bin(x: float, platform_x1: float, platform_x2: float, margin: float = _EDGE_MARGIN) -> int:
    if x <= platform_x1 + margin:
        return 0
    if x >= platform_x2 - margin:
        return 2
    return 1


def _encode_state(context: BrainContext, target) -> tuple[int, int, int, int, int, int]:
    dx = target.x - context.x
    dy = target.y - context.y
    return (
        _bin(dx, _DX_BREAKS),
        _bin(dy, _DY_BREAKS),
        int(context.in_air),
        int(target.action in _THREAT_ACTIONS),
        _edge_bin(context.x, context.stage_platform_x1, context.stage_platform_x2),
        int(context.action == Action.HITSTUN.value),
    )


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

# (left, right, up, attack)
_ACTIONS: tuple[tuple[bool, bool, bool, bool], ...] = (
    (False, False, False, False),  # 0 NONE
    (True,  False, False, False),  # 1 LEFT
    (False, True,  False, False),  # 2 RIGHT
    (False, False, True,  False),  # 3 UP
    (False, False, False, True),   # 4 ATTACK
    (True,  False, False, True),   # 5 LEFT+ATTACK
    (False, True,  False, True),   # 6 RIGHT+ATTACK
    (False, False, True,  True),   # 7 UP+ATTACK
)


# ---------------------------------------------------------------------------
# QTableBot
# ---------------------------------------------------------------------------

# ctx:annotate QTableBot
class QTableBot(CharacterBrain):
    """Tabular Q-learning brain. Reuse one instance across matches to keep learning."""

    BRAIN_NAME = "Q-Table Bot"

    N_ACTIONS = len(_ACTIONS)

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        dmg_scale: float = 0.01,
        ko_reward: float = 1.0,
        death_penalty: float = 1.0,
        time_penalty: float = 0.0,
        q_table_path: str | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.dmg_scale = dmg_scale
        self.ko_reward = ko_reward
        self.death_penalty = death_penalty
        self.time_penalty = time_penalty

        self.q_table: dict[tuple[int, ...], list[float]] = {}
        self.episode_reward = 0.0

        # (state, action, snapshot) from the previous tick, awaiting this tick's
        # reward/next-state before it can be used in a Q-update.
        self._pending: tuple[tuple[int, ...], int, tuple[float, int, float, int]] | None = None

        if q_table_path is not None and os.path.exists(q_table_path):
            self.load(q_table_path)

    # ------------------------------------------------------------------

    @property
    def q_table_size(self) -> int:
        return len(self.q_table)

    def new_episode(self) -> None:
        """Call between matches: clears the pending transition and episode reward."""
        self._pending = None
        self.episode_reward = 0.0

    # ------------------------------------------------------------------

    def think(self, context: BrainContext, out: InputState) -> None:
        out.clear()
        if not context.opponents:
            return
        target = context.opponents[0]

        state = _encode_state(context, target)
        snapshot = (context.damage_pct, context.stocks, target.damage_pct, target.stocks)

        if self._pending is not None:
            prev_state, prev_action, prev_snapshot = self._pending
            reward = self._reward(prev_snapshot, snapshot)
            self.episode_reward += reward
            self._update_q(prev_state, prev_action, reward, state)

        action = self._select_action(state)
        self._apply_action(action, out)
        self._pending = (state, action, snapshot)

    # ------------------------------------------------------------------

    def _reward(
        self,
        prev_snapshot: tuple[float, int, float, int],
        snapshot: tuple[float, int, float, int],
    ) -> float:
        _prev_self_dmg, prev_self_stocks, prev_opp_dmg, prev_opp_stocks = prev_snapshot
        _self_dmg, self_stocks, opp_dmg, opp_stocks = snapshot

        reward = -self.time_penalty
        dmg_dealt = max(0.0, opp_dmg - prev_opp_dmg)
        reward += dmg_dealt * self.dmg_scale
        if opp_stocks < prev_opp_stocks:
            reward += self.ko_reward
        if self_stocks < prev_self_stocks:
            reward -= self.death_penalty
        return reward

    def _select_action(self, state: tuple[int, ...]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.N_ACTIONS)
        q = self.q_table.get(state)
        if q is None:
            return random.randrange(self.N_ACTIONS)
        best = max(q)
        candidates = [i for i, v in enumerate(q) if v == best]
        return random.choice(candidates)

    def _update_q(self, state: tuple[int, ...], action: int, reward: float, next_state: tuple[int, ...]) -> None:
        q = self.q_table.setdefault(state, [0.0] * self.N_ACTIONS)
        next_q = self.q_table.get(next_state)
        next_max = max(next_q) if next_q is not None else 0.0
        q[action] += self.alpha * (reward + self.gamma * next_max - q[action])

    @staticmethod
    def _apply_action(action: int, out: InputState) -> None:
        out.left, out.right, out.up, out.attack = _ACTIONS[action]

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q_table": {"_".join(str(x) for x in k): v for k, v in self.q_table.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.q_table = {
            tuple(int(p) for p in k.split("_")): v
            for k, v in data.get("q_table", {}).items()
        }
