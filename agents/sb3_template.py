"""
SB3BrainTemplate — wraps a trained Stable-Baselines3 checkpoint as a brain.

Usage:
    1. Train a model against SmeshLiteEnv and save it:
           model.save("agents/checkpoints/my_model.zip")
    2. match.set_player_brain(1, SB3BrainTemplate("agents/checkpoints/my_model.zip"))

Requires the optional 'train' extra: pip install -e ".[train]"

Like PlayerInput's deferred pygame import, the stable_baselines3 import is
deferred to __init__ so this file can still be discovered/imported by the
brain registry even if SB3 isn't installed — only instantiating it requires
the dependency.
"""
from __future__ import annotations

import numpy as np

from smeshlite.core.brain import (
    CharacterBrain, BrainContext, InputState, brain_context_to_obs,
)


class SB3BrainTemplate(CharacterBrain):
    """
    Drives inputs from a trained SB3 model (e.g. PPO) using the same
    22-float observation vector as SmeshLiteEnv (Box(22,)).

    The action is expected to be MultiBinary(4) — [left, right, up, attack],
    matching SmeshLiteEnv.action_space.
    """

    BRAIN_NAME = "SB3 Agent (template)"

    def __init__(self, checkpoint_path: str = "agents/checkpoints/model.zip", deterministic: bool = True) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "SB3BrainTemplate requires stable-baselines3. "
                "Install with: pip install -e \".[train]\""
            ) from exc

        self._model = PPO.load(checkpoint_path)
        self._deterministic = deterministic

    def think(self, context: BrainContext, out: InputState) -> None:
        obs = np.array(brain_context_to_obs(context), dtype=np.float32)
        action, _ = self._model.predict(obs, deterministic=self._deterministic)

        out.left   = bool(action[0])
        out.right  = bool(action[1])
        out.up     = bool(action[2])
        out.attack = bool(action[3])
