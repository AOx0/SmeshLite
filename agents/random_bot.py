"""
RandomBot — presses random buttons every frame.

Useful as a baseline opponent / smoke test for the brain registry.
"""
from __future__ import annotations

import random

from smeshlite.core.brain import CharacterBrain, BrainContext, InputState


class RandomBot(CharacterBrain):
    """Mashes random inputs each tick."""

    BRAIN_NAME = "Random Bot"

    def think(self, context: BrainContext, out: InputState) -> None:
        out.left   = random.random() < 0.5
        out.right  = random.random() < 0.5
        out.up     = random.random() < 0.1
        out.attack = random.random() < 0.2
