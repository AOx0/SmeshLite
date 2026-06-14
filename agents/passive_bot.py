"""PassiveBot -- a brain that does nothing (stands still).

Useful as a training opponent to teach agents to approach and
initiate combat, preventing the passive-opponent failure mode
where the agent waits for the opponent to come to it.
"""
from __future__ import annotations

from smeshlite.core.brain import CharacterBrain, BrainContext, InputState


class PassiveBot(CharacterBrain):
    """Stands completely still. Never moves, jumps, or attacks."""

    BRAIN_NAME = "Passive Bot"

    def think(self, context: BrainContext, out: InputState) -> None:
        out.clear()
