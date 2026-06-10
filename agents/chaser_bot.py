"""
ChaserBot — simple rule-based bot.

Walks toward the nearest opponent and attacks once in range.
Demonstrates reading BrainContext.opponents without any ML dependencies.
"""
from __future__ import annotations

from smeshlite.core.brain import CharacterBrain, BrainContext, InputState


ATTACK_RANGE = 120.0


class ChaserBot(CharacterBrain):
    """Walks toward the closest opponent; attacks when close enough."""

    BRAIN_NAME = "Chaser Bot"

    def think(self, context: BrainContext, out: InputState) -> None:
        out.clear()
        if not context.opponents:
            return

        target = min(context.opponents, key=lambda o: abs(o.x - context.x))
        dx = target.x - context.x

        if abs(dx) < ATTACK_RANGE:
            out.attack = True
        elif dx > 0:
            out.right = True
        else:
            out.left = True

        # Jump if the opponent is significantly above us
        if target.y - context.y > 50.0:
            out.up = True
