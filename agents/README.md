# /agents — drop-in brain implementations

Drop a `.py` file in this folder and SmeshLite will auto-discover any
`CharacterBrain` subclass defined in it.

## Convention

- Define a class that subclasses `smeshlite.core.brain.CharacterBrain`
  and overrides `think(self, context: BrainContext, out: InputState) -> None`.
- Optionally set a class attribute `BRAIN_NAME = "My Cool Bot"` for a
  friendly display name. If omitted, the class name is used.
- Only classes *defined in your file* are registered — importing and
  re-exporting another module's brain won't register it again.
- If your file fails to import (syntax error, missing dependency), it is
  skipped with a warning — it won't break discovery of other agents.

## Using your brain

```python
from smeshlite.core.match import Match, MatchConfig

match = Match(MatchConfig())
match.reset(n_players=2)

# By name (auto-discovered from /agents):
match.set_player_brain(1, "My Cool Bot")

# Or import directly:
from agents.chaser_bot import ChaserBot
match.set_player_brain(1, ChaserBot)   # class — instantiated for you
match.set_player_brain(1, ChaserBot()) # or pass an instance directly
```

List all discovered brains:

```python
from smeshlite.core.match import Match
print(Match.list_available_brains())
```

## Examples in this folder

- `random_bot.py` — presses random buttons each frame.
- `chaser_bot.py` — simple rule-based bot that walks toward its opponent
  and attacks when close.
- `sb3_template.py` — wraps a trained Stable-Baselines3 `.zip` checkpoint
  as a brain, using `brain_context_to_obs()` to build the same 22-float
  observation vector that `SmeshLiteEnv` produces.
