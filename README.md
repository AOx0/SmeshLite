# SmeshLite 🥊

SmeshLite is a Python port of the fighting-game engine from **ULTRA MEGA SMESH**
(a Scratch 3.0 platform fighter) into a [Gymnasium](https://gymnasium.farama.org/)
reinforcement-learning environment.

It faithfully reproduces the original game's physics, 4-button input scheme, and
attack/hitbox system for the **MINIUM** character, and exposes it both as:

- a **playable 2-player sandbox** (`demo.py`) for humans, and
- a **Gym environment** (`SmeshLiteEnv`) for training RL agents.

---

## ✨ Features

- Faithful physics: gravity, friction, jumping, variable-gravity float / fast-fall
- 5-attack combat system with real per-attack hitboxes parsed from the original SVG sprites
- Charge-damage multiplier on smash attacks
- Pygame rendering (human window or `rgb_array` for recording)
- 1 or 2 player matches, with stocks, time limit, and respawn/kill-zone logic
- Pluggable "brain" system — wire up keyboard input, scripted AI, or an RL policy

---

## 📦 Requirements

- **Python 3.11+**
- [pygame](https://www.pygame.org/) (only needed for rendering / playing the demo)

---

## 🚀 Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd SmeshLite
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   ```

   Activate it:

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - macOS / Linux: `source .venv/bin/activate`

3. **Install the package**

   For just the core RL environment:

   ```bash
   pip install -e .
   ```

   To also play the local demo / use the renderer (recommended):

   ```bash
   pip install -e ".[render]"
   ```

   To additionally install Stable-Baselines3 for training agents:

   ```bash
   pip install -e ".[render,train]"
   ```

---

## 🎮 Playing the demo

Run the 2-player sandbox:

```bash
python demo.py
```

**Controls (4-button scheme, matches the original game):**

| Player | Left/Right | Jump | Attack / Charge |
|--------|-----------|------|-----------------|
| P1     | `A` / `D` | `W`  | `S`             |
| P2     | `←` / `→` | `↑`  | `↓`             |

**Pause menu** (`ESC` to open):

| Key | Action |
|-----|--------|
| `ESC` / `Space` / `Enter` | Resume |
| `R` | Restart match |
| `Q` | Quit |

---

## 🤖 Using the Gym environment

```python
from smeshlite import SmeshLiteEnv

env = SmeshLiteEnv(render_mode="human")  # or "none" / "rgb_array"
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

- **Observation space**: `Box(22,)` — 11 floats for each of the 2 players (self + opponent)
- **Action space**: `MultiBinary(4)` — `[left, right, up, attack]`
- **Reward** (configurable via `reward_config`): damage dealt per step, `+1.0` on KO,
  `-1.0` on being KO'd

For self-play or full multi-agent control, set custom brains directly via
`env.match.characters[i].set_brain(...)` (see `smeshlite/core/brain.py`).

---

## 🧪 Running tests

```bash
pip install -e ".[render]"
pytest
```

---

## 📁 Project structure

```
smeshlite/
├── env.py                  # SmeshLiteEnv (Gymnasium environment)
├── core/
│   ├── character.py         # Physics, state machine, combat
│   ├── brain.py              # Input/controller abstractions (player, AI, RL)
│   ├── match.py               # Match tick coordinator, hitbox resolution
│   └── stage.py                # Platform collision, stage bounds
├── data/
│   ├── character_def.py     # Character JSON/SVG loading & hitbox parsing
│   ├── stage_data.py          # Stage layout
│   └── characters/Minium/      # MINIUM stats, attacks, costumes, sprites
└── render/
    ├── renderer.py           # Pygame renderer (human / rgb_array)
    └── sprite_loader.py        # SVG sprite loading

demo.py        # Player-controlled 2P sandbox
format_json.py # Pretty-prints the original Scratch project.json for reference
tests/         # Pytest test suite
```

---

## 📜 Source

The original Scratch 3.0 game lives in `ULTRA MEGA SMESH/project.json`. It's only
needed if you want to add new characters or dig into the original game logic —
run `format_json.py` to produce a readable `project_formatted.json`.
