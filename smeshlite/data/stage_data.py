"""
Stage 0 geometry — approximated from project.json data.

Sources:
  - Spawn points: (Stage) Spawn Points[0] = "px1:-245;py1:-10;px2:245;py2:-10;..."
  - Kill zone: BoundariesX=700, BoundariesY=350 (Stage variables)
  - (Stage) BottomGroundY[0] = -25
  - Ground costume "stage_1", rotation center (244, -47.6) → ~488px wide stage sprite
  - Semisolid costume "stage_1", rotation center (241.6, 68.5)
  - Target positions from targetstages[0] used as platform height hints

Platform format: (x_left, x_right, y_top) — axis-aligned horizontal platform.
Characters land on y_top when falling from above.
Semisolid platforms can be passed through from below.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Platform:
    x1: float       # left edge
    x2: float       # right edge
    y: float        # surface Y (characters stand at this Y)
    semisolid: bool = False  # passable from below


@dataclass(frozen=True)
class StageData:
    platforms: tuple[Platform, ...]
    spawn_p1: tuple[float, float]
    spawn_p2: tuple[float, float]
    kill_x: float   # |x| > kill_x → dead
    kill_y: float   # y < -kill_y → dead


# Stage 0 — "Battlefield" style with a wide main platform and two side platforms
STAGE_0 = StageData(
    platforms=(
        Platform(x1=-320.0, x2=320.0, y=0.0,    semisolid=False),  # main ground
        Platform(x1=-200.0, x2=-60.0, y=85.0,   semisolid=True),   # left platform
        Platform(x1=60.0,   x2=200.0, y=85.0,   semisolid=True),   # right platform
    ),
    # Spawn above main platform (y=0). Character origin sits PHYS_HALF_H ≈ 68
    # above the platform; add a small buffer so gravity settles cleanly.
    spawn_p1=(-245.0, 70.0),
    spawn_p2=( 245.0, 70.0),
    kill_x=700.0,
    kill_y=350.0,
)

# Default stage used by the environment
DEFAULT_STAGE = STAGE_0
