"""
Brain-defined raycast sensors ("eyes").

A CharacterBrain may declare a tuple of RaycastSensor specs via its SENSORS class
attribute. Each tick, Character._build_context() resolves them to world-space rays
and casts them against the stage (platforms + kill-zone boundary) via Stage.raycast,
producing a RaycastResult per sensor name in BrainContext.sensors.

Local-space conventions:
  - origin/direction as a (dx, dy) tuple are in the character's local space: dx is
    mirrored by `facing` (so "ahead" sensors stay ahead when the character turns
    around); dy is not mirrored.
  - origin/direction == "velocity" makes the sensor dynamic: the probe point follows
    (vx, vy) (so it reaches further ahead the faster the character moves), and/or the
    ray points in the direction of travel. Zero velocity falls back to no offset /
    straight down.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RaycastSensor:
    """A brain-defined probe: a ray cast from `origin` in `direction`, up to `max_dist`."""
    name: str
    origin: tuple[float, float] | str = (0.0, 0.0)
    direction: tuple[float, float] | str = (0.0, -1.0)
    max_dist: float = 100.0


@dataclass(frozen=True)
class RaycastResult:
    """Result of casting one RaycastSensor against the stage."""
    origin: tuple[float, float]   # world-space start point
    end: tuple[float, float]      # world-space hit point (or max-range point)
    distance: float               # 0..max_dist
    max_dist: float
    hit: bool

    @property
    def value(self) -> float:
        """Normalized closeness in [0,1]; 1.0 = hit right at origin, 0.0 = no hit."""
        if not self.hit or self.max_dist <= 0:
            return 0.0
        return max(0.0, 1.0 - self.distance / self.max_dist)


def evaluate_sensors(specs, character, stage) -> dict[str, RaycastResult]:
    """Resolve and cast each RaycastSensor for `character` against `stage`."""
    results: dict[str, RaycastResult] = {}

    for spec in specs:
        if spec.origin == "velocity":
            ox = character.x + character.vx
            oy = character.y + character.vy
        else:
            off_x, off_y = spec.origin
            ox = character.x + off_x * character.facing
            oy = character.y + off_y

        if spec.direction == "velocity":
            dx, dy = character.vx, character.vy
            if dx == 0.0 and dy == 0.0:
                dx, dy = 0.0, -1.0
        else:
            dx, dy = spec.direction
            dx *= character.facing

        results[spec.name] = stage.raycast(ox, oy, dx, dy, spec.max_dist)

    return results
