"""
Stage: static geometry + collision resolution.

Coordinates follow the original Scratch convention:
  +X = right, +Y = up, origin at stage center.

Collision model:
  - Solid platforms: block from all sides (characters land on top, stop at sides/bottom)
  - Semisolid platforms: only block from above (passthrough from sides/below)
"""
from __future__ import annotations
import math
from smeshlite.data.stage_data import StageData, Platform
from smeshlite.core.sensors import RaycastResult


class Stage:
    def __init__(self, data: StageData) -> None:
        self.data = data
        self.platforms: tuple[Platform, ...] = data.platforms
        self.spawn_p1 = data.spawn_p1
        self.spawn_p2 = data.spawn_p2
        self.kill_x = data.kill_x
        self.kill_y = data.kill_y

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def resolve_ground(
        self,
        x: float,
        y: float,
        vy: float,
        half_w: float = 20.0,
        half_h: float = 20.0,
        falling_through: bool = False,
    ) -> tuple[float, float, bool]:
        """
        Check if the character overlaps any platform from above.

        Args:
            x, y        : character center position (bottom = y - half_h)
            vy          : vertical velocity (negative = falling)
            half_w      : character half-width for horizontal overlap check
            half_h      : character half-height; feet are at (y - half_h)
            falling_through: if True, skip semisolid platforms

        Returns:
            (new_y, new_vy, on_ground):
              new_y     : corrected center Y (snapped to platform surface)
              new_vy    : corrected vy (zeroed on landing)
              on_ground : True if standing on a platform this frame
        """
        feet = y - half_h
        on_ground = False

        for plat in self.platforms:
            if plat.semisolid and falling_through:
                continue
            if plat.semisolid and vy > 0:
                # Moving upward through semisolid — skip
                continue
            # Horizontal overlap
            if x + half_w <= plat.x1 or x - half_w >= plat.x2:
                continue
            # Vertical: feet crossed through surface this frame.
            # Window = one frame of travel + margin; clamp to 2× half_h to
            # avoid snapping when deeply embedded (e.g. after a hard knock).
            window = min(abs(vy) + 2.0, half_h * 2)
            if feet <= plat.y and feet >= plat.y - window:
                y = plat.y + half_h
                vy = 0.0
                on_ground = True
                break  # resolve only the topmost platform

        return y, vy, on_ground

    def resolve_walls(
        self,
        x: float,
        vx: float,
        half_w: float = 20.0,
    ) -> tuple[float, float]:
        """
        Bounce off solid platform side edges.

        TODO: not yet called from Character.  The original Scratch game
        bounces SpeedX to ±PhysicsConstants.wall_bounce_x on wall contact;
        this port has not wired that up yet.
        """
        for plat in self.platforms:
            if plat.semisolid:
                continue
            # Left wall of platform
            if abs(x - plat.x1) < half_w and vx > 0:
                x = plat.x1 - half_w
                vx = -vx * 0.5
            # Right wall of platform
            elif abs(x - plat.x2) < half_w and vx < 0:
                x = plat.x2 + half_w
                vx = -vx * 0.5
        return x, vx

    def is_dead(self, x: float, y: float) -> bool:
        """Return True if the character is outside the kill zone."""
        return abs(x) > self.kill_x or y < -self.kill_y

    def clamp_to_bounds(
        self,
        x: float,
        vx: float,
        half_w: float = 20.0,
        bounce_speed: float = 15.0,
    ) -> tuple[float, float]:
        """
        Prevent characters from clipping through stage wall boundaries.

        TODO: not yet called from Character.  Pairs with resolve_walls()
        for the unimplemented wall-bounce mechanic.
        """
        if x - half_w < -self.kill_x:
            x = -self.kill_x + half_w
            vx = bounce_speed
        elif x + half_w > self.kill_x:
            x = self.kill_x - half_w
            vx = -bounce_speed
        return x, vx

    # ------------------------------------------------------------------
    # Raycast sensors
    # ------------------------------------------------------------------

    def raycast(self, ox: float, oy: float, dx: float, dy: float, max_dist: float) -> RaycastResult:
        """
        Cast a ray from (ox, oy) in direction (dx, dy) up to max_dist.

        Hits platform top edges and the kill-zone boundary (the three segments a
        character must cross to die: the left/right blast walls and the bottom).
        Returns the closest hit, or hit=False with end at the max-range point.
        """
        mag = math.hypot(dx, dy)
        if mag == 0.0:
            dx, dy = 0.0, -1.0
            mag = 1.0
        dx, dy = dx / mag, dy / mag

        segments = [(plat.x1, plat.y, plat.x2, plat.y) for plat in self.platforms]
        kx, ky = self.kill_x, self.kill_y
        segments.append((-kx, -ky, -kx, ky))   # left blast wall
        segments.append(( kx, -ky,  kx, ky))   # right blast wall
        segments.append((-kx, -ky,  kx, -ky))  # bottom blast floor

        best_t = max_dist
        hit = False
        for ax, ay, bx, by in segments:
            t = _ray_segment_intersect(ox, oy, dx, dy, ax, ay, bx, by)
            if t is not None and 0.0 <= t < best_t:
                best_t = t
                hit = True

        end = (ox + dx * best_t, oy + dy * best_t)
        return RaycastResult(origin=(ox, oy), end=end, distance=best_t, max_dist=max_dist, hit=hit)


def _ray_segment_intersect(
    ox: float, oy: float, dx: float, dy: float,
    ax: float, ay: float, bx: float, by: float,
) -> float | None:
    """
    Intersect ray (origin (ox,oy), direction (dx,dy), unit length) with segment A-B.
    Returns the ray parameter t (distance along the ray) if it hits within the
    segment, else None.
    """
    ex, ey = bx - ax, by - ay
    diffx, diffy = ax - ox, ay - oy

    det = ex * dy - ey * dx
    if det == 0.0:
        return None  # parallel

    t = (ex * diffy - ey * diffx) / det
    s = (dx * diffy - dy * diffx) / det

    if t >= 0.0 and 0.0 <= s <= 1.0:
        return t
    return None
