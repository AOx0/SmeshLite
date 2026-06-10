"""
Pygame renderer for SmeshLite.

Characters are drawn with their actual MINIUM SVG vector sprites.
Stage and HUD use monochromatic geometry.
SVG loading is attempted once per costume; if it fails, a colored box is drawn
instead so the renderer always works even without the asset folder.

Coordinate mapping:
  Game: +X right, +Y up, origin at center
  Screen: (0,0) top-left, Y axis flipped

Scale: PIXELS_PER_UNIT converts game units to screen pixels.
"""
from __future__ import annotations

SCREEN_W = 800
SCREEN_H = 600
PIXELS_PER_UNIT = 0.8
ORIGIN_X = SCREEN_W // 2
ORIGIN_Y = SCREEN_H // 2 + 60

BG_COLOR         = (15,  15,  15)
PLATFORM_COLOR   = (200, 200, 200)
SEMISOLID_COLOR  = (120, 120, 120)
PLAYER_COLORS    = [(240, 240, 240), (160, 160, 160), (100, 200, 100), (200, 100, 100)]
HUD_COLOR        = (200, 200, 200)
HITBOX_COLOR     = (255, 60,  60)    # debug hitbox overlay (semi-transparent)
SENSOR_HIT_COLOR = (255, 220, 60)    # sensor ray that hit something
SENSOR_MISS_COLOR = (70, 70, 90)     # sensor ray with no hit


def _to_screen(x: float, y: float) -> tuple[int, int]:
    sx = int(ORIGIN_X + x * PIXELS_PER_UNIT)
    sy = int(ORIGIN_Y - y * PIXELS_PER_UNIT)
    return sx, sy


class Renderer:
    def __init__(self, render_mode: str = "human") -> None:
        import pygame
        self._pygame = pygame
        self.render_mode = render_mode
        pygame.init()
        pygame.display.set_caption("SmeshLite")
        if render_mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        else:
            self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
        self.font = pygame.font.SysFont("monospace", 14)
        self.clock = pygame.time.Clock()

        pass  # sprite loading is handled per-character via char.sprite_loader

    # ------------------------------------------------------------------

    def render(self, match) -> "np.ndarray | None":
        pg = self._pygame
        screen = self.screen

        screen.fill(BG_COLOR)

        # Stage platforms
        for plat in match.stage.platforms:
            x1, y1 = _to_screen(plat.x1, plat.y)
            x2, _  = _to_screen(plat.x2, plat.y)
            h      = 10
            color  = SEMISOLID_COLOR if plat.semisolid else PLATFORM_COLOR
            pg.draw.rect(screen, color, (min(x1, x2), y1, abs(x2 - x1), h))

        # Kill-zone markers
        kx_l, _ = _to_screen(-match.stage.kill_x, 0)
        kx_r, _ = _to_screen( match.stage.kill_x, 0)
        pg.draw.line(screen, (60, 20, 20), (kx_l, 0), (kx_l, SCREEN_H), 1)
        pg.draw.line(screen, (60, 20, 20), (kx_r, 0), (kx_r, SCREEN_H), 1)

        # Characters
        from smeshlite.core.character import Action
        for i, char in enumerate(match.characters):
            sx, sy = _to_screen(char.x, char.y)
            self._draw_character(screen, char, i, sx, sy)
            self._draw_sensors(screen, char)

        self._draw_hud(screen, match)

        if self.render_mode == "human":
            pg.display.flip()
            self.clock.tick(60)
            return None
        else:
            import numpy as np
            return np.transpose(
                pg.surfarray.array3d(screen), (1, 0, 2)
            ).copy()

    # ------------------------------------------------------------------

    def _draw_character(self, screen, char, player_idx: int, sx: int, sy: int) -> None:
        from smeshlite.core.character import Action
        pg = self._pygame

        costume_idx = char.get_current_costume_index()
        flip = (char.facing < 0)

        info = char.sprite_loader.get_sprite(costume_idx, flip=flip)

        if info is not None and char.action != Action.DEAD:
            # Blit: position sprite so rotation-center aligns with character origin
            blit_x = sx - info.anchor_x
            blit_y = sy - info.anchor_y

            # Invincibility flash: every 6 frames draw at half alpha
            surf = info.surface
            if char.invincibility > 0 and (char.anim_frame % 6) < 3:
                surf = surf.copy()
                surf.set_alpha(100)

            screen.blit(surf, (blit_x, blit_y))

        else:
            # Fallback: colored rectangle
            half = int(char._def.stats.size / 2 * PIXELS_PER_UNIT)
            if char.action == Action.DEAD:
                color = (50, 50, 50)
            elif char.action == Action.HITSTUN:
                color = (255, 80, 80)
            else:
                color = PLAYER_COLORS[player_idx % len(PLAYER_COLORS)]
            pg.draw.rect(screen, color, (sx - half, sy - half, half * 2, half * 2))
            # Facing arrow
            tip_x = sx + int(char.facing * (half + 5))
            pg.draw.polygon(screen, color, [
                (tip_x, sy),
                (sx + int(char.facing * half), sy - 6),
                (sx + int(char.facing * half), sy + 6),
            ])

        # Velocity arrow (debug, thin)
        vsx = sx + int(char.vx * PIXELS_PER_UNIT * 2)
        vsy = sy - int(char.vy * PIXELS_PER_UNIT * 2)
        pg.draw.line(screen, (80, 80, 200), (sx, sy), (vsx, vsy), 1)

    # ------------------------------------------------------------------

    def _draw_sensors(self, screen, char) -> None:
        """Debug overlay: draw each of the brain's RaycastSensor results as a line."""
        pg = self._pygame
        for result in char._sensor_results.values():
            start = _to_screen(*result.origin)
            end = _to_screen(*result.end)
            color = SENSOR_HIT_COLOR if result.hit else SENSOR_MISS_COLOR
            pg.draw.line(screen, color, start, end, 1)
            if result.hit:
                pg.draw.circle(screen, color, end, 3)

    # ------------------------------------------------------------------

    def _draw_hud(self, screen, match) -> None:
        pg = self._pygame

        frame_txt = self.font.render(f"Frame: {match.frame}", True, HUD_COLOR)
        screen.blit(frame_txt, (8, 8))

        for i, char in enumerate(match.characters):
            x_base = 50 + i * 350
            y_base = SCREEN_H - 60
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            dmg_txt = self.font.render(
                f"P{i}  {char.damage_pct:.0f}%  {'★' * char.stocks}",
                True, color,
            )
            screen.blit(dmg_txt, (x_base, y_base))
            act_txt = self.font.render(
                f"act:{char.action.name}  c:{char.get_current_costume_index()}",
                True, (120, 120, 120),
            )
            screen.blit(act_txt, (x_base, y_base + 18))

        if match.done:
            msg = f"P{match.winner} WINS!" if match.winner is not None else "TIMEOUT"
            win_txt = self.font.render(msg, True, (255, 255, 0))
            screen.blit(
                win_txt,
                (SCREEN_W // 2 - win_txt.get_width() // 2, SCREEN_H // 2 - 20),
            )

    def close(self) -> None:
        self._pygame.quit()
