"""
SpriteLoader — converts a character's SVG costume files into scaled pygame Surfaces.

One SpriteLoader instance per Character; surfaces are cached by (costume_idx, flip).
Characters expose their loader via char.sprite_loader (lazy-init property).

Scale pipeline:
  SVG pixel  →  × SVG_TO_GAME (0.5)  →  game unit  →  × PIXELS_PER_UNIT (0.8)  →  screen pixel
  Combined: SVG_TO_SCREEN = 0.4
"""
from __future__ import annotations

import io
from typing import NamedTuple


_PIXELS_PER_UNIT: float = 0.8
_SVG_TO_GAME: float = 0.5
SVG_TO_SCREEN: float = _SVG_TO_GAME * _PIXELS_PER_UNIT   # 0.4


class SpriteInfo(NamedTuple):
    surface: object   # pygame.Surface
    anchor_x: int     # rotation-center x in screen pixels (facing right)
    anchor_y: int     # rotation-center y in screen pixels


class SpriteLoader:
    """Per-character sprite cache, keyed to a CharacterDef folder."""

    def __init__(self, char_def) -> None:
        self._def = char_def
        self._cache: dict[int, SpriteInfo | None] = {}
        self._flip_cache: dict[int, SpriteInfo | None] = {}

    # ------------------------------------------------------------------

    def get_sprite(self, costume_idx: int, flip: bool = False) -> SpriteInfo | None:
        """
        Return the cached SpriteInfo for the given costume, loading it if needed.
        flip=True returns the horizontally mirrored surface (facing left).
        Returns None if the asset could not be loaded.
        """
        import pygame
        cache = self._flip_cache if flip else self._cache

        if costume_idx not in self._cache:
            info = self._load_costume(costume_idx)
            self._cache[costume_idx] = info
            if info is not None:
                flipped_surf = pygame.transform.flip(info.surface, True, False)
                w = info.surface.get_width()
                self._flip_cache[costume_idx] = SpriteInfo(
                    flipped_surf,
                    w - info.anchor_x,
                    info.anchor_y,
                )
            else:
                self._flip_cache[costume_idx] = None

        return cache.get(costume_idx)

    def clear(self) -> None:
        self._cache.clear()
        self._flip_cache.clear()

    # ------------------------------------------------------------------

    def _load_costume(self, costume_idx: int) -> SpriteInfo | None:
        import pygame
        path = self._def.sprite_path(costume_idx)
        if not __import__("os").path.isfile(path):
            return None

        raw = self._try_load_svg(path)
        if raw is None:
            return None

        w_orig, h_orig = raw.get_size()
        w_new = max(1, round(w_orig * SVG_TO_SCREEN))
        h_new = max(1, round(h_orig * SVG_TO_SCREEN))

        scaled = (
            pygame.transform.smoothscale(raw, (w_new, h_new))
            if (w_new != w_orig or h_new != h_orig)
            else raw
        )

        meta = self._def.costumes[costume_idx]
        anchor_x = round(meta.rc_x * SVG_TO_SCREEN)
        anchor_y = round(meta.rc_y * SVG_TO_SCREEN)

        return SpriteInfo(scaled, anchor_x, anchor_y)

    @staticmethod
    def _try_load_svg(path: str) -> "pygame.Surface | None":
        import pygame
        try:
            return pygame.image.load(path).convert_alpha()
        except Exception:
            pass
        try:
            import cairosvg
            png_bytes = cairosvg.svg2png(url=path)
            return pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
        except Exception:
            pass
        return None
