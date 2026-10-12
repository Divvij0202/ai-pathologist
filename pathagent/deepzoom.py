"""Deep Zoom (DZI) tiles from a pyramid slide, for the OpenSeadragon viewer.

Each DZI tile is cut from the nearest finer slide level and resized with
OpenCV, then JPEG-encoded with cv2.imencode.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from .slide import Slide

TILE = 254
OVERLAP = 1


class DeepZoom:
    def __init__(self, slide: Slide, tile: int = TILE, overlap: int = OVERLAP, quality: int = 80):
        self.s = slide
        self.tile, self.overlap, self.quality = tile, overlap, quality
        w, h = slide.dimensions
        self.max_level = int(math.ceil(math.log2(max(w, h))))
        self._tile = lru_cache(maxsize=2048)(self._render)

    def dzi(self) -> str:
        w, h = self.s.dimensions
        return (f'<?xml version="1.0" encoding="UTF-8"?><Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
                f'Format="jpeg" Overlap="{self.overlap}" TileSize="{self.tile}"><Size Width="{w}" Height="{h}"/></Image>')

    def level_size(self, level: int):
        w, h = self.s.dimensions
        scale = 2 ** (self.max_level - level)
        return max(1, math.ceil(w / scale)), max(1, math.ceil(h / scale))

    def tile_jpeg(self, level: int, col: int, row: int) -> bytes | None:
        if level < 0 or level > self.max_level:
            return None
        return self._tile(level, col, row)

    def _render(self, level: int, col: int, row: int) -> bytes | None:
        lw, lh = self.level_size(level)
        x0 = col * self.tile - (self.overlap if col else 0)
        y0 = row * self.tile - (self.overlap if row else 0)
        if x0 >= lw or y0 >= lh:
            return None
        x1 = min(lw, (col + 1) * self.tile + self.overlap)
        y1 = min(lh, (row + 1) * self.tile + self.overlap)
        ds = 2 ** (self.max_level - level)  # level-0 pixels per DZI pixel
        # finest slide level that is still coarser than or equal to what we need
        sl = 0
        for lv in self.s.levels:
            if lv.downsample <= ds * 1.0001:
                sl = lv.index
        sds = self.s.levels[sl].downsample
        f = ds / sds
        rx, ry = int(x0 * f), int(y0 * f)
        rw, rh = max(1, int(math.ceil((x1 - x0) * f))), max(1, int(math.ceil((y1 - y0) * f)))
        img = self.s.read_region(sl, rx, ry, rw, rh)
        img = cv2.resize(img, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA if f > 1 else cv2.INTER_LINEAR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else None


def heat_rgba(grid: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO, alpha_min: float = 0.0,
              normalize: bool = False, max_side: int = 2048) -> bytes:
    """Encode a probability grid (nan = unseen) as a transparent PNG heatmap."""
    g = grid.astype(np.float32)
    seen = np.isfinite(g)
    v = np.nan_to_num(g, nan=0)
    if normalize and seen.any():
        lo, hi = v[seen].min(), v[seen].max()
        v = (v - lo) / (hi - lo + 1e-6)
    color = cv2.applyColorMap((np.clip(v, 0, 1) * 255).astype(np.uint8), colormap)
    a = np.where(seen, np.clip(0.25 + 0.6 * v, 0, 0.85), 0) * (v >= alpha_min)
    rgba = np.dstack([color, (a * 255).astype(np.uint8)])
    scale = max(1, min(8, max_side // max(g.shape)))
    rgba = cv2.resize(rgba, (g.shape[1] * scale, g.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", rgba)
    return buf.tobytes()
