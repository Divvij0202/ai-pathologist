"""Whole-slide access.

CAMELYON16 slides are tiled, JPEG-compressed pyramid TIFFs (0.3 to 4 GB each),
far too large for cv2.imread. We use tifffile only to locate and decode the
512x512 tiles a region needs; every pixel is handed straight to OpenCV as a
BGR uint8 array.

A slide can be a local path or an S3/HTTPS URL. Remote slides are read with
HTTP range requests, so only the tiles we look at are ever transferred. On AWS
in us-west-2 (next to the public bucket) this costs nothing.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import tifffile
import zarr

CAMELYON_BUCKET_URL = "https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON16"


def camelyon_url(name: str, kind: str = "images") -> str:
    """URL for a CAMELYON16 file, e.g. camelyon_url("tumor_005")."""
    name = name.removesuffix(".tif")
    if kind == "images":
        return f"{CAMELYON_BUCKET_URL}/images/{name}.tif"
    if kind == "masks":
        return f"{CAMELYON_BUCKET_URL}/masks/{name}_mask.tif"
    if kind == "annotations":
        return f"{CAMELYON_BUCKET_URL}/annotations/{name}.xml"
    raise ValueError(kind)


def _open_file(path: str):
    if path.startswith("s3://"):
        # s3://camelyon-dataset/CAMELYON16/... -> public HTTPS endpoint (no AWS creds needed)
        bucket, _, key = path[5:].partition("/")
        path = f"https://{bucket}.s3.us-west-2.amazonaws.com/{key}"
    if path.startswith(("http://", "https://")):
        import fsspec

        # blockcache keeps recently read byte ranges (tile index, hot tiles) in memory
        return fsspec.open(path, block_size=1 << 18, cache_type="blockcache").open()
    return open(path, "rb")


@dataclass
class Level:
    index: int
    width: int
    height: int
    downsample: float


class Slide:
    """Read regions of a pyramid TIFF at any level as BGR numpy arrays.

    Coordinates passed to read_region are always in the pixel space of the
    requested level (not level 0), which keeps tile maths simple.
    """

    def __init__(self, path: str):
        self.path = path
        self._fh = _open_file(path)
        self._tif = tifffile.TiffFile(self._fh)
        self._lock = threading.Lock()  # tifffile file handles are not thread safe
        series = self._tif.series[0]
        self._zarr = zarr.open(series.aszarr(), mode="r")
        pages = series.levels if series.levels else [series]
        w0 = pages[0].shape[1]
        self.levels = [
            Level(i, p.shape[1], p.shape[0], w0 / p.shape[1]) for i, p in enumerate(pages)
        ]
        self.mpp = self._read_mpp()

    # ---- metadata -------------------------------------------------------
    def _read_mpp(self) -> float:
        """Microns per pixel at level 0."""
        page = self._tif.pages[0]
        desc = page.description or ""
        # Philips DP export used by CAMELYON16 stores DICOM PIXEL_SPACING in mm
        m = re.search(r'PIXEL_SPACING"[^>]*>&quot;([0-9.eE-]+)&quot;', desc)
        if m:
            return float(m.group(1)) * 1000.0
        xres = page.tags.get("XResolution")
        unit = page.tags.get("ResolutionUnit")
        if xres is not None and unit is not None:
            num, den = xres.value
            if num:
                px_per_unit = num / den
                if unit.value == 3:  # centimetre
                    return 10000.0 / px_per_unit
                if unit.value == 2:  # inch
                    return 25400.0 / px_per_unit
        return 0.243  # CAMELYON16 scanners are ~0.23-0.25 um/px at 40x

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.levels[0].width, self.levels[0].height

    def level_mpp(self, level: int) -> float:
        return self.mpp * self.levels[level].downsample

    def best_level_for_mpp(self, target_mpp: float) -> int:
        """Finest level whose resolution is not finer than target (so we only downsample)."""
        best = 0
        for lv in self.levels:
            if self.mpp * lv.downsample <= target_mpp * 1.05:
                best = lv.index
        return best

    def thumbnail_level(self, max_side: int = 4096) -> int:
        for lv in self.levels:
            if max(lv.width, lv.height) <= max_side:
                return lv.index
        return len(self.levels) - 1

    # ---- pixels ---------------------------------------------------------
    def read_region(self, level: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Return BGR uint8 (h, w, 3). Areas outside the slide are white."""
        lv = self.levels[level]
        out = np.full((h, w, 3), 255, np.uint8)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, lv.width), min(y + h, lv.height)
        if x1 <= x0 or y1 <= y0:
            return out
        with self._lock:
            rgb = np.asarray(self._zarr[str(level)][y0:y1, x0:x1])
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        out[y0 - y : y1 - y, x0 - x : x1 - x] = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
        return out

    def read_level(self, level: int) -> np.ndarray:
        lv = self.levels[level]
        return self.read_region(level, 0, 0, lv.width, lv.height)

    def close(self):
        self._tif.close()
        self._fh.close()


class Mask:
    """CAMELYON16 pixel masks: 0 background, 1 normal tissue, 2 tumour."""

    def __init__(self, path: str):
        self._fh = _open_file(path)
        self._tif = tifffile.TiffFile(self._fh)
        series = self._tif.series[0]
        self._zarr = zarr.open(series.aszarr(), mode="r")
        pages = series.levels if series.levels else [series]
        w0 = pages[0].shape[1]
        self.levels = [Level(i, p.shape[1], p.shape[0], w0 / p.shape[1]) for i, p in enumerate(pages)]

    def read_level(self, level: int) -> np.ndarray:
        a = np.asarray(self._zarr[str(level)][:])
        return a if a.ndim == 2 else a[..., 0]

    def level_for_downsample(self, downsample: float) -> int:
        return min(range(len(self.levels)), key=lambda i: abs(self.levels[i].downsample - downsample))


@lru_cache(maxsize=8)
def open_slide(path: str) -> Slide:
    return Slide(path)
