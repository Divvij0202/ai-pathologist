"""Tissue detection on a low-resolution thumbnail with OpenCV.

HSV saturation -> Otsu threshold -> morphological open/close -> contours.
Glass is bright and unsaturated; H&E tissue is saturated pink/purple. Pen
marks (blue/green/black ink) are removed by hue and value rules.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TissueRegion:
    id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h in thumbnail pixels
    area_px: int
    contour: np.ndarray


def tissue_mask(thumb_bgr: np.ndarray, min_area_frac: float = 0.0005) -> np.ndarray:
    """Binary uint8 mask (255 = tissue) for a BGR thumbnail."""
    hsv = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_blur = cv2.GaussianBlur(s, (5, 5), 0)
    otsu, mask = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu < 15:  # nearly blank thumbnail: Otsu splits noise, so use a floor
        _, mask = cv2.threshold(s_blur, 15, 255, cv2.THRESH_BINARY)

    # Pen ink: blue/green hues, or very dark pixels. H&E sits in pink-purple hues
    # (OpenCV hue ~120-180 and 0-10), so hue 35-110 is never stain.
    ink = ((h > 35) & (h < 110) & (s > 40)) | (v < 40)
    mask[ink] = 0
    # Black scanner borders and out-of-slide padding
    mask[(v < 15)] = 0

    k = max(3, int(round(min(thumb_bgr.shape[:2]) / 300)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * k, 3 * k)))

    # Drop specks and fill holes inside tissue pieces
    min_area = min_area_frac * mask.size
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    keep = [c for c in contours if cv2.contourArea(c) >= min_area]
    cv2.drawContours(clean, keep, -1, 255, thickness=cv2.FILLED)
    # Re-cut genuine holes (glass inside a ring of tissue) using the raw mask
    return cv2.bitwise_and(clean, cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel))


def tissue_regions(mask: np.ndarray, min_area_frac: float = 0.0005) -> list[TissueRegion]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * mask.size
    regions = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        regions.append(TissueRegion(len(regions) + 1, tuple(int(v) for v in cv2.boundingRect(c)), int(a), c))
    return regions


def blur_score(bgr: np.ndarray) -> float:
    """Variance of the Laplacian; low values mean an out-of-focus region."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tissue_fraction(bgr: np.ndarray) -> float:
    """Fraction of a tile that is tissue (cheap per-tile check at high zoom)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[..., 1] > 25) & (hsv[..., 2] > 40)))
