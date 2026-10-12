"""Stain handling in OpenCV: Reinhard and Macenko normalization, H&E deconvolution.

Slides from different labs and scanners differ in stain colour, which is one
of the main reasons a PCam-trained classifier misfires on new slides.
"""
from __future__ import annotations

import cv2
import numpy as np

# Reinhard target statistics in OpenCV 8-bit LAB, measured on PCam-like H&E tiles.
# Recompute with `reinhard_stats` on your own reference tiles for best results.
REINHARD_TARGET = (np.array([168.0, 156.0, 113.0]), np.array([35.0, 12.0, 9.0]))


def reinhard_stats(bgr: np.ndarray, tissue: np.ndarray | None = None):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if tissue is not None:
        lab = lab[tissue.reshape(-1) > 0]
    if len(lab) < 16:
        return None
    return lab.mean(0), lab.std(0) + 1e-6


def reinhard(bgr: np.ndarray, target=REINHARD_TARGET, source=None) -> np.ndarray:
    """Match LAB mean/std of tissue pixels to a target. `source` lets a whole
    region share one set of statistics so neighbouring tiles stay consistent."""
    if source is None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        source = reinhard_stats(bgr, ((hsv[..., 1] > 25) & (hsv[..., 2] < 245)).astype(np.uint8))
        if source is None:
            return bgr
    mu_s, sd_s = source
    mu_t, sd_t = target
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = (lab - mu_s) * (sd_t / sd_s) + mu_t
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---- Macenko -----------------------------------------------------------
MACENKO_REF_STAINS = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])  # RGB x (H, E)
MACENKO_REF_MAX = np.array([1.9705, 1.0308])


def _od(bgr: np.ndarray, io: float = 240.0) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
    return -np.log((rgb + 1.0) / io)


def macenko_stains(bgr: np.ndarray, beta: float = 0.15, alpha: float = 1.0):
    """Estimate the H and E stain vectors (3x2) and max concentrations."""
    od = _od(bgr)
    od_t = od[(od > beta).all(axis=1)]
    if len(od_t) < 50:
        return None
    # Eigenvectors of the OD covariance via OpenCV
    _, _, eigvecs = cv2.eigen(np.cov(od_t.T).astype(np.float64))
    plane = eigvecs[:2].T[:, ::-1]  # 3x2, two largest
    proj = od_t @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    stains = np.stack([v1, v2], 1) if v1[0] > v2[0] else np.stack([v2, v1], 1)
    stains *= np.sign(stains.sum(0, keepdims=True))
    conc, *_ = np.linalg.lstsq(stains, od.T, rcond=None)
    max_c = np.percentile(conc, 99, axis=1)
    return stains, max_c


def macenko(bgr: np.ndarray, source=None, io: float = 240.0) -> np.ndarray:
    est = source or macenko_stains(bgr)
    if est is None:
        return bgr
    stains, max_c = est
    od = _od(bgr, io)
    conc, *_ = np.linalg.lstsq(stains, od.T, rcond=None)
    conc *= (MACENKO_REF_MAX / np.maximum(max_c, 1e-6))[:, None]
    rgb = io * np.exp(-(MACENKO_REF_STAINS @ conc))
    rgb = np.clip(rgb.T.reshape(bgr.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def hematoxylin(bgr: np.ndarray) -> np.ndarray:
    """Hematoxylin optical density (nuclei) using the reference H&E vectors, float32 HxW."""
    od = _od(bgr)
    conc, *_ = np.linalg.lstsq(MACENKO_REF_STAINS, od.T, rcond=None)
    return conc[0].reshape(bgr.shape[:2]).astype(np.float32)
