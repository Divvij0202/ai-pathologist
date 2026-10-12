"""Fit the low-zoom survey scorer: logistic regression on quadratic expansions of
OpenCV colour/texture features of ~3.9 um/px tiles, labelled by the CAMELYON16 masks.
(OpenCV 5 moved cv2.ml to contrib, so the model is a small JSON of weights.)

    python -m scripts.fit_survey_prior --slides tumor_001 tumor_002 ... --out models/survey_prior.json

Tiles are 1/16 of a cell (4x4 per cell), the same grid survey_slide scores.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import json
from pathlib import Path

from pathagent.classifier import expand, predict, survey_features
from pathagent.slide import Mask, Slide, camelyon_url
from pathagent.tissue import tissue_mask

TILE_UM = 768 * 0.972 / 4  # survey sub-tile size in microns


def slide_samples(name: str, max_neg: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = Slide(camelyon_url(name))
    level = s.best_level_for_mpp(3.9)
    tl = s.thumbnail_level(2048)
    thumb = s.read_level(tl)
    tm = tissue_mask(thumb)
    lv = s.levels[level]
    px = TILE_UM / s.level_mpp(level)  # tile size at survey level
    ds_t = s.levels[tl].downsample / lv.downsample
    mask = Mask(camelyon_url(name, "masks")) if name.startswith("tumor") else None
    if mask is not None:
        ml = mask.level_for_downsample(lv.downsample * 4)
        gt = (mask.read_level(ml) == 2).astype(np.float32)
        gt_scale = lv.downsample / mask.levels[ml].downsample
    X, y = [], []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tm)
    for i in range(1, n):
        x0, y0, w, h, area = stats[i]
        if area < 200:
            continue
        X0, Y0, W, H = int(x0 * ds_t), int(y0 * ds_t), int(w * ds_t), int(h * ds_t)
        img = s.read_region(level, X0, Y0, W, H)
        for ty in np.arange(0, H - px, px):
            for tx in np.arange(0, W - px, px):
                t = img[int(ty):int(ty + px), int(tx):int(tx + px)]
                cy, cx = int((Y0 + ty + px / 2) / ds_t), int((X0 + tx + px / 2) / ds_t)
                if cy >= tm.shape[0] or cx >= tm.shape[1] or not tm[cy, cx]:
                    continue
                frac = 0.0
                if mask is not None:
                    gy0, gx0 = int((Y0 + ty) * gt_scale), int((X0 + tx) * gt_scale)
                    g = gt[gy0:gy0 + int(px * gt_scale) + 1, gx0:gx0 + int(px * gt_scale) + 1]
                    frac = float(g.mean()) if g.size else 0.0
                if 0.1 < frac < 0.5:
                    continue  # ambiguous edge tiles
                X.append(survey_features(t)); y.append(1.0 if frac >= 0.5 else 0.0)
    X, y = np.array(X, np.float32), np.array(y, np.float32)
    neg = np.nonzero(y == 0)[0]
    if len(neg) > max_neg:
        drop = rng.choice(neg, len(neg) - max_neg, replace=False)
        keep = np.setdiff1d(np.arange(len(y)), drop)
        X, y = X[keep], y[keep]
    print(f"{name}: {int(y.sum())} tumour, {int((1 - y).sum())} normal tiles", flush=True)
    return X, y


def fit_logreg(X, y, l2=1e-3, iters=3000, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Q = expand((X - mu) / sd)
    w = np.zeros(Q.shape[1]); b = 0.0
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    sw = np.where(y > 0.5, pos_w, 1.0); sw /= sw.mean()
    for _ in range(iters):  # full-batch gradient descent, class balanced
        p = 1 / (1 + np.exp(-(Q @ w + b)))
        g = sw * (p - y)
        w -= lr * (Q.T @ g / len(y) + l2 * w); b -= lr * g.mean()
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": float(b), "type": "quadratic_logreg"}


def auc(s, l):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = l > 0.5; n1, n0 = p.sum(), (~p).sum()
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--out", default="models/survey_prior.json")
    a = ap.parse_args()
    with ThreadPoolExecutor(6) as ex:
        tr = list(ex.map(slide_samples, a.slides))
        va = list(ex.map(slide_samples, a.val))
    X = np.concatenate([t[0] for t in tr]); y = np.concatenate([t[1] for t in tr])
    np.savez_compressed(a.out.replace(".json", "_features.npz"), X=X, y=y)
    model = fit_logreg(X, y)
    Path(a.out).write_text(json.dumps(model))
    print("train AUC", round(auc(predict(model, X), y), 3))
    for name, (Xv, yv) in zip(a.val, va):
        if yv.sum() and (1 - yv).sum():
            print(name, "val AUC", round(auc(predict(model, Xv), yv), 3))
    print("saved", a.out)


if __name__ == "__main__":
    main()
