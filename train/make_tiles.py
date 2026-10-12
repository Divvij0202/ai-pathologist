"""Cut labelled 96x96 tiles straight from CAMELYON16 slides using the official masks.

Two uses:
  1. A quick stand-in training set when PCam is not at hand (this is how the
     pipeline was smoke-tested before the real PCam model existed).
  2. Fine-tuning a PCam model on real slide tiles, the fallback in the plan for
     when the PCam model misfires on whole slides.

Tiles are read remotely from the public bucket; nothing else is downloaded.

    python -m train.make_tiles --slides tumor_001 tumor_002 normal_001 --per-slide 400 --out data/c16_tiles.npz
"""
from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from pathagent.slide import Mask, Slide, camelyon_url

_local = threading.local()


def _slide(name):
    cache = getattr(_local, "slides", None)
    if cache is None:
        cache = _local.slides = {}
    if name not in cache:
        cache[name] = Slide(camelyon_url(name))
    return cache[name]


def sample_coords(name: str, per_slide: int, rng: np.random.Generator, mpp: float = 0.972,
                  tissue_negatives: bool = False):
    """Return [(name, level, x, y, label)] with x, y the tile's top-left at the analysis level."""
    slide = Slide(camelyon_url(name))
    if name.startswith("normal") and tissue_negatives:
        # hard negatives: anywhere our own OpenCV tissue mask says tissue, including capsule
        # edges, fat borders and stroma that the official (eroded) normal masks under-sample
        from pathagent.tissue import tissue_mask
        tl = slide.thumbnail_level(4096)
        m = (tissue_mask(slide.read_level(tl)) > 0).astype(np.uint8)
        mds = slide.levels[tl].downsample
    else:
        mask = Mask(camelyon_url(name, "masks"))
        ml = mask.level_for_downsample(32)
        m = mask.read_level(ml)
        mds = mask.levels[ml].downsample
    level = slide.best_level_for_mpp(mpp)
    lds = slide.levels[level].downsample
    k = np.ones((3, 3), np.uint8)
    tumour = cv2.erode((m == 2).astype(np.uint8), k)
    normal = m.astype(np.uint8) if (name.startswith("normal") and tissue_negatives) else cv2.erode((m == 1).astype(np.uint8), k)
    out = []
    for lab, mm, n in ((1, tumour, per_slide // 2 if name.startswith("tumor") else 0),
                       (0, normal, per_slide // 2 if name.startswith("tumor") else per_slide)):
        ys, xs = np.nonzero(mm)
        if len(ys) == 0 or n == 0:
            continue
        sel = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
        for i in sel:
            # random offset inside the mask pixel, then centre a 96 px tile there
            x0 = (xs[i] + rng.random()) * mds / lds - 48
            y0 = (ys[i] + rng.random()) * mds / lds - 48
            out.append((name, level, int(x0), int(y0), lab))
    slide.close()
    return out


def read_tile(job):
    name, level, x, y, lab = job
    t = _slide(name).read_region(level, x, y, 96, 96)
    return t, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", required=True)
    ap.add_argument("--per-slide", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="data/c16_tiles.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tissue-negatives", action="store_true", help="normal slides: sample from the OpenCV tissue mask")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    jobs = []
    for s in a.slides:
        j = sample_coords(s, a.per_slide, rng, tissue_negatives=a.tissue_negatives)
        print(f"{s}: {sum(x[4] for x in j)} tumour, {sum(1 - x[4] for x in j)} normal", flush=True)
        jobs += j
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(read_tile, jobs))
    X = np.stack([r[0] for r in res])  # BGR uint8 N x 96 x 96 x 3
    y = np.array([r[1] for r in res], np.uint8)
    groups = np.array([j[0] for j in jobs])
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, x=X, y=y, slide=groups)
    print("saved", a.out, X.shape, "positives", int(y.sum()))


if __name__ == "__main__":
    main()
