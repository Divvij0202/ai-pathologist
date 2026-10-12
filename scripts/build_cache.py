"""Read every tissue cell of a slide once and store all tile probabilities.

This is the exhaustive sliding window, done fast (parallel S3 reads), and saved
so that every search strategy can then be replayed against the same classifier
outputs in seconds: `VisionService(..., cache_dir=...)`.

    python -m scripts.build_cache --slides test_001 test_002 --cache eval/cache/standin --workers 12

Caches are tied to the classifier that made them: use one cache folder per model.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pathagent.classifier import load_models
from pathagent.slide import Slide, camelyon_url
from pathagent.vision import VisionService

_local = threading.local()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    clf, scorer = load_models(a.models)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "model.json").write_text(json.dumps({"classifier": clf.meta, "survey": scorer.kind}, default=str))
    vision = VisionService(clf, scorer, out_dir="/tmp/cache_runs")
    for name in a.slides:
        out = cache / f"{name}.npz"
        if out.exists():
            print(f"{name}: cached already")
            continue
        t0 = time.time()
        url = camelyon_url(name)
        vision.survey_slide(url, name)
        st = vision.states[name]
        cells = sorted(st.cell_region)

        def read(cell):
            sl = getattr(_local, "slides", {}).get(url)
            if sl is None:
                _local.slides = {url: Slide(url)}
                sl = _local.slides[url]
            # _read_cell only needs geometry from the state; give it this thread's handle
            return cell, vision._read_cell(type("S", (), {"slide": sl, "cell0": st.cell0})(), cell)

        T = vision.cfg.cell_tiles
        with ThreadPoolExecutor(a.workers) as ex:
            for cell, img in ex.map(read, cells):
                _, grid, _ = vision.classify_cell(img)
                r, c = cell
                st.tile_prob[r * T:(r + 1) * T, c * T:(c + 1) * T] = grid
        np.savez_compressed(out, tile_prob=st.tile_prob, survey=st.survey, cell_tissue=st.cell_tissue)
        print(f"{name}: {len(cells)} cells in {time.time() - t0:.0f}s", flush=True)
        vision.states.pop(name)


if __name__ == "__main__":
    main()
