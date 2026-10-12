"""Throughput of the cv.dnn tile classifier on this machine (run it on Graviton and on x86).

    python -m scripts.benchmark --model models/tile_classifier.onnx --threads 4

Tiles per second sets the look budget you can afford per slide: one look = up to 64 tiles.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import cv2
import numpy as np

from pathagent.classifier import TileClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/tile_classifier.onnx")
    ap.add_argument("--threads", type=int, default=0, help="0 = OpenCV default")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    if a.threads:
        cv2.setNumThreads(a.threads)
    clf = TileClassifier(a.model, batch_size=a.batch)
    rng = np.random.default_rng(0)
    tiles = [rng.integers(0, 255, (96, 96, 3), dtype=np.uint8) for _ in range(a.n)]
    clf.predict(tiles[: a.batch])  # warm up
    t0 = time.perf_counter()
    clf.predict(tiles)
    dt = time.perf_counter() - t0
    res = {"machine": platform.machine(), "processor": platform.processor(), "opencv": cv2.__version__,
           "threads": cv2.getNumThreads(), "model": clf.name, "batch": a.batch,
           "tiles_per_s": round(a.n / dt, 1), "ms_per_look_64_tiles": round(64 / (a.n / dt) * 1000, 1)}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
