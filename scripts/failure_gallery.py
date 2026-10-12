"""Failure gallery: the worst misses and false alarms, rendered with OpenCV.

Reads eval/results/<tag>/rows.csv (from eval_suite.py) and the exhaustive caches,
picks the largest reference lesions the chosen strategy missed and the strongest
false-positive clusters on normal slides, and renders each as a crop with the
reference outline (green) and the classifier's tile heatmap.

    python -m scripts.failure_gallery --tag standin --cache eval/cache/standin --mode best_first --budget 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pathagent.classifier import SurveyScorer
from pathagent.evaluate import reference_lesions
from pathagent.slide import Slide, camelyon_url
from pathagent.vision import VisionConfig, VisionService

CFG = VisionConfig()


def render(slide: Slide, bbox0, tile_prob, gt_labels, gt_id, tile0, caption, size=640):
    x, y, w, h = bbox0
    side = max(w, h) * 1.6 + tile0 * 4
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    lvl = 0
    for lv in slide.levels:
        if side / lv.downsample >= size:
            lvl = lv.index
    ds = slide.levels[lvl].downsample
    img = slide.read_region(lvl, int(x0 / ds), int(y0 / ds), int(side / ds), int(side / ds))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    s = size / side  # output px per level-0 px
    # tile-grid window covering the crop
    tx0, ty0 = int(np.floor(x0 / tile0)), int(np.floor(y0 / tile0))
    n = int(np.ceil(side / tile0)) + 2
    H, W = tile_prob.shape
    win = np.full((n, n), np.nan, np.float32)
    gwin = np.zeros((n, n), np.uint8)
    ys, xs = slice(max(ty0, 0), min(ty0 + n, H)), slice(max(tx0, 0), min(tx0 + n, W))
    win[ys.start - ty0:ys.stop - ty0, xs.start - tx0:xs.stop - tx0] = tile_prob[ys, xs]
    if gt_labels is not None:
        g = gt_labels[ys, xs]
        gwin[ys.start - ty0:ys.stop - ty0, xs.start - tx0:xs.stop - tx0] = ((g == gt_id) if gt_id else (g > 0)).astype(np.uint8)
    big = int(round(n * tile0 * s))
    off = (int(round((tx0 * tile0 - x0) * s)), int(round((ty0 * tile0 - y0) * s)))
    heat = cv2.resize(np.nan_to_num(win, nan=0), (big, big), interpolation=cv2.INTER_NEAREST)
    seen = cv2.resize(np.isfinite(win).astype(np.uint8), (big, big), interpolation=cv2.INTER_NEAREST) > 0
    gtm = cv2.resize(gwin, (big, big), interpolation=cv2.INTER_NEAREST)
    canvas_h = np.zeros((size, size), np.float32); canvas_s = np.zeros((size, size), bool); canvas_g = np.zeros((size, size), np.uint8)
    sx0, sy0 = max(off[0], 0), max(off[1], 0)
    ex, ey = min(off[0] + big, size), min(off[1] + big, size)
    canvas_h[sy0:ey, sx0:ex] = heat[sy0 - off[1]:ey - off[1], sx0 - off[0]:ex - off[0]]
    canvas_s[sy0:ey, sx0:ex] = seen[sy0 - off[1]:ey - off[1], sx0 - off[0]:ex - off[0]]
    canvas_g[sy0:ey, sx0:ex] = gtm[sy0 - off[1]:ey - off[1], sx0 - off[0]:ex - off[0]]
    color = cv2.applyColorMap((canvas_h * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    m = canvas_s & (canvas_h >= 0.5)
    out = img.copy()
    out[m] = cv2.addWeighted(img, 0.55, color, 0.45, 0)[m]
    cnts, _ = cv2.findContours(canvas_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (40, 200, 40), 2)
    bar = np.full((48, size, 3), 255, np.uint8)
    for i, line in enumerate(caption[:2]):
        cv2.putText(bar, line, (8, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([out, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--mode", default="best_first")
    ap.add_argument("--budget", default="40")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    import json
    rows = [r for r in json.load(open(f"eval/results/{a.tag}/runs.json"))
            if r["mode"] == a.mode and str(r["budget"]) == str(a.budget)]
    out = Path(f"eval/results/{a.tag}/gallery"); out.mkdir(parents=True, exist_ok=True)
    vision = VisionService(None, SurveyScorer())  # only for lesion maths on cached grids
    cases = []
    for r in rows:
        name = r["slide"]
        c = np.load(Path(a.cache) / f"{name}.npz")
        tp = c["tile_prob"]
        slide = Slide(camelyon_url(name))
        tile0 = CFG.tile_px * CFG.analysis_mpp / slide.mpp
        if r["label"] in ("macro", "micro"):
            labels, info = reference_lesions(name, tuple(tp.shape), tile0, mpp0=slide.mpp)
            found = {l["id"] for l in r["lesions"] if l["found_at_look"]}
            for l in sorted(info, key=lambda l: -l["diameter_um"]):
                if l.get("size_class") == "itc":
                    continue
                ys, xs = np.nonzero(labels == l["id"])
                if len(ys) == 0:
                    continue
                hit_any = np.nanmax(np.where(labels == l["id"], np.nan_to_num(tp, nan=0), 0)) >= 0.5
                cases.append({"kind": "found" if l["id"] in found else "missed lesion", "slide": name,
                              "score": l["diameter_um"], "bbox": (xs.min() * tile0, ys.min() * tile0,
                              (np.ptp(xs) + 1) * tile0, (np.ptp(ys) + 1) * tile0), "gt": (labels, l["id"]),
                              "detail": f"{l['diameter_um']:.0f} um lesion; classifier {'does' if hit_any else 'does not'} "
                                        f"fire on it when viewed", "tp": tp, "slide_obj": slide, "tile0": tile0,
                              "classifier_hit": bool(hit_any)})
        if r["label"] == "negative":
            pos = (np.nan_to_num(tp, nan=0) >= 0.5).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(pos, connectivity=8)
            if n > 1:
                k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                x, y, w, h, area = stats[k]
                cases.append({"kind": "false alarm", "slide": name, "score": float(area),
                              "bbox": (x * tile0, y * tile0, w * tile0, h * tile0), "gt": None,
                              "detail": f"{area} tiles >= 0.5 on a normal slide", "tp": tp, "slide_obj": slide,
                              "tile0": tile0, "classifier_hit": True})
    missed = [c for c in cases if c["kind"] == "missed lesion"]
    # misses the classifier itself can't see are model failures; ones it can see are search failures
    missed.sort(key=lambda c: -c["score"])
    fas = sorted([c for c in cases if c["kind"] == "false alarm"], key=lambda c: -c["score"])
    pick = missed[: a.n // 2] + fas[: a.n - min(len(missed), a.n // 2)]
    md = [f"# Failure gallery ({a.tag}, {a.mode}, budget {a.budget})\n",
          "Green outline: reference tumour. Tinted tiles: classifier probability >= 0.5 (exhaustive scan).\n"]
    for i, c in enumerate(pick, 1):
        why = ("search failure: the classifier sees it, the search never looked there" if c["kind"] == "missed lesion"
               and c["classifier_hit"] else "model failure: the classifier does not fire on it" if c["kind"] == "missed lesion"
               else "model failure: tissue mistaken for tumour")
        img = render(c["slide_obj"], c["bbox"], c["tp"], *(c["gt"] or (None, 0)), c["tile0"],
                     [f"{i}. {c['slide']}: {c['kind']}", c["detail"]])
        fn = f"{i:02d}_{c['slide']}_{c['kind'].replace(' ', '_')}.jpg"
        cv2.imwrite(str(out / fn), img)
        md.append(f"## {i}. {c['slide']}: {c['kind']}\n\n{c['detail']}. Likely cause: {why}.\n\n![]({fn})\n")
    (out / "README.md").write_text("\n".join(md))
    print(f"wrote {len(pick)} cases to {out}")


if __name__ == "__main__":
    main()
