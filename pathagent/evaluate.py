"""Score a finished search against the CAMELYON16 reference masks.

Lesion-level: reference tumour regions closer than 75 um are merged into one
lesion (the CAMELYON16 FROC convention, ~5 tumour cells), then split into
macrometastases (> 2 mm), micrometastases (0.2-2 mm) and isolated tumour cells
(ITC, < 0.2 mm). Headline recall counts macro + micro; ITCs are reported
separately (CAMELYON16 does not penalise missing them). A lesion counts as found
when a viewed tile with probability >= threshold overlaps it (one tile of slack). The efficiency curve replays the looks in order: fraction of
reference lesions found vs fraction of tissue cells viewed.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .slide import Mask, camelyon_url
from .vision import VisionService


@lru_cache(maxsize=64)
def reference_lesions(name: str, tile_grid_shape, tile0: float, min_diam_um: float = 0.0, mpp0: float = 0.243):
    """Rasterise the reference tumour mask onto the tile grid. Returns (labels, info)."""
    mask = Mask(camelyon_url(name, "masks"))
    lvl = mask.level_for_downsample(tile0 / 4)  # a few mask pixels per tile
    m = (mask.read_level(lvl) == 2).astype(np.uint8)
    ds = mask.levels[lvl].downsample
    r = max(1, int(round(75 / (ds * mpp0) / 2)))  # merge regions closer than 75 um
    merged = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    H, W = tile_grid_shape
    info = []
    labels = np.zeros((H, W), np.int32)
    for i in range(1, n):
        comp = ((lab == i) & (m > 0)).astype(np.uint8)
        if not comp.any():
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.vstack(cnts))
        diam = max(rw, rh, 1) * ds * mpp0
        if diam < min_diam_um:
            continue
        # area-resample onto the tile grid: any overlap counts
        x, y, w, h, _ = stats[i]
        tx0, ty0 = int(x * ds // tile0), int(y * ds // tile0)
        tx1, ty1 = int((x + w) * ds // tile0) + 1, int((y + h) * ds // tile0) + 1
        sub = cv2.resize(comp[y:y + h, x:x + w].astype(np.float32), (tx1 - tx0, ty1 - ty0), interpolation=cv2.INTER_AREA)
        hit = sub > 0
        region = labels[ty0:ty1, tx0:tx1]
        hh, ww = region.shape
        region[hit[:hh, :ww]] = len(info) + 1
        info.append({"id": len(info) + 1, "diameter_um": float(diam), "macro": diam > 2000,
                     "size_class": "macro" if diam > 2000 else "micro" if diam >= 200 else "itc"})
    return labels, info


def evaluate(vision: VisionService, session: str, threshold: float = 0.5) -> dict:
    st = vision.states[session]
    name = st.slide_path.split("/")[-1].removesuffix(".tif")
    tile0 = vision.cfg.tile_px * vision.cfg.analysis_mpp / st.slide.mpp
    out = {"slide": name}
    if name.startswith("normal"):
        labels, info = np.zeros(st.tile_prob.shape, np.int32), []
    else:
        try:
            labels, info = reference_lesions(name, tuple(st.tile_prob.shape), tile0, mpp0=st.slide.mpp)
        except Exception as e:  # test slides that are negative have no mask
            labels, info = np.zeros(st.tile_prob.shape, np.int32), []
            out["mask_note"] = f"no reference mask ({type(e).__name__}); treated as negative"
    T = vision.cfg.cell_tiles
    lab_d = cv2.dilate(labels.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)  # one tile slack
    found_at = {}
    main_ids = {l["id"] for l in info if l["size_class"] != "itc"}
    total_cells = max(len(st.cell_region), 1)
    curve = []
    seen = set()
    for i, lk in enumerate(st.looks):
        r, c = lk.cell
        seen.add(lk.cell)
        block = np.nan_to_num(st.tile_prob[r * T:(r + 1) * T, c * T:(c + 1) * T], nan=0) >= threshold
        ids = set(np.unique(lab_d[r * T:(r + 1) * T, c * T:(c + 1) * T][block])) - {0}
        for j in ids:
            found_at.setdefault(int(j), i + 1)
        n_main = sum(1 for j in found_at if j in main_ids)
        curve.append((round(len(seen) / total_cells, 4), round(n_main / max(len(main_ids), 1), 4)))
    # false-positive tiles: positive tiles not touching any reference lesion
    pos = np.nan_to_num(st.tile_prob, nan=0) >= threshold
    fp_tiles = int((pos & (lab_d == 0)).sum())
    macro = [l for l in info if l["size_class"] == "macro"]
    micro = [l for l in info if l["size_class"] == "micro"]
    itc = [l for l in info if l["size_class"] == "itc"]
    main = macro + micro
    out.update({
        "reference_lesions": len(main),
        "found": sum(l["id"] in found_at for l in main),
        "itc_found": f"{sum(l['id'] in found_at for l in itc)}/{len(itc)}",
        "macro_found": f"{sum(l['id'] in found_at for l in macro)}/{len(macro)}",
        "micro_found": f"{sum(l['id'] in found_at for l in micro)}/{len(micro)}",
        "first_hit_look": min(found_at.values()) if found_at else None,
        "looks": len(st.looks),
        "fraction_tissue_viewed": round(len(st.visited) / total_cells, 4),
        "false_positive_tiles": fp_tiles,
        "slide_score": round(vision.slide_score(session), 3),
        "efficiency_curve": curve,
        "lesions": [{**l, "found_at_look": found_at.get(l["id"])} for l in sorted(info, key=lambda l: -l["diameter_um"])],
    })
    return out
