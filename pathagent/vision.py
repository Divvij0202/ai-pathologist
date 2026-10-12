"""The vision service: everything that touches pixels.

The agent never sees pixels. It calls these methods (directly, or over HTTP
through service/app.py), and each call returns small JSON-able summaries plus
snapshot image paths.

Geometry
--------
* Analysis resolution is PCam's: ~0.97 um/px, 96 px tiles (~93 um).
* A *cell* is an 8x8 block of tiles (768 px, ~0.75 mm). One cell read at high
  zoom is one *look*, the unit of the search budget.
* A *region* is a group of tissue cells: either a hotspot (connected cells
  with high survey scores) or a sweep block (a 4x4-cell patch of the rest).
* Internally everything is stored on the cell/tile grid; positions are
  reported in microns and as fractions of the slide (0..1) so the LLM can
  talk about "upper left" without pixel maths.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import stain
from .classifier import SurveyScorer, TileClassifier
from .slide import Slide
from .tissue import blur_score, tissue_mask


@dataclass
class VisionConfig:
    analysis_mpp: float = 0.972  # PCam resolution
    tile_px: int = 96
    cell_tiles: int = 8
    thumb_max_side: int = 2048
    cell_tissue_min: float = 0.10  # fraction of a cell that must be tissue
    tile_tissue_min: float = 0.25
    hotspot_quantile: float = 0.90  # survey cells above this quantile seed hotspots
    sweep_block: int = 4  # sweep regions are sweep_block x sweep_block cells
    positive_prob: float = 0.5  # calibrated tile probability counted as tumour
    borderline: tuple[float, float] = (0.25, 0.75)
    blur_min: float = 30.0  # Laplacian variance below this flags a blurry look
    stain_normalize: str = "none"  # none | reinhard | macenko for normal looks

    @property
    def cell_px(self) -> int:
        return self.tile_px * self.cell_tiles


@dataclass
class Region:
    id: int
    kind: str  # hotspot | sweep
    cells: list[tuple[int, int]]  # (row, col) on the cell grid
    survey_max: float
    survey_mean: float
    center: tuple[float, float]  # (x, y) as fractions of the tissue bounding box
    bbox_um: tuple[float, float, float, float]

    def public(self, visited: set) -> dict:
        n_vis = sum(1 for c in self.cells if c in visited)
        return {
            "region_id": self.id,
            "kind": self.kind,
            "n_cells": len(self.cells),
            "cells_viewed": n_vis,
            "survey_max": round(self.survey_max, 3),
            "survey_mean": round(self.survey_mean, 3),
            "center_xy": [round(self.center[0], 3), round(self.center[1], 3)],
            "location": describe_location(*self.center),
        }


def describe_location(x: float, y: float) -> str:
    v = "upper" if y < 1 / 3 else "lower" if y > 2 / 3 else "middle"
    h = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "centre"
    return "centre" if (v, h) == ("middle", "centre") else f"{v} {h}"


@dataclass
class Look:
    n: int
    cell: tuple[int, int]
    region_id: int
    max_prob: float
    top3_mean: float
    n_tumour_tiles: int
    n_tissue_tiles: int
    blur: float
    refined: bool
    snapshot: str
    seconds: float


@dataclass
class SlideState:
    slide_path: str
    slide: Slide
    run_dir: Path
    thumb: np.ndarray
    thumb_level: int
    mask: np.ndarray
    grid_rows: int
    grid_cols: int
    cell0: float  # cell size in level-0 pixels
    cell_tissue: np.ndarray  # rows x cols tissue fraction
    survey: np.ndarray  # rows x cols survey score (nan = not tissue)
    regions: dict[int, Region]
    cell_region: dict[tuple[int, int], int]
    tile_prob: np.ndarray  # (rows*T) x (cols*T), nan = not looked at / not tissue
    visited: set = field(default_factory=set)
    looks: list[Look] = field(default_factory=list)
    frontier: list[tuple[int, int]] = field(default_factory=list)
    survey_seconds: float = 0.0


class VisionService:
    def __init__(self, classifier: TileClassifier | None, scorer: SurveyScorer, config: VisionConfig | None = None,
                 out_dir: str | Path = "runs"):
        self.clf = classifier
        self.scorer = scorer
        self.cfg = config or VisionConfig()
        self.out_dir = Path(out_dir)
        self.states: dict[str, SlideState] = {}

    # ------------------------------------------------------------------
    # survey_slide
    # ------------------------------------------------------------------
    def survey_slide(self, slide_path: str, session: str | None = None) -> dict:
        t0 = time.time()
        cfg = self.cfg
        slide = Slide(slide_path)
        session = session or Path(slide_path).stem
        run_dir = self.out_dir / session
        (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)

        tl = slide.thumbnail_level(cfg.thumb_max_side)
        thumb = slide.read_level(tl)
        mask = tissue_mask(thumb)

        w0, h0 = slide.dimensions
        cell0 = cfg.cell_px * cfg.analysis_mpp / slide.mpp
        rows, cols = math.ceil(h0 / cell0), math.ceil(w0 / cell0)
        # tissue fraction per cell: pad the mask to whole cells, then area-average
        ds = slide.levels[tl].downsample
        pad = np.zeros((int(round(rows * cell0 / ds)), int(round(cols * cell0 / ds))), np.uint8)
        hh, ww = min(pad.shape[0], mask.shape[0]), min(pad.shape[1], mask.shape[1])
        pad[:hh, :ww] = mask[:hh, :ww]
        cell_tissue = cv2.resize(pad.astype(np.float32) / 255, (cols, rows), interpolation=cv2.INTER_AREA)
        tissue_cells = cell_tissue >= cfg.cell_tissue_min

        survey = self._survey_scores(slide, tissue_cells, cell0)
        regions, cell_region = self._make_regions(survey, tissue_cells, rows, cols, cell0, slide)

        T = cfg.cell_tiles
        st = SlideState(slide_path, slide, run_dir, thumb, tl, mask, rows, cols, cell0, cell_tissue, survey,
                        regions, cell_region, np.full((rows * T, cols * T), np.nan, np.float32))
        st.survey_seconds = time.time() - t0
        self.states[session] = st
        cv2.imwrite(str(run_dir / "survey.jpg"), self.render_overlay(session, show_survey=True))

        ranked = sorted(regions.values(), key=lambda r: -r.survey_max)
        return {
            "session": session,
            "slide": Path(slide_path).name,
            "slide_size_mm": [round(w0 * slide.mpp / 1000, 1), round(h0 * slide.mpp / 1000, 1)],
            "tissue_cells": int(tissue_cells.sum()),
            "tissue_area_mm2": round(float(tissue_cells.sum()) * (cfg.cell_px * cfg.analysis_mpp / 1000) ** 2, 1),
            "n_regions": len(regions),
            "n_hotspots": sum(r.kind == "hotspot" for r in regions.values()),
            "survey_model": self.scorer.kind,
            "regions": [r.public(st.visited) for r in ranked],
            "overview_image": str(run_dir / "survey.jpg"),
            "seconds": round(st.survey_seconds, 2),
        }

    def _survey_scores(self, slide: Slide, tissue_cells: np.ndarray, cell0: float) -> np.ndarray:
        """Score each tissue cell at low zoom. Reads one bounding box per tissue
        piece rather than one request per cell."""
        rows, cols = tissue_cells.shape
        level = slide.best_level_for_mpp(self.scorer.mpp)
        ds = slide.levels[level].downsample
        cell_l = cell0 / ds  # cell size in survey-level pixels
        sub = 4 if self.scorer.model is None else max(1, int(round(cell_l / self.scorer.model.input_size)))
        survey = np.full((rows, cols), np.nan, np.float32)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_cells.astype(np.uint8), connectivity=8)
        for lab in range(1, n):
            c0, r0, cw, rh, _ = stats[lab]
            x, y = int(c0 * cell_l), int(r0 * cell_l)
            w, h = int(math.ceil(cw * cell_l)), int(math.ceil(rh * cell_l))
            img = slide.read_region(level, x, y, w, h)
            tiles, where = [], []
            for r in range(r0, r0 + rh):
                for c in range(c0, c0 + cw):
                    if labels[r, c] != lab:
                        continue
                    cy, cx = int((r - r0) * cell_l), int((c - c0) * cell_l)
                    cell_img = img[cy : cy + int(cell_l), cx : cx + int(cell_l)]
                    step = max(1, cell_img.shape[0] // sub)
                    for i in range(sub):
                        for j in range(sub):
                            t = cell_img[i * step : (i + 1) * step, j * step : (j + 1) * step]
                            if t.size:
                                tiles.append(t)
                                where.append((r, c))
            if not tiles:
                continue
            scores = self.scorer.score(tiles)
            for (r, c), s in zip(where, scores):
                survey[r, c] = s if np.isnan(survey[r, c]) else max(survey[r, c], s)
        return survey

    def _make_regions(self, survey, tissue_cells, rows, cols, cell0, slide):
        cfg = self.cfg
        vals = survey[tissue_cells]
        regions: dict[int, Region] = {}
        cell_region: dict[tuple[int, int], int] = {}
        if vals.size == 0:
            return regions, cell_region
        thr = float(np.nanquantile(vals, cfg.hotspot_quantile))
        hot = (tissue_cells & (np.nan_to_num(survey, nan=-1) >= thr)).astype(np.uint8)
        n, labels = cv2.connectedComponents(hot, connectivity=8)
        cell_um = cell0 * slide.mpp
        # positions are given relative to the tissue's bounding box, which is what
        # a pathologist means by "lower left" when looking at the slide
        tr, tc = np.nonzero(tissue_cells)
        r_lo, r_span = tr.min(), tr.max() - tr.min() + 1
        c_lo, c_span = tc.min(), tc.max() - tc.min() + 1

        def add(kind, cells):
            s = np.array([survey[c] for c in cells], np.float32)
            rs = np.array([c[0] for c in cells]); cs = np.array([c[1] for c in cells])
            rid = len(regions) + 1
            regions[rid] = Region(
                rid, kind, sorted(cells, key=lambda c: -np.nan_to_num(survey[c])),
                float(np.nanmax(s)), float(np.nanmean(s)),
                (float((cs.mean() + 0.5 - c_lo) / c_span), float((rs.mean() + 0.5 - r_lo) / r_span)),
                (float(cs.min() * cell_um), float(rs.min() * cell_um),
                 float((cs.max() - cs.min() + 1) * cell_um), float((rs.max() - rs.min() + 1) * cell_um)),
            )
            for c in cells:
                cell_region[c] = rid

        for lab in range(1, n):
            add("hotspot", [tuple(map(int, rc)) for rc in zip(*np.nonzero(labels == lab))])
        b = cfg.sweep_block
        for br in range(0, rows, b):
            for bc in range(0, cols, b):
                cells = [(r, c) for r in range(br, min(br + b, rows)) for c in range(bc, min(bc + b, cols))
                         if tissue_cells[r, c] and (r, c) not in cell_region]
                if cells:
                    add("sweep", cells)
        # Renumber by survey score so region 1 is the most suspicious
        order = sorted(regions.values(), key=lambda r: -r.survey_max)
        remap = {r.id: i + 1 for i, r in enumerate(order)}
        regions = {remap[r.id]: Region(remap[r.id], r.kind, r.cells, r.survey_max, r.survey_mean, r.center, r.bbox_um)
                   for r in order}
        cell_region = {c: remap[i] for c, i in cell_region.items()}
        return regions, cell_region

    def region_near(self, session: str, x: float, y: float) -> int | None:
        """Region whose nearest unviewed cell is closest to a tissue-relative point."""
        st = self.states[session]
        cells = list(st.cell_region)
        if not cells:
            return None
        rs = np.array([c[0] for c in cells]); cs = np.array([c[1] for c in cells])
        r_lo, r_span = rs.min(), rs.max() - rs.min() + 1
        c_lo, c_span = cs.min(), cs.max() - cs.min() + 1
        d = ((cs + 0.5 - c_lo) / c_span - x) ** 2 + ((rs + 0.5 - r_lo) / r_span - y) ** 2
        d = d + np.array([1.0 if c in st.visited else 0.0 for c in cells])  # prefer unviewed
        return st.cell_region[cells[int(np.argmin(d))]]

    # ------------------------------------------------------------------
    # zoom_region / refine_region
    # ------------------------------------------------------------------
    def next_cells(self, session: str, region_id: int, k: int) -> list[tuple[int, int]]:
        """Pick the next unvisited cells for a region: frontier cells first
        (neighbours of tumour found so far), then by survey score."""
        st = self.states[session]
        reg = st.regions[region_id]
        in_reg = set(reg.cells)
        picks = [c for c in st.frontier if c not in st.visited and (c in in_reg or self._touches(c, in_reg))]
        picks += [c for c in reg.cells if c not in st.visited and c not in picks]
        return picks[:k]

    @staticmethod
    def _touches(c, cells):
        return any((c[0] + dr, c[1] + dc) in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1))

    def look(self, session: str, cell: tuple[int, int], refine: bool = False) -> Look:
        """Read one cell at analysis resolution and classify its tiles."""
        if self.clf is None:
            raise RuntimeError("No tile classifier loaded (models/tile_classifier.onnx)")
        t0 = time.time()
        st, cfg = self.states[session], self.cfg
        img = self._read_cell(st, cell)
        T, P = cfg.cell_tiles, cfg.tile_px
        mode = "macenko" if refine else cfg.stain_normalize
        if mode == "macenko":
            img = stain.macenko(img, source=stain.macenko_stains(img))
        elif mode == "reinhard":
            img = stain.reinhard(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        tis = ((hsv[..., 1] > 25) & (hsv[..., 2] > 40)).astype(np.float32)
        tile_tis = cv2.resize(tis, (T, T), interpolation=cv2.INTER_AREA)
        tiles, idx = [], []
        for i in range(T):
            for j in range(T):
                if tile_tis[i, j] >= cfg.tile_tissue_min:
                    tiles.append(img[i * P : (i + 1) * P, j * P : (j + 1) * P])
                    idx.append((i, j))
        probs = (self.clf.predict_tta(tiles) if refine else self.clf.predict(tiles)) if tiles else np.zeros(0)
        grid = np.full((T, T), np.nan, np.float32)
        for (i, j), p in zip(idx, probs):
            grid[i, j] = p
        r, c = cell
        st.tile_prob[r * T : (r + 1) * T, c * T : (c + 1) * T] = grid
        st.visited.add(cell)

        pos = np.nan_to_num(grid, nan=0) >= cfg.positive_prob
        if pos.sum() >= 2:  # tumour reaching a cell edge: queue the neighbour across that edge
            for dr, dc, edge in ((-1, 0, pos[0].sum() >= 2), (1, 0, pos[-1].sum() >= 2),
                                 (0, -1, pos[:, 0].sum() >= 2), (0, 1, pos[:, -1].sum() >= 2)):
                nb = (r + dr, c + dc)
                if edge and nb in st.cell_region and nb not in st.visited and nb not in st.frontier:
                    st.frontier.append(nb)

        n = len(st.looks) + 1
        snap = st.run_dir / "snapshots" / f"look_{n:03d}_r{r}_c{c}.jpg"
        cv2.imwrite(str(snap), self._snapshot(img, grid))
        top = np.sort(probs)[::-1][:3] if len(probs) else np.zeros(1)
        lk = Look(n, cell, st.cell_region.get(cell, 0), float(probs.max()) if len(probs) else 0.0,
                  float(top.mean()), int(pos.sum()), len(tiles), blur_score(img), refine, str(snap),
                  time.time() - t0)
        st.looks.append(lk)
        return lk

    def _read_cell(self, st: SlideState, cell) -> np.ndarray:
        cfg, slide = self.cfg, st.slide
        level = slide.best_level_for_mpp(cfg.analysis_mpp)
        ds = slide.levels[level].downsample
        size_l = int(round(st.cell0 / ds))
        x, y = int(round(cell[1] * st.cell0 / ds)), int(round(cell[0] * st.cell0 / ds))
        img = slide.read_region(level, x, y, size_l, size_l)
        if size_l != cfg.cell_px:
            img = cv2.resize(img, (cfg.cell_px, cfg.cell_px), interpolation=cv2.INTER_AREA)
        return img

    def _snapshot(self, img: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Cell image with tumour-probability tiles tinted and outlined."""
        P = self.cfg.tile_px
        heat = cv2.resize(np.nan_to_num(grid, nan=0), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        alpha = (cv2.resize((~np.isnan(grid)).astype(np.float32), (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST) * 0.35)[..., None]
        out = (img * (1 - alpha) + color * alpha).astype(np.uint8)
        for i, j in zip(*np.nonzero(np.nan_to_num(grid, nan=0) >= self.cfg.positive_prob)):
            cv2.rectangle(out, (j * P, i * P), ((j + 1) * P - 1, (i + 1) * P - 1), (0, 0, 255), 2)
        return cv2.resize(out, (384, 384), interpolation=cv2.INTER_AREA)

    def zoom_region(self, session: str, region_id: int, looks: int = 1) -> dict:
        st = self.states[session]
        if region_id not in st.regions:
            return {"error": f"unknown region {region_id}"}
        cells = self.next_cells(session, region_id, looks)
        if not cells:
            return {"region_id": region_id, "note": "every cell in this region has already been viewed",
                    "looks": []}
        res = [self.look(session, c) for c in cells]
        return {"region_id": region_id, "looks": [self._look_public(l) for l in res],
                **self._region_summary(st, region_id)}

    def refine_region(self, session: str, region_id: int, max_looks: int = 2) -> dict:
        """Re-read the most borderline viewed cells of a region with Macenko
        stain normalisation and 8-way test-time augmentation."""
        st, (lo, hi) = self.states[session], self.cfg.borderline
        cand = [l for l in st.looks if l.region_id == region_id and not l.refined and lo <= l.max_prob <= hi]
        cand = sorted(cand, key=lambda l: abs(l.max_prob - 0.5))[:max_looks]
        if not cand:
            return {"region_id": region_id, "note": "no borderline cells to refine in this region", "looks": []}
        out = []
        for l in cand:
            new = self.look(session, l.cell, refine=True)
            out.append({**self._look_public(new), "before_max_prob": round(l.max_prob, 3)})
        return {"region_id": region_id, "looks": out, **self._region_summary(st, region_id)}

    def _look_public(self, l: Look) -> dict:
        d = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(l).items()}
        d["cell"] = list(l.cell)
        if l.blur < self.cfg.blur_min:
            d["warning"] = "region looks out of focus"
        return d

    def _region_summary(self, st: SlideState, region_id: int) -> dict:
        reg = st.regions[region_id]
        T = self.cfg.cell_tiles
        probs = np.concatenate([st.tile_prob[r * T:(r + 1) * T, c * T:(c + 1) * T].ravel()
                                for r, c in reg.cells if (r, c) in st.visited] or [np.zeros(0)])
        probs = probs[~np.isnan(probs)]
        return {
            "region_cells_viewed": sum(c in st.visited for c in reg.cells),
            "region_cells_total": len(reg.cells),
            "region_max_prob": round(float(probs.max()), 3) if probs.size else None,
            "region_tumour_tiles": int((probs >= self.cfg.positive_prob).sum()),
            "frontier_cells_pending": sum(1 for c in st.frontier if c not in st.visited),
        }

    # ------------------------------------------------------------------
    # measure_lesion
    # ------------------------------------------------------------------
    def lesions(self, session: str, threshold: float | None = None) -> list[dict]:
        """Merge tumour tiles into lesions with OpenCV contours and measure them."""
        st, cfg = self.states[session], self.cfg
        thr = cfg.positive_prob if threshold is None else threshold
        prob = np.nan_to_num(st.tile_prob, nan=0)
        binary = (prob >= thr).astype(np.uint8)
        if not binary.any():
            return []
        tile_um = cfg.tile_px * cfg.analysis_mpp
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        T = cfg.cell_tiles
        H, W = st.tile_prob.shape
        out = []
        for lab in range(1, n):
            comp = (labels == lab).astype(np.uint8)
            # Upsample 4x before contouring so minAreaRect sees tile corners, not centres
            up = cv2.resize(comp, (W * 4, H * 4), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(np.vstack(cnts))
            p = prob[comp > 0]
            ys, xs = np.nonzero(comp)
            # does it touch a cell nobody has looked at yet?
            dil = cv2.dilate(comp, np.ones((3, 3), np.uint8))
            ring = (dil > 0) & (comp == 0)
            unexplored = bool(np.isnan(st.tile_prob[ring]).any() and any(
                (int(y) // T, int(x) // T) not in st.visited and (int(y) // T, int(x) // T) in st.cell_region
                for y, x in zip(*np.nonzero(ring))))
            diam = max(rw, rh) / 4 * tile_um
            w0, h0 = st.slide.dimensions
            px0 = cfg.tile_px * cfg.analysis_mpp / st.slide.mpp  # tile size in level-0 px
            out.append({
                "n_tiles": int(comp.sum()),
                "max_prob": round(float(p.max()), 3),
                "top3_mean": round(float(np.sort(p)[::-1][:3].mean()), 3),
                "largest_diameter_um": round(float(diam), 0),
                "area_mm2": round(float(comp.sum()) * (tile_um / 1000) ** 2, 3),
                "size_class": "macrometastasis (>2 mm)" if diam > 2000 else
                              "micrometastasis (0.2-2 mm)" if diam >= 200 else "isolated tumour cells (<0.2 mm)",
                "center_xy": [round(float(cx / 4 * px0 / w0), 4), round(float(cy / 4 * px0 / h0), 4)],
                "center_level0": [int(cx / 4 * px0), int(cy / 4 * px0)],
                "bbox_level0": [int(xs.min() * px0), int(ys.min() * px0), int((xs.max() - xs.min() + 1) * px0),
                                int((ys.max() - ys.min() + 1) * px0)],
                "regions": sorted({st.cell_region.get((int(y) // T, int(x) // T), 0) for y, x in zip(ys, xs)}),
                "touches_unviewed_tissue": unexplored,
            })
        out.sort(key=lambda d: (-d["top3_mean"], -d["n_tiles"]))
        for i, d in enumerate(out):
            d["lesion_id"] = i + 1
        return out

    def measure_lesion(self, session: str, region_id: int | None = None) -> dict:
        les = self.lesions(session)
        if region_id is not None:
            les = [l for l in les if region_id in l["regions"]]
        return {"region_id": region_id, "lesions": les[:10], "n_lesions": len(les)}

    # ------------------------------------------------------------------
    # slide-level numbers and overlays
    # ------------------------------------------------------------------
    def slide_score(self, session: str) -> float:
        """Slide-level tumour score: best lesion's mean of its top-3 tile probabilities.
        Requiring 3 tiles damps single-tile false alarms."""
        les = self.lesions(session)
        return max((l["top3_mean"] for l in les if l["n_tiles"] >= 2), default=
                   max((l["max_prob"] * 0.5 for l in les), default=0.0))

    def coverage(self, session: str) -> dict:
        st = self.states[session]
        total = len(st.cell_region)
        return {"cells_viewed": len(st.visited), "tissue_cells": total,
                "fraction_viewed": round(len(st.visited) / max(total, 1), 4)}

    def render_overlay(self, session: str, show_survey: bool = False, path: bool = True) -> np.ndarray:
        """Thumbnail with tissue outline, survey/tumour heatmap and numbered search path."""
        st = self.states[session]
        out = st.thumb.copy()
        H, W = out.shape[:2]
        ds = st.slide.levels[st.thumb_level].downsample
        cell_t = st.cell0 / ds
        cnts, _ = cv2.findContours(st.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (80, 160, 80), 1)
        layer = st.survey if show_survey else st.tile_prob
        scale = cell_t if show_survey else cell_t / self.cfg.cell_tiles
        if np.isfinite(layer).any():
            lay = np.nan_to_num(layer, nan=0)
            if show_survey:
                fin = layer[np.isfinite(layer)]
                lay = np.clip((lay - fin.min()) / (np.ptp(fin) + 1e-6), 0, 1) * np.isfinite(layer)
            big = cv2.resize(lay.astype(np.float32), (int(layer.shape[1] * scale), int(layer.shape[0] * scale)),
                             interpolation=cv2.INTER_NEAREST)[:H, :W]
            bigm = cv2.resize(np.isfinite(layer).astype(np.uint8), (big.shape[1], big.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
            heat = cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_JET)
            roi = out[: big.shape[0], : big.shape[1]]
            blend = cv2.addWeighted(roi, 0.5, heat, 0.5, 0)
            roi[bigm] = blend[bigm]
        if path:
            prev = None
            for lk in st.looks:
                r, c = lk.cell
                p1 = (int(c * cell_t), int(r * cell_t))
                p2 = (int((c + 1) * cell_t), int((r + 1) * cell_t))
                col = (0, 0, 255) if lk.n_tumour_tiles else (255, 255, 255)
                cv2.rectangle(out, p1, p2, col, 1)
                ctr = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                if prev is not None:
                    cv2.line(out, prev, ctr, (255, 200, 0), 1, cv2.LINE_AA)
                prev = ctr
            for lk in st.looks:
                r, c = lk.cell
                cv2.putText(out, str(lk.n), (int(c * cell_t) + 1, int((r + 1) * cell_t) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def save_state(self, session: str):
        st = self.states[session]
        np.save(st.run_dir / "tile_prob.npy", st.tile_prob)
        np.save(st.run_dir / "survey.npy", st.survey)
        (st.run_dir / "looks.json").write_text(json.dumps([self._look_public(l) for l in st.looks], indent=1))
        cv2.imwrite(str(st.run_dir / "overlay.jpg"), self.render_overlay(session))
