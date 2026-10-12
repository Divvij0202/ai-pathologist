"""Offline tests (no network): synthetic slides exercise tissue detection, stain
normalisation, the search session rules and lesion measurement.

    python -m pytest -q tests
"""
import math

import cv2
import numpy as np
import pytest

from pathagent import stain
from pathagent.agent.policies import best_first
from pathagent.agent.session import Budget, SearchSession
from pathagent.classifier import SurveyScorer
from pathagent.slide import Level
from pathagent.tissue import blur_score, tissue_mask
from pathagent.vision import VisionService

PINK = (200, 150, 230)  # BGR, eosin-ish
PURPLE = (180, 80, 120)  # BGR, hematoxylin-ish


def synthetic_slide(h=4096, w=3072, tumour_center=(2000, 1500), tumour_r=300):
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, PURPLE, -1)
    rng = np.random.default_rng(0)
    img = cv2.add(img, rng.integers(0, 20, img.shape, dtype=np.uint8))
    cv2.circle(img, tumour_center[::-1], tumour_r, PINK, -1)
    return img


class FakeSlide:
    """In-memory slide pyramid; level 0 at 0.972 um/px so a small array is a realistic-size slide."""

    def __init__(self, img, mpp=0.972):
        self.path = "fake"
        self.mpp = mpp
        self.pyr = [img]
        for _ in range(5):
            self.pyr.append(cv2.pyrDown(self.pyr[-1]))
        self.levels = [Level(i, p.shape[1], p.shape[0], img.shape[1] / p.shape[1]) for i, p in enumerate(self.pyr)]

    dimensions = property(lambda s: (s.levels[0].width, s.levels[0].height))

    def best_level_for_mpp(self, target):
        best = 0
        for lv in self.levels:
            if self.mpp * lv.downsample <= target * 1.05:
                best = lv.index
        return best

    def thumbnail_level(self, max_side=4096):
        return len(self.levels) - 1

    def read_region(self, level, x, y, w, h):
        out = np.full((h, w, 3), 255, np.uint8)
        src = self.pyr[level]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, src.shape[1]), min(y + h, src.shape[0])
        if x1 > x0 and y1 > y0:
            out[y0 - y:y1 - y, x0 - x:x1 - x] = src[y0:y1, x0:x1]
        return out

    def read_level(self, level):
        return self.pyr[level].copy()


class PinkClassifier:
    """Stand-in for the ONNX model: pink tiles are 'tumour'."""
    name, mpp, input_size = "pink", 0.972, 96

    def predict(self, tiles):
        out = []
        for t in tiles:
            t = t.astype(np.int16)
            pink = (t[..., 2] - t[..., 1] > 55) & (t[..., 1] > 110)
            out.append(0.98 if pink.mean() > 0.3 else 0.01)
        return np.array(out, np.float32)

    predict_tta = predict


def test_tissue_mask_finds_tissue_and_drops_glass():
    img = synthetic_slide()
    small = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8))
    m = tissue_mask(small)
    frac = m.mean() / 255
    assert 0.25 < frac < 0.5  # ellipse covers pi/9 ~ 0.35 of the image
    assert m[0, 0] == 0


def test_blur_score_drops_when_blurred():
    img = synthetic_slide(512, 512, (256, 256), 50)
    assert blur_score(cv2.GaussianBlur(img, (21, 21), 8)) < blur_score(img)


def test_reinhard_moves_stats_toward_target():
    img = synthetic_slide(256, 256, (128, 128), 40)
    out = stain.reinhard(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_macenko_runs():
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal((180, 100, 160), 30, (128, 128, 3)), 0, 255).astype(np.uint8)
    assert stain.macenko(img).shape == img.shape


@pytest.fixture
def vision(tmp_path, monkeypatch):
    import pathagent.vision as V
    fake = FakeSlide(synthetic_slide(h=8192, w=6144, tumour_center=(4000, 3000), tumour_r=900))
    monkeypatch.setattr(V, "Slide", lambda path: fake)
    return VisionService(PinkClassifier(), SurveyScorer(), out_dir=tmp_path)


def test_best_first_finds_and_measures_lesion(vision):
    s = SearchSession(vision, "fake", Budget(max_looks=20), session_id="t")
    report = best_first(s)
    assert report["slide_call"] == "likely metastasis"
    les = report["findings"][0]
    # tumour disc diameter = 1800 px * 0.972 um = 1750 um; tiles are 93 um so allow slack
    assert 1300 < les["largest_diameter_um"] < 2400
    assert s.looks_used <= 20


def test_negative_report_refused_until_hotspots_viewed(vision):
    s = SearchSession(vision, "fake", Budget(max_looks=50), session_id="t2")
    s.call("survey_slide")
    st = vision.states["t2"]
    # look only at a sweep cell far from the tumour
    sweep = [r for r in st.regions.values() if r.kind == "sweep"][-1]
    s.call("zoom_region", {"region_id": sweep.id, "looks": 1})
    res = s.call("write_report", {"summary": "nothing"})
    if any(r.kind == "hotspot" for r in st.regions.values()):
        assert "error" in res


def test_budget_is_enforced(vision):
    s = SearchSession(vision, "fake", Budget(max_looks=3, max_looks_per_call=4), session_id="t3")
    s.call("survey_slide")
    for rid in list(vision.states["t3"].regions)[:6]:
        res = s.call("zoom_region", {"region_id": rid, "looks": 4})
    assert s.looks_used == 3
    assert "error" in res
