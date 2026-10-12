"""Tile classifier run through OpenCV's DNN module (cv.dnn) from an ONNX file.

The model is trained on PCam (96x96 px tiles at ~0.97 um/px) by
train/train_pcam.py and exported with a JSON sidecar holding the input size,
normalisation and the calibration temperature:

    models/tile_classifier.onnx
    models/tile_classifier.json   {"input_size": 96, "mean": [...], "std": [...],
                                   "temperature": 1.3, "mpp": 0.972}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TileClassifier:
    def __init__(self, onnx_path: str | os.PathLike, batch_size: int = 64, threads: int | None = None):
        onnx_path = Path(onnx_path)
        meta_path = onnx_path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.input_size = int(self.meta.get("input_size", 96))
        self.mean = np.array(self.meta.get("mean", IMAGENET_MEAN), np.float32)
        self.std = np.array(self.meta.get("std", IMAGENET_STD), np.float32)
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.mpp = float(self.meta.get("mpp", 0.972))
        self.batch_size = batch_size
        if threads:
            cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        self.name = onnx_path.stem

    def _blob(self, tiles_bgr: list[np.ndarray]) -> np.ndarray:
        s = self.input_size
        # blobFromImages: resize, BGR->RGB, scale to [0,1], NCHW
        blob = cv2.dnn.blobFromImages(tiles_bgr, scalefactor=1.0 / 255, size=(s, s), swapRB=True, crop=False)
        blob -= self.mean.reshape(1, 3, 1, 1)
        blob /= self.std.reshape(1, 3, 1, 1)
        return blob

    def logits(self, tiles_bgr: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(tiles_bgr), self.batch_size):
            self.net.setInput(self._blob(tiles_bgr[i : i + self.batch_size]))
            y = self.net.forward().astype(np.float32)
            # Accept either one logit (tumour) or two logits (normal, tumour)
            y = y.reshape(len(tiles_bgr[i : i + self.batch_size]), -1)
            out.append(y[:, 0] if y.shape[1] == 1 else y[:, 1] - y[:, 0])
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def predict(self, tiles_bgr: list[np.ndarray]) -> np.ndarray:
        """Calibrated tumour probability per tile."""
        if not tiles_bgr:
            return np.zeros(0, np.float32)
        z = self.logits(tiles_bgr) / self.temperature
        return 1.0 / (1.0 + np.exp(-z))

    def predict_tta(self, tiles_bgr: list[np.ndarray]) -> np.ndarray:
        """Test-time augmentation: average over the 8 dihedral transforms."""
        if not tiles_bgr:
            return np.zeros(0, np.float32)
        logit_sum = np.zeros(len(tiles_bgr), np.float32)
        for k in range(4):
            for flip in (False, True):
                aug = []
                for t in tiles_bgr:
                    a = np.rot90(t, k)
                    aug.append(np.ascontiguousarray(cv2.flip(a, 1) if flip else a))
                logit_sum += self.logits(aug)
        z = logit_sum / 8 / self.temperature
        return 1.0 / (1.0 + np.exp(-z))


def survey_features(t: np.ndarray) -> np.ndarray:
    """Cheap OpenCV colour/texture features for one low-zoom tile (BGR)."""
    from .stain import _od, MACENKO_REF_STAINS
    lab = cv2.cvtColor(t, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    tis = ((hsv[..., 1] > 25) & (hsv[..., 2] > 40)).reshape(-1)
    if tis.sum() < 8:
        return np.zeros(14, np.float32)
    od = _od(t)
    conc = np.linalg.lstsq(MACENKO_REF_STAINS, od.T, rcond=None)[0]
    h, e = conc[0][tis], conc[1][tis]
    gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F).reshape(-1)[tis]
    L = lab[tis]
    return np.array([L[:, 0].mean(), L[:, 1].mean(), L[:, 2].mean(), L[:, 0].std(), L[:, 1].std(), L[:, 2].std(),
                     hsv[..., 1].reshape(-1)[tis].mean(), h.mean(), h.std(), e.mean(), np.mean(h > 0.6),
                     lap.std(), tis.mean(), np.percentile(L[:, 0], 10)], np.float32)


def expand(Z: np.ndarray) -> np.ndarray:
    """Linear + pairwise quadratic terms of standardised features."""
    iu = np.triu_indices(Z.shape[1])
    return np.concatenate([Z, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]], axis=1)


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    Z = (X - np.array(model["mu"])) / np.array(model["sd"])
    z = np.clip(expand(Z) @ np.array(model["w"]) + model["b"], -30, 30)
    return 1 / (1 + np.exp(-z))


class SurveyScorer:
    """Coarse suspicion score for a low-zoom tile.

    If a survey model (trained on PCam downsampled 4x, see train_pcam.py
    --downsample 4) is available it is used via cv.dnn. Otherwise a hand-built
    OpenCV colour/texture prior is used. Metastatic breast carcinoma in a lymph
    node tends to look paler and pinker than the dense, dark lymphocyte sea
    around it, with larger nuclei, so the prior favours tissue with moderate
    hematoxylin and high local contrast. It is a weak prior, only meant to
    order the search, never to make a call.
    """

    def __init__(self, model: TileClassifier | None = None, prior_path: str | os.PathLike | None = None):
        self.model = model
        self.prior = None
        if model is None and prior_path and Path(prior_path).exists():
            # logistic model on survey_features, fitted by scripts/fit_survey_prior.py
            self.prior = json.loads(Path(prior_path).read_text())

    @property
    def kind(self) -> str:
        return "onnx" if self.model else "opencv-features-fitted" if self.prior else "opencv-prior"

    @property
    def mpp(self) -> float:
        return self.model.mpp if self.model else 3.9

    def score(self, tiles_bgr: list[np.ndarray]) -> np.ndarray:
        if self.model is not None:
            return self.model.predict(tiles_bgr)
        if self.prior is not None:
            X = np.stack([survey_features(t) for t in tiles_bgr]).astype(np.float32)
            return predict(self.prior, X).astype(np.float32)
        return np.array([self._prior(t) for t in tiles_bgr], np.float32)

    @staticmethod
    def _prior(t: np.ndarray) -> float:
        lab = cv2.cvtColor(t, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
        tissue = (hsv[..., 1] > 25) & (hsv[..., 2] > 40)
        if tissue.mean() < 0.3:
            return 0.0
        L = lab[..., 0][tissue]
        a = lab[..., 1][tissue]
        gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
        contrast = cv2.Laplacian(gray, cv2.CV_32F)[tissue].std()
        # Dense lymphocytes: dark (low L) and blue-purple (low a). Tumour: mid L, pinker.
        lightness = np.clip((np.median(L) - 110) / 80, 0, 1)
        pinkness = np.clip((np.median(a) - 135) / 25, 0, 1)
        texture = np.clip(contrast / 40, 0, 1)
        return float(0.45 * lightness + 0.35 * pinkness + 0.2 * texture)


def load_models(model_dir: str | os.PathLike = "models"):
    """Return (TileClassifier | None, SurveyScorer)."""
    model_dir = Path(model_dir)
    tile = model_dir / "tile_classifier.onnx"
    survey = model_dir / "survey_classifier.onnx"
    clf = TileClassifier(tile) if tile.exists() else None
    scorer = SurveyScorer(TileClassifier(survey, batch_size=256) if survey.exists() else None,
                          prior_path=model_dir / "survey_prior.json")
    return clf, scorer
