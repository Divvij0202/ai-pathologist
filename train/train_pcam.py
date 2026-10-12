"""Train the tile classifier, calibrate it, and export ONNX for cv.dnn.

Runs on one GPU (an RTX 4060 8 GB is plenty) or, slowly, on CPU.

Data sources (pick one):
  --pcam DIR   PatchCamelyon .h5 files (camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5)
               from https://github.com/basveeling/pcam (Zenodo record 2546921), gunzipped.
  --npz FILE.. tiles cut from CAMELYON16 slides by train/make_tiles.py (split by slide).

Examples:
  # the real model (about 1-2 h on an RTX 4060 for 5 epochs of ResNet-18)
  python -m train.train_pcam --pcam ~/pcam --arch resnet18 --pretrained --epochs 5 --out models/tile_classifier.onnx
  # the low-zoom survey model: same data, downsampled 4x to ~3.9 um/px (24 px tiles)
  python -m train.train_pcam --pcam ~/pcam --arch small --downsample 4 --epochs 5 --out models/survey_classifier.onnx
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data
class H5Split(Dataset):
    """Lazy PCam split. Opens the h5 file per worker (h5py handles are not fork safe)."""

    def __init__(self, x_path, y_path, limit=None):
        import h5py
        self.x_path, self.y_path = str(x_path), str(y_path)
        with h5py.File(self.y_path, "r") as f:
            self.y = f["y"][:].reshape(-1).astype(np.float32)
        self.n = len(self.y) if limit is None else min(limit, len(self.y))
        self._x = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._x is None:
            import h5py
            self._x = h5py.File(self.x_path, "r")["x"]
        return self._x[i], self.y[i]  # RGB uint8 96x96x3


class ArraySplit(Dataset):
    def __init__(self, x_rgb, y):
        self.x, self.y = x_rgb, y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_data(args):
    if args.pcam:
        d = Path(args.pcam).expanduser()
        f = lambda s, k: d / f"camelyonpatch_level_2_split_{s}_{k}.h5"
        return (H5Split(f("train", "x"), f("train", "y"), args.limit),
                H5Split(f("valid", "x"), f("valid", "y"), args.limit and args.limit // 8),
                H5Split(f("test", "x"), f("test", "y")))
    import cv2
    zs = [np.load(f) for f in args.npz]
    x = np.concatenate([z["x"] for z in zs]); y = np.concatenate([z["y"] for z in zs])
    slides = np.concatenate([z["slide"] for z in zs])
    # drop glass/fat tiles with the same tissue rule the vision service uses at inference
    keep = np.array([_tissue_frac(t, cv2) >= 0.25 for t in x])
    print(f"kept {keep.sum()}/{len(keep)} tiles with tissue")
    x, y, slides = x[keep][..., ::-1].copy(), y[keep], slides[keep]  # make_tiles saves BGR; train in RGB like PCam
    names = sorted(set(slides))
    rng = np.random.default_rng(0)
    rng.shuffle(names)
    # split by slide so validation never sees a training slide
    n_val = max(2, len(names) // 5)
    val_names, test_names = set(names[:n_val // 2 or 1]), set(names[n_val // 2 or 1:n_val])
    va = np.isin(slides, list(val_names)); te = np.isin(slides, list(test_names)); tr = ~(va | te)
    print("slides  val:", sorted(val_names), " test:", sorted(test_names))
    return ArraySplit(x[tr], y[tr]), ArraySplit(x[va], y[va]), ArraySplit(x[te], y[te])


def _tissue_frac(bgr, cv2):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[..., 1] > 25) & (hsv[..., 2] > 40)))


class GPUAugment(nn.Module):
    """Flips, 90-degree rotations and colour jitter, done on the GPU batch."""

    def __init__(self, downsample=1, jitter=0.1):
        super().__init__()
        self.ds, self.jitter = downsample, jitter
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x_uint8, train: bool):
        x = x_uint8.permute(0, 3, 1, 2).float() / 255
        if self.ds > 1:
            x = F.interpolate(x, scale_factor=1 / self.ds, mode="area")
        if train:
            k = int(torch.randint(4, (1,)))
            x = torch.rot90(x, k, (2, 3))
            if torch.rand(1) < 0.5:
                x = x.flip(3)
            b = x.shape[0]
            j = self.jitter
            bright = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * j * 2.5
            contrast = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * j * 5
            sat = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * j * 2.5
            hue_shift = (torch.rand(b, 3, 1, 1, device=x.device) * 2 - 1) * j * 0.4
            x = x * bright
            m = x.mean((1, 2, 3), keepdim=True)
            x = (x - m) * contrast + m
            g = x.mean(1, keepdim=True)
            x = (x - g) * sat + g
            x = (x + hue_shift).clamp(0, 1)
        return (x - self.mean) / self.std


# ---------------------------------------------------------------- models
class SmallNet(nn.Module):
    """Tiny CNN for the survey model or CPU smoke tests."""

    def __init__(self, width=32):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        w = width
        self.f = nn.Sequential(blk(3, w), nn.MaxPool2d(2), blk(w, 2 * w), nn.MaxPool2d(2), blk(2 * w, 4 * w),
                               nn.MaxPool2d(2), blk(4 * w, 8 * w))
        self.head = nn.Linear(8 * w, 1)

    def forward(self, x):
        return self.head(self.f(x).mean((2, 3)))


def build_model(arch, pretrained):
    import torchvision
    if arch == "small":
        return SmallNet()
    if arch == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    if arch == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
        return m
    raise ValueError(arch)


# ---------------------------------------------------------------- loops
@torch.no_grad()
def predict_logits(model, aug, loader, device):
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        x = aug(x.to(device, non_blocking=True), train=False)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            zs.append(model(x).float().view(-1).cpu())
        ys.append(y.view(-1))
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def fit_temperature(logits, labels):
    """Temperature scaling on the validation split (Guo et al. 2017)."""
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp())


def ece(probs, labels, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.any():
            e += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def metrics(logits, labels, T=1.0):
    p = 1 / (1 + np.exp(-logits / T))
    return {"acc": float(((p > 0.5) == (labels > 0.5)).mean()), "auc": auc(logits, labels), "ece": ece(p, labels),
            "n": int(len(labels))}


class ExportWrapper(nn.Module):
    """Wraps the model so the ONNX graph takes normalised NCHW float input
    (what cv.dnn.blobFromImages + mean/std gives) and returns one logit."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def export_onnx(model, size, out, meta):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = ExportWrapper(model).cpu().eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, str(out), input_names=["input"], output_names=["logit"],
                      dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17, dynamo=False)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    # sanity check: OpenCV must load it and agree with PyTorch
    import cv2
    net = cv2.dnn.readNetFromONNX(str(out))
    x = np.random.randn(4, 3, size, size).astype(np.float32)
    net.setInput(x)
    y_cv = net.forward().reshape(-1)
    with torch.no_grad():
        y_pt = model(torch.from_numpy(x)).numpy().reshape(-1)
    print(f"cv.dnn vs torch max abs diff: {np.abs(y_cv - y_pt).max():.2e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcam")
    src.add_argument("--npz", nargs="+")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0", "small"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--downsample", type=int, default=1, help="4 = survey model at ~3.9 um/px")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training patches")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="models/tile_classifier.onnx")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    train_ds, val_ds, test_ds = load_data(args)
    print("patches  train", len(train_ds), " val", len(val_ds), " test", len(test_ds))
    kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, **kw)
    test_dl = DataLoader(test_ds, **kw)

    model = build_model(args.arch, args.pretrained).to(device).to(memory_format=torch.channels_last)
    aug = GPUAugment(args.downsample).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            x = aug(x.to(device, non_blocking=True), train=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device).float().view(-1, 1)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y); n += len(y)
            if i % 200 == 0:
                print(f"  ep {ep} step {i}/{len(train_dl)} loss {tot / max(n, 1):.4f}", flush=True)
        zv, yv = predict_logits(model, aug, val_dl, device)
        m = metrics(zv, yv)
        print(f"epoch {ep}: train loss {tot / n:.4f}  val {m}  ({time.time() - t0:.0f}s)", flush=True)
        if m["auc"] > best:
            best, best_state = m["auc"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    zv, yv = predict_logits(model, aug, val_dl, device)
    T = fit_temperature(zv, yv)
    zt, yt = predict_logits(model, aug, test_dl, device)
    report = {"val_uncalibrated": metrics(zv, yv), "val_calibrated": metrics(zv, yv, T),
              "test_calibrated": metrics(zt, yt, T), "temperature": T}
    print(json.dumps(report, indent=2))

    size = 96 // args.downsample
    meta = {"input_size": size, "mean": list(MEAN), "std": list(STD), "temperature": T,
            "mpp": 0.972 * args.downsample, "arch": args.arch, "pretrained": args.pretrained,
            "data": "pcam" if args.pcam else [Path(f).name for f in args.npz], "metrics": report}
    out = export_onnx(model, size, args.out, meta)
    torch.save(best_state, Path(args.out).with_suffix(".pt"))
    print("exported", out)


if __name__ == "__main__":
    main()
