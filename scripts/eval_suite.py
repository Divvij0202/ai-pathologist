"""Evaluation suite: search strategies x look budgets on a fixed slide subset.

Uses the exhaustive caches from scripts/build_cache.py, so every strategy sees
identical classifier outputs and a full run takes minutes, not hours.

    python -m scripts.eval_suite --cache eval/cache/standin --tag standin
    python -m scripts.eval_suite --cache eval/cache/resnet18 --tag resnet18 --modes best_first agent --backend bedrock

Writes eval/results/<tag>/: rows.csv, summary.md, summary.json, efficiency_curve.png
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from pathagent.agent.policies import best_first, sliding_window
from pathagent.agent.session import Budget, SearchSession
from pathagent.classifier import load_models
from pathagent.evaluate import evaluate
from pathagent.slide import camelyon_url
from pathagent.vision import VisionService

GRID = np.linspace(0, 1, 101)


def labels(ref_csv: str) -> dict:
    out = {}
    for r in csv.DictReader(open(ref_csv)):
        out[r["image"].removesuffix(".tif")] = {"tumour": r["type"] == "tumor", "class": r["class"]}
    return out


def auc(scores, y):
    scores, y = np.asarray(scores, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    for v in np.unique(scores):
        m = scores == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_one(vision, name, mode, budget, backend):
    sid = f"{name}_{mode}_{budget}"
    s = SearchSession(vision, camelyon_url(name), Budget(max_looks=budget), session_id=sid)
    if mode == "agent":
        from pathagent.agent.llm import PathologistAgent
        PathologistAgent(s, backend=backend, show_images=False).run()
    elif mode == "best_first":
        best_first(s)
    else:
        sliding_window(s)
    ev = evaluate(vision, sid)
    ev["mode"], ev["budget"], ev["slide_call"] = mode, budget, s.report["slide_call"]
    vision.states.pop(sid, None)
    return ev


def curve_on_grid(curve, n_lesions):
    """Step-interpolate (fraction viewed, recall) onto GRID; recall stays flat after the run ends."""
    if not curve:
        return np.zeros_like(GRID)
    xs = np.array([0.0] + [c[0] for c in curve]); ys = np.array([0.0] + [c[1] for c in curve])
    idx = np.searchsorted(xs, GRID, side="right") - 1
    return ys[np.clip(idx, 0, len(ys) - 1)]


def plot(curves: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"sliding_window": "#eb6834", "best_first": "#2a78d6", "agent": "#1baf7a"}
    names = {"sliding_window": "Sliding window (raster order)", "best_first": "Fixed best-first", "agent": "LLM agent"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for mode, (x, y) in curves.items():
        ax.plot(x * 100, y * 100, color=colors.get(mode, "#888"), lw=2, label=names.get(mode, mode))
        ax.annotate(names.get(mode, mode), (x[-1] * 100, y[-1] * 100), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color="#333", va="center")
    ax.set_xlabel("Tissue viewed at high zoom (%)")
    ax.set_ylabel("Reference lesions found (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.grid(color="#e6e6e6", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Lesions found vs tissue viewed", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", default="eval/test_subset.txt")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--modes", nargs="+", default=["sliding_window", "best_first"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 20, 40, 80])
    ap.add_argument("--backend", default="bedrock")
    ap.add_argument("--models", default="models")
    ap.add_argument("--reference", default="eval/reference.csv")
    a = ap.parse_args()

    names = [l.strip() for l in open(a.slides) if l.strip()]
    names = [n for n in names if (Path(a.cache) / f"{n}.npz").exists()]
    lab = labels(a.reference)
    clf, scorer = load_models(a.models)
    vision = VisionService(clf, scorer, out_dir=f"/tmp/eval_runs/{a.tag}", cache_dir=a.cache)
    out = Path("eval/results") / a.tag
    out.mkdir(parents=True, exist_ok=True)

    rows, curves = [], {}
    for mode in a.modes:
        budgets = [100000] if mode == "sliding_window" else a.budgets
        for b in budgets:
            for name in names:
                ev = run_one(vision, name, mode, b, a.backend)
                ev["label"] = lab.get(name, {}).get("class", "?")
                rows.append(ev)
                print(f"{mode:15s} b={b:<6d} {name}  found {ev['found']}/{ev['reference_lesions']}  "
                      f"viewed {ev['fraction_tissue_viewed']:.1%}  score {ev['slide_score']:.2f}", flush=True)
        # pooled efficiency curve at the largest budget
        big = [r for r in rows if r["mode"] == mode and r["budget"] == max(budgets)]
        w = np.array([r["reference_lesions"] for r in big], float)
        if w.sum():
            ys = np.array([curve_on_grid(r["efficiency_curve"], r["reference_lesions"]) for r in big])
            reach = max(r["fraction_tissue_viewed"] for r in big)
            m = GRID <= reach + 1e-9
            curves[mode] = (GRID[m], (ys * w[:, None]).sum(0)[m] / w.sum())

    # summary table
    summary = []
    for mode in a.modes:
        for b in sorted({r["budget"] for r in rows if r["mode"] == mode}):
            rs = [r for r in rows if r["mode"] == mode and r["budget"] == b]
            tum = [r for r in rs if r["label"] in ("macro", "micro")]
            n_les = sum(r["reference_lesions"] for r in rs)
            mac = [tuple(map(int, r["macro_found"].split("/"))) for r in rs]
            mic = [tuple(map(int, r["micro_found"].split("/"))) for r in rs]
            summary.append({
                "mode": mode, "budget": "all" if b >= 100000 else b, "slides": len(rs),
                "slide_auc": auc([r["slide_score"] for r in rs], [r["label"] in ("macro", "micro") for r in rs]),
                "tumour_slides_with_a_hit": f"{sum(r['found'] > 0 for r in tum)}/{len(tum)}",
                "lesion_recall": round(sum(r["found"] for r in rs) / max(n_les, 1), 3),
                "macro_recall": f"{sum(f for f, _ in mac)}/{sum(t for _, t in mac)}",
                "micro_recall": f"{sum(f for f, _ in mic)}/{sum(t for _, t in mic)}",
                "mean_tissue_viewed": round(float(np.mean([r["fraction_tissue_viewed"] for r in rs])), 3),
                "fp_tiles_per_slide": round(float(np.mean([r["false_positive_tiles"] for r in rs])), 1),
            })
    with open(out / "rows.csv", "w", newline="") as f:
        keys = ["slide", "label", "mode", "budget", "reference_lesions", "found", "macro_found", "micro_found",
                "first_hit_look", "looks", "fraction_tissue_viewed", "false_positive_tiles", "slide_score", "slide_call"]
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    (out / "runs.json").write_text(json.dumps([{k: r[k] for k in ("slide", "label", "mode", "budget", "lesions",
                                                                     "found", "reference_lesions")} for r in rows]))
    (out / "summary.json").write_text(json.dumps({"summary": summary, "slides": names,
                                                  "model": json.loads((Path(a.cache) / "model.json").read_text())
                                                  if (Path(a.cache) / "model.json").exists() else None}, indent=1))
    hdr = list(summary[0].keys())
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join("" if v is None else str(v) for v in s.values()) + " |" for s in summary]
    (out / "summary.md").write_text(f"# Evaluation: {a.tag}\n\n{len(names)} slides from {a.slides}\n\n" + "\n".join(md) + "\n")
    if curves:
        plot(curves, out / "efficiency_curve.png")
    print("\n".join(md))
    print("wrote", out)


if __name__ == "__main__":
    main()
