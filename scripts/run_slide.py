"""Search one slide from the command line.

    python -m scripts.run_slide --slide tumor_016 --mode best_first --budget 40
    python -m scripts.run_slide --slide test_001 --mode agent --backend bedrock --budget 60
    python -m scripts.run_slide --slide tumor_016 --mode agent --chat "check the lower left again"
"""
from __future__ import annotations

import argparse
import json
import time

from pathagent.agent.policies import best_first, sliding_window
from pathagent.agent.session import Budget, SearchSession
from pathagent.classifier import load_models
from pathagent.slide import camelyon_url
from pathagent.vision import VisionService


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True, help="CAMELYON16 name (tumor_016) or a path/URL")
    ap.add_argument("--mode", default="best_first", choices=["agent", "best_first", "sliding_window"])
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--backend", default="bedrock", choices=["bedrock", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--no-images", action="store_true", help="don't show snapshots to the LLM")
    ap.add_argument("--chat", action="append", default=[], help="pathologist follow-up(s) after the report")
    ap.add_argument("--models", default="models")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--session", default=None)
    a = ap.parse_args()

    clf, scorer = load_models(a.models)
    if clf is None:
        raise SystemExit(f"no tile classifier in {a.models}/ (train one with train/train_pcam.py)")
    vision = VisionService(clf, scorer, out_dir=a.runs)
    path = a.slide if "/" in a.slide else camelyon_url(a.slide)
    sid = a.session or f"{a.slide.split('/')[-1].removesuffix('.tif')}_{a.mode}"
    s = SearchSession(vision, path, Budget(max_looks=a.budget), session_id=sid)
    t0 = time.time()

    def show(kind, d):
        if kind == "thought":
            print(f"  [agent] {d['text']}")
        elif kind == "tool":
            print(f"  -> {d['tool']}({json.dumps(d['args'])})  # {d['reason']}")
        elif kind == "result":
            r = d["result"]
            if "error" in r:
                print(f"     ! {r['error']}")
            elif d["tool"] in ("zoom_region", "refine_region"):
                for l in r.get("looks", []):
                    print(f"     look {l['n']}: cell {l['cell']} max p={l['max_prob']:.2f} tumour tiles={l['n_tumour_tiles']}")
        elif kind == "error":
            print(f"  ! {d['message']}")

    if a.mode == "agent":
        from pathagent.agent.llm import PathologistAgent
        agent = PathologistAgent(s, backend=a.backend, model=a.model, effort=a.effort,
                                 show_images=not a.no_images, on_event=show)
        report = agent.run()
        for msg in a.chat:
            print(f"\nPathologist: {msg}")
            report = agent.chat(msg)
        print("tokens:", agent.usage)
    elif a.mode == "best_first":
        report = best_first(s)
    else:
        report = sliding_window(s)

    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, indent=1))
    for f in report.get("findings", []):
        print(f"  finding: lesion {f['lesion_id']} {f['size_class']}, {f['largest_diameter_um']:.0f} um, "
              f"p={f['top3_mean']:.2f}, regions {f['regions']}")
    print(f"done in {time.time() - t0:.1f}s; run folder: {vision.states[sid].run_dir}")


if __name__ == "__main__":
    main()
