"""Run a search on one or more slides and score each against the reference masks.

    python -m scripts.evaluate_run --slide tumor_016 tumor_027 normal_010 --mode best_first --budget 40
    python -m scripts.evaluate_run --slide tumor_016 --mode sliding_window --budget 10000

Writes runs/<slide>_<mode>/eval.json and prints a one-line summary per slide.
"""
from __future__ import annotations

import argparse
import json

from pathagent.agent.policies import best_first, sliding_window
from pathagent.agent.session import Budget, SearchSession
from pathagent.classifier import load_models
from pathagent.evaluate import evaluate
from pathagent.slide import camelyon_url
from pathagent.vision import VisionService


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", nargs="+", required=True)
    ap.add_argument("--mode", default="best_first", choices=["agent", "best_first", "sliding_window"])
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--backend", default="bedrock")
    ap.add_argument("--models", default="models")
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    clf, scorer = load_models(a.models)
    vision = VisionService(clf, scorer, out_dir=a.runs)
    rows = []
    for name in a.slide:
        sid = f"{name}_{a.mode}_{a.budget}"
        s = SearchSession(vision, camelyon_url(name), Budget(max_looks=a.budget), session_id=sid)
        if a.mode == "agent":
            from pathagent.agent.llm import PathologistAgent
            PathologistAgent(s, backend=a.backend).run()
        elif a.mode == "best_first":
            best_first(s)
        else:
            sliding_window(s)
        ev = evaluate(vision, sid)
        ev["report"] = {k: s.report[k] for k in ("slide_call", "slide_score", "looks_used", "coverage", "seconds")}
        (vision.states[sid].run_dir / "eval.json").write_text(json.dumps(ev, indent=1))
        rows.append(ev)
        print(f"{name:12s} call={s.report['slide_call']!r:55s} score={ev['slide_score']:.2f} "
              f"found {ev['found']}/{ev['reference_lesions']} (macro {ev['macro_found']}, micro {ev['micro_found']}) "
              f"first hit look {ev['first_hit_look']}  viewed {ev['fraction_tissue_viewed']:.1%}  "
              f"FP tiles {ev['false_positive_tiles']}  {s.report['seconds']}s", flush=True)
        vision.states.pop(sid, None)


if __name__ == "__main__":
    main()
