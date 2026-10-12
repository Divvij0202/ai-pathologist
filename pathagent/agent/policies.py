"""Non-LLM search strategies, run through the same session and tools so the
comparison with the agent is fair (same budget, same logging).

* best_first: follow tumour to its edges first (frontier), else the highest
  survey score anywhere. This is the fallback if the LLM agent misbehaves.
* sliding_window: look at every tissue cell in raster order (the exhaustive
  baseline; give it a budget >= the number of tissue cells).
"""
from __future__ import annotations

import numpy as np

from .session import SearchSession


def best_first(s: SearchSession, stop_early: bool = False, max_follow: int = 4) -> dict:
    """max_follow caps how many looks are spent tracing one lesion's edges before
    moving to the next seed, so one big lesion can't eat the whole budget."""
    if not s.surveyed:
        s.call("survey_slide", reason="start", actor="best_first")
    st = s.vision.states[s.session_id]
    follow = 0
    while s.looks_left > 0:
        pending = [c for c in st.frontier if c not in st.visited]
        if pending and follow < max_follow:
            cell = pending[0]
            follow += 1
            why = "follow tumour across the cell edge"
        else:
            cand = [c for c in st.cell_region if c not in st.visited]
            if not cand:
                break
            cell = max(cand, key=lambda c: np.nan_to_num(st.survey[c], nan=-1))
            follow = 0
            st.frontier = [c for c in st.frontier if c != cell]
            why = f"highest remaining survey score {st.survey[cell]:.2f}"
        rid = st.cell_region[cell]
        # put this cell at the head of the frontier so zoom_region picks it
        st.frontier.insert(0, cell)
        s.call("zoom_region", {"region_id": rid, "looks": 1}, reason=why, actor="best_first")
        if stop_early:
            les = s.vision.lesions(s.session_id)
            if any(l["top3_mean"] >= s.rules.early_stop_lesion and l["n_tiles"] >= 3 and not l["touches_unviewed_tissue"]
                   for l in les):
                break
    return s.call("write_report", {"summary": "Fixed best-first search (no LLM)."}, reason="done", actor="best_first")


def sliding_window(s: SearchSession) -> dict:
    if not s.surveyed:
        s.call("survey_slide", reason="start", actor="sliding_window")
    st = s.vision.states[s.session_id]
    for cell in sorted(st.cell_region):
        if s.looks_left <= 0:
            break
        if cell in st.visited:
            continue
        st.frontier.insert(0, cell)
        s.call("zoom_region", {"region_id": st.cell_region[cell], "looks": 1}, reason="raster scan",
               actor="sliding_window")
    return s.call("write_report", {"summary": "Exhaustive sliding window (no LLM)."}, reason="done",
                  actor="sliding_window")
