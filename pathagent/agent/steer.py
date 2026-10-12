"""Rule-based handling of pathologist instructions, used when no LLM is available.

Understands a location ("lower left", "top right", "centre"), "region 4",
"lesion 2", and "again" / "more" / "higher zoom" (which turn into refine).
"""
from __future__ import annotations

import re

from .session import SearchSession

_V = {"upper": 0.17, "top": 0.17, "middle": 0.5, "center": 0.5, "centre": 0.5, "lower": 0.83, "bottom": 0.83}
_H = {"left": 0.17, "middle": 0.5, "center": 0.5, "centre": 0.5, "right": 0.83}


def parse_location(text: str):
    t = text.lower()
    y = next((v for k, v in _V.items() if re.search(rf"\b{k}\b", t)), None)
    x = next((v for k, v in _H.items() if re.search(rf"\b{k}\b", t)), None)
    if x is None and y is None:
        return None
    return [0.5 if x is None else x, 0.5 if y is None else y]


def handle(s: SearchSession, text: str, extra_looks: int = 8) -> dict:
    s.add_budget(extra_looks)
    t = text.lower()
    actions = []
    m = re.search(r"region\s+(\d+)", t)
    lesion = re.search(r"lesion\s+(\d+)", t)
    if lesion:
        les = s.vision.lesions(s.session_id)
        k = int(lesion.group(1)) - 1
        if 0 <= k < len(les) and les[k]["regions"]:
            m = re.match(r"(\d+)", str(les[k]["regions"][0]))
    wants_refine = any(w in t for w in ("again", "re-check", "recheck", "higher zoom", "closer", "refine"))
    if m:
        rid = int(m.group(1))
        if wants_refine:
            actions.append(s.call("refine_region", {"region_id": rid}, reason=f"pathologist: {text}", actor="pathologist"))
        actions.append(s.call("zoom_region", {"region_id": rid, "looks": 4}, reason=f"pathologist: {text}",
                              actor="pathologist"))
    else:
        xy = parse_location(t) or [0.5, 0.5]
        for _ in range(2):
            actions.append(s.call("zoom_region", {"near_xy": xy, "looks": 4}, reason=f"pathologist: {text}",
                                  actor="pathologist"))
    s.call("measure_lesion", {}, reason="update measurements", actor="pathologist")
    return s.call("write_report", {"summary": f"Re-checked after pathologist request: '{text}'."},
                  reason="update report", actor="pathologist")
