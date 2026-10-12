"""A search session: the agent's six tools, with budgets and safety rules in code.

The LLM chooses strategy, but it cannot argue its way past anything here:
* every high-zoom look is counted against a hard budget;
* the slide-level call is computed from the evidence by code, not written by the LLM;
* a negative call is refused until every hotspot has been looked at (or the budget is spent);
* borderline results that were never refined go to the human review list automatically.
Every tool call is logged with the agent's stated reason, for the replay.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..vision import VisionService


@dataclass
class Budget:
    max_looks: int = 60
    max_looks_per_call: int = 4
    max_refines: int = 10


@dataclass
class Rules:
    positive_call: float = 0.85  # slide score at or above: "likely metastasis"
    review_call: float = 0.5  # between review_call and positive_call: "suspicious, review"
    early_stop_lesion: float = 0.9  # a lesion this strong (>= 3 tiles) allows stopping before the budget is used


@dataclass
class Flag:
    target: str
    reason: str
    by: str  # agent | rule


@dataclass
class LogEntry:
    step: int
    t: float
    actor: str
    tool: str
    args: dict
    reason: str
    result: dict
    looks_used: int


class BudgetExceeded(Exception):
    pass


class SearchSession:
    def __init__(self, vision: VisionService, slide_path: str, budget: Budget | None = None,
                 rules: Rules | None = None, session_id: str | None = None):
        self.vision = vision
        self.slide_path = slide_path
        self.budget = budget or Budget()
        self.rules = rules or Rules()
        self.session_id = session_id or Path(slide_path).stem
        self.flags: list[Flag] = []
        self.log: list[LogEntry] = []
        self.refines_used = 0
        self.report: dict | None = None
        self.surveyed = False
        self.t0 = time.time()

    # ----------------------------------------------------------- bookkeeping
    @property
    def looks_used(self) -> int:
        st = self.vision.states.get(self.session_id)
        return len(st.looks) if st else 0

    @property
    def looks_left(self) -> int:
        return self.budget.max_looks - self.looks_used

    def add_budget(self, looks: int):
        """Used when the pathologist asks for more work after the report."""
        self.budget.max_looks += looks
        self.report = None

    def call(self, tool: str, args: dict | None = None, reason: str = "", actor: str = "agent") -> dict:
        args = dict(args or {})
        fn = getattr(self, f"tool_{tool}", None)
        if fn is None:
            result = {"error": f"unknown tool {tool}"}
        else:
            try:
                result = fn(**args)
            except BudgetExceeded as e:
                result = {"error": str(e)}
            except TypeError as e:
                result = {"error": f"bad arguments: {e}"}
        if "error" not in result and tool != "write_report":
            result["looks_left"] = self.looks_left
        self.log.append(LogEntry(len(self.log) + 1, round(time.time() - self.t0, 2), actor, tool, args, reason,
                                 _compact(result), self.looks_used))
        return result

    # ----------------------------------------------------------- tools
    def tool_survey_slide(self) -> dict:
        res = self.vision.survey_slide(self.slide_path, self.session_id)
        self.surveyed = True
        res["budget"] = {"max_looks": self.budget.max_looks, "max_looks_per_call": self.budget.max_looks_per_call,
                         "cell_size_mm": round(self.vision.cfg.cell_px * self.vision.cfg.analysis_mpp / 1000, 2)}
        # keep the LLM's context small: top regions only, the rest summarised
        regs = res["regions"]
        res["regions"] = regs[:25]
        if len(regs) > 25:
            res["regions_not_shown"] = len(regs) - 25
        return res

    def _need_survey(self):
        if not self.surveyed:
            raise BudgetExceeded("call survey_slide first")

    def tool_zoom_region(self, region_id: int | None = None, looks: int = 1, near_xy: list | None = None) -> dict:
        self._need_survey()
        if region_id is None and near_xy is not None:
            region_id = self.vision.region_near(self.session_id, float(near_xy[0]), float(near_xy[1]))
        if region_id is None:
            return {"error": "give region_id or near_xy"}
        looks = max(1, min(int(looks), self.budget.max_looks_per_call))
        if self.looks_left <= 0:
            raise BudgetExceeded("look budget used up; call write_report")
        looks = min(looks, self.looks_left)
        return self.vision.zoom_region(self.session_id, int(region_id), looks)

    def tool_refine_region(self, region_id: int) -> dict:
        self._need_survey()
        if self.refines_used >= self.budget.max_refines:
            raise BudgetExceeded("refine budget used up")
        if self.looks_left <= 0:
            raise BudgetExceeded("look budget used up; call write_report")
        res = self.vision.refine_region(self.session_id, int(region_id), max_looks=min(2, self.looks_left))
        self.refines_used += 1 if res.get("looks") else 0
        return res

    def tool_measure_lesion(self, region_id: int | None = None) -> dict:
        self._need_survey()
        return self.vision.measure_lesion(self.session_id, None if region_id is None else int(region_id))

    def tool_flag_for_human(self, target: str, reason: str) -> dict:
        self.flags.append(Flag(str(target), reason, "agent"))
        return {"flagged": str(target), "review_list_size": len(self.flags)}

    def tool_get_status(self) -> dict:
        """Not one of the six plan tools: a free read of where the search stands."""
        self._need_survey()
        st = self.vision.states[self.session_id]
        unviewed_hot = [r.id for r in st.regions.values() if r.kind == "hotspot" and not any(c in st.visited for c in r.cells)]
        return {"coverage": self.vision.coverage(self.session_id), "slide_score": round(self.vision.slide_score(self.session_id), 3),
                "unviewed_hotspots": unviewed_hot, "lesions": self.vision.lesions(self.session_id)[:5],
                "flags": [asdict(f) for f in self.flags]}

    def tool_write_report(self, summary: str = "") -> dict:
        self._need_survey()
        v, sid, R = self.vision, self.session_id, self.rules
        st = v.states[sid]
        score = v.slide_score(sid)
        lesions = v.lesions(sid)
        strong = [l for l in lesions if l["top3_mean"] >= R.early_stop_lesion and l["n_tiles"] >= 3]
        unviewed_hot = [r.id for r in st.regions.values()
                        if r.kind == "hotspot" and not any(c in st.visited for c in r.cells)]
        if self.looks_left > 0 and not strong and unviewed_hot:
            return {"error": "report refused: no strong lesion found yet and hotspot regions "
                             f"{unviewed_hot[:10]} have not been viewed. Look at them or use up the budget first."}

        # rule-based flags
        lo, hi = v.cfg.borderline
        refined_cells = {l.cell for l in st.looks if l.refined}
        for l in st.looks:
            if lo <= l.max_prob <= hi and l.cell not in refined_cells and not l.refined:
                self.flags.append(Flag(f"cell {list(l.cell)} (region {l.region_id})",
                                       f"borderline tile probability {l.max_prob:.2f}, never refined", "rule"))
            if l.blur < v.cfg.blur_min:
                self.flags.append(Flag(f"cell {list(l.cell)} (region {l.region_id})", "out of focus", "rule"))
        for les in lesions:
            if les["touches_unviewed_tissue"] and les["top3_mean"] >= R.review_call:
                self.flags.append(Flag(f"lesion {les['lesion_id']}", "lesion runs into tissue that was not viewed; "
                                       "size may be underestimated", "rule"))

        if score >= R.positive_call:
            call = "likely metastasis"
        elif score >= R.review_call:
            call = "suspicious, needs pathologist review"
        else:
            call = "no suspicious regions found within the look budget"
        top = [l for l in lesions if l["top3_mean"] >= R.review_call][:5]
        v.save_state(sid)
        self.report = {
            "slide": Path(self.slide_path).name,
            "slide_call": call,
            "slide_score": round(score, 3),
            "review_regions": sorted({r for l in top for r in l["regions"]}),
            "findings": top,
            "coverage": v.coverage(sid),
            "looks_used": self.looks_used,
            "budget": self.budget.max_looks,
            "human_review": [asdict(f) for f in _dedupe(self.flags)],
            "agent_summary": summary,
            "overlay_image": str(st.run_dir / "overlay.jpg"),
            "seconds": round(time.time() - self.t0, 1),
            "disclaimer": "Research prototype, second reader only. A pathologist must review every slide.",
        }
        self.save()
        return self.report

    # ----------------------------------------------------------- persistence
    def save(self):
        st = self.vision.states.get(self.session_id)
        if not st:
            return
        (st.run_dir / "decision_log.json").write_text(json.dumps([asdict(e) for e in self.log], indent=1))
        if self.report:
            (st.run_dir / "report.json").write_text(json.dumps(self.report, indent=1))


def _dedupe(flags):
    seen, out = set(), []
    for f in flags:
        k = (f.target, f.reason)
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def _compact(result: dict) -> dict:
    """Keep the log readable: drop long lists from logged results."""
    out = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 6:
            out[k] = v[:6] + [f"... {len(v) - 6} more"]
        else:
            out[k] = v
    return out
