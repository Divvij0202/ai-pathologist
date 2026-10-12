"""The LLM agent: Claude chooses the search strategy through tool calls.

Backends
  bedrock    Claude on Amazon Bedrock via the Anthropic SDK's Mantle client (default on AWS)
  anthropic  the Claude API directly (ANTHROPIC_API_KEY)

The loop is written by hand (not the SDK tool runner) because the session
needs to inject the pathologist's follow-up messages and fall back to the
fixed best-first search if the model stops early, refuses, or errors.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2

from .policies import best_first
from .session import SearchSession

SYSTEM = """You are a slide-reading assistant working for a pathologist. You search one H&E whole-slide image \
of a sentinel lymph node for breast cancer metastases. You never see the full-resolution slide yourself; \
you use tools that read it for you.

How the slide is organised:
- survey_slide gives a low-zoom map: tissue split into regions. "hotspot" regions had high low-zoom suspicion; \
"sweep" regions are the rest of the tissue in blocks. Locations are relative to the tissue (x, y in 0..1, y down).
- A cell is about 0.75 mm of tissue. One high-zoom look reads one cell and classifies its 96-pixel tiles with a \
calibrated tumour classifier. Looks are the budget, so spend them deliberately.
- When tumour reaches the edge of a cell, the neighbouring cell is queued as "frontier" and zoom_region on that \
region will follow it first.

Working method:
1. Survey. Then plan: which hotspots first, how much budget to keep for sweeping the rest.
2. Zoom into the most suspicious regions. Follow positive findings to their edges, then measure_lesion.
3. Borderline results (max tile probability roughly 0.25-0.75) deserve refine_region, which re-reads with stain \
normalisation and test-time augmentation. If still unclear, flag_for_human.
4. If nothing turns up in the hotspots, sweep the remaining tissue broadly rather than going deeper in one place. \
Small metastases (micrometastases, under 2 mm) are the ones people miss.
5. Stop when the evidence is strong (a clear lesion found and measured) or the budget runs low, then write_report \
with a short plain-language summary of what you did and why.

The code computes the slide-level call and enforces the budget; you cannot override them, and write_report will \
tell you if a rule stops you from finishing. You are a second reader: say what you saw, never give a diagnosis. \
Keep each text reply to one or two sentences explaining your next step; that text is shown in the replay."""

TOOLS = [
    {"name": "survey_slide", "description": "Low-zoom survey: tissue mask, regions ranked by coarse suspicion. Free. Call once first.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "zoom_region",
     "description": "High-zoom look at the next 1-4 unviewed cells of a region (frontier cells first, then by survey score). "
                    "Costs one look per cell. Give region_id, or near_xy=[x,y] (tissue-relative 0..1) to look near a location.",
     "input_schema": {"type": "object", "properties": {
         "region_id": {"type": "integer"}, "looks": {"type": "integer", "minimum": 1, "maximum": 4},
         "near_xy": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
         "reason": {"type": "string", "description": "one short sentence: why this region now"}},
         "required": ["reason"], "additionalProperties": False}},
    {"name": "refine_region",
     "description": "Re-read up to 2 borderline cells already viewed in a region with Macenko stain normalisation and 8-way "
                    "test-time augmentation. Costs one look per cell re-read.",
     "input_schema": {"type": "object", "properties": {"region_id": {"type": "integer"}, "reason": {"type": "string"}},
                      "required": ["region_id", "reason"], "additionalProperties": False}},
    {"name": "measure_lesion",
     "description": "Merge tumour tiles found so far into lesions (OpenCV contours) and measure largest diameter in microns. "
                    "Free. Optionally limit to one region.",
     "input_schema": {"type": "object", "properties": {"region_id": {"type": "integer"}, "reason": {"type": "string"}},
                      "additionalProperties": False}},
    {"name": "flag_for_human", "description": "Add a region, cell or lesion to the pathologist's review list with a reason. Free.",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}, "reason": {"type": "string"}},
                      "required": ["target", "reason"], "additionalProperties": False}},
    {"name": "get_status", "description": "Coverage, current slide score, unviewed hotspots, top lesions and flags. Free.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "write_report",
     "description": "Finish: the code computes the slide call from the evidence; you supply a 2-4 sentence summary of the "
                    "search and what the pathologist should look at.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                      "required": ["summary"], "additionalProperties": False}},
]


def make_client(backend: str):
    import anthropic
    if backend == "bedrock":
        from anthropic import AnthropicBedrockMantle
        return AnthropicBedrockMantle(aws_region=os.environ.get("AWS_REGION", "us-west-2"))
    return anthropic.Anthropic()


def default_model(backend: str) -> str:
    env = os.environ.get("PATHAGENT_MODEL")
    if env:
        return env
    return "anthropic.claude-opus-5-5" if backend == "bedrock" else "claude-opus-5-5"


class PathologistAgent:
    def __init__(self, session: SearchSession, backend: str = "bedrock", model: str | None = None,
                 effort: str = "medium", show_images: bool = True, max_turns: int = 60, on_event=None):
        self.s = session
        self.backend = backend
        self.client = make_client(backend)
        self.model = model or default_model(backend)
        self.effort = effort
        self.show_images = show_images
        self.max_turns = max_turns
        self.messages: list[dict] = []
        self.on_event = on_event or (lambda kind, data: None)
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    # ------------------------------------------------------------------
    def run(self, task: str | None = None) -> dict:
        task = task or (f"Search slide {Path(self.s.slide_path).name} for metastases. "
                        f"Budget: {self.s.budget.max_looks} high-zoom looks.")
        self.messages.append({"role": "user", "content": task})
        return self._loop()

    def chat(self, text: str, extra_looks: int = 8) -> dict:
        """A pathologist instruction after (or during) the search, e.g. 'check the lower left again'."""
        self.s.add_budget(extra_looks)
        self.s.log_user = getattr(self.s, "log_user", []) + [text]
        self.messages.append({"role": "user", "content":
                              f"Pathologist: {text}\n(You have {extra_looks} extra looks for this. "
                              "Do what was asked, then call write_report again with an updated summary.)"})
        return self._loop()

    # ------------------------------------------------------------------
    def _create(self):
        kw = dict(model=self.model, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=self.messages,
                  output_config={"effort": self.effort})
        if self.backend == "anthropic":
            # Server-side fallback if the request is declined by a safety classifier
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                    extra_body={"fallbacks": "default"}, **kw)
        return self.client.messages.create(**kw)

    def _loop(self) -> dict:
        for _ in range(self.max_turns):
            try:
                resp = self._create()
            except Exception as e:  # network, throttling, auth: never leave the pathologist with nothing
                self.on_event("error", {"message": f"LLM call failed ({type(e).__name__}: {e}); finishing with best-first"})
                return self._fallback(f"LLM unavailable: {type(e).__name__}")
            u = getattr(resp, "usage", None)
            if u:
                self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
            if resp.stop_reason == "refusal":
                return self._fallback("model declined")
            content = [b.model_dump(exclude_none=True) for b in resp.content]
            self.messages.append({"role": "assistant", "content": content})
            text = " ".join(b.text for b in resp.content if b.type == "text").strip()
            if text:
                self.on_event("thought", {"text": text})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                if self.s.report:
                    return self.s.report
                self.messages.append({"role": "user", "content": "Continue the search with the tools, or call write_report."})
                continue
            results = []
            for tu in tool_uses:
                args = dict(tu.input or {})
                reason = args.pop("reason", text[:300])
                self.on_event("tool", {"tool": tu.name, "args": args, "reason": reason})
                out = self.s.call(tu.name, args, reason=reason, actor="llm")
                self.on_event("result", {"tool": tu.name, "result": out})
                results.append(self._tool_result(tu.id, tu.name, out))
            self.messages.append({"role": "user", "content": results})
            if self.s.report and any(t.name == "write_report" for t in tool_uses):
                return self.s.report
        return self._fallback("turn limit reached")

    def _tool_result(self, tool_use_id: str, name: str, out: dict) -> dict:
        body = [{"type": "text", "text": json.dumps(_for_llm(out))}]
        if self.show_images and name in ("zoom_region", "refine_region"):
            # Let Claude see the most suspicious snapshot of this call (agentic vision)
            looks = [l for l in out.get("looks", []) if l.get("max_prob", 0) >= 0.25]
            if looks:
                best = max(looks, key=lambda l: l["max_prob"])
                img = cv2.imread(best["snapshot"])
                if img is not None:
                    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (256, 256)), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    body.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                             "data": base64.b64encode(buf).decode()}})
                    body.append({"type": "text", "text": f"Snapshot of look {best['n']} (red boxes = tiles >= 0.5)."})
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": body, **({"is_error": True} if "error" in out else {})}

    def _fallback(self, why: str) -> dict:
        self.s.call("flag_for_human", {"target": "whole slide", "reason": f"agent fell back to best-first search: {why}"},
                    reason=why, actor="system")
        return best_first(self.s)


def _for_llm(out: dict) -> dict:
    """Strip file paths and noisy fields from tool results before they reach the model."""
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items() if k not in ("snapshot", "overview_image", "overlay_image",
                                                                    "center_level0", "bbox_level0", "seconds")}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return clean(out)
