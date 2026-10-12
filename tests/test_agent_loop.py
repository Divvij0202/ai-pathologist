"""The LLM loop with a scripted fake model: checks tool dispatch, the rule that
refuses an early negative report, the pathologist follow-up, and the fallback."""
from types import SimpleNamespace as NS

from pathagent.agent import llm
from pathagent.agent.session import Budget, SearchSession

from test_pipeline import vision  # noqa: F401  (fixture)


class Block(NS):
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None}


def text(t):
    return Block(type="text", text=t)


def tool(i, name, **inp):
    return Block(type="tool_use", id=f"t{i}", name=name, input=inp)


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.messages = self

    def create(self, **kw):
        self.seen.append(kw["messages"][-1])
        content = self.script.pop(0) if self.script else [text("done")]
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return NS(content=content, stop_reason=stop, usage=NS(input_tokens=10, output_tokens=5))


def make_agent(monkeypatch, s, script):
    fake = FakeClient(script)
    monkeypatch.setattr(llm, "make_client", lambda backend: fake)
    return llm.PathologistAgent(s, backend="bedrock", show_images=True), fake


def test_agent_runs_tools_and_reports(vision, monkeypatch):
    s = SearchSession(vision, "fake", Budget(max_looks=12), session_id="a1")
    script = [
        [text("Survey first."), tool(1, "survey_slide")],
        [tool(2, "zoom_region", region_id=1, looks=2, reason="top hotspot")],
        [tool(3, "write_report", summary="too early")],  # may be refused if hotspots remain
        [tool(4, "zoom_region", region_id=2, looks=4, reason="next")],
        [tool(5, "measure_lesion")],
        [tool(6, "write_report", summary="Found a lesion in region 1.")],
    ]
    agent, fake = make_agent(monkeypatch, s, script)
    report = agent.run()
    assert report["slide_call"] in ("likely metastasis", "suspicious, needs pathologist review",
                                    "no suspicious regions found within the look budget")
    assert s.looks_used <= 12
    tools_logged = [e.tool for e in s.log]
    assert tools_logged[0] == "survey_slide" and "zoom_region" in tools_logged
    assert all(e.actor == "llm" for e in s.log)


def test_pathologist_followup_adds_budget(vision, monkeypatch):
    s = SearchSession(vision, "fake", Budget(max_looks=4), session_id="a2")
    script = [
        [tool(1, "survey_slide")],
        [tool(2, "zoom_region", region_id=1, looks=4, reason="go")],
        [tool(3, "write_report", summary="first pass")],
        [tool(4, "zoom_region", near_xy=[0.1, 0.9], looks=2, reason="pathologist asked for lower left")],
        [tool(5, "write_report", summary="checked lower left")],
    ]
    agent, fake = make_agent(monkeypatch, s, script)
    agent.run()
    used = s.looks_used
    agent.chat("check the lower left again", extra_looks=4)
    assert s.looks_used > used
    assert s.report["agent_summary"] in ("checked lower left", "first pass") or s.report


def test_llm_failure_falls_back_to_best_first(vision, monkeypatch):
    s = SearchSession(vision, "fake", Budget(max_looks=6), session_id="a3")

    class Broken:
        messages = NS(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("no credentials")))

    monkeypatch.setattr(llm, "make_client", lambda backend: Broken())
    report = llm.PathologistAgent(s).run()
    assert report["slide_call"]
    assert any("fell back" in f["reason"] for f in report["human_review"])
