"""HTTP service: the vision tools plus agent runs, for the web viewer and the live endpoint.

    uvicorn service.app:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health
  GET  /slides                       demo slides available
  POST /runs {slide, mode, budget}   start a search (mode: agent | best_first | sliding_window)
  GET  /runs/{id}                    status, decision log, report (poll this for the live replay)
  POST /runs/{id}/chat {text}        pathologist instruction, e.g. "check the lower left again"
  GET  /runs/{id}/overlay.jpg        thumbnail with heatmap and search path
  GET  /runs/{id}/snapshots/{name}   one high-zoom snapshot
  GET  /runs/{id}/heat.png           tumour (or ?layer=survey) heatmap as a transparent PNG
  GET  /dz/{slide}.dzi               Deep Zoom descriptor for OpenSeadragon
  GET  /dz/{slide}_files/{l}/{c}_{r}.jpg  Deep Zoom tiles
  POST /tools/{tool}                 raw tool call on a run (for debugging and other agents)
"""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from pathagent.agent.llm import PathologistAgent
from pathagent.agent import steer
from pathagent.agent.policies import best_first, sliding_window
from pathagent.agent.session import Budget, SearchSession
from pathagent.classifier import load_models
from functools import lru_cache

from pathagent.deepzoom import DeepZoom, heat_rgba
from pathagent.slide import Slide, camelyon_url
from pathagent.vision import VisionService

MODEL_DIR = os.environ.get("PATHAGENT_MODELS", "models")
RUN_DIR = Path(os.environ.get("PATHAGENT_RUNS", "runs"))
BACKEND = os.environ.get("PATHAGENT_LLM", "bedrock")
DEMO_SLIDES = os.environ.get("PATHAGENT_DEMO_SLIDES", "tumor_026,tumor_027,normal_010,test_001,test_002,tumor_005").split(",")

app = FastAPI(title="AI Pathologist Agent")
from fastapi.staticfiles import StaticFiles  # noqa: E402
app.mount("/static", StaticFiles(directory=Path(__file__).with_name("static")), name="static")
clf, scorer = load_models(MODEL_DIR)
vision = VisionService(clf, scorer, out_dir=RUN_DIR)
runs: dict[str, dict] = {}
lock = threading.Lock()  # one search at a time per process keeps memory and S3 reads predictable


class RunRequest(BaseModel):
    slide: str
    mode: str = "agent"
    budget: int = 60


class ChatRequest(BaseModel):
    text: str
    extra_looks: int = 8


def _resolve(slide: str) -> str:
    if slide.startswith(("s3://", "http://", "https://", "/")):
        return slide
    return camelyon_url(slide)


def _worker(run_id: str):
    r = runs[run_id]
    s: SearchSession = r["session"]

    def on_event(kind, data):
        r["events"].append({"kind": kind, **data})

    with lock:
        r["status"] = "running"
        try:
            if r["mode"] == "agent" and BACKEND != "none":
                agent = PathologistAgent(s, backend=BACKEND, on_event=on_event)
                r["agent"] = agent
                agent.run()
            elif r["mode"] in ("best_first", "agent"):
                best_first(s)
            else:
                sliding_window(s)
            r["status"] = "done"
        except Exception as e:  # surface the error to the viewer
            r["status"] = "error"
            r["error"] = f"{type(e).__name__}: {e}"


@lru_cache(maxsize=16)
def _dz(slide: str) -> DeepZoom:
    return DeepZoom(Slide(_resolve(slide)))


@app.get("/dz/{slide}.dzi")
def dzi(slide: str):
    return Response(_dz(slide).dzi(), media_type="application/xml")


@app.get("/dz/{slide}_files/{level}/{tile}")
def dz_tile(slide: str, level: int, tile: str):
    col, row = (int(v) for v in tile.split(".")[0].split("_"))
    data = _dz(slide).tile_jpeg(level, col, row)
    if data is None:
        raise HTTPException(404)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/runs/{run_id}/heat.png")
def heat(run_id: str, layer: str = "tumour"):
    st = vision.states.get(run_id)
    if st is None:
        raise HTTPException(404)
    png = heat_rgba(st.survey, cv2.COLORMAP_VIRIDIS, normalize=True) if layer == "survey" else heat_rgba(st.tile_prob)
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return {"ok": True, "classifier": clf.name if clf else None,
            "survey": scorer.kind, "opencv": cv2.__version__, "llm": BACKEND}


@app.get("/slides")
def slides():
    return {"demo_slides": DEMO_SLIDES}


@app.post("/runs")
def start_run(req: RunRequest):
    if clf is None:
        raise HTTPException(503, "no tile classifier loaded")
    run_id = uuid.uuid4().hex[:10]
    s = SearchSession(vision, _resolve(req.slide), Budget(max_looks=req.budget), session_id=run_id)
    runs[run_id] = {"id": run_id, "slide": req.slide, "mode": req.mode, "status": "queued", "session": s,
                    "events": []}
    threading.Thread(target=_worker, args=(run_id,), daemon=True).start()
    return {"run_id": run_id}


@app.get("/runs/{run_id}")
def get_run(run_id: str, since: int = 0):
    r = runs.get(run_id) or HTTPException(404)
    if isinstance(r, HTTPException):
        raise r
    s: SearchSession = r["session"]
    st = vision.states.get(run_id)
    return {
        "id": run_id, "slide": r["slide"], "mode": r["mode"], "status": r["status"], "error": r.get("error"),
        "looks_used": s.looks_used, "budget": s.budget.max_looks,
        "log": [e.__dict__ for e in s.log[since:]], "events": r["events"],
        "report": s.report,
        "geometry": _geometry(st) if st else None,
        "looks": [_look_geo(st, l) for l in st.looks] if st else [],
        "thoughts": [e for e in r["events"] if e["kind"] in ("thought", "error", "user")],
    }


def _geometry(st):
    w0, h0 = st.slide.dimensions
    rs = [c[0] for c in st.cell_region] or [0]
    cs = [c[1] for c in st.cell_region] or [0]
    bbox = [min(cs) * st.cell0, min(rs) * st.cell0, (max(cs) - min(cs) + 1) * st.cell0, (max(rs) - min(rs) + 1) * st.cell0]
    return {"width": w0, "height": h0, "rows": st.grid_rows, "cols": st.grid_cols, "cell0": st.cell0,
            "mpp": st.slide.mpp, "tissue_bbox": bbox}


def _look_geo(st, l):
    r, c = l.cell
    return {"n": l.n, "x": c * st.cell0, "y": r * st.cell0, "size": st.cell0, "max_prob": round(l.max_prob, 3),
            "tumour_tiles": l.n_tumour_tiles, "region_id": l.region_id, "refined": l.refined,
            "snapshot": Path(l.snapshot).name}


@app.post("/runs/{run_id}/chat")
def chat(run_id: str, req: ChatRequest):
    r = runs.get(run_id)
    if not r:
        raise HTTPException(404)
    if r["status"] in ("running", "queued"):
        raise HTTPException(409, "still working")

    def go():
        with lock:
            r["status"] = "running"
            r["events"].append({"kind": "user", "text": req.text})
            try:
                if "agent" in r:
                    r["agent"].chat(req.text, req.extra_looks)
                else:  # no LLM in this run: rule-based steering
                    steer.handle(r["session"], req.text, req.extra_looks)
                r["status"] = "done"
            except Exception as e:
                r["status"], r["error"] = "error", str(e)
    threading.Thread(target=go, daemon=True).start()
    return {"ok": True}


@app.get("/runs/{run_id}/overlay.jpg")
def overlay(run_id: str, survey: bool = False):
    if run_id not in vision.states:
        raise HTTPException(404)
    ok, buf = cv2.imencode(".jpg", vision.render_overlay(run_id, show_survey=survey))
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/runs/{run_id}/snapshots/{name}")
def snapshot(run_id: str, name: str):
    p = (RUN_DIR / run_id / "snapshots" / Path(name).name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/tools/{tool}")
def raw_tool(tool: str, body: dict):
    r = runs.get(body.pop("run_id", ""))
    if not r:
        raise HTTPException(404, "run_id")
    return r["session"].call(tool, body, reason=body.pop("reason", "api"), actor="api")


@app.get("/", response_class=HTMLResponse)
def index():
    viewer = Path(__file__).with_name("viewer.html")
    return viewer.read_text() if viewer.exists() else "<p>AI Pathologist Agent API. See /docs.</p>"
