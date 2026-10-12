# AI Pathologist Agent

An agent that reads a whole-slide image the way a pathologist does: it surveys the slide at low zoom, picks
suspicious areas, zooms in only where needed, follows tumour to its edges, and stops when it has enough
evidence. It works on CAMELYON16 sentinel lymph node slides (breast cancer metastases).

**Research prototype and second reader only. Not a diagnostic device. A pathologist reviews every slide.**

Built for the OpenCV AI Competition 2026: OpenCV 5 does all the image analysis, and it runs on AWS Graviton
in us-west-2 next to the public CAMELYON16 bucket.

## How it works

```
 Claude (Bedrock)  ──tool calls──▶  SearchSession (budget + safety rules, decision log)
                                          │
                                          ▼
                                   VisionService (OpenCV 5)
                   survey · zoom · refine · measure · overlays
                                          │  HTTP range reads, only the tiles needed
                                          ▼
                         s3://camelyon-dataset/CAMELYON16 (public, us-west-2)
```

* **Slides stream from S3.** `pathagent/slide.py` reads the pyramid TIFF's tile index over HTTP range requests,
  so a 3 GB slide is never downloaded. Survey of one slide reads ~20-40 MB; each high-zoom look ~0.5 MB.
* **OpenCV 5 does the pixel work** (`pathagent/tissue.py`, `stain.py`, `vision.py`, `classifier.py`):
  HSV + Otsu tissue mask with morphology and contours; Laplacian-variance blur check; Reinhard (LAB) and
  Macenko stain normalisation; the ONNX tile classifier through `cv.dnn` with `blobFromImages`; lesion
  measurement with connected components, contours and `minAreaRect`; heatmaps with `applyColorMap`.
* **The grid.** Tiles are 96 px at 0.97 µm/px (PCam's resolution). A *cell* is 8x8 tiles (~0.75 mm); one cell
  read at high zoom is one *look*, the unit of budget. A *region* is a hotspot (connected high-suspicion cells
  from the low-zoom survey) or a sweep block (4x4 cells of the remaining tissue).
* **The agent** (`pathagent/agent/llm.py`) gets six tools: `survey_slide`, `zoom_region`, `refine_region`,
  `measure_lesion`, `flag_for_human`, `write_report` (plus a free `get_status`). It also sees the snapshot of
  its most suspicious look in each zoom.
* **Code owns the rules** (`pathagent/agent/session.py`): hard look budget; the slide-level call is computed from
  the evidence, not written by the LLM; a negative report is refused until every hotspot was viewed; unrefined
  borderline cells, blurry looks and lesions that run off into unviewed tissue are auto-flagged for review.
* **Baselines** (`pathagent/agent/policies.py`): fixed best-first search (also the fallback if the LLM fails)
  and an exhaustive sliding window, through the same tools and logging, for the agent-vs-fixed comparison.

## Quick start

```bash
pip install -r requirements.txt
# put a model in models/ (see Training), then:
python -m scripts.run_slide --slide tumor_016 --mode best_first --budget 40
python -m scripts.run_slide --slide tumor_016 --mode agent --backend bedrock --budget 40 \
       --chat "check the lower left again"
python -m scripts.evaluate_run --slide tumor_016 --mode best_first --budget 40   # score against the reference mask
uvicorn service.app:app --port 8000                                          # API + viewer
```

Agent backends: `bedrock` (Claude on Amazon Bedrock, uses your AWS credentials, region `AWS_REGION`, default
us-west-2) or `anthropic` (`ANTHROPIC_API_KEY`). Model defaults to Claude Opus 5.5; override with
`PATHAGENT_MODEL` or `--model`.

## Training (on your GPU)

```bash
pip install -r requirements-train.txt
# PCam: https://github.com/basveeling/pcam (Zenodo 2546921). Download and gunzip the 6 *_x.h5 / *_y.h5 files.
python -m train.train_pcam --pcam ~/pcam --arch resnet18 --pretrained --epochs 5 --batch 256 \
       --out models/tile_classifier.onnx
# optional survey model (~3.9 um/px, 24 px input): makes the low-zoom survey a learned score
python -m train.train_pcam --pcam ~/pcam --arch small --downsample 4 --epochs 5 --out models/survey_classifier.onnx
```

The script calibrates with temperature scaling on the PCam validation split, exports ONNX (opset 17), checks
that `cv.dnn` matches PyTorch, and writes `tile_classifier.json` with the temperature and test metrics.
To fine-tune on real slide tiles: `python -m train.make_tiles --slides tumor_001 ... --out data/c16.npz`
then `train_pcam --npz data/c16.npz`.

## Deploy on Graviton

```bash
EMAIL=you@example.com LIMIT=50 ./deploy/budget_alarm.sh
REPO_URL=https://github.com/<you>/ai-pathologist.git ./deploy/launch_graviton.sh
./deploy/teardown.sh
```

## Data and licences

CAMELYON16 (CC0) and PatchCamelyon (CC0). Cite: Ehteshami Bejnordi et al., JAMA 2017 (CAMELYON16);
Veeling et al., MICCAI 2018 (PCam). Tumour slides with non-exhaustive annotations (listed in the dataset
README) are excluded from evaluation.
