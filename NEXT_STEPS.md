# Next steps (status Oct 12, 2026)

## Where things stand

The whole pipeline runs end to end on real CAMELYON16 slides streamed from S3. The tile classifier in `models/`
is a **stand-in** (small CNN trained on CPU on ~12k tiles cut from 24 training slides), so the numbers
below show the plumbing works, not final accuracy.

First run, fixed best-first search, 40 looks, held-out slides:

| slide | result | tissue viewed | time |
|---|---|---|---|
| tumor_026 | found 4/4 macrometastases, 0/11 small ones; 3 false-positive tiles | 8% | 37 s |
| tumor_026, exhaustive sliding window | 4/4 macro, 10/11 small; 1868 false-positive tiles | 100% | 288 s |
| tumor_027 | missed its one small lesion; 326 false-positive tiles | 11% | 36 s |
| normal_010 | wrongly called positive; 105 false-positive tiles | 59% | 31 s |

What this says: the low-zoom survey is strong (held-out tile AUC 0.97 to 0.998) and finds big lesions fast.
Small lesions need budget spent on sweeping, which is exactly the decision the LLM agent should make.
The stand-in classifier fires on sinus histiocytes and pink stroma; the PCam model should fix most of that.

## 1. Train the real classifier on your RTX 4060

```bash
git clone <repo> && cd ai-pathologist
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements-train.txt
# PCam from https://github.com/basveeling/pcam (Zenodo record 2546921): the 6 files
#   camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5.gz   then gunzip them into ~/pcam
python -m train.train_pcam --pcam ~/pcam --arch resnet18 --pretrained --epochs 5 --batch 256 --workers 6 \
       --out models/tile_classifier.onnx
```

Send back `models/tile_classifier.onnx` and `models/tile_classifier.json` (drop them in the project files).
If it misfires on whole slides, fine-tune on real slide tiles:
`python -m train.train_pcam --npz data/c16_tiles.npz data/c16_hardneg.npz ...` (make them with `train/make_tiles.py`).

## 2. Run the LLM agent (needs your AWS account)

Enable Claude in Amazon Bedrock (us-west-2), then:

```bash
aws configure            # or use an instance role
python -m scripts.run_slide --slide tumor_026 --mode agent --backend bedrock --budget 40 \
       --chat "check the lower left again"
```

## 3. Deploy on Graviton

```bash
EMAIL=you@example.com LIMIT=50 ./deploy/budget_alarm.sh
REPO_URL=https://github.com/<you>/ai-pathologist.git ./deploy/launch_graviton.sh
```

## Built since (Oct 12, evening)

- Web viewer at `/` of the API: deep zoom slide, tumour and survey heatmaps, numbered search path with replay,
  findings you can click to fly to, decision log with snapshots, and pathologist chat (rule-based when no LLM).
  Try it: `PATHAGENT_LLM=none uvicorn service.app:app --port 8000` then open http://localhost:8000.
  Screenshot: `docs/viewer.png`.
- Tile cache + evaluation suite: `scripts/build_cache.py` scans each slide once; `scripts/eval_suite.py` replays
  every strategy and budget on the cache and writes tables and the efficiency curve to `eval/results/<tag>/`.
- Failure gallery: `scripts/failure_gallery.py`.
- Drafts: `docs/technical_report.md`, `docs/video_script.md`.

When your PCam model arrives, the whole evaluation reruns with:

```bash
python -m scripts.build_cache --slides $(cat eval/test_subset.txt) --cache eval/cache/resnet18
python -m scripts.eval_suite --cache eval/cache/resnet18 --tag resnet18 --modes sliding_window best_first agent
python -m scripts.failure_gallery --tag resnet18 --cache eval/cache/resnet18
```

## Still to build

- Graviton vs x86 benchmark numbers (needs your AWS account: `python -m scripts.benchmark` on each).
- Final numbers in the report once the PCam model and the agent runs exist.
