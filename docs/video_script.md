# Demo video script (4:30)

Record the screen of the viewer at 1440 x 900. Keep the disclaimer visible in every live shot.

## 0:00 to 0:30 · The problem
**Screen:** tumor_026 at full zoom on a single cell, then a slow zoom out until the whole slide fits.
**Say:** "This is one lymph node slide from a breast cancer patient. It is two hundred thousand pixels tall.
A pathologist has to check all of it for cancer cells that have spread, and the small spots are easy to miss.
We built an agent that reads the slide the way a pathologist does, and shows its work."

## 0:30 to 2:30 · Live run
1. **0:30** Pick tumor_026, mode *LLM agent*, budget 40, press Start. Turn on *Low-zoom survey*.
   "First it surveys the slide at low zoom. OpenCV finds the tissue and scores every patch from colour and texture."
2. **0:50** Survey off, heatmap on. The first looks appear.
   "Now it zooms in only where it's suspicious. Each box is one look at high magnification: an ONNX model running
   in OpenCV's DNN module classifies 64 tiles."
3. **1:15** Read two of the agent's notes from the decision log aloud. Click a snapshot thumbnail.
   "Here it found tumour and followed it to its edge, then decided it had seen enough of this one and moved on."
4. **1:45** A refine or flag appears. "This one was borderline, so it re-read it with stain normalisation and
   eight rotations. Still unsure, so it flagged it for the pathologist instead of guessing."
5. **2:05** Report appears. Click Lesion 1 to fly to it. "Likely metastasis, four lesions, sizes measured with
   OpenCV contours, and it looked at only N percent of the tissue."
6. **2:15** Type *check the lower left again*. "The pathologist can steer it." Show it zooming there.

## 2:30 to 3:30 · Does it work?
**Screen:** `efficiency_curve.png`, then the summary table, then one gallery image.
**Say:** "On 40 held-out test slides, the agent finds 58 percent of metastases while viewing 29 percent of the tissue (stand-in numbers; update with the final model).
A raster scan has found 35 percent at that point but raises far more false alarms. Here's an honest failure: a 1.5 mm metastasis
that the classifier would have caught, but the search never looked there."

## 3:30 to 4:15 · How it's built
**Screen:** README architecture diagram, then a terminal with `scripts/benchmark.py` on Graviton.
**Say:** "The slides never leave Amazon S3; we read only the tiles we need. The vision service runs OpenCV 5 on a
Graviton instance in the same region, Claude on Bedrock makes the strategy calls, and the budget and safety rules
live in code, so the model can't argue past them. On Graviton one slide costs about $A and takes B seconds."

## 4:15 to 4:30 · Limits
**Say:** "It's a research prototype and a second reader. It was trained on one dataset from two hospitals, small
metastases are still its weak spot, and a pathologist signs off on every slide."
