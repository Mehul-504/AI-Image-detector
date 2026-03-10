# AI Detector MVP Scaffold

This repository now contains a production-oriented scaffold for an authenticity-risk
pipeline covering image and video, with image-first transformer integration.

## What is implemented

- Category score schema (`risk`, `confidence`) for each signal.
- Image/video weight profiles aligned with the architecture discussion.
- Fusion engine:
  - confidence-aware weighted aggregation
  - missing-signal weight redistribution
  - hard overrides for provenance/watermark
  - 3-class matrix verdict
- Layered pipeline with detector interfaces for:
  - deterministic/classical forensic signals (including PRNU)
  - CLIP image layer (real-vs-AI prompt scoring)
  - forensic transformer image layer (checkpoint-configurable)
  - placeholder video transformer layers (until video models are plugged in)
- FastAPI endpoint to run analysis.
- Local web UI route (`/`) for browser-based image testing.
- Persistent run logs for every CLI/API/web request.
- Dataset curation pipeline (dedupe + corruption filtering + compression augmentation).
- Unit tests for fusion logic.

## Project layout

- `src/aidetector/schemas.py`: contracts and enums
- `src/aidetector/config.py`: weights and thresholds
- `src/aidetector/fusion.py`: fusion + decision matrix
- `src/aidetector/detectors/provenance_layer.py`: provenance branch layer
- `src/aidetector/detectors/watermark_layer.py`: watermark branch layer
- `src/aidetector/detectors/metadata_layer.py`: metadata branch layer
- `src/aidetector/detectors/prnu_layer.py`: PRNU branch layer
- `src/aidetector/detectors/frequency_layer.py`: frequency branch layer
- `src/aidetector/detectors/spatial_layer.py`: spatial branch layer
- `src/aidetector/detectors/tamper_localization_layer.py`: tamper branch layer
- `src/aidetector/detectors/temporal_consistency_layer.py`: temporal branch layer (video)
- `src/aidetector/detectors/image_clip_layer.py`: CLIP transformer layer (image)
- `src/aidetector/detectors/image_forensic_transformer_layer.py`: forensic transformer layer (image)
- `src/aidetector/detectors/video_clip_layer.py`: CLIP placeholder layer (video)
- `src/aidetector/detectors/video_forensic_transformer_layer.py`: forensic placeholder layer (video)
- `src/aidetector/pipeline.py`: orchestration
- `src/aidetector/api.py`: HTTP API
- `tests/test_fusion.py`: fusion behavior tests

## Quick start (core, no external dependencies)

```bash
python3 -m unittest discover -s tests -v
```

Run analyzer from JSON payload:

```bash
PYTHONPATH=src python3 -m aidetector.cli --input payload.json
```

Sample payload file:

- `payload.image.sample.json`

## Image mode on local Mac (M4/MPS)

Install image dependencies in your local environment:

```bash
pip install -e ".[image]"
```

`[image]` now includes:

- `numpy` (FFT/DFT and forensic math)
- `opencv-python-headless` (video frame sampling/temporal analysis)
- `torch`, `transformers`, `Pillow` (model inference + image IO)

Optional env vars:

```bash
export AIDETECTOR_DEVICE=auto
export AIDETECTOR_CLIP_MODEL_ID=openai/clip-vit-base-patch32
# optional override (default model id is used if not set)
export AIDETECTOR_IMAGE_FORENSIC_MODEL=dima806/ai_vs_real_image_detection
# optional: force transformers to local cache/path only (offline mode)
export AIDETECTOR_HF_LOCAL_ONLY=1
```

Notes:

- `AIDETECTOR_DEVICE=auto` picks `mps` on Apple Silicon when available.
- CLIP works with the default model id.
- Forensic transformer defaults to `dima806/ai_vs_real_image_detection`.
- You can override forensic model with `AIDETECTOR_IMAGE_FORENSIC_MODEL` (local path or HF id).
- Video forensic defaults to the same model unless `AIDETECTOR_VIDEO_FORENSIC_MODEL` is set.
- Model loading now prefers:
  1) explicit local path,
  2) local Hugging Face cache snapshot,
  3) remote model id.
- If dependencies/checkpoint are missing, the pipeline returns neutral scores with rationale instead of failing.

## Detector Logic Coverage

Implemented branch logic (no pass-through placeholders in normal media runs):

- `provenance_layer.py`: embedded/sidecar provenance marker parsing + declaration heuristics
- `watermark_layer.py`: watermark metadata markers + periodic spectral watermark heuristics
- `metadata_layer.py`: EXIF/metadata heuristics
- `signal_utils.py`: shared FFT/DFT/ELA utilities + transport compression signals
- `prnu_layer.py`: residual stationarity + edge-correlation + spectral periodicity
- `frequency_layer.py`: FFT/DFT radial features + block artifacts
- `spatial_layer.py`: edge/smoothness/clipping/channel-correlation forensics
- `tamper_localization_layer.py`: ELA-based localization anomaly scoring
- `temporal_consistency_layer.py`: frame-difference DFT temporal consistency scoring
- `image_clip_layer.py`: CLIP semantic anomaly scoring
- `image_forensic_transformer_layer.py`: multicrop forensic classifier + metadata/consensus/compression calibration
- `video_clip_layer.py`: keyframe CLIP aggregation
- `video_forensic_transformer_layer.py`: keyframe forensic aggregation + audio-visual proxy score

## Optional API mode

If `fastapi` and `uvicorn` are installed:

1. Start API:
   `uvicorn aidetector.api:app --reload`
2. Call endpoint:
   `POST /v1/analyze` with the same payload shape as below.
3. Open local web app:
   `http://127.0.0.1:8000/`

## Local Web App

Install optional dependencies:

```bash
pip install -e ".[api,image]"
```

Run:

```bash
uvicorn aidetector.api:app --reload
```

Then open:

- `http://127.0.0.1:8000/` (web UI)
- `http://127.0.0.1:8000/v1/logs/recent?limit=20` (recent run summaries)

Web UI behavior:

- Upload an image file.
- Click **Analyze Image**.
- Backend stores upload in `logs/uploads/`, runs analysis, and logs result in `logs/runs/`.

## Run logs

Every run from CLI/API/Web is stored to:

- `logs/runs/<run_id>.json` full record (payload/result/error)
- `logs/runs_index.jsonl` append-only run summary index
- `logs/uploads/` reserved for uploaded media artifacts

The CLI/API response now includes:

- `run_id`
- `log_file`

## Dataset Evaluation

Evaluate on local labeled folders:

- `dataset/real/*`
- `dataset/ai_generated/*`
- `dataset/ai_edited/*`

Run:

```bash
PYTHONPATH=src python -m aidetector.eval_dataset --dataset-root dataset
```

Optional JSON output:

```bash
PYTHONPATH=src python -m aidetector.eval_dataset --dataset-root dataset --out logs/evals/latest.json
```

Train learned fusion model from labeled folders:

```bash
PYTHONPATH=src python -m aidetector.train_fusion --dataset-root dataset --out models/fusion_model.json
```

Curate and expand dataset before training (recommended):

```bash
PYTHONPATH=src python -m aidetector.curate_dataset \
  --dataset-root dataset \
  --output-root dataset_curated \
  --report logs/dataset/curation_report.json
```

Then train on curated data:

```bash
PYTHONPATH=src python -m aidetector.train_fusion --dataset-root dataset_curated --out models/fusion_model.json
```

One-command cycle (curate + train + balanced/strict eval):

```bash
PYTHONPATH=src python -m aidetector.run_training_cycle --dataset-root dataset --curated-root dataset_curated --model-out models/fusion_model.json
```

Enable learned fusion during inference:

```bash
export AIDETECTOR_FUSION_MODEL_PATH=/absolute/path/to/models/fusion_model.json
```

Optional mode profile (balanced/strict):

```bash
export AIDETECTOR_MODE_PROFILE=balanced
# or strict
```

## Payload example

```json
{
  "media_type": "image",
  "media_uri": "/absolute/path/to/image.jpg",
  "provided_scores": {
    "provenance": {"risk": 10, "confidence": 90},
    "metadata": {"risk": 18, "confidence": 72}
  }
}
```

`provided_scores` are optional. If you omit `clip_anomaly` and `forensic_transformer`,
the image detectors will infer them from `media_uri`.
