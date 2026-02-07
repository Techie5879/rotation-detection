# Rotation Detection Pipeline (Metadata-Agnostic, No Tesseract OSD)

This is an attempt at vibe prompting, vibe coding, vibe training and vibe fixing this. I have barely read code of this repo - have just skimmed through AGENTS.md and some of the train scripts a little bit. Maybe "vibe" isn't exactly right because this involved hella hella prompting to get this almost right the first try. This isn't one shot but is damn close lmao.
Most of the runs/tests are also autonomous. Maybe this works? We shall see.

> "Just accept AI Love into your heart and die from the greatest increase of productivity ever."
> 
> — Primeagen, 2026



This repo now contains an end-to-end PDF orientation pipeline that avoids both orientation shortcuts:

- No `/Rotate` metadata truth assumptions.
- No Tesseract OSD usage.

## What It Includes

- Synthetic dataset maker that:
  - Renders each source page to an image.
  - Applies random page-wise rotation from `{0, 90, 180, 270}`.
  - Re-wraps rotated page images into a fresh PDF.
  - Verifies output pages have neutral `/Rotate` metadata.
  - Saves explicit split folders: `train/`, `val/`, `test/` with `pages/`, `pdfs/`, `labels.jsonl`, `manifest.json`.
  - Saves root `manifest.json` and `labels.all.jsonl` index.
  - Uses multiprocessing with `spawn` and up to 12 workers.
- Orientation detectors:
  - `heuristic`: classical CV baseline (fast, weaker on hard 180 cases).
  - `torch`: trainable 4-class CNN (recommended for robust 180 handling).
- Evaluator:
  - Compares predictions vs synthetic labels.
  - Outputs overall accuracy, per-angle accuracy, and confusion matrix.
- Experiment runner:
  - Runs full benchmark: dataset -> heuristic detect/eval -> torch train/detect/eval.

## Install

Dependencies were installed with `uv pip` and frozen into `requirements.txt`.

If you need to recreate the environment:

```bash
uv pip install -r requirements.txt
```

## Quick Start

Generate a dataset from `test_pdfs/`:

```bash
uv run main.py make-dataset \
  --input-dir test_pdfs \
  --output-root datasets \
  --dataset-name run_seed42 \
  --seed 42 \
  --angles 0 90 180 270 \
  --rotate-probability 0.7 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --class-balance uniform \
  --min-val-docs 8 \
  --min-test-docs 8 \
  --log-every-pages 200
```

Dataset layout:

```text
datasets/run_seed42/
  manifest.json
  labels.all.jsonl
  train/
    manifest.json
    labels.jsonl
    pages/
    pdfs/
  val/
    manifest.json
    labels.jsonl
    pages/
    pdfs/
  test/
    manifest.json
    labels.jsonl
    pages/
    pdfs/
```

Run heuristic detection on the test split:

```bash
uv run main.py detect \
  --method heuristic \
  --manifest-path datasets/run_seed42/test/manifest.json \
  --log-every-pages 200 \
  --output-jsonl datasets/run_seed42/predictions.heuristic.jsonl
```

Evaluate heuristic predictions:

```bash
uv run main.py evaluate \
  --manifest-path datasets/run_seed42/test/manifest.json \
  --predictions-path datasets/run_seed42/predictions.heuristic.jsonl \
  --output-json datasets/run_seed42/report.heuristic.json
```

Train torch detector from explicit train/val splits:

```bash
uv run main.py train \
  --manifest-path datasets/run_seed42/manifest.json \
  --checkpoint-path models/orientation_cnn.pt \
  --epochs 10 \
  --batch-size 256 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.0001 \
  --log-every-batches 50 \
  --device auto
```

Run torch detection and evaluate:

```bash
uv run main.py detect \
  --method torch \
  --manifest-path datasets/run_seed42/test/manifest.json \
  --checkpoint-path models/orientation_cnn.pt \
  --output-jsonl datasets/run_seed42/predictions.torch.jsonl

uv run main.py evaluate \
  --manifest-path datasets/run_seed42/test/manifest.json \
  --predictions-path datasets/run_seed42/predictions.torch.jsonl \
  --output-json datasets/run_seed42/report.torch.json
```

One-shot experiment run:

```bash
uv run main.py run-experiment \
  --input-dir test_pdfs \
  --output-root runs \
  --seed 42 \
  --train-epochs 10 \
  --train-batch-size 96 \
  --log-every-pages 200 \
  --log-every-batches 20 \
  --skip-heuristic
```

Incremental scale-up (recommended for long runs):

```bash
uv run main.py run-curriculum \
  --input-dir test_pdfs \
  --output-root runs \
  --curriculum-name scaleup_v1 \
  --stage-max-pages-per-doc 600 1000 2000 0 \
  --train-epochs 5 \
  --train-batch-size 96 \
  --log-every-pages 200 \
  --log-every-batches 20 \
  --skip-heuristic \
  --stop-on-regression \
  --regression-tolerance 0.02
```

Ingest a subset of GCS zip files (uses `gcloud storage cp`), then extract PDFs:

```bash
uv run python scripts/ingest_gcs_zips.py \
  --gcs-prefix gs://ew-ny-dump/ew-ny-dump \
  --dest-root test_pdfs/gcs-ew-ny-dump \
  --sample-size 20 \
  --seed 42 \
  --max-zip-mb 250
```

## Final Run

Use this copy/paste flow for the final dataset + training + test-only evaluation run.

Set reusable paths first:

```bash
RUN_ROOT="runs"
RUN_NAME="final_run_dataset"
DATASET_DIR="${RUN_ROOT}/${RUN_NAME}"
MODEL_PATH="${DATASET_DIR}/models/orientation_cnn.pt"
```

1) Make dataset (uniform class balance + guarded val/test splits):

```bash
uv run main.py make-dataset \
  --input-dir test_pdfs \
  --output-root "${RUN_ROOT}" \
  --dataset-name "${RUN_NAME}" \
  --seed 42 \
  --dpi 96 \
  --rotate-probability 0.7 \
  --angles 0 90 180 270 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --class-balance uniform \
  --min-val-docs 8 \
  --min-test-docs 8 \
  --log-every-pages 100
```

2) Train model (early stopping enabled, using latest successful LR=0.0006):

```bash
uv run main.py train \
  --manifest-path "${DATASET_DIR}/manifest.json" \
  --checkpoint-path "${MODEL_PATH}" \
  --epochs 10 \
  --batch-size 256 \
  --learning-rate 0.0006 \
  --weight-decay 0.0001 \
  --image-size 256 \
  --seed 42 \
  --num-workers 0 \
  --device auto \
  --log-every-batches 5 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.0001
```

3) Evaluate saved model on test split only (no training in this step):

```bash
uv run python scripts/test_saved_model.py \
  --checkpoint-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_DIR}" \
  --split test \
  --batch-size 256 \
  --device auto \
  --num-workers 0 \
  --log-every-batches 5 \
  --output-dir "${DATASET_DIR}/saved_model_eval_test"
```

Expected key outputs:

- Dataset log: `${DATASET_DIR}/dataset.log`
- Training log: `${MODEL_PATH}.train.log`
- Checkpoint: `${MODEL_PATH}`
- Test eval report: `${DATASET_DIR}/saved_model_eval_test/report.json`
- Test predictions: `${DATASET_DIR}/saved_model_eval_test/predictions.jsonl`

## Notes

- `rotation_deg` in labels means clockwise rotation applied during synthetic generation.
- A detector predicts the same `rotation_deg` class (`0/90/180/270`).
- To correct a predicted page back to upright, rotate it counterclockwise by the predicted angle.
- Commands now use tee-style logging to both terminal and files:
  - `make-dataset`: `<dataset_dir>/dataset.log`
  - `run-experiment`: `<experiment_dir>/run.log`
  - `run-curriculum`: `<curriculum_dir>/curriculum.log` and per-stage `<stage_dir>/run.log`
  - `train`: `<checkpoint>.train.log`
  - `detect`: `<predictions>.log`
  - `evaluate`: `<report>.eval.log`
- Training/testing/curriculum now include memory profiling logs (`rss`, `ru_maxrss`, Python heap, and torch device memory when available).
