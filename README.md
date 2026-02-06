# Rotation Detection Pipeline (Metadata-Agnostic, No Tesseract OSD)

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
  --rotate-probability 1.0 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
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
  --batch-size 64 \
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

Prepare dataset:

```bash
uv run main.py make-dataset --input-dir test_pdfs --output-root runs --dataset-name final_run_dataset --seed 42 --dpi 96 --rotate-probability 1.0 --angles 0 90 180 270 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --log-every-pages 200
```

Train model:

```bash
uv run main.py train --manifest-path runs/final_run_dataset/manifest.json --checkpoint-path runs/final_run_dataset/models/orientation_cnn.pt --epochs 10 --batch-size 96 --learning-rate 0.0003 --weight-decay 0.0001 --image-size 256 --seed 42 --num-workers 0 --device auto --log-every-batches 20
```

Evaluate saved model on test split:

```bash
uv run python scripts/test_saved_model.py --checkpoint-path runs/final_run_dataset/models/orientation_cnn.pt --dataset-path runs/final_run_dataset --split test --batch-size 128 --device auto --num-workers 0 --log-every-batches 20 --output-dir runs/final_run_dataset/saved_model_eval_final
```

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
