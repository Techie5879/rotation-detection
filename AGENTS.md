# Rotation Detection Project Agent Guide

## Project Intent
- Build a robust PDF page rotation detection pipeline that does **not** rely on PDF rotation metadata (`/Rotate`) and does **not** use Tesseract OSD.
- Target production behavior where many PDFs have missing, incorrect, or unusable rotation metadata.
- Prioritize reliable detection of 0/90/180/270 degree orientation, including difficult 180-degree cases.

## Hard Constraints
- No metadata cheating:
  - Do not use PDF page rotation tags (`/Rotate`) as orientation truth.
  - Dataset generation must strip/reset page rotation metadata to neutral values.
- No Tesseract OSD for orientation detection.
- Assume page rotations are cardinal (0, 90, 180, 270) unless explicitly testing otherwise.
- Keep synthetic data generation and evaluation reproducible (fixed seeds, saved manifests).

## Why OCRmyPDF + Tesseract OSD Fail Here
- OCRmyPDF autorotation is driven by OCR engine orientation detection (typically Tesseract OSD), so it inherits OSD failure modes.
- Tesseract OSD can be weak on 180-degree cases, especially with symmetric layouts, sparse text, mixed scripts, or low text density.
- OSD couples script/orientation inference; wrong script assumptions can reduce orientation confidence or produce wrong angle.
- OCRmyPDF uses a confidence gate for rotation, so low-confidence pages are left unchanged even when actually misoriented.
- OCRmyPDF reads and carries page `/Rotate` state for rendering/grafting bookkeeping; in production PDFs with missing or bad metadata, this is not reliable as a truth source.

## Current Environment (captured on 2026-02-07)
- Host OS: macOS 26.2 (Build 25C56)
- Kernel: Darwin 25.2.0
- Architecture: arm64
- CPU: Apple M4 Max
- Memory: 68719476736 bytes (64 GB)
- GPU: Apple M4 Max, 40 cores
- Metal: Metal 4 supported
- Active Apple developer tools path: `/Library/Developer/CommandLineTools`
- Full Xcode app: not active/installed for this shell (`xcodebuild` unavailable under current developer dir)

## Python/Tooling Baseline
- Package manager/runtime workflow: `uv` only
- `uv` version: 0.9.0
- Project Python requirement: `>=3.13` (`pyproject.toml`)
- Active environment after first `uv run`: `.venv/` created with CPython 3.13.8
- Currently installed project packages: none

## GPU ML Package Install Policy
- For GPU-accelerated ML packages (for example: `torch`, `torchvision`, `torchaudio`, `mlx`), install with `uv pip install ...`.
- Do not use `uv add` for these packages. We have seen dependency-resolution and wheel-compatibility issues when trying to manage these through `pyproject.toml`.
- It is acceptable that these installs do not update `pyproject.toml` or `uv.lock`.
- After any ML package install/update, manually refresh `requirements.txt` so the environment is reproducible:
  - `uv pip freeze > requirements.txt`

## ML Acceleration Status
- Apple Metal is available on this machine.
- MLX package status: not installed (`mlx_installed=False`)
- PyTorch package status: not installed (`torch_installed=False`)
- MPS runtime note:
  - MPS backend can be used only after installing a framework that exposes it (typically PyTorch).
  - As of now, Metal hardware is present but no Python ML framework is installed in this env.

## PDF/Test Data Inventory
- Root test directory: `test_pdfs/`
- Current PDF files discovered:
  - `test_pdfs/6c199e86-deaf-48f0-ab1a-c21b10a17c3e/6c199e86-deaf-48f0-ab1a-c21b10a17c3e.pdf`
  - `test_pdfs/EW-2-22626634/EW-2-22626634.pdf`
  - `test_pdfs/EW-2-22696801/EW-2-22696801.pdf`
  - `test_pdfs/EW-2-22697100/EW-2-22697100.pdf`
  - `test_pdfs/EW-2-22697155/EW-2-22697155.pdf`
  - `test_pdfs/EW-2-22697411/EW-2-22697411.pdf`
  - `test_pdfs/EW-2-22714892/EW-2-22714892.pdf`
  - `test_pdfs/one-leidos/BI-Serrano (medical records).pdf`
  - `test_pdfs/one-leidos-small/BI-Serrano (medical records)_p0001-0060_p0800-0890.pdf`

## Planned Workstreams (no implementation yet)
- Dataset maker:
  - Randomly rotate selected pages by 0/90/180/270.
  - Strip/reset rotation metadata in outputs.
  - Save labels/manifests with per-page ground truth rotations.
- Rotation detector:
  - Metadata-agnostic and Tesseract-OSD-free orientation inference.
  - Strong handling of 180-degree ambiguity.
- Evaluator:
  - Compare predicted per-page rotations vs synthetic ground truth.
  - Report accuracy by angle class and document.

## Execution Rules for Agents
- Do not introduce orientation shortcuts from PDF metadata.
- Keep all experiments deterministic when possible.
- Prefer scripts that can be rerun end-to-end.
- Record assumptions and failure modes explicitly.

A possible helpful resource: https://yousry.medium.com/correcting-image-orientation-using-convolutional-neural-networks-bf0f7be3a762