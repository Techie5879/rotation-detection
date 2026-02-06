"""Run orientation detection over PDF pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

from .detectors.heuristic import predict_rotation_heuristic
from .detectors.torch_detector import load_trained_model, predict_rotation_torch
from .logging_utils import tee_output
from .manifest import write_jsonl
from .pdf_ops import iter_rendered_pages
from .utils import discover_input_pdfs, load_json, stable_doc_id


@dataclass(slots=True)
class DetectionTarget:
    doc_id: str
    pdf_path: Path
    expected_pages: int | None = None


def _targets_from_manifest(manifest_path: Path) -> list[DetectionTarget]:
    manifest = load_json(manifest_path)
    dataset_root = manifest_path.parent

    targets: list[DetectionTarget] = []
    for doc in manifest.get("documents", []):
        doc_id = str(doc["doc_id"])
        pdf_rel = Path(str(doc["output_pdf"]))
        expected_pages = int(doc.get("page_count", 0)) if doc.get("page_count") is not None else None
        targets.append(
            DetectionTarget(
                doc_id=doc_id,
                pdf_path=(dataset_root / pdf_rel).resolve(),
                expected_pages=expected_pages,
            )
        )
    return targets


def _targets_from_inputs(input_pdfs: list[str], input_dir: str | None) -> list[DetectionTarget]:
    discovered = discover_input_pdfs(input_pdfs, input_dir)
    targets: list[DetectionTarget] = []
    for item in discovered:
        targets.append(
            DetectionTarget(
                doc_id=stable_doc_id(item.source_key, item.path.stem),
                pdf_path=item.path,
                expected_pages=None,
            )
        )
    return targets


def run_detection(
    *,
    method: str,
    output_jsonl: str,
    dpi: int,
    manifest_path: str | None,
    input_pdfs: list[str],
    input_dir: str | None,
    checkpoint_path: str | None,
    device: str,
    max_pages_per_doc: int | None,
    log_every_pages: int,
) -> Path:
    """Detect orientation for every page of selected PDFs."""
    output_path = Path(output_jsonl).expanduser().resolve()
    log_path = output_path.with_suffix(output_path.suffix + ".log")

    if manifest_path:
        targets = _targets_from_manifest(Path(manifest_path).expanduser().resolve())
    else:
        targets = _targets_from_inputs(input_pdfs, input_dir)

    if not targets:
        raise RuntimeError("No detection targets found.")

    with tee_output(log_path):
        print(f"[detect] output={output_path}")

        predictor: Callable[[Any], dict[str, Any]]
        model_payload: dict[str, Any] | None = None
        model_device = None

        if method == "heuristic":
            predictor = predict_rotation_heuristic
        elif method == "torch":
            if not checkpoint_path:
                raise RuntimeError("--checkpoint-path is required when --method torch is used.")
            model, model_payload, model_device = load_trained_model(
                Path(checkpoint_path).expanduser().resolve(),
                device=device,
            )

            def _torch_predict(image):
                assert model_payload is not None
                assert model_device is not None
                return predict_rotation_torch(image, model=model, device=model_device, payload=model_payload)

            predictor = _torch_predict
        else:
            raise ValueError(f"Unsupported detection method '{method}'.")

        all_records: list[dict[str, Any]] = []
        total_expected_pages = 0
        expected_known = True
        for target in targets:
            if target.expected_pages is None:
                expected_known = False
                continue
            total_expected_pages += (
                min(target.expected_pages, max_pages_per_doc)
                if max_pages_per_doc is not None
                else target.expected_pages
            )

        overall_processed = 0
        started_at = time.perf_counter()

        if expected_known:
            print(f"[detect] start method={method} targets={len(targets)} total_pages={total_expected_pages}")
        else:
            print(f"[detect] start method={method} targets={len(targets)}")

        for target in targets:
            page_count = 0
            doc_started_at = time.perf_counter()
            for page_index, image in iter_rendered_pages(target.pdf_path, dpi=dpi):
                if max_pages_per_doc is not None and page_index >= max_pages_per_doc:
                    break

                prediction = predictor(image)
                record = {
                    "doc_id": target.doc_id,
                    "pdf_path": str(target.pdf_path),
                    "page_index": page_index,
                    "predicted_rotation_deg": int(prediction["predicted_rotation_deg"]),
                    "confidence": float(prediction.get("confidence", 0.0)),
                    "method": method,
                }
                if "probabilities" in prediction:
                    record["probabilities"] = prediction["probabilities"]
                if "scores" in prediction:
                    record["scores"] = prediction["scores"]
                all_records.append(record)
                page_count += 1
                overall_processed += 1

                if log_every_pages > 0 and overall_processed % log_every_pages == 0:
                    elapsed = time.perf_counter() - started_at
                    rate = overall_processed / max(elapsed, 1e-9)
                    if expected_known:
                        print(
                            "[detect] progress "
                            f"method={method} overall={overall_processed}/{total_expected_pages} rate_pps={rate:.2f}"
                        )
                    else:
                        print(
                            "[detect] progress "
                            f"method={method} overall={overall_processed} rate_pps={rate:.2f}"
                        )

            doc_elapsed = time.perf_counter() - doc_started_at
            print(
                f"detected doc_id={target.doc_id} pages={page_count} elapsed_s={doc_elapsed:.1f} pdf={target.pdf_path}"
            )

        write_jsonl(all_records, output_path)
        total_elapsed = time.perf_counter() - started_at
        print(
            f"[detect] completed method={method} pages={overall_processed} elapsed_s={total_elapsed:.1f} "
            f"avg_pps={overall_processed / max(total_elapsed, 1e-9):.2f}"
        )
        print(f"Predictions written: {output_path}")
    return output_path
