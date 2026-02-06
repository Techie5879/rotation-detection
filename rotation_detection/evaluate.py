"""Evaluation against synthetic ground truth labels."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import CARDINAL_ANGLES
from .logging_utils import tee_output
from .manifest import read_jsonl
from .utils import dump_json, load_json


def _ground_truth_from_manifest(manifest_path: Path) -> dict[tuple[str, int], int]:
    manifest = load_json(manifest_path)
    gt: dict[tuple[str, int], int] = {}

    for doc in manifest.get("documents", []):
        doc_id = str(doc["doc_id"])
        for page in doc.get("pages", []):
            key = (doc_id, int(page["page_index"]))
            gt[key] = int(page["rotation_deg"]) % 360
    return gt


def _predictions_map(predictions: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    mapping: dict[tuple[str, int], int] = {}
    for row in predictions:
        key = (str(row["doc_id"]), int(row["page_index"]))
        mapping[key] = int(row["predicted_rotation_deg"]) % 360
    return mapping


def run_evaluation(
    *,
    manifest_path: str,
    predictions_path: str,
    output_json: str,
) -> Path:
    """Compare predictions with manifest ground truth and emit a report."""
    manifest_abs = Path(manifest_path).expanduser().resolve()
    preds_abs = Path(predictions_path).expanduser().resolve()
    output_abs = Path(output_json).expanduser().resolve()
    log_path = output_abs.with_suffix(".eval.log")

    with tee_output(log_path):
        print(f"[eval] manifest={manifest_abs}")
        print(f"[eval] predictions={preds_abs}")

        gt = _ground_truth_from_manifest(manifest_abs)
        predictions = _predictions_map(read_jsonl(preds_abs))

        confusion: dict[int, dict[int, int]] = {
            angle: {pred_angle: 0 for pred_angle in CARDINAL_ANGLES} for angle in CARDINAL_ANGLES
        }
        per_angle_support: defaultdict[int, int] = defaultdict(int)
        per_angle_correct: defaultdict[int, int] = defaultdict(int)

        total = len(gt)
        found = 0
        correct = 0
        invalid_predictions = 0

        missing: list[dict[str, Any]] = []
        for key, true_angle in gt.items():
            per_angle_support[true_angle] += 1
            predicted = predictions.get(key)
            if predicted is None:
                missing.append({"doc_id": key[0], "page_index": key[1], "true_rotation_deg": true_angle})
                continue

            found += 1
            if predicted not in CARDINAL_ANGLES:
                invalid_predictions += 1
                continue

            confusion[true_angle][predicted] += 1
            if predicted == true_angle:
                correct += 1
                per_angle_correct[true_angle] += 1

        extra_predictions = [
            {"doc_id": key[0], "page_index": key[1], "predicted_rotation_deg": pred}
            for key, pred in predictions.items()
            if key not in gt
        ]

        per_angle = {}
        for angle in CARDINAL_ANGLES:
            support = per_angle_support[angle]
            angle_correct = per_angle_correct[angle]
            per_angle[str(angle)] = {
                "support": support,
                "correct": angle_correct,
                "accuracy": (angle_correct / support) if support else 0.0,
            }

        report = {
            "manifest_path": str(manifest_abs),
            "predictions_path": str(preds_abs),
            "total_pages": total,
            "predictions_found": found,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
            "accuracy_on_found": (correct / found) if found else 0.0,
            "invalid_predictions": invalid_predictions,
            "missing_predictions": len(missing),
            "extra_predictions": len(extra_predictions),
            "per_angle": per_angle,
            "confusion_matrix": {
                str(true_angle): {str(pred_angle): count for pred_angle, count in row.items()}
                for true_angle, row in confusion.items()
            },
            "missing_examples": missing[:50],
            "extra_examples": extra_predictions[:50],
        }

        dump_json(report, output_abs)
        print(f"Evaluation report: {output_abs}")
        print(f"accuracy={report['accuracy']:.4f} found={found}/{total} correct={correct}")
    return output_abs
