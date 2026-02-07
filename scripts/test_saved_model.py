"""Simple evaluator for a saved torch orientation model.

This script loads a trained checkpoint, reads a split labels file from a generated
dataset directory, runs batched inference, prints live progress logs, and writes
predictions + metrics to disk.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from rotation_detection.constants import CARDINAL_ANGLES, INDEX_TO_ANGLE
from rotation_detection.detectors.torch_detector import load_trained_model
from rotation_detection.logging_utils import tee_output
from rotation_detection.memory_profile import format_memory, log_memory, snapshot_memory
from rotation_detection.manifest import read_jsonl, write_jsonl
from rotation_detection.pdf_ops import rotate_metadata_violations
from rotation_detection.utils import dump_json, load_json, utc_timestamp


def _resolve_split_dir(dataset_path: Path, split: str) -> Path:
    if (dataset_path / "labels.jsonl").exists() and (dataset_path / "manifest.json").exists():
        return dataset_path
    split_dir = dataset_path / split
    if (split_dir / "labels.jsonl").exists() and (split_dir / "manifest.json").exists():
        return split_dir
    raise RuntimeError(
        f"Could not resolve split labels from {dataset_path}. Expected either "
        f"{dataset_path}/labels.jsonl or {dataset_path}/{split}/labels.jsonl"
    )


def _resolve_pdf_path(split_dir: Path, dataset_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    split_candidate = split_dir / candidate
    if split_candidate.exists():
        return split_candidate
    root_candidate = dataset_root / candidate
    if root_candidate.exists():
        return root_candidate
    return split_candidate


def _collect_split_pdfs(split_dir: Path, dataset_root: Path, rows: list[dict]) -> list[Path]:
    manifest_path = split_dir / "manifest.json"
    pdfs: list[Path] = []
    seen: set[Path] = set()

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        for doc in manifest.get("documents", []):
            if "output_pdf" not in doc:
                continue
            resolved = _resolve_pdf_path(split_dir, dataset_root, str(doc["output_pdf"]))
            if resolved not in seen:
                seen.add(resolved)
                pdfs.append(resolved)
        if pdfs:
            return pdfs

    for row in rows:
        if "output_pdf" not in row:
            continue
        resolved = _resolve_pdf_path(split_dir, dataset_root, str(row["output_pdf"]))
        if resolved not in seen:
            seen.add(resolved)
            pdfs.append(resolved)
    return pdfs


def _assert_neutral_rotation_metadata(split_dir: Path, dataset_root: Path, rows: list[dict]) -> dict:
    pdf_paths = _collect_split_pdfs(split_dir, dataset_root, rows)
    if not pdf_paths:
        raise RuntimeError("Metadata check failed: no PDF files were discovered for this split.")

    checked = 0
    violation_count = 0
    violation_examples: list[dict] = []

    progress = tqdm(
        pdf_paths,
        total=len(pdf_paths),
        desc="[saved-model-test] metadata",
        unit="pdf",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for pdf_path in progress:
        checked += 1
        violations = rotate_metadata_violations(pdf_path)
        if violations:
            violation_count += len(violations)
            for row in violations[:10]:
                violation_examples.append(
                    {
                        "pdf_path": str(pdf_path),
                        "page_index": int(row["page_index"]),
                        "rotate": int(row["rotate"]),
                    }
                )
        progress.set_postfix(violations=violation_count)
    progress.close()

    result = {
        "checked_pdfs": checked,
        "violating_pages": violation_count,
        "violation_examples": violation_examples[:50],
    }
    if violation_count > 0:
        raise RuntimeError(
            "Metadata check failed: non-zero /Rotate entries detected. "
            f"violating_pages={violation_count} examples={result['violation_examples'][:5]}"
        )
    return result


def _normalize_image(image: Image.Image, image_size: int, mean: tuple[float, float, float], std: tuple[float, float, float]):
    import numpy as np

    resized = image.convert("RGB").resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


class PageDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        split_dir: Path,
        dataset_root: Path,
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ):
        self.rows = rows
        self.split_dir = split_dir
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        raw_path = Path(str(row["image_path"]))
        if raw_path.is_absolute():
            image_path = raw_path
        else:
            candidate_split = self.split_dir / raw_path
            candidate_root = self.dataset_root / raw_path
            if candidate_split.exists():
                image_path = candidate_split
            elif candidate_root.exists():
                image_path = candidate_root
            else:
                image_path = candidate_split
        image = Image.open(image_path).convert("RGB")
        tensor = _normalize_image(image, self.image_size, self.mean, self.std)
        target_angle = int(row["rotation_deg"]) % 360
        target_idx = CARDINAL_ANGLES.index(target_angle)
        return tensor, target_idx, str(row["doc_id"]), int(row["page_index"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved orientation model on a dataset split")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-path", required=True, help="Dataset root (with train/val/test) or split folder")
    parser.add_argument("--split", default="test", help="Split name when dataset-path points to root")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    split_dir = _resolve_split_dir(dataset_path, args.split)
    labels_path = split_dir / "labels.jsonl"

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else split_dir / f"saved_model_eval_{utc_timestamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    with tee_output(log_path):
        print(f"[saved-model-test] checkpoint={checkpoint_path}")
        print(f"[saved-model-test] split_dir={split_dir}")
        print(f"[saved-model-test] labels={labels_path}")
        log_memory("[saved-model-test] start")

        rows = read_jsonl(labels_path)
        if not rows:
            raise RuntimeError(f"No labels found in {labels_path}")

        metadata_check = _assert_neutral_rotation_metadata(
            split_dir=split_dir,
            dataset_root=split_dir.parent,
            rows=rows,
        )
        print(
            f"[saved-model-test] metadata_check passed checked_pdfs={metadata_check['checked_pdfs']}"
        )

        model, payload, device = load_trained_model(checkpoint_path, device=args.device)
        image_size = int(payload.get("image_size", 320))
        mean = tuple(payload.get("mean", (0.5, 0.5, 0.5)))
        std = tuple(payload.get("std", (0.5, 0.5, 0.5)))

        num_workers = args.num_workers
        if num_workers > 0:
            print(
                "[saved-model-test] num_workers>0 requested, but this environment has "
                "worker-shared-memory issues; using num_workers=0"
            )
            num_workers = 0

        ds = PageDataset(
            rows,
            split_dir=split_dir,
            dataset_root=split_dir.parent,
            image_size=image_size,
            mean=mean,
            std=std,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

        confusion = {t: {p: 0 for p in CARDINAL_ANGLES} for t in CARDINAL_ANGLES}
        predictions: list[dict] = []

        total = 0
        correct = 0
        started = time.perf_counter()

        total_batches = len(loader)
        progress = tqdm(
            loader,
            total=total_batches,
            desc="[saved-model-test] batches",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for batch_idx, (images, targets_idx, doc_ids, page_indices) in enumerate(progress, start=1):
            images = images.to(device)
            targets_idx = targets_idx.to(device)

            with torch.no_grad():
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(dim=1)

            batch_size = int(images.shape[0])
            total += batch_size
            correct += int((pred_idx == targets_idx).sum().item())
            running_acc = correct / max(total, 1)
            progress.set_postfix(acc=f"{running_acc:.4f}")

            probs_cpu = probs.detach().cpu()
            pred_cpu = pred_idx.detach().cpu().tolist()
            target_cpu = targets_idx.detach().cpu().tolist()

            for i in range(batch_size):
                true_angle = INDEX_TO_ANGLE[int(target_cpu[i])]
                pred_angle = INDEX_TO_ANGLE[int(pred_cpu[i])]
                confidence = float(probs_cpu[i, int(pred_cpu[i])].item())
                confusion[true_angle][pred_angle] += 1

                predictions.append(
                    {
                        "doc_id": str(doc_ids[i]),
                        "page_index": int(page_indices[i]),
                        "true_rotation_deg": true_angle,
                        "predicted_rotation_deg": pred_angle,
                        "confidence": confidence,
                    }
                )

            if args.log_every_batches > 0 and (
                batch_idx % args.log_every_batches == 0 or batch_idx == total_batches
            ):
                elapsed = time.perf_counter() - started
                rate = total / max(elapsed, 1e-9)
                mem = snapshot_memory(device=device)
                print(
                    f"[saved-model-test] progress batch={batch_idx}/{total_batches} "
                    f"pages={total}/{len(rows)} acc={running_acc:.4f} rate_pps={rate:.2f} "
                    f"{format_memory(mem)}"
                )

        progress.close()

        elapsed = time.perf_counter() - started

        per_angle = {}
        for true_angle in CARDINAL_ANGLES:
            support = sum(confusion[true_angle].values())
            correct_angle = confusion[true_angle][true_angle]
            per_angle[str(true_angle)] = {
                "support": support,
                "correct": correct_angle,
                "accuracy": (correct_angle / support) if support else 0.0,
            }

        report = {
            "checkpoint_path": str(checkpoint_path),
            "dataset_path": str(dataset_path),
            "split_dir": str(split_dir),
            "labels_path": str(labels_path),
            "metadata_check": metadata_check,
            "total_pages": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
            "batch_size": args.batch_size,
            "device": str(device),
            "elapsed_s": elapsed,
            "avg_pages_per_s": total / max(elapsed, 1e-9),
            "per_angle": per_angle,
            "confusion_matrix": {
                str(t): {str(p): int(c) for p, c in row.items()} for t, row in confusion.items()
            },
            "predictions_path": str(output_dir / "predictions.jsonl"),
            "memory_end": snapshot_memory(device=device),
        }

        write_jsonl(predictions, output_dir / "predictions.jsonl")
        dump_json(report, output_dir / "report.json")

        print(f"[saved-model-test] complete accuracy={report['accuracy']:.4f}")
        print(f"[saved-model-test] {format_memory(report['memory_end'])}")
        print(f"[saved-model-test] report={output_dir / 'report.json'}")
        print(f"[saved-model-test] predictions={output_dir / 'predictions.jsonl'}")
        log_memory("[saved-model-test] done", device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
