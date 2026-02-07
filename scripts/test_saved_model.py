"""Simple evaluator for a saved torch orientation model.

This script loads a trained checkpoint, reads a split labels file from a generated
dataset directory, runs batched inference, prints live progress logs, and writes
predictions + metrics to disk.
"""

from __future__ import annotations

import argparse
import math
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

from rotation_detection.constants import ANGLE_TO_INDEX, CARDINAL_ANGLES, INDEX_TO_ANGLE
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
    parser.add_argument(
        "--postprocess-rotation-tta",
        action="store_true",
        help=(
            "Enable rotation-consistency postprocessing: for low-confidence/low-margin pages, "
            "run model on 0/90/180/270 rotated views and aggregate back to base orientation."
        ),
    )
    parser.add_argument(
        "--postprocess-confidence-threshold",
        type=float,
        default=0.75,
        help="Candidate threshold for postprocessing when confidence is below this value",
    )
    parser.add_argument(
        "--postprocess-margin-threshold",
        type=float,
        default=0.10,
        help="Candidate threshold for postprocessing when (top1 - top2) probability margin is below this value",
    )
    parser.add_argument(
        "--postprocess-max-pages",
        type=int,
        default=0,
        help="Max number of pages to postprocess (0 means no limit)",
    )
    return parser


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return float(values[0])
    if q >= 1.0:
        return float(values[-1])
    idx = (len(values) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(values[lo])
    frac = idx - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        }
    sorted_values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return {
        "mean": float(mean),
        "std": float(math.sqrt(var)),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "p05": float(_percentile(sorted_values, 0.05)),
        "p50": float(_percentile(sorted_values, 0.50)),
        "p95": float(_percentile(sorted_values, 0.95)),
    }


def _calibration_stats(
    confidences: list[float],
    correct_flags: list[int],
    *,
    bins: int = 15,
) -> dict[str, object]:
    if not confidences:
        return {"ece": 0.0, "mce": 0.0, "bins": bins, "histogram": []}

    total = len(confidences)
    bucket_counts = [0] * bins
    bucket_conf_sum = [0.0] * bins
    bucket_correct_sum = [0] * bins

    for conf, is_correct in zip(confidences, correct_flags):
        idx = min(bins - 1, max(0, int(conf * bins)))
        bucket_counts[idx] += 1
        bucket_conf_sum[idx] += conf
        bucket_correct_sum[idx] += int(is_correct)

    histogram: list[dict[str, float | int]] = []
    ece = 0.0
    mce = 0.0
    for i in range(bins):
        lower = i / bins
        upper = (i + 1) / bins
        count = bucket_counts[i]
        if count > 0:
            avg_conf = bucket_conf_sum[i] / count
            acc = bucket_correct_sum[i] / count
        else:
            avg_conf = 0.0
            acc = 0.0
        gap = abs(acc - avg_conf)
        ece += (count / total) * gap
        mce = max(mce, gap)
        histogram.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(count),
                "avg_confidence": float(avg_conf),
                "accuracy": float(acc),
                "gap": float(gap),
            }
        )

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bins": int(bins),
        "histogram": histogram,
    }


def _rotation_consistency_probs(model, images: torch.Tensor) -> torch.Tensor:
    """Aggregate predictions from rotated views back to base orientation labels."""
    accum = torch.zeros((images.shape[0], len(CARDINAL_ANGLES)), device=images.device, dtype=torch.float32)
    for delta in CARDINAL_ANGLES:
        if delta == 0:
            rotated = images
        else:
            k = -(delta // 90)
            rotated = torch.rot90(images, k=k, dims=(2, 3))
        probs_delta = torch.softmax(model(rotated), dim=1)
        mapped_indices = [ANGLE_TO_INDEX[(angle + delta) % 360] for angle in CARDINAL_ANGLES]
        accum += probs_delta[:, mapped_indices]
    return accum / float(len(CARDINAL_ANGLES))


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
        print(
            "[saved-model-test] postprocess "
            f"rotation_tta={bool(args.postprocess_rotation_tta)} "
            f"conf_thr={float(args.postprocess_confidence_threshold):.3f} "
            f"margin_thr={float(args.postprocess_margin_threshold):.3f} "
            f"max_pages={int(args.postprocess_max_pages)}"
        )
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
        base_correct = 0
        nll_sum = 0.0
        brier_sum = 0.0
        confidences: list[float] = []
        correct_flags: list[int] = []
        true_logprobs: list[float] = []
        pred_logprobs: list[float] = []

        postprocess_enabled = bool(args.postprocess_rotation_tta)
        postprocess_conf_thr = float(args.postprocess_confidence_threshold)
        postprocess_margin_thr = float(args.postprocess_margin_threshold)
        postprocess_budget = int(args.postprocess_max_pages)
        postprocess_remaining = postprocess_budget if postprocess_budget > 0 else None
        postprocess_candidates = 0
        postprocess_applied = 0
        postprocess_changed = 0
        postprocess_skipped_budget = 0
        postprocess_fixed = 0
        postprocess_regressed = 0

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
                log_probs = torch.log_softmax(logits, dim=1)
                pred_idx = logits.argmax(dim=1)

            base_pred_idx = pred_idx.clone()
            base_probs = probs.clone()
            postprocessed_flags = [False] * int(images.shape[0])
            postprocess_changed_flags = [False] * int(images.shape[0])

            if postprocess_enabled:
                with torch.no_grad():
                    top2 = torch.topk(probs, k=2, dim=1).values
                    base_conf = top2[:, 0]
                    base_margin = top2[:, 0] - top2[:, 1]
                    candidate_mask = (base_conf < postprocess_conf_thr) | (base_margin < postprocess_margin_thr)
                    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)

                    if candidate_indices.numel() > 0:
                        postprocess_candidates += int(candidate_indices.numel())
                        if postprocess_remaining is not None:
                            allow = max(0, min(int(candidate_indices.numel()), int(postprocess_remaining)))
                            if allow < int(candidate_indices.numel()):
                                postprocess_skipped_budget += int(candidate_indices.numel()) - allow
                            if allow == 0:
                                candidate_indices = candidate_indices[:0]
                            else:
                                candidate_indices = candidate_indices[:allow]
                                postprocess_remaining -= allow

                    if candidate_indices.numel() > 0:
                        images_pp = images.index_select(0, candidate_indices)
                        refined_probs = _rotation_consistency_probs(model, images_pp)
                        probs.index_copy_(0, candidate_indices, refined_probs)
                        log_probs.index_copy_(0, candidate_indices, torch.log(refined_probs.clamp_min(1e-12)))
                        refined_pred_idx = refined_probs.argmax(dim=1)
                        pred_idx.index_copy_(0, candidate_indices, refined_pred_idx)

                        postprocess_applied += int(candidate_indices.numel())
                        changed_tensor = refined_pred_idx != base_pred_idx.index_select(0, candidate_indices)
                        postprocess_changed += int(changed_tensor.sum().item())

                        for local_i, row_idx in enumerate(candidate_indices.detach().cpu().tolist()):
                            postprocessed_flags[int(row_idx)] = True
                            postprocess_changed_flags[int(row_idx)] = bool(changed_tensor[local_i].item())

            batch_size = int(images.shape[0])
            total += batch_size
            base_correct += int((base_pred_idx == targets_idx).sum().item())
            correct += int((pred_idx == targets_idx).sum().item())
            running_acc = correct / max(total, 1)
            progress.set_postfix(acc=f"{running_acc:.4f}")

            probs_cpu = probs.detach().cpu()
            base_probs_cpu = base_probs.detach().cpu()
            log_probs_cpu = log_probs.detach().cpu()
            pred_cpu = pred_idx.detach().cpu().tolist()
            target_cpu = targets_idx.detach().cpu().tolist()

            for i in range(batch_size):
                true_idx = int(target_cpu[i])
                pred_cls_idx = int(pred_cpu[i])
                true_angle = INDEX_TO_ANGLE[true_idx]
                pred_angle = INDEX_TO_ANGLE[pred_cls_idx]
                base_angle = INDEX_TO_ANGLE[int(base_pred_idx[i].item())]

                prob_row = probs_cpu[i]
                log_prob_row = log_probs_cpu[i]
                confidence = float(prob_row[pred_cls_idx].item())
                true_logprob = float(log_prob_row[true_idx].item())
                pred_logprob = float(log_prob_row[pred_cls_idx].item())

                one_hot = torch.zeros_like(prob_row)
                one_hot[true_idx] = 1.0
                brier = float(torch.mean((prob_row - one_hot) ** 2).item())
                is_correct = int(pred_cls_idx == true_idx)
                base_is_correct = int(int(base_pred_idx[i].item()) == true_idx)

                if postprocessed_flags[i]:
                    if base_is_correct == 0 and is_correct == 1:
                        postprocess_fixed += 1
                    elif base_is_correct == 1 and is_correct == 0:
                        postprocess_regressed += 1

                nll_sum += -true_logprob
                brier_sum += brier
                confidences.append(confidence)
                correct_flags.append(is_correct)
                true_logprobs.append(true_logprob)
                pred_logprobs.append(pred_logprob)

                confusion[true_angle][pred_angle] += 1

                prob_map = {
                    str(angle): float(prob_row[ANGLE_TO_INDEX[angle]].item()) for angle in CARDINAL_ANGLES
                }
                logprob_map = {
                    str(angle): float(log_prob_row[ANGLE_TO_INDEX[angle]].item())
                    for angle in CARDINAL_ANGLES
                }

                predictions.append(
                    {
                        "doc_id": str(doc_ids[i]),
                        "page_index": int(page_indices[i]),
                        "true_rotation_deg": true_angle,
                        "base_predicted_rotation_deg": base_angle,
                        "predicted_rotation_deg": pred_angle,
                        "confidence": confidence,
                        "base_confidence": float(base_probs_cpu[i, int(base_pred_idx[i].item())].item()),
                        "logprob_true": true_logprob,
                        "logprob_pred": pred_logprob,
                        "logprobs": logprob_map,
                        "probabilities": prob_map,
                        "postprocessed": bool(postprocessed_flags[i]),
                        "postprocess_changed": bool(postprocess_changed_flags[i]),
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

        calibration = _calibration_stats(confidences, correct_flags, bins=15)
        nll_mean = nll_sum / max(total, 1)
        brier_mean = brier_sum / max(total, 1)
        accuracy_before_postprocess = base_correct / max(total, 1)

        report = {
            "checkpoint_path": str(checkpoint_path),
            "dataset_path": str(dataset_path),
            "split_dir": str(split_dir),
            "labels_path": str(labels_path),
            "metadata_check": metadata_check,
            "total_pages": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
            "accuracy_before_postprocess": accuracy_before_postprocess,
            "accuracy_delta_postprocess": ((correct / total) - accuracy_before_postprocess) if total else 0.0,
            "batch_size": args.batch_size,
            "device": str(device),
            "elapsed_s": elapsed,
            "avg_pages_per_s": total / max(elapsed, 1e-9),
            "postprocess": {
                "enabled": postprocess_enabled,
                "method": "rotation_tta_consistency",
                "confidence_threshold": postprocess_conf_thr,
                "margin_threshold": postprocess_margin_thr,
                "max_pages": int(postprocess_budget),
                "candidates": int(postprocess_candidates),
                "applied": int(postprocess_applied),
                "changed_predictions": int(postprocess_changed),
                "skipped_budget": int(postprocess_skipped_budget),
                "fixed_errors": int(postprocess_fixed),
                "regressed_errors": int(postprocess_regressed),
            },
            "nll_mean": nll_mean,
            "brier_mean": brier_mean,
            "calibration": calibration,
            "confidence_stats": _summarize_values(confidences),
            "true_logprob_stats": _summarize_values(true_logprobs),
            "pred_logprob_stats": _summarize_values(pred_logprobs),
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
        print(
            "[saved-model-test] postprocess "
            f"base_acc={report['accuracy_before_postprocess']:.4f} "
            f"delta={report['accuracy_delta_postprocess']:.4f} "
            f"applied={report['postprocess']['applied']} "
            f"changed={report['postprocess']['changed_predictions']} "
            f"fixed={report['postprocess']['fixed_errors']} "
            f"regressed={report['postprocess']['regressed_errors']}"
        )
        print(
            "[saved-model-test] calibration "
            f"nll={report['nll_mean']:.4f} brier={report['brier_mean']:.4f} "
            f"ece15={float(report['calibration']['ece']):.4f} "
            f"mce15={float(report['calibration']['mce']):.4f}"
        )
        print(f"[saved-model-test] {format_memory(report['memory_end'])}")
        print(f"[saved-model-test] report={output_dir / 'report.json'}")
        print(f"[saved-model-test] predictions={output_dir / 'predictions.jsonl'}")
        log_memory("[saved-model-test] done", device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
