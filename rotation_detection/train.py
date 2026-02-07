"""Training entrypoint for the torch orientation detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .detectors.torch_detector import TrainConfig, train_orientation_model
from .logging_utils import tee_output
from .memory_profile import log_memory, snapshot_memory
from .manifest import read_jsonl
from .utils import dump_json, load_json


def _resolve_labels_path(manifest_path: str | None, labels_path: str | None) -> Path:
    if labels_path:
        return Path(labels_path).expanduser().resolve()
    if manifest_path:
        manifest_abs = Path(manifest_path).expanduser().resolve()
        manifest = load_json(manifest_abs)
        rel = Path(str(manifest.get("labels_path", "labels.jsonl")))
        return (manifest_abs.parent / rel).resolve()
    raise RuntimeError("Provide either --labels-path or --manifest-path for training.")


def _resolve_split_labels(
    manifest_path: str,
    train_split: str,
    val_split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, Path, dict[str, Any]]:
    manifest_abs = Path(manifest_path).expanduser().resolve()
    manifest = load_json(manifest_abs)
    root = manifest_abs.parent
    splits = manifest.get("splits")

    if not isinstance(splits, dict):
        labels_abs = _resolve_labels_path(manifest_path, None)
        return read_jsonl(labels_abs), None, labels_abs.parent, {"mode": "single_manifest_labels"}

    train_entry = splits.get(train_split)
    if not isinstance(train_entry, dict):
        labels_abs = _resolve_labels_path(manifest_path, None)
        return read_jsonl(labels_abs), None, labels_abs.parent, {"mode": "single_manifest_labels"}

    train_labels_abs = (root / Path(str(train_entry["labels_path"]))).resolve()
    train_rows = read_jsonl(train_labels_abs)

    val_rows = None
    val_entry = splits.get(val_split_name)
    if isinstance(val_entry, dict):
        val_labels_abs = (root / Path(str(val_entry["labels_path"]))).resolve()
        if val_labels_abs.exists():
            val_rows = read_jsonl(val_labels_abs)

    metadata = {
        "mode": "split_manifest_labels",
        "train_split": train_split,
        "val_split": val_split_name,
        "train_labels_path": str(train_labels_abs),
        "val_labels_path": str((root / Path(str(val_entry["labels_path"]))).resolve()) if isinstance(val_entry, dict) else None,
    }
    return train_rows, val_rows, root, metadata


def run_training(
    *,
    manifest_path: str | None,
    labels_path: str | None,
    checkpoint_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    image_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
    device: str,
    max_samples: int | None,
    train_split: str,
    val_split_name: str,
    log_every_batches: int,
    early_stopping_patience: int = 2,
    early_stopping_min_delta: float = 1e-4,
) -> Path:
    """Train a torch classifier from dataset labels."""
    checkpoint_abs = Path(checkpoint_path).expanduser().resolve()
    log_path = checkpoint_abs.with_suffix(".train.log")

    with tee_output(log_path):
        print(f"[train] checkpoint={checkpoint_abs}")
        start_mem = snapshot_memory()
        print(f"[train] start memory {start_mem}")

        if labels_path:
            labels_abs = _resolve_labels_path(manifest_path, labels_path)
            dataset_root = labels_abs.parent
            labels = read_jsonl(labels_abs)
            val_labels = None
            source_meta = {"mode": "explicit_labels", "labels_path": str(labels_abs)}
        elif manifest_path:
            manifest_abs = Path(manifest_path).expanduser().resolve()
            manifest_payload = load_json(manifest_abs)
            print(
                "[train] data_sampling "
                f"class_balance={manifest_payload.get('class_balance', 'unknown')} "
                f"split_strategy={manifest_payload.get('split_strategy', 'unknown')} "
                f"min_val_docs={manifest_payload.get('min_val_docs', 'unknown')} "
                f"min_test_docs={manifest_payload.get('min_test_docs', 'unknown')}"
            )
            labels, val_labels, dataset_root, source_meta = _resolve_split_labels(
                manifest_path=manifest_path,
                train_split=train_split,
                val_split_name=val_split_name,
            )
        else:
            raise RuntimeError("Provide either --labels-path or --manifest-path for training.")

        if max_samples is not None and max_samples > 0:
            labels = labels[:max_samples]
            if val_labels is not None:
                val_labels = val_labels[: max(1, max_samples // 4)]

        print(
            f"[train] rows train={len(labels)} val={(len(val_labels) if val_labels is not None else 'auto')}"
        )

        config = TrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            image_size=image_size,
            val_split=val_split,
            seed=seed,
            num_workers=num_workers,
            device=device,
            log_every_batches=log_every_batches,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )

        report = train_orientation_model(
            labels=labels,
            dataset_root=dataset_root,
            checkpoint_path=checkpoint_abs,
            config=config,
            val_labels=val_labels,
        )
        report["data_source"] = source_meta
        report["memory_before_train"] = start_mem
        report["memory_after_train"] = snapshot_memory()
        report_path = checkpoint_abs.with_suffix(".report.json")
        dump_json(report, report_path)
        print(f"Training report: {report_path}")
        log_memory("[train] final")
    return checkpoint_abs
