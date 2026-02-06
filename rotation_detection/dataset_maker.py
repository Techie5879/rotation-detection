"""Synthetic dataset generation with baked-in page rotations."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any
import time
from tqdm.auto import tqdm

from pypdf import PdfReader, PdfWriter

from .constants import CARDINAL_ANGLES
from .logging_utils import tee_output
from .manifest import write_jsonl
from .pdf_ops import append_image_as_pdf_page, iter_rendered_pages, rotate_image_clockwise, rotate_metadata_violations
from .utils import (
    PdfInput,
    discover_input_pdfs,
    dump_json,
    set_global_seed,
    stable_doc_id,
    utc_timestamp,
    validate_angles,
)


SPLITS: tuple[str, str, str] = ("train", "val", "test")


def _wrap_tqdm(iterable, *, total: int | None, desc: str, unit: str, leave: bool = True):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        mininterval=0.5,
    )


def _sample_rotation(rng: random.Random, angles: tuple[int, ...], rotate_probability: float) -> int:
    if rng.random() > rotate_probability:
        return 0
    return rng.choice(angles)


def _normalize_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative.")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("At least one split ratio must be > 0.")
    return {name: value / total for name, value in ratios.items()}


def _effective_page_count(path: Path, max_pages_per_doc: int | None) -> int:
    count = len(PdfReader(str(path)).pages)
    if max_pages_per_doc is not None:
        count = min(count, max_pages_per_doc)
    return count


def _assign_document_splits(
    inputs: list[PdfInput],
    *,
    page_counts: dict[str, int],
    split_ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign each input document to train/val/test with page-aware balancing."""
    rng = random.Random(seed)

    order = list(inputs)
    rng.shuffle(order)
    order.sort(key=lambda item: page_counts[item.source_key], reverse=True)

    n_docs = len(order)

    raw_doc_targets = {split: split_ratios[split] * n_docs for split in SPLITS}
    doc_targets = {split: int(raw_doc_targets[split]) for split in SPLITS}

    for split in SPLITS:
        if n_docs >= len(SPLITS):
            doc_targets[split] = max(1, doc_targets[split])

    assigned_so_far = sum(doc_targets.values())
    if assigned_so_far > n_docs:
        while assigned_so_far > n_docs:
            split = max(SPLITS, key=lambda name: doc_targets[name])
            if doc_targets[split] > 1:
                doc_targets[split] -= 1
                assigned_so_far -= 1
            else:
                break
    elif assigned_so_far < n_docs:
        remainders = {
            split: raw_doc_targets[split] - int(raw_doc_targets[split]) for split in SPLITS
        }
        while assigned_so_far < n_docs:
            split = max(SPLITS, key=lambda name: remainders[name])
            doc_targets[split] += 1
            remainders[split] = -1.0
            assigned_so_far += 1

    total_pages = sum(page_counts.values())
    target_pages = {split: split_ratios[split] * total_pages for split in SPLITS}

    assigned_pages: defaultdict[str, int] = defaultdict(int)
    assigned_docs: defaultdict[str, int] = defaultdict(int)
    assignments: dict[str, str] = {}

    cursor = 0
    if n_docs >= 3 and all(doc_targets[split] > 0 for split in SPLITS):
        for split in SPLITS:
            item = order[cursor]
            assignments[item.source_key] = split
            assigned_pages[split] += page_counts[item.source_key]
            assigned_docs[split] += 1
            cursor += 1

    for item in order[cursor:]:
        best_split = None
        best_score = None
        for split in SPLITS:
            docs_remaining = doc_targets[split] - assigned_docs[split]
            if docs_remaining <= 0:
                continue
            remaining = target_pages[split] - assigned_pages[split]
            score = (docs_remaining, remaining)
            if best_score is None or score > best_score:
                best_score = score
                best_split = split

        if best_split is None:
            for split in SPLITS:
                remaining = target_pages[split] - assigned_pages[split]
                score = (remaining, -assigned_docs[split])
                if best_score is None or score > best_score:
                    best_score = score
                    best_split = split

        assert best_split is not None
        assignments[item.source_key] = best_split
        assigned_pages[best_split] += page_counts[item.source_key]
        assigned_docs[best_split] += 1

    return assignments


def build_dataset(
    inputs: list[PdfInput],
    output_dir: Path,
    *,
    seed: int,
    dpi: int,
    angles: tuple[int, ...],
    rotate_probability: float,
    max_pages_per_doc: int | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    log_every_pages: int,
) -> dict[str, Any]:
    """Create synthetic rotated PDFs with explicit train/val/test split folders."""
    if rotate_probability < 0 or rotate_probability > 1:
        raise ValueError("rotate_probability must be between 0 and 1.")

    set_global_seed(seed)
    rng = random.Random(seed)

    split_ratios = _normalize_split_ratios(train_ratio, val_ratio, test_ratio)
    full_page_counts = {item.source_key: _effective_page_count(item.path, None) for item in inputs}
    target_page_counts = {
        item.source_key: _effective_page_count(item.path, max_pages_per_doc) for item in inputs
    }
    split_by_source = _assign_document_splits(
        inputs,
        page_counts=full_page_counts,
        split_ratios=split_ratios,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output_dir / split / "pages").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "pdfs").mkdir(parents=True, exist_ok=True)

    all_labels_records: list[dict[str, Any]] = []
    all_docs_records: list[dict[str, Any]] = []
    split_labels_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    split_docs_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    split_page_counts: defaultdict[str, int] = defaultdict(int)
    total_target_pages = sum(target_page_counts.values())
    processed_total_pages = 0
    started_at = time.perf_counter()

    docs_iter = _wrap_tqdm(inputs, total=len(inputs), desc="[dataset] documents", unit="doc")
    overall_bar = tqdm(
        total=total_target_pages,
        desc="[dataset] pages",
        unit="page",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    try:
        for source in docs_iter:
            split = split_by_source[source.source_key]
            doc_id = stable_doc_id(source.source_key, source.path.stem)

            output_pdf_rel_root = Path(split) / "pdfs" / f"{doc_id}.pdf"
            output_pdf_rel_split = Path("pdfs") / f"{doc_id}.pdf"
            output_pdf_abs = output_dir / output_pdf_rel_root

            writer = PdfWriter()
            root_doc_pages: list[dict[str, Any]] = []
            split_doc_pages: list[dict[str, Any]] = []
            doc_target_pages = target_page_counts[source.source_key]
            doc_started_at = time.perf_counter()
            print(
                f"[dataset] start split={split} doc_id={doc_id} target_pages={doc_target_pages}"
            )

            page_iter = _wrap_tqdm(
                iter_rendered_pages(source.path, dpi=dpi),
                total=doc_target_pages,
                desc=f"[dataset] {doc_id[:24]}",
                unit="page",
                leave=False,
            )
            for page_index, image in page_iter:
                if max_pages_per_doc is not None and page_index >= max_pages_per_doc:
                    break

                rotation_deg = _sample_rotation(rng, angles, rotate_probability)
                rotated_image = rotate_image_clockwise(image, rotation_deg)

                image_rel_root = Path(split) / "pages" / f"{doc_id}_p{page_index + 1:05d}.png"
                image_rel_split = Path("pages") / f"{doc_id}_p{page_index + 1:05d}.png"
                image_abs = output_dir / image_rel_root
                rotated_image.save(image_abs, format="PNG", optimize=True)

                append_image_as_pdf_page(writer, rotated_image, dpi=dpi)

                record = {
                    "doc_id": doc_id,
                    "split": split,
                    "page_index": page_index,
                    "rotation_deg": rotation_deg,
                    "source_pdf": str(source.path),
                    "output_pdf": str(output_pdf_rel_root),
                    "image_path": str(image_rel_root),
                    "width": rotated_image.width,
                    "height": rotated_image.height,
                }
                all_labels_records.append(record)
                split_labels_records[split].append(record)

                root_doc_pages.append(
                    {
                        "page_index": page_index,
                        "rotation_deg": rotation_deg,
                        "image_path": str(image_rel_root),
                        "width": rotated_image.width,
                        "height": rotated_image.height,
                    }
                )
                split_doc_pages.append(
                    {
                        "page_index": page_index,
                        "rotation_deg": rotation_deg,
                        "image_path": str(image_rel_split),
                        "width": rotated_image.width,
                        "height": rotated_image.height,
                    }
                )

                processed_total_pages += 1
                overall_bar.update(1)
                if log_every_pages > 0 and len(root_doc_pages) % log_every_pages == 0:
                    elapsed = time.perf_counter() - started_at
                    rate = processed_total_pages / max(elapsed, 1e-9)
                    print(
                        "[dataset] progress "
                        f"doc_id={doc_id} split={split} doc_pages={len(root_doc_pages)}/{doc_target_pages} "
                        f"overall={processed_total_pages}/{total_target_pages} "
                        f"rate_pps={rate:.2f}"
                    )

            if not root_doc_pages:
                continue

            with output_pdf_abs.open("wb") as handle:
                writer.write(handle)

            violations = rotate_metadata_violations(output_pdf_abs)
            if violations:
                raise RuntimeError(
                    f"Rotation metadata leak detected in {output_pdf_abs}: {violations}."
                )

            split_page_counts[split] += len(root_doc_pages)
            doc_elapsed = time.perf_counter() - doc_started_at
            print(
                f"[dataset] done split={split} doc_id={doc_id} pages={len(root_doc_pages)} "
                f"elapsed_s={doc_elapsed:.1f}"
            )

            all_docs_records.append(
                {
                    "doc_id": doc_id,
                    "split": split,
                    "source_pdf": str(source.path),
                    "output_pdf": str(output_pdf_rel_root),
                    "page_count": len(root_doc_pages),
                    "pages": root_doc_pages,
                }
            )
            split_docs_records[split].append(
                {
                    "doc_id": doc_id,
                    "split": split,
                    "source_pdf": str(source.path),
                    "output_pdf": str(output_pdf_rel_split),
                    "page_count": len(split_doc_pages),
                    "pages": split_doc_pages,
                }
            )
    finally:
        overall_bar.close()

    split_index = {}
    for split in SPLITS:
        split_dir = output_dir / split
        split_manifest = {
            "dataset_id": output_dir.name,
            "split": split,
            "created_at_utc": utc_timestamp(),
            "seed": seed,
            "dpi": dpi,
            "angles": list(angles),
            "rotate_probability": rotate_probability,
            "documents": split_docs_records[split],
            "labels_path": "labels.jsonl",
        }
        dump_json(split_manifest, split_dir / "manifest.json")
        write_jsonl(split_labels_records[split], split_dir / "labels.jsonl")

        split_index[split] = {
            "manifest_path": f"{split}/manifest.json",
            "labels_path": f"{split}/labels.jsonl",
            "documents": len(split_docs_records[split]),
            "pages": split_page_counts[split],
        }

    manifest = {
        "dataset_id": output_dir.name,
        "created_at_utc": utc_timestamp(),
        "seed": seed,
        "dpi": dpi,
        "angles": list(angles),
        "rotate_probability": rotate_probability,
        "split_strategy": "document_page_weighted",
        "split_ratios": split_ratios,
        "splits": split_index,
        "documents": all_docs_records,
        "labels_path": "labels.all.jsonl",
    }

    dump_json(manifest, output_dir / "manifest.json")
    write_jsonl(all_labels_records, output_dir / "labels.all.jsonl")
    total_elapsed = time.perf_counter() - started_at
    print(
        f"[dataset] completed total_pages={processed_total_pages} elapsed_s={total_elapsed:.1f} "
        f"avg_pps={processed_total_pages / max(total_elapsed, 1e-9):.2f}"
    )
    return manifest


def run_make_dataset(
    *,
    input_pdfs: list[str],
    input_dir: str | None,
    output_root: str,
    dataset_name: str | None,
    seed: int,
    dpi: int,
    rotate_probability: float,
    angles: list[int],
    max_pages_per_doc: int | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    log_every_pages: int,
) -> Path:
    """CLI wrapper for dataset generation."""
    selected = discover_input_pdfs(input_pdfs, input_dir)
    if not selected:
        raise RuntimeError("No input PDFs found. Provide --input-dir or --input-pdf.")

    parsed_angles = validate_angles(angles or list(CARDINAL_ANGLES))

    output_root_path = Path(output_root).expanduser().resolve()
    run_name = dataset_name or f"synthetic_{utc_timestamp()}_seed{seed}"
    dataset_dir = output_root_path / run_name

    log_path = dataset_dir / "dataset.log"
    with tee_output(log_path):
        print(f"[dataset] run_dir={dataset_dir}")
        manifest = build_dataset(
            selected,
            dataset_dir,
            seed=seed,
            dpi=dpi,
            angles=parsed_angles,
            rotate_probability=rotate_probability,
            max_pages_per_doc=max_pages_per_doc,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            log_every_pages=log_every_pages,
        )

        print(f"Dataset created: {dataset_dir}")
        print(f"Documents: {len(manifest['documents'])}")
        for split, info in manifest["splits"].items():
            print(f"Split {split}: docs={info['documents']} pages={info['pages']}")
        print(f"All labels: {dataset_dir / 'labels.all.jsonl'}")
        print(f"Manifest: {dataset_dir / 'manifest.json'}")

    return dataset_dir
