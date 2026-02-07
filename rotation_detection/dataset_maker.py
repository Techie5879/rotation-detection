"""Synthetic dataset generation with baked-in page rotations."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
import os
import random
from collections import defaultdict
from pathlib import Path
import time
from typing import Any

from pypdf import PdfReader, PdfWriter
from tqdm.auto import tqdm

from .constants import CARDINAL_ANGLES
from .logging_utils import tee_output
from .manifest import write_jsonl
from .pdf_ops import append_image_as_pdf_page, rotate_image_clockwise, rotate_metadata_violations
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
DATASET_MAX_WORKERS = 12
PAGE_POOL_MIN_PAGES = 64


def _sample_rotation(rng: random.Random, angles: tuple[int, ...], rotate_probability: float) -> int:
    if rng.random() > rotate_probability:
        return 0
    return rng.choice(angles)


def _balanced_rotation_sequence(total: int, angles: tuple[int, ...], rng: random.Random) -> list[int]:
    if total <= 0:
        return []

    classes = list(angles)
    base = total // len(classes)
    rem = total % len(classes)

    counts = {angle: base for angle in classes}
    remainder_classes = list(classes)
    rng.shuffle(remainder_classes)
    for idx in range(rem):
        counts[remainder_classes[idx]] += 1

    sequence: list[int] = []
    for angle in classes:
        sequence.extend([angle] * counts[angle])
    rng.shuffle(sequence)
    return sequence


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


def _stable_doc_seed(seed: int, source_key: str) -> int:
    digest = hashlib.sha1(f"{seed}:{source_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _resolve_worker_count(job_count: int) -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(DATASET_MAX_WORKERS, cpu, job_count))


def _select_page_indices(
    *,
    total_pages: int,
    max_pages_per_doc: int | None,
    seed: int,
    source_key: str,
) -> list[int]:
    del seed
    del source_key
    if total_pages <= 0:
        return []
    if max_pages_per_doc is None or max_pages_per_doc <= 0 or total_pages <= max_pages_per_doc:
        return list(range(total_pages))
    return list(range(max_pages_per_doc))


def _assign_document_splits(
    inputs: list[PdfInput],
    *,
    page_counts: dict[str, int],
    split_ratios: dict[str, float],
    seed: int,
    min_val_docs: int,
    min_test_docs: int,
) -> dict[str, str]:
    """Assign each input document to train/val/test with size stratification."""
    rng = random.Random(seed)

    order = sorted(inputs, key=lambda item: page_counts[item.source_key], reverse=True)

    n_docs = len(order)
    raw_doc_targets = {split: split_ratios[split] * n_docs for split in SPLITS}
    doc_targets = {split: int(raw_doc_targets[split]) for split in SPLITS}

    if n_docs >= len(SPLITS):
        doc_targets["train"] = max(1, doc_targets["train"])
        doc_targets["val"] = max(1, doc_targets["val"], min_val_docs)
        doc_targets["test"] = max(1, doc_targets["test"], min_test_docs)

    assigned_so_far = sum(doc_targets.values())
    if assigned_so_far > n_docs:
        minimums = {
            "train": 1 if n_docs >= 3 else 0,
            "val": 1 if n_docs >= 3 else 0,
            "test": 1 if n_docs >= 3 else 0,
        }
        while assigned_so_far > n_docs:
            split = max(SPLITS, key=lambda name: doc_targets[name])
            if doc_targets[split] > minimums[split]:
                doc_targets[split] -= 1
                assigned_so_far -= 1
            else:
                reduced = False
                for alt in SPLITS:
                    if doc_targets[alt] > minimums[alt]:
                        doc_targets[alt] -= 1
                        assigned_so_far -= 1
                        reduced = True
                        break
                if not reduced:
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

    assigned_pages: dict[str, int] = {split: 0 for split in SPLITS}
    assigned_docs: dict[str, int] = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}

    bin_count = max(1, min(6, n_docs))
    bins: list[list[PdfInput]] = [[] for _ in range(bin_count)]
    for idx, item in enumerate(order):
        bins[idx % bin_count].append(item)
    for bucket in bins:
        rng.shuffle(bucket)

    for bucket in bins:
        for item in bucket:
            best_split = None
            best_score = None

            for split in SPLITS:
                docs_remaining = doc_targets[split] - assigned_docs[split]
                if docs_remaining <= 0:
                    continue
                remaining_pages = target_pages[split] - assigned_pages[split]
                score = (docs_remaining, remaining_pages)
                if best_score is None or score > best_score:
                    best_score = score
                    best_split = split

            if best_split is None:
                for split in SPLITS:
                    remaining_pages = target_pages[split] - assigned_pages[split]
                    score = (remaining_pages, -assigned_docs[split])
                    if best_score is None or score > best_score:
                        best_score = score
                        best_split = split

            assert best_split is not None
            assignments[item.source_key] = best_split
            assigned_pages[best_split] += page_counts[item.source_key]
            assigned_docs[best_split] += 1

    return assignments


def _build_rotation_plan(
    *,
    page_indices: list[int],
    rng: random.Random,
    angles: tuple[int, ...],
    rotate_probability: float,
    class_balance: str,
) -> dict[int, int]:
    if not page_indices:
        return {}

    if class_balance == "uniform":
        sequence = _balanced_rotation_sequence(len(page_indices), angles, rng)
    else:
        sequence = [_sample_rotation(rng, angles, rotate_probability) for _ in page_indices]

    return {page_idx: sequence[idx] for idx, page_idx in enumerate(page_indices)}


def _format_angle_counts(labels: list[dict[str, Any]]) -> str:
    counts = {angle: 0 for angle in CARDINAL_ANGLES}
    for row in labels:
        angle = int(row["rotation_deg"]) % 360
        if angle in counts:
            counts[angle] += 1
    return " ".join(f"{angle}:{counts[angle]}" for angle in CARDINAL_ANGLES)


def _max_doc_share(labels: list[dict[str, Any]]) -> float:
    if not labels:
        return 0.0
    per_doc: dict[str, int] = defaultdict(int)
    for row in labels:
        per_doc[str(row["doc_id"])] += 1
    max_pages = max(per_doc.values()) if per_doc else 0
    return max_pages / max(len(labels), 1)


def _validate_doc_share_limit(name: str, value: float) -> None:
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}.")


def _render_page(task: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(str(task["source_path"]))
    page_index = int(task["page_index"])
    dpi = int(task["dpi"])
    rotation_deg = int(task["rotation_deg"])
    image_abs = Path(str(task["image_abs"]))
    image_rel_root = str(task["image_rel_root"])
    image_rel_split = str(task["image_rel_split"])

    import pypdfium2 as pdfium

    started = time.perf_counter()
    pdf = pdfium.PdfDocument(str(source_path))
    try:
        page = pdf[page_index]
        bitmap = page.render(scale=(dpi / 72.0), rotation=0)
        image = bitmap.to_pil()
        rotated_image = rotate_image_clockwise(image, rotation_deg)
        rotated_image.save(image_abs, format="PNG", compress_level=3)
        return {
            "page_index": page_index,
            "rotation_deg": rotation_deg,
            "image_abs": str(image_abs),
            "image_rel_root": image_rel_root,
            "image_rel_split": image_rel_split,
            "width": int(rotated_image.width),
            "height": int(rotated_image.height),
            "elapsed_s": time.perf_counter() - started,
        }
    finally:
        pdf.close()


def _process_document_job(job: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(str(job["source_path"]))
    source_key = str(job["source_key"])
    split = str(job["split"])
    doc_id = str(job["doc_id"])
    output_dir = Path(str(job["output_dir"]))
    dpi = int(job["dpi"])
    seed = int(job["seed"])
    rotate_probability = float(job["rotate_probability"])
    angles = tuple(int(a) for a in job["angles"])
    max_pages_per_doc = int(job["max_pages_per_doc"]) if job["max_pages_per_doc"] is not None else None
    class_balance = str(job.get("class_balance", "random"))

    output_pdf_rel_root = Path(split) / "pdfs" / f"{doc_id}.pdf"
    output_pdf_rel_split = Path("pdfs") / f"{doc_id}.pdf"
    output_pdf_abs = output_dir / output_pdf_rel_root

    rng = random.Random(_stable_doc_seed(seed, source_key))
    selected_page_indices = sorted(int(i) for i in job.get("selected_page_indices", []))
    if not selected_page_indices:
        total_pages = _effective_page_count(source_path, max_pages_per_doc)
        selected_page_indices = list(range(total_pages))

    rotation_plan = _build_rotation_plan(
        page_indices=selected_page_indices,
        rng=rng,
        angles=angles,
        rotate_probability=rotate_probability,
        class_balance=class_balance,
    )

    writer = PdfWriter()
    labels_records: list[dict[str, Any]] = []
    root_doc_pages: list[dict[str, Any]] = []
    split_doc_pages: list[dict[str, Any]] = []

    page_tasks: list[dict[str, Any]] = []
    for page_index in selected_page_indices:
        rotation_deg = int(rotation_plan[page_index])
        image_rel_root = Path(split) / "pages" / f"{doc_id}_p{page_index + 1:05d}.png"
        image_rel_split = Path("pages") / f"{doc_id}_p{page_index + 1:05d}.png"
        image_abs = output_dir / image_rel_root
        page_tasks.append(
            {
                "source_path": str(source_path),
                "page_index": page_index,
                "dpi": dpi,
                "rotation_deg": rotation_deg,
                "image_abs": str(image_abs),
                "image_rel_root": str(image_rel_root),
                "image_rel_split": str(image_rel_split),
            }
        )

    started = time.perf_counter()
    rendered_by_page: dict[int, dict[str, Any]] = {}
    if page_tasks:
        page_worker_count = _resolve_worker_count(len(page_tasks))
        use_page_pool = len(page_tasks) >= PAGE_POOL_MIN_PAGES and page_worker_count > 1
        completed = 0
        page_started = time.perf_counter()

        if use_page_pool:
            print(
                f"[dataset] doc={doc_id} split={split} page_pool=spawn workers={page_worker_count} pages={len(page_tasks)}"
            )
            spawn_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=page_worker_count, mp_context=spawn_ctx) as executor:
                futures = [executor.submit(_render_page, task) for task in page_tasks]
                for future in as_completed(futures):
                    rendered = future.result(timeout=120)
                    page_index = int(rendered["page_index"])
                    rendered_by_page[page_index] = rendered
                    completed += 1
                    if completed == len(page_tasks) or completed % 25 == 0:
                        elapsed = time.perf_counter() - page_started
                        rate = completed / max(elapsed, 1e-9)
                        print(
                            f"[dataset] doc_progress doc_id={doc_id} pages={completed}/{len(page_tasks)} "
                            f"rate_pps={rate:.2f}"
                        )
        else:
            print(
                f"[dataset] doc={doc_id} split={split} page_pool=off workers=1 pages={len(page_tasks)}"
            )
            for task in page_tasks:
                rendered = _render_page(task)
                page_index = int(rendered["page_index"])
                rendered_by_page[page_index] = rendered
                completed += 1
                if completed == len(page_tasks) or completed % 25 == 0:
                    elapsed = time.perf_counter() - page_started
                    rate = completed / max(elapsed, 1e-9)
                    print(
                        f"[dataset] doc_progress doc_id={doc_id} pages={completed}/{len(page_tasks)} "
                        f"rate_pps={rate:.2f}"
                    )

    from PIL import Image

    for page_index in selected_page_indices:
        rendered = rendered_by_page[page_index]
        image_abs = Path(str(rendered["image_abs"]))
        with Image.open(image_abs) as rotated_image:
            append_image_as_pdf_page(writer, rotated_image, dpi=dpi)

        image_rel_root = str(rendered["image_rel_root"])
        image_rel_split = str(rendered["image_rel_split"])
        rotation_deg = int(rendered["rotation_deg"])
        width = int(rendered["width"])
        height = int(rendered["height"])

        labels_records.append(
            {
                "doc_id": doc_id,
                "split": split,
                "page_index": page_index,
                "rotation_deg": rotation_deg,
                "source_pdf": str(source_path),
                "output_pdf": str(output_pdf_rel_root),
                "image_path": image_rel_root,
                "width": width,
                "height": height,
            }
        )

        root_doc_pages.append(
            {
                "page_index": page_index,
                "rotation_deg": rotation_deg,
                "image_path": image_rel_root,
                "width": width,
                "height": height,
            }
        )
        split_doc_pages.append(
            {
                "page_index": page_index,
                "rotation_deg": rotation_deg,
                "image_path": image_rel_split,
                "width": width,
                "height": height,
            }
        )

    page_count = len(root_doc_pages)
    if page_count == 0:
        return {
            "doc_id": doc_id,
            "split": split,
            "page_count": 0,
            "elapsed_s": time.perf_counter() - started,
            "labels_records": [],
            "root_doc_record": None,
            "split_doc_record": None,
        }

    with output_pdf_abs.open("wb") as handle:
        writer.write(handle)

    violations = rotate_metadata_violations(output_pdf_abs)
    if violations:
        raise RuntimeError(f"Rotation metadata leak detected in {output_pdf_abs}: {violations}.")

    return {
        "doc_id": doc_id,
        "split": split,
        "page_count": page_count,
        "elapsed_s": time.perf_counter() - started,
        "labels_records": labels_records,
        "root_doc_record": {
            "doc_id": doc_id,
            "split": split,
            "source_pdf": str(source_path),
            "output_pdf": str(output_pdf_rel_root),
            "page_count": page_count,
            "pages": root_doc_pages,
        },
        "split_doc_record": {
            "doc_id": doc_id,
            "split": split,
            "source_pdf": str(source_path),
            "output_pdf": str(output_pdf_rel_split),
            "page_count": page_count,
            "pages": split_doc_pages,
        },
    }


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
    class_balance: str,
    min_val_docs: int,
    min_test_docs: int,
    max_val_doc_share: float,
    max_test_doc_share: float,
) -> dict[str, Any]:
    """Create synthetic rotated PDFs with explicit train/val/test split folders."""
    if rotate_probability < 0 or rotate_probability > 1:
        raise ValueError("rotate_probability must be between 0 and 1.")
    if class_balance not in {"random", "uniform"}:
        raise ValueError("class_balance must be one of: random, uniform")
    if min_val_docs < 1 or min_test_docs < 1:
        raise ValueError("min_val_docs and min_test_docs must be >= 1")
    _validate_doc_share_limit("max_val_doc_share", max_val_doc_share)
    _validate_doc_share_limit("max_test_doc_share", max_test_doc_share)

    set_global_seed(seed)

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
        min_val_docs=min_val_docs,
        min_test_docs=min_test_docs,
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
    next_log_threshold = log_every_pages if log_every_pages > 0 else None

    jobs: list[dict[str, Any]] = []
    for source in inputs:
        split = split_by_source[source.source_key]
        doc_id = stable_doc_id(source.source_key, source.path.stem)
        selected_page_indices = _select_page_indices(
            total_pages=full_page_counts[source.source_key],
            max_pages_per_doc=max_pages_per_doc,
            seed=seed,
            source_key=source.source_key,
        )
        jobs.append(
            {
                "source_path": str(source.path),
                "source_key": source.source_key,
                "split": split,
                "doc_id": doc_id,
                "output_dir": str(output_dir),
                "dpi": dpi,
                "seed": seed,
                "rotate_probability": rotate_probability,
                "angles": list(angles),
                "max_pages_per_doc": max_pages_per_doc,
                "selected_page_indices": selected_page_indices,
                "class_balance": class_balance,
            }
        )

    print(
        "[dataset] document_loop=sequential page_render_pool=spawn "
        f"page_pool_min_pages={PAGE_POOL_MIN_PAGES} max_workers_cap={DATASET_MAX_WORKERS}"
    )

    docs_bar = tqdm(
        total=len(jobs),
        desc="[dataset] documents",
        unit="doc",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    pages_bar = tqdm(
        total=total_target_pages,
        desc="[dataset] pages",
        unit="page",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    try:
        for job in jobs:
            try:
                result = _process_document_job(job)
            except Exception as exc:
                raise RuntimeError(
                    f"Dataset worker failed for source={job['source_path']} doc_id={job['doc_id']}"
                ) from exc

            docs_bar.update(1)
            page_count = int(result["page_count"])
            pages_bar.update(page_count)
            processed_total_pages += page_count

            split = str(result["split"])
            if page_count > 0:
                split_page_counts[split] += page_count
                labels_records = result["labels_records"]
                all_labels_records.extend(labels_records)
                split_labels_records[split].extend(labels_records)

                root_doc_record = result["root_doc_record"]
                split_doc_record = result["split_doc_record"]
                if root_doc_record is not None:
                    all_docs_records.append(root_doc_record)
                if split_doc_record is not None:
                    split_docs_records[split].append(split_doc_record)

                print(
                    f"[dataset] done split={split} doc_id={result['doc_id']} pages={page_count} "
                    f"elapsed_s={float(result['elapsed_s']):.1f}"
                )
            else:
                print(f"[dataset] skipped split={split} doc_id={result['doc_id']} pages=0")

            if next_log_threshold is not None and processed_total_pages >= next_log_threshold:
                elapsed = time.perf_counter() - started_at
                rate = processed_total_pages / max(elapsed, 1e-9)
                print(
                    "[dataset] progress "
                    f"overall={processed_total_pages}/{total_target_pages} rate_pps={rate:.2f}"
                )
                next_log_threshold += log_every_pages
    finally:
        docs_bar.close()
        pages_bar.close()

    all_labels_records.sort(key=lambda row: (str(row["doc_id"]), int(row["page_index"])))
    all_docs_records.sort(key=lambda row: str(row["doc_id"]))
    for split in SPLITS:
        split_labels_records[split].sort(key=lambda row: (str(row["doc_id"]), int(row["page_index"])))
        split_docs_records[split].sort(key=lambda row: str(row["doc_id"]))

    split_angle_counts = {
        split: _format_angle_counts(split_labels_records[split]) for split in SPLITS
    }
    split_doc_shares = {
        split: _max_doc_share(split_labels_records[split]) for split in SPLITS
    }

    val_doc_share = float(split_doc_shares["val"])
    test_doc_share = float(split_doc_shares["test"])
    if val_doc_share > max_val_doc_share:
        raise RuntimeError(
            "Validation split is document-dominated: "
            f"max_doc_share={val_doc_share:.3f} > allowed={max_val_doc_share:.3f}. "
            "Increase --min-val-docs, lower --max-pages-per-doc, change --seed, "
            "or relax --max-val-doc-share."
        )
    if test_doc_share > max_test_doc_share:
        raise RuntimeError(
            "Test split is document-dominated: "
            f"max_doc_share={test_doc_share:.3f} > allowed={max_test_doc_share:.3f}. "
            "Increase --min-test-docs, lower --max-pages-per-doc, change --seed, "
            "or relax --max-test-doc-share."
        )

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

        angle_counts = split_angle_counts[split]
        doc_share = split_doc_shares[split]
        print(
            f"[dataset] split_proof split={split} docs={len(split_docs_records[split])} "
            f"pages={split_page_counts[split]} angles={angle_counts} max_doc_share={doc_share:.3f}"
        )

    manifest = {
        "dataset_id": output_dir.name,
        "created_at_utc": utc_timestamp(),
        "seed": seed,
        "dpi": dpi,
        "angles": list(angles),
        "rotate_probability": rotate_probability,
        "class_balance": class_balance,
        "split_strategy": "document_page_weighted",
        "split_ratios": split_ratios,
        "min_val_docs": int(min_val_docs),
        "min_test_docs": int(min_test_docs),
        "max_val_doc_share": float(max_val_doc_share),
        "max_test_doc_share": float(max_test_doc_share),
        "splits": split_index,
        "documents": all_docs_records,
        "labels_path": "labels.all.jsonl",
    }

    dump_json(manifest, output_dir / "manifest.json")
    write_jsonl(all_labels_records, output_dir / "labels.all.jsonl")

    print(
        f"[dataset] sampling_proof class_balance={class_balance} split_strategy=document_page_weighted "
        f"min_val_docs={min_val_docs} min_test_docs={min_test_docs} "
        f"max_val_doc_share={max_val_doc_share:.3f} max_test_doc_share={max_test_doc_share:.3f}"
    )

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
    class_balance: str = "random",
    min_val_docs: int = 1,
    min_test_docs: int = 1,
    max_val_doc_share: float = 0.35,
    max_test_doc_share: float = 0.35,
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
        print(
            f"[dataset] requested_sampling class_balance={class_balance} "
            f"min_val_docs={min_val_docs} min_test_docs={min_test_docs} "
            f"max_val_doc_share={max_val_doc_share:.3f} max_test_doc_share={max_test_doc_share:.3f}"
        )
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
            class_balance=class_balance,
            min_val_docs=min_val_docs,
            min_test_docs=min_test_docs,
            max_val_doc_share=max_val_doc_share,
            max_test_doc_share=max_test_doc_share,
        )

        print(f"Dataset created: {dataset_dir}")
        print(f"Documents: {len(manifest['documents'])}")
        for split, info in manifest["splits"].items():
            print(f"Split {split}: docs={info['documents']} pages={info['pages']}")
        print(f"All labels: {dataset_dir / 'labels.all.jsonl'}")
        print(f"Manifest: {dataset_dir / 'manifest.json'}")

    return dataset_dir
