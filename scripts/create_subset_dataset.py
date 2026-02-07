"""Create a smaller, balanced dataset view from an existing generated run.

The script does not copy images/PDFs. It writes new manifests/labels that point
to absolute paths in the source dataset, while capping pages per document per split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _cap_rows_per_doc(
    rows: list[dict[str, Any]],
    cap_per_doc: int,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["doc_id"])].append(row)

    selected: list[dict[str, Any]] = []
    for doc_id in sorted(grouped):
        doc_rows = sorted(grouped[doc_id], key=lambda r: int(r["page_index"]))
        if cap_per_doc <= 0 or len(doc_rows) <= cap_per_doc:
            picked = doc_rows
        else:
            rng = random.Random(f"{seed}:{split}:{doc_id}")
            picked = rng.sample(doc_rows, k=cap_per_doc)
            picked.sort(key=lambda r: int(r["page_index"]))
        selected.extend(picked)

    selected.sort(key=lambda r: (str(r["doc_id"]), int(r["page_index"])))
    return selected


def _abspath_if_relative(root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def _materialize_paths(source_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["image_path"] = _abspath_if_relative(source_root, str(row["image_path"]))
        if "output_pdf" in row:
            item["output_pdf"] = _abspath_if_relative(source_root, str(row["output_pdf"]))
        out.append(item)
    return out


def _build_split_manifest(
    source_manifest: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    rows_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        rows_by_doc[str(row["doc_id"])].append(row)

    source_docs = {str(doc["doc_id"]): doc for doc in source_manifest.get("documents", [])}
    documents: list[dict[str, Any]] = []

    for doc_id in sorted(rows_by_doc):
        doc_rows = sorted(rows_by_doc[doc_id], key=lambda r: int(r["page_index"]))
        source_doc = source_docs.get(doc_id)
        output_pdf = str(doc_rows[0].get("output_pdf", ""))
        pages: list[dict[str, Any]] = []

        source_pages_by_idx: dict[int, dict[str, Any]] = {}
        if isinstance(source_doc, dict):
            for page in source_doc.get("pages", []):
                source_pages_by_idx[int(page["page_index"])] = page

        for row in doc_rows:
            page_idx = int(row["page_index"])
            source_page = source_pages_by_idx.get(page_idx, {})
            pages.append(
                {
                    "page_index": page_idx,
                    "rotation_deg": int(row["rotation_deg"]) % 360,
                    "width": int(row.get("width", source_page.get("width", 0))),
                    "height": int(row.get("height", source_page.get("height", 0))),
                    "image_path": str(row["image_path"]),
                }
            )

        documents.append(
            {
                "doc_id": doc_id,
                "output_pdf": output_pdf,
                "page_count": len(doc_rows),
                "pages": pages,
            }
        )

    return {
        "dataset_id": source_manifest.get("dataset_id", "subset"),
        "angles": source_manifest.get("angles", [0, 90, 180, 270]),
        "split": split,
        "documents": documents,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create capped per-doc subset from an existing dataset run")
    parser.add_argument("--source-root", required=True, help="Existing dataset root, e.g. runs/final_run_dataset")
    parser.add_argument("--output-root", required=True, help="New subset dataset root")
    parser.add_argument("--cap-per-doc", type=int, default=500, help="Max pages per document per split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_root_manifest = _load_json(source_root / "manifest.json")

    all_rows: list[dict[str, Any]] = []
    split_stats: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        source_split_dir = source_root / split
        source_split_manifest = _load_json(source_split_dir / "manifest.json")
        source_split_labels = _read_jsonl(source_split_dir / "labels.jsonl")

        selected = _cap_rows_per_doc(
            rows=source_split_labels,
            cap_per_doc=args.cap_per_doc,
            seed=args.seed,
            split=split,
        )
        selected = _materialize_paths(source_root=source_root, rows=selected)

        split_manifest = _build_split_manifest(
            source_manifest=source_split_manifest,
            selected_rows=selected,
            split=split,
        )

        split_dir = output_root / split
        _write_json(split_dir / "manifest.json", split_manifest)
        _write_jsonl(split_dir / "labels.jsonl", selected)

        doc_count = len(split_manifest["documents"])
        page_count = len(selected)
        split_stats[split] = {"documents": doc_count, "pages": page_count}
        all_rows.extend(selected)

    all_rows.sort(key=lambda r: (str(r.get("split", "")), str(r["doc_id"]), int(r["page_index"])))
    _write_jsonl(output_root / "labels.all.jsonl", all_rows)

    root_manifest = {
        "dataset_id": output_root.name,
        "angles": source_root_manifest.get("angles", [0, 90, 180, 270]),
        "labels_path": "labels.all.jsonl",
        "splits": {
            split: {
                "manifest_path": f"{split}/manifest.json",
                "labels_path": f"{split}/labels.jsonl",
                "documents": int(split_stats[split]["documents"]),
                "pages": int(split_stats[split]["pages"]),
            }
            for split in SPLITS
        },
        "source_root": str(source_root),
        "cap_per_doc": int(args.cap_per_doc),
        "seed": int(args.seed),
    }
    _write_json(output_root / "manifest.json", root_manifest)

    print(f"subset_root={output_root}")
    for split in SPLITS:
        print(
            f"{split}: docs={split_stats[split]['documents']} pages={split_stats[split]['pages']} cap={args.cap_per_doc}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
