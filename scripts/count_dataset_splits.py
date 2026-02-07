"""Count documents and pages in train/val/test dataset splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("train", "val", "test")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_from_split_manifest(split_manifest_path: Path) -> tuple[int, int]:
    payload = _load_json(split_manifest_path)
    docs = payload.get("documents", [])
    doc_count = len(docs)
    page_count = sum(int(doc.get("page_count", 0)) for doc in docs)
    return doc_count, page_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Count split document/page totals for a dataset run")
    parser.add_argument("--dataset-root", required=True, help="Dataset root path, e.g. runs/final_run_dataset")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    root_manifest = dataset_root / "manifest.json"

    print(f"dataset_root={dataset_root}")

    if root_manifest.exists():
        payload = _load_json(root_manifest)
        splits = payload.get("splits")
        if isinstance(splits, dict):
            print("counts_from_root_manifest:")
            for split in SPLITS:
                info = splits.get(split, {})
                docs = int(info.get("documents", 0))
                pages = int(info.get("pages", 0))
                print(f"  {split}: docs={docs} pages={pages}")

    print("counts_from_split_manifests:")
    for split in SPLITS:
        split_manifest = dataset_root / split / "manifest.json"
        if not split_manifest.exists():
            print(f"  {split}: missing_manifest={split_manifest}")
            continue
        docs, pages = _count_from_split_manifest(split_manifest)
        print(f"  {split}: docs={docs} pages={pages}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
