"""Download a subset of GCS zip files and extract PDFs.

Uses `gcloud storage cp` (not gsutil) for downloads.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rotation_detection.logging_utils import tee_output
from rotation_detection.utils import utc_timestamp
from tqdm.auto import tqdm


def _iter_with_progress(items, *, desc: str, unit: str):
    return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True, mininterval=0.5)


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True)


def list_zip_uris(gcs_prefix: str) -> list[str]:
    pattern = gcs_prefix.rstrip("/") + "/*.zip"
    output = _run(["gcloud", "storage", "ls", pattern])
    return [line.strip() for line in output.splitlines() if line.strip().endswith(".zip")]


def gcs_object_size(uri: str) -> int:
    output = _run(["gcloud", "storage", "du", uri]).strip()
    if not output:
        return 0
    return int(output.split()[0])


def select_subset(
    uris: list[str],
    *,
    sample_size: int,
    seed: int,
    max_zip_bytes: int | None,
) -> tuple[list[str], dict[str, int]]:
    rng = random.Random(seed)
    shuffled = list(uris)
    rng.shuffle(shuffled)

    chosen: list[str] = []
    sizes: dict[str, int] = {}

    iter_uris = _iter_with_progress(shuffled, desc="Selecting zips", unit="zip")
    for uri in iter_uris:
        if max_zip_bytes is not None:
            size = gcs_object_size(uri)
            if size <= 0:
                continue
            if size > max_zip_bytes:
                continue
            sizes[uri] = size
        chosen.append(uri)
        if len(chosen) >= sample_size:
            break

    if max_zip_bytes is None:
        for uri in chosen:
            sizes[uri] = gcs_object_size(uri)

    return sorted(chosen), sizes


def download_zips(uris: list[str], zips_dir: Path) -> list[Path]:
    zips_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for uri in _iter_with_progress(uris, desc="Downloading zips", unit="zip"):
        filename = uri.rsplit("/", 1)[-1]
        local_path = zips_dir / filename
        if local_path.exists() and local_path.stat().st_size > 0:
            downloaded.append(local_path)
            continue
        subprocess.check_call(["gcloud", "storage", "cp", uri, str(zips_dir)])
        downloaded.append(local_path)

    return downloaded


def _safe_join(base: Path, member: str) -> Path:
    destination = (base / member).resolve()
    if not str(destination).startswith(str(base.resolve())):
        raise RuntimeError(f"Unsafe zip entry path detected: {member}")
    return destination


def extract_pdfs(zip_paths: list[Path], extracted_dir: Path) -> list[Path]:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    extracted_pdfs: list[Path] = []

    for zip_path in _iter_with_progress(zip_paths, desc="Extracting PDFs", unit="zip"):
        target_root = extracted_dir / zip_path.stem
        target_root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                if not info.filename.lower().endswith(".pdf"):
                    continue
                destination = _safe_join(target_root, info.filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info, "r") as src, destination.open("wb") as dst:
                    dst.write(src.read())
                extracted_pdfs.append(destination)

    return extracted_pdfs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a subset of GCS zip files into test_pdfs")
    parser.add_argument("--gcs-prefix", default="gs://ew-ny-dump/ew-ny-dump")
    parser.add_argument("--dest-root", default="test_pdfs/gcs-ew-ny-dump")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-zip-mb", type=int, default=250)
    parser.add_argument("--selection-manifest", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    dest_root = Path(args.dest_root).expanduser().resolve()
    zips_dir = dest_root / "zips"
    extracted_dir = dest_root / "extracted"
    manifests_dir = dest_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    log_path = manifests_dir / f"ingest_{utc_timestamp()}.log"
    with tee_output(log_path):
        print(f"[ingest] dest_root={dest_root}")
        print(f"[ingest] gcs_prefix={args.gcs_prefix}")

        if args.selection_manifest:
            selected = [
                line.strip()
                for line in Path(args.selection_manifest).expanduser().resolve().read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            sizes = {uri: gcs_object_size(uri) for uri in selected}
            print(f"[ingest] loaded selection manifest: {args.selection_manifest} ({len(selected)} uris)")
        else:
            uris = list_zip_uris(args.gcs_prefix)
            print(f"[ingest] listed {len(uris)} zip objects")
            selected, sizes = select_subset(
                uris,
                sample_size=args.sample_size,
                seed=args.seed,
                max_zip_bytes=args.max_zip_mb * 1024 * 1024,
            )
            print(f"[ingest] selected {len(selected)} zips (max_zip_mb={args.max_zip_mb})")

        selected_name = f"selected_seed{args.seed}_n{len(selected)}_max{args.max_zip_mb}mb"
        txt_path = manifests_dir / f"{selected_name}.txt"
        txt_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

        downloaded_zips = download_zips(selected, zips_dir)
        extracted_pdfs = extract_pdfs(downloaded_zips, extracted_dir)

        summary = {
            "gcs_prefix": args.gcs_prefix,
            "selection_manifest": str(txt_path),
            "selected_count": len(selected),
            "selected_uris": selected,
            "sizes_bytes": sizes,
            "downloaded_count": len(downloaded_zips),
            "downloaded_paths": [str(path) for path in downloaded_zips],
            "extracted_pdf_count": len(extracted_pdfs),
            "extracted_pdf_paths": [str(path) for path in extracted_pdfs],
            "zips_dir": str(zips_dir),
            "extracted_dir": str(extracted_dir),
            "log_path": str(log_path),
        }
        summary_path = manifests_dir / f"ingest_summary_{utc_timestamp()}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        total_bytes = sum(sizes.values())
        print(f"[ingest] downloaded_zips={len(downloaded_zips)} total_gib={total_bytes / (1024 ** 3):.3f}")
        print(f"[ingest] extracted_pdfs={len(extracted_pdfs)}")
        print(f"[ingest] summary={summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
