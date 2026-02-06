"""Manifest and prediction read/write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records from JSONL file."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows
