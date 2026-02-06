"""Utility helpers for deterministic, reproducible pipeline runs."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import CARDINAL_ANGLES


@dataclass(slots=True)
class PdfInput:
    """Represents one source PDF selected for processing."""

    path: Path
    source_key: str


def utc_timestamp() -> str:
    """Return a compact UTC timestamp string suitable for file names."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def sanitize_name(value: str) -> str:
    """Produce filesystem-safe names from arbitrary strings."""
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return collapsed.strip("-._") or "item"


def stable_doc_id(source_key: str, stem: str) -> str:
    """Create a deterministic document id from a stable key."""
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:10]
    return f"{sanitize_name(stem)[:40]}-{digest}"


def validate_angles(angles: Iterable[int]) -> tuple[int, ...]:
    """Validate and normalize angle choices to unique, ordered cardinal values."""
    normalized: list[int] = []
    for raw in angles:
        angle = raw % 360
        if angle not in CARDINAL_ANGLES:
            msg = f"Unsupported angle {raw}. Only cardinal angles {CARDINAL_ANGLES} are allowed."
            raise ValueError(msg)
        if angle not in normalized:
            normalized.append(angle)
    if not normalized:
        raise ValueError("Angle list cannot be empty.")
    return tuple(normalized)


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for random and numpy (plus torch if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def discover_input_pdfs(input_paths: Iterable[str], input_dir: str | None) -> list[PdfInput]:
    """Resolve input PDFs from explicit paths plus optional directory scan."""
    discovered: list[PdfInput] = []
    seen: set[Path] = set()

    for raw in input_paths:
        path = Path(raw).expanduser().resolve()
        if path.suffix.lower() != ".pdf":
            continue
        if path not in seen and path.exists():
            seen.add(path)
            discovered.append(PdfInput(path=path, source_key=str(path)))

    if input_dir:
        root = Path(input_dir).expanduser().resolve()
        if root.exists():
            for path in sorted(root.rglob("*.pdf")):
                if path not in seen:
                    seen.add(path)
                    discovered.append(PdfInput(path=path, source_key=str(path.relative_to(root))))

    return discovered


def dump_json(data: dict, output_path: Path) -> None:
    """Write JSON with stable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Read JSON object from file."""
    return json.loads(path.read_text(encoding="utf-8"))
