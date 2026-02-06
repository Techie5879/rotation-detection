"""Lightweight memory profiling helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import tracemalloc
from typing import Any


def ensure_tracemalloc_started() -> None:
    if not tracemalloc.is_tracing():
        tracemalloc.start()


def _ps_rss_mb() -> float | None:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
        if not output:
            return None
        return float(output) / 1024.0
    except Exception:
        return None


def _ru_max_rss_mb() -> float | None:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return value / (1024.0 * 1024.0)
        return value / 1024.0
    except Exception:
        return None


def _torch_memory_mb(device: Any | None) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "torch_allocated_mb": None,
        "torch_reserved_mb": None,
        "torch_mps_driver_mb": None,
    }
    if device is None:
        return result

    try:
        import torch

        if getattr(device, "type", "") == "cuda":
            result["torch_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
            result["torch_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
        elif getattr(device, "type", "") == "mps":
            current_alloc = getattr(torch.mps, "current_allocated_memory", None)
            driver_alloc = getattr(torch.mps, "driver_allocated_memory", None)
            if callable(current_alloc):
                result["torch_allocated_mb"] = float(current_alloc()) / (1024.0 * 1024.0)
            if callable(driver_alloc):
                result["torch_mps_driver_mb"] = float(driver_alloc()) / (1024.0 * 1024.0)
    except Exception:
        return result

    return result


def snapshot_memory(device: Any | None = None) -> dict[str, float | None]:
    ensure_tracemalloc_started()
    py_current, py_peak = tracemalloc.get_traced_memory()
    snapshot = {
        "rss_mb": _ps_rss_mb(),
        "ru_maxrss_mb": _ru_max_rss_mb(),
        "py_current_mb": py_current / (1024.0 * 1024.0),
        "py_peak_mb": py_peak / (1024.0 * 1024.0),
    }
    snapshot.update(_torch_memory_mb(device))
    return snapshot


def format_memory(snapshot: dict[str, float | None]) -> str:
    parts: list[str] = []
    keys = [
        "rss_mb",
        "ru_maxrss_mb",
        "py_current_mb",
        "py_peak_mb",
        "torch_allocated_mb",
        "torch_reserved_mb",
        "torch_mps_driver_mb",
    ]
    for key in keys:
        value = snapshot.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value:.1f}")
    return " ".join(parts)


def log_memory(prefix: str, device: Any | None = None) -> dict[str, float | None]:
    snap = snapshot_memory(device=device)
    print(f"{prefix} memory {format_memory(snap)}")
    return snap
