"""Tee-style logging helpers for terminal + file output."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator, TextIO


class _TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO):
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        written = self.primary.write(data)
        self.secondary.write(data)
        self.primary.flush()
        self.secondary.flush()
        return written

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self.primary, "encoding", None)


@contextmanager
def tee_output(log_path: Path) -> Iterator[None]:
    """Mirror stdout/stderr to a log file while preserving terminal output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = _TeeStream(original_stdout, handle)  # type: ignore[assignment]
        sys.stderr = _TeeStream(original_stderr, handle)  # type: ignore[assignment]
        print(f"[logging] tee enabled log_file={log_path}")

        try:
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
