"""Progress reporting and cooperative cancellation."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, TextIO


class CancelledError(RuntimeError):
    """Raised when the user cancels a run."""


@dataclass
class RunProgress:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    metric: str = ""
    frame: int | None = None
    total: int | None = None
    message: str = ""
    started_at: float = field(default_factory=time.monotonic)
    _listeners: list[Callable[["RunProgress"], None]] = field(default_factory=list)

    def on_update(self, fn: Callable[["RunProgress"], None]) -> None:
        self._listeners.append(fn)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError("Run cancelled")

    def cancel(self) -> None:
        self.cancel_event.set()

    def update(
        self,
        metric: str | None = None,
        frame: int | None = None,
        total: int | None = None,
        message: str | None = None,
    ) -> None:
        if metric is not None:
            self.metric = metric
        if frame is not None:
            self.frame = frame
        if total is not None:
            self.total = total
        if message is not None:
            self.message = message
        for fn in self._listeners:
            fn(self)

    def format_line(self) -> str:
        elapsed = self.elapsed()
        parts = [f"{elapsed:6.1f}s"]
        if self.metric:
            parts.append(self.metric)
        if self.frame is not None and self.total:
            parts.append(f"frame {self.frame}/{self.total}")
        elif self.frame is not None:
            parts.append(f"frame {self.frame}")
        if self.message:
            parts.append(self.message)
        return " | ".join(parts)


def attach_console(progress: RunProgress, stream: TextIO | None = None) -> None:
    out = stream or sys.stderr

    def _print(p: RunProgress) -> None:
        out.write("\r" + p.format_line() + "    ")
        out.flush()

    progress.on_update(_print)
