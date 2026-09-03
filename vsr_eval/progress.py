"""Progress reporting and cooperative cancellation."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence, TextIO

from .paths import unique_stem

PENDING = "pending"
RUNNING = "running"
DONE = "done"
ERROR = "error"
SKIPPED = "skipped"

_TERMINAL = {DONE, ERROR, SKIPPED}
_MARK = {
    PENDING: "○",
    RUNNING: "→",
    DONE: "✓",
    ERROR: "✕",
    SKIPPED: "–",
}


@dataclass
class StepState:
    key: str
    label: str
    status: str = PENDING
    error: str = ""


@dataclass
class FileState:
    stem: str
    path: str
    status: str = PENDING
    steps: list[StepState] = field(default_factory=list)
    error: str = ""

    def display_name(self) -> str:
        return Path(self.path).name or self.stem

    def step(self, key: str) -> StepState | None:
        for item in self.steps:
            if item.key == key:
                return item
        return None


def steps_for_metrics(metrics: Sequence[str]) -> list[StepState]:
    """Pipeline work units for one distorted file, matching how scoring actually runs."""
    wanted = {m.strip().lower() for m in metrics if m and m.strip()}
    steps = [StepState("validate", "Validate")]
    ffmpeg_labels = [name for name, key in (("PSNR", "psnr"), ("SSIM", "ssim")) if key in wanted]
    if ffmpeg_labels:
        steps.append(StepState("psnr_ssim", "+".join(ffmpeg_labels)))
    vmaf_labels = [name for name, key in (("VMAF", "vmaf"), ("MS-SSIM", "ms_ssim")) if key in wanted]
    if vmaf_labels:
        steps.append(StepState("vmaf", "+".join(vmaf_labels)))
    perc_labels = [name for name, key in (("LPIPS", "lpips"), ("ERQA", "erqa")) if key in wanted]
    if perc_labels:
        if len(perc_labels) == 2:
            key = "lpips_erqa"
        else:
            key = "lpips" if perc_labels[0] == "LPIPS" else "erqa"
        steps.append(StepState(key, "+".join(perc_labels)))
    return steps


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
    setup: list[StepState] = field(default_factory=list)
    files: list[FileState] = field(default_factory=list)
    current_stem: str | None = None
    finished: bool = False
    _listeners: list[Callable[["RunProgress"], None]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_update(self, fn: Callable[["RunProgress"], None]) -> None:
        with self._lock:
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

    def set_plan(self, dist_paths: Iterable[str | Path], metrics: Sequence[str]) -> None:
        """Build the file × step checklist before work starts."""
        used: set[str] = set()
        files: list[FileState] = []
        for raw in dist_paths:
            path = str(raw)
            stem = unique_stem(path, used)
            files.append(FileState(stem=stem, path=path, steps=steps_for_metrics(metrics)))
        with self._lock:
            self.setup = [
                StepState("ffmpeg", "Check FFmpeg"),
                StepState("probe", "Probe videos"),
                StepState("reports", "Write reports"),
            ]
            self.files = files
            self.current_stem = None
            self.finished = False
        self._notify()

    def begin_setup(self, key: str, message: str | None = None) -> None:
        with self._lock:
            step = _find_step(self.setup, key)
            if step:
                step.status = RUNNING
                self.metric = step.label
            if message is not None:
                self.message = message
        self._notify()

    def finish_setup(self, key: str, status: str = DONE, error: str | None = None) -> None:
        with self._lock:
            step = _find_step(self.setup, key)
            if step:
                step.status = status
                if error:
                    step.error = error
        self._notify()

    def begin_file(self, stem: str) -> None:
        with self._lock:
            self.current_stem = stem
            found = _find_file(self.files, stem)
            if found:
                found.status = RUNNING
            self.message = stem
            self.frame = None
        self._notify()

    def begin_step(self, key: str, message: str | None = None) -> None:
        with self._lock:
            current = _find_file(self.files, self.current_stem)
            if current:
                step = current.step(key)
                if step:
                    step.status = RUNNING
                    self.metric = step.label
            if message is not None:
                self.message = message
            self.frame = 0
        self._notify()

    def finish_step(self, key: str, status: str = DONE, error: str | None = None) -> None:
        with self._lock:
            current = _find_file(self.files, self.current_stem)
            if current:
                step = current.step(key)
                if step:
                    step.status = status
                    if error:
                        step.error = error
                        current.error = current.error or error
        self._notify()

    def finish_file(self, stem: str, status: str = DONE, error: str | None = None) -> None:
        with self._lock:
            found = _find_file(self.files, stem)
            if found:
                found.status = status
                if error:
                    found.error = error
                for step in found.steps:
                    if step.status == RUNNING:
                        step.status = status if status in _TERMINAL else DONE
                        if error and status == ERROR:
                            step.error = step.error or error
                    elif step.status == PENDING:
                        step.status = SKIPPED
        self._notify()

    def complete(self, message: str = "Done") -> None:
        with self._lock:
            self.finished = True
            self.message = message
            self.metric = "done"
            for step in self.setup:
                if step.status == PENDING:
                    step.status = SKIPPED
                elif step.status == RUNNING:
                    step.status = DONE
            for item in self.files:
                if item.status == PENDING:
                    item.status = SKIPPED
                    for step in item.steps:
                        if step.status == PENDING:
                            step.status = SKIPPED
        self._notify()

    def update(
        self,
        metric: str | None = None,
        frame: int | None = None,
        total: int | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            if metric is not None:
                self.metric = metric
            if frame is not None:
                self.frame = frame
            if total is not None:
                self.total = total
            if message is not None:
                self.message = message
        self._notify()

    def counts(self) -> tuple[int, int, int, int, int]:
        """files_done, files_total, steps_done, steps_total, steps_remaining."""
        with self._lock:
            return self._counts_unlocked()

    def completed_file_count(self) -> int:
        with self._lock:
            return sum(1 for item in self.files if item.status in {DONE, ERROR, SKIPPED})

    def terminal_files(self) -> list[FileState]:
        with self._lock:
            return [self._copy_file(item) for item in self.files if item.status in _TERMINAL]

    def format_line(self) -> str:
        elapsed = self.elapsed()
        with self._lock:
            metric = self.metric
            frame = self.frame
            total = self.total
            message = self.message
            files = list(self.files)
            current = self.current_stem
            files_done, files_total, steps_done, steps_total, remaining = self._counts_unlocked()
        parts = [f"{elapsed:6.1f}s"]
        if files_total:
            idx = next((i for i, item in enumerate(files) if item.stem == current), None)
            shown = (idx + 1) if idx is not None else max(files_done, 1)
            parts.append(f"file {shown}/{files_total}")
            if current:
                parts.append(current)
        if metric:
            parts.append(metric)
        if frame is not None and total:
            parts.append(f"frame {frame}/{total}")
        elif frame is not None:
            parts.append(f"frame {frame}")
        if steps_total:
            parts.append(f"{steps_done}/{steps_total} steps")
            if remaining:
                parts.append(f"{remaining} left")
        if message and message not in parts and not message.lower().startswith("frame="):
            parts.append(message)
        return " | ".join(parts)

    def format_markdown(self) -> str:
        elapsed = self.elapsed()
        with self._lock:
            metric = self.metric
            frame = self.frame
            total = self.total
            message = self.message
            files = [self._copy_file(item) for item in self.files]
            setup = [StepState(s.key, s.label, s.status, s.error) for s in self.setup]
            current = self.current_stem
            finished = self.finished
            files_done, files_total, steps_done, steps_total, remaining = self._counts_unlocked()

        if not files and not setup:
            line = self.format_line()
            return line or "Starting…"

        title = "Finished" if finished else ("Cancelled" if self.cancelled() else "Running")
        lines = [f"### {title} — {elapsed:.1f}s"]
        if steps_total:
            bar = _bar(steps_done, steps_total)
            lines.append(
                f"**Files** {files_done}/{files_total} done · "
                f"**Steps** {steps_done}/{steps_total} done · "
                f"**{remaining} remaining**"
            )
            lines.append(f"`{bar}`")
        current_file = next((item for item in files if item.stem == current), None)
        if not finished and current_file and current_file.status == RUNNING:
            now = f"**Now:** `{current_file.display_name()}`"
            running = next((s for s in current_file.steps if s.status == RUNNING), None)
            label = running.label if running else (metric or "")
            if label:
                now += f" — **{label}**"
            if frame is not None and total:
                now += f" · frame {frame}/{total}"
            elif frame is not None:
                now += f" · frame {frame}"
            lines.append("")
            lines.append(now)
        elif not finished and metric:
            lines.append("")
            lines.append(f"**Now:** **{metric}**")
            if frame is not None and total:
                lines[-1] += f" · frame {frame}/{total}"

        detail = (message or "").strip()
        if (
            detail
            and not finished
            and not detail.lower().startswith("frame=")
            and detail != (current_file.stem if current_file else "")
            and detail != (current_file.display_name() if current_file else "")
        ):
            lines.append(f"_{detail}_")

        if setup:
            lines.append("")
            lines.append("#### Setup")
            for step in setup:
                extra = f" — {step.error}" if step.error else ""
                lines.append(f"- {_MARK.get(step.status, '?')} {step.label}{extra}")

        treated = [item for item in files if item.status in {RUNNING, DONE, ERROR, SKIPPED}]
        lines.append("")
        lines.append("#### Files treated")
        if not treated:
            lines.append("_None yet — queued files are under **Still to do**._")
        else:
            for item in treated:
                lines.append(_file_line(item, current_stem=current, frame=frame, total=total))

        todo_lines = _still_to_do(files, setup)
        lines.append("")
        lines.append("#### Still to do")
        if not todo_lines:
            lines.append("_Nothing remaining._")
        else:
            lines.extend(todo_lines)

        return "\n".join(lines)

    def _notify(self) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            fn(self)

    def _counts_unlocked(self) -> tuple[int, int, int, int, int]:
        files_total = len(self.files)
        files_done = sum(1 for item in self.files if item.status in _TERMINAL)
        all_steps: list[StepState] = list(self.setup)
        for item in self.files:
            all_steps.extend(item.steps)
        steps_total = len(all_steps)
        steps_done = sum(1 for step in all_steps if step.status in _TERMINAL)
        remaining = sum(1 for step in all_steps if step.status == PENDING)
        return files_done, files_total, steps_done, steps_total, remaining

    @staticmethod
    def _copy_file(item: FileState) -> FileState:
        return FileState(
            stem=item.stem,
            path=item.path,
            status=item.status,
            error=item.error,
            steps=[StepState(s.key, s.label, s.status, s.error) for s in item.steps],
        )


def _find_step(steps: list[StepState], key: str) -> StepState | None:
    for step in steps:
        if step.key == key:
            return step
    return None


def _find_file(files: list[FileState], stem: str | None) -> FileState | None:
    if not stem:
        return None
    for item in files:
        if item.stem == stem:
            return item
    return None


def _bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return ""
    filled = int(round(width * min(1.0, max(0.0, done / total))))
    filled = min(width, filled)
    return "█" * filled + "░" * (width - filled)


def _file_line(
    item: FileState,
    *,
    current_stem: str | None,
    frame: int | None,
    total: int | None,
) -> str:
    mark = _MARK.get(item.status, "?")
    name = item.display_name()
    heading = f"**`{name}`**" if item.stem == current_stem and item.status == RUNNING else f"`{name}`"
    bits: list[str] = []
    for step in item.steps:
        if item.status == RUNNING:
            if step.status == DONE:
                bits.append(f"{step.label} ✓")
            elif step.status == RUNNING:
                frame_bit = ""
                if frame is not None and total:
                    frame_bit = f" {frame}/{total}"
                elif frame is not None:
                    frame_bit = f" {frame}"
                bits.append(f"**{step.label}**{frame_bit}")
            elif step.status == ERROR:
                bits.append(f"{step.label} ✕")
            elif step.status == SKIPPED:
                bits.append(f"{step.label} skipped")
            else:
                bits.append(f"{step.label} remaining")
        else:
            if step.status == DONE:
                bits.append(step.label)
            elif step.status == ERROR:
                bits.append(f"{step.label} ✕")
            elif step.status == SKIPPED:
                bits.append(f"{step.label} skipped")
            elif step.status == RUNNING:
                bits.append(f"{step.label}…")
            else:
                bits.append(f"{step.label} remaining")
    detail = " — " + " · ".join(bits) if bits else ""
    err = f" — {item.error}" if item.error else ""
    return f"- {mark} {heading}{detail}{err}"


def _still_to_do(files: list[FileState], setup: list[StepState]) -> list[str]:
    lines: list[str] = []
    for item in files:
        pending = [s.label for s in item.steps if s.status == PENDING]
        if pending:
            lines.append(f"- `{item.display_name()}` — {', '.join(pending)}")
    for step in setup:
        if step.status == PENDING:
            lines.append(f"- {step.label}")
    return lines


def attach_console(progress: RunProgress, stream: TextIO | None = None) -> None:
    out = stream or sys.stderr
    state = {"done": 0}

    def _print(p: RunProgress) -> None:
        n = p.completed_file_count()
        if n > state["done"]:
            newly = p.terminal_files()
            for item in newly[state["done"] :]:
                mark = _MARK.get(item.status, "?")
                done_labels = ", ".join(
                    s.label for s in item.steps if s.status == DONE
                ) or item.status
                out.write(f"\n  {mark} {item.stem}  {done_labels}\n")
            state["done"] = n
        out.write("\r" + p.format_line() + "    ")
        out.flush()

    progress.on_update(_print)
