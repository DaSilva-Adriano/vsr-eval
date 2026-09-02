"""Launch the configured FFmpeg binary (never PATH) with live progress."""

from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path

from .ffmpeg_check import FFmpegError
from .progress import CancelledError, RunProgress

_FRAME_RE = re.compile(r"frame=\s*(\d+)")


def kill_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass


def run_ffmpeg(
    ffmpeg: str | Path,
    args: list[str],
    *,
    cwd: str | Path | None = None,
    progress: RunProgress | None = None,
    metric: str = "",
    total_frames: int | None = None,
    log_fh=None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Run ffmpeg with an absolute executable. Returns combined stderr text."""
    exe = Path(ffmpeg)
    cmd = [str(exe), "-hide_banner", "-nostdin", "-y", *args]
    if log_fh is not None:
        log_fh.write("CMD: " + " ".join(cmd) + "\n")
        log_fh.flush()

    env = None
    if extra_env:
        import os
        env = os.environ.copy()
        env.update(extra_env)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
    except OSError as exc:
        raise FFmpegError(f"Failed to launch FFmpeg at {exe}: {exc}") from exc

    stderr_chunks: list[str] = []
    lock = threading.Lock()

    def _consume_stderr() -> None:
        assert proc.stderr is not None
        buf = b""
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                n = buf.find(b"\n")
                r = buf.find(b"\r")
                if n < 0 and r < 0:
                    break
                if n < 0:
                    idx = r
                elif r < 0:
                    idx = n
                else:
                    idx = min(n, r)
                line = buf[:idx].decode("utf-8", errors="replace")
                buf = buf[idx + 1 :]
                with lock:
                    stderr_chunks.append(line + "\n")
                if log_fh is not None:
                    log_fh.write(line + "\n")
                if progress is not None:
                    m = _FRAME_RE.search(line)
                    if m:
                        progress.update(
                            metric=metric or progress.metric,
                            frame=int(m.group(1)),
                            total=total_frames,
                            message=line.strip()[:120],
                        )

        if buf:
            line = buf.decode("utf-8", errors="replace")
            with lock:
                stderr_chunks.append(line)
            if log_fh is not None:
                log_fh.write(line)

    reader = threading.Thread(target=_consume_stderr, daemon=True)
    reader.start()

    stdout_data = b""
    try:
        while True:
            if progress is not None:
                progress.check_cancel()
            try:
                proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
    except CancelledError:
        kill_process(proc)
        reader.join(timeout=2)
        raise
    finally:
        if proc.stdout is not None:
            try:
                stdout_data = proc.stdout.read() or b""
            except OSError:
                stdout_data = b""

    reader.join(timeout=5)
    text = "".join(stderr_chunks)
    if stdout_data:
        text += stdout_data.decode("utf-8", errors="replace")

    if proc.returncode not in (0, None) and proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-40:])
        raise FFmpegError(
            f"FFmpeg exited with code {proc.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"{tail}"
        )
    return text


def software_input_args(path: str | Path, codec: str | None = None) -> list[str]:
    """Decode flags that keep metrics on the software HEVC path (no NVDEC)."""
    args = ["-hwaccel", "none"]
    if codec and codec.lower() in {"hevc", "h265"}:
        args.extend(["-c:v", "hevc"])
    args.extend(["-i", str(path)])
    return args


def segment_filters(start_frame: int = 0, max_frames: int | None = None, extra: str | None = None) -> str:
    parts = ["settb=AVTB"]
    trim_bits = []
    sf = int(start_frame or 0)
    if sf:
        trim_bits.append(f"start_frame={sf}")
    if max_frames is not None:
        # FFmpeg 9 trim has start_frame / end_frame (exclusive). nb_frames was removed.
        trim_bits.append(f"end_frame={sf + int(max_frames)}")
    if trim_bits:
        parts.append("trim=" + ":".join(trim_bits))
    parts.append("setpts=PTS-STARTPTS")
    if extra:
        parts.append(extra)
    return ",".join(parts)
