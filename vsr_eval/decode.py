"""Paired software HEVC decode to RGB24 via FFmpeg pipes (no PNG dumps)."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import numpy as np

from .ffmpeg_check import FFmpegError
from .ffmpeg_run import kill_process, segment_filters
from .progress import CancelledError, RunProgress


class DecodeError(RuntimeError):
    pass


def _drain_stderr(proc: subprocess.Popen, acc: list[str]) -> None:
    if proc.stderr is None:
        return
    try:
        for chunk in iter(lambda: proc.stderr.read(4096), b""):
            acc.append(chunk.decode("utf-8", errors="replace"))
    except OSError:
        pass


def _rgb_cmd(
    ffmpeg: Path,
    path: Path,
    codec: str | None,
    start_frame: int,
    max_frames: int | None,
    stride: int,
) -> list[str]:
    extra = None
    if stride and stride > 1:
        extra = f"select=eq(mod(n\\,{int(stride)})\\,0)"
    vf = segment_filters(start_frame, max_frames, extra=extra)
    vf = vf + ",format=rgb24"
    cmd = [str(ffmpeg), "-hide_banner", "-nostdin", "-hwaccel", "none"]
    if codec and codec.lower() in {"hevc", "h265"}:
        cmd.extend(["-c:v", "hevc"])
    cmd.extend([
        "-i", str(path),
        "-an", "-sn",
        "-vf", vf,
        "-fps_mode", "passthrough",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ])
    return cmd


class PairedRGBDecoder:
    """Yield aligned RGB uint8 frames (H, W, 3) from ref and dist using software decode."""

    def __init__(
        self,
        ffmpeg: str | Path,
        ref: str | Path,
        dist: str | Path,
        *,
        width: int,
        height: int,
        start_frame: int = 0,
        max_frames: int | None = None,
        stride: int = 1,
        ref_codec: str | None = None,
        dist_codec: str | None = None,
        progress: RunProgress | None = None,
        log_fh=None,
    ) -> None:
        self.ffmpeg = Path(ffmpeg)
        self.ref = Path(ref)
        self.dist = Path(dist)
        self.width = int(width)
        self.height = int(height)
        self.start_frame = int(start_frame)
        self.max_frames = max_frames
        self.stride = max(1, int(stride))
        self.ref_codec = ref_codec
        self.dist_codec = dist_codec
        self.progress = progress
        self.log_fh = log_fh
        self.frame_bytes = self.width * self.height * 3
        self._ref_p: subprocess.Popen | None = None
        self._dist_p: subprocess.Popen | None = None
        self._ref_err: list[str] = []
        self._dist_err: list[str] = []

    def __enter__(self) -> "PairedRGBDecoder":
        ref_cmd = _rgb_cmd(self.ffmpeg, self.ref, self.ref_codec, self.start_frame, self.max_frames, self.stride)
        dist_cmd = _rgb_cmd(self.ffmpeg, self.dist, self.dist_codec, self.start_frame, self.max_frames, self.stride)
        if self.log_fh is not None:
            self.log_fh.write("DECODE REF: " + " ".join(ref_cmd) + "\n")
            self.log_fh.write("DECODE DIST: " + " ".join(dist_cmd) + "\n")
            self.log_fh.flush()
        self._ref_p = subprocess.Popen(
            ref_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_bytes
        )
        self._dist_p = subprocess.Popen(
            dist_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_bytes
        )
        threading.Thread(target=_drain_stderr, args=(self._ref_p, self._ref_err), daemon=True).start()
        threading.Thread(target=_drain_stderr, args=(self._dist_p, self._dist_err), daemon=True).start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for p in (self._ref_p, self._dist_p):
            if p is not None:
                kill_process(p)

    def _read_exact(self, proc: subprocess.Popen, label: str) -> np.ndarray | None:
        assert proc.stdout is not None
        buf = bytearray()
        needed = self.frame_bytes
        while len(buf) < needed:
            if self.progress is not None:
                self.progress.check_cancel()
            chunk = proc.stdout.read(needed - len(buf))
            if not chunk:
                if buf:
                    raise DecodeError(
                        f"{label} ended mid-frame ({len(buf)}/{needed} bytes). "
                        + "".join(self._ref_err[-8:] + self._dist_err[-8:])
                    )
                return None
            buf.extend(chunk)
        arr = np.frombuffer(buf, dtype=np.uint8)
        return arr.reshape((self.height, self.width, 3))

    def frames(self):
        if self._ref_p is None or self._dist_p is None:
            raise DecodeError("decoder not started")
        i = 0
        while True:
            if self.progress is not None:
                self.progress.check_cancel()
            try:
                ref = self._read_exact(self._ref_p, "reference")
                dist = self._read_exact(self._dist_p, "distorted")
            except CancelledError:
                raise
            if ref is None and dist is None:
                break
            if ref is None or dist is None:
                raise DecodeError(
                    "Unbalanced RGB decode: one stream ended before the other. "
                    "Clips may have different frame counts."
                )
            source = self.start_frame + i * self.stride
            yield source, ref, dist
            i += 1
            if self.max_frames is not None and self.stride > 1:
                # trim already limited the parent stream; select keeps every Nth
                pass
            if self.max_frames is not None and self.stride == 1 and i >= self.max_frames:
                break
