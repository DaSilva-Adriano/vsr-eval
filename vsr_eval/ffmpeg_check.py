"""Startup FFmpeg capability check (libvmaf / psnr / ssim)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_FILTERS = ("psnr", "ssim", "libvmaf")


class FFmpegError(RuntimeError):
    pass


@dataclass
class FFmpegStatus:
    ffmpeg_path: Path
    ok: bool
    version_line: str = ""
    filters_found: dict[str, bool] = field(default_factory=dict)
    error: str = ""
    raw_version: str = ""

    def require(self) -> None:
        if not self.ok:
            raise FFmpegError(self.error or "FFmpeg check failed")


def _run(exe: Path, args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    if not exe.is_file():
        raise FFmpegError(
            f"FFmpeg executable not found:\n  {exe}\n"
            "Set the absolute path in settings (do not rely on PATH)."
        )
    try:
        return subprocess.run(
            [str(exe), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        raise FFmpegError(f"Failed to launch {exe}: {exc}") from exc


def check_ffmpeg(ffmpeg_path: str | Path) -> FFmpegStatus:
    exe = Path(ffmpeg_path)
    status = FFmpegStatus(ffmpeg_path=exe, ok=False, filters_found={f: False for f in REQUIRED_FILTERS})
    try:
        ver = _run(exe, ["-version"])
    except FFmpegError as exc:
        status.error = str(exc)
        return status
    blob = (ver.stdout or "") + (ver.stderr or "")
    status.raw_version = blob
    first = next((ln.strip() for ln in blob.splitlines() if ln.strip()), "")
    status.version_line = first
    if ver.returncode != 0 and not first.lower().startswith("ffmpeg version"):
        status.error = (
            f"`ffmpeg -version` failed (exit {ver.returncode}).\n"
            f"{blob.strip() or '(no output)'}"
        )
        return status

    try:
        filt = _run(exe, ["-hide_banner", "-filters"], timeout=60)
    except FFmpegError as extra:
        status.error = str(extra)
        return status
    filt_text = (filt.stdout or "") + (filt.stderr or "")
    for name in REQUIRED_FILTERS:
        # Filter table: flags name description. Match a dedicated column token.
        found = False
        for line in filt_text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == name:
                found = True
                break
        status.filters_found[name] = found

    missing = [n for n, ok in status.filters_found.items() if not ok]
    if missing:
        status.error = (
            f"This FFmpeg build is missing required video filters: {', '.join(missing)}.\n"
            f"Path: {exe}\n"
            "Need a full build with --enable-libvmaf (Gyan full_build is expected).\n"
            "PSNR/SSIM/VMAF cannot run until libvmaf, psnr, and ssim are listed in `ffmpeg -filters`."
        )
        status.ok = False
        return status

    if "libvmaf" not in (ver.stdout + ver.stderr) and "--enable-libvmaf" not in blob:
        # Version banner on Gyan includes --enable-libvmaf; still OK if the filter exists.
        pass

    status.ok = True
    return status


def check_ffprobe(ffprobe_path: str | Path) -> None:
    exe = Path(ffprobe_path)
    if not exe.is_file():
        raise FFmpegError(
            f"FFprobe executable not found:\n  {exe}\n"
            "Set the absolute path in settings (do not rely on PATH)."
        )
    proc = _run(exe, ["-version"])
    if proc.returncode != 0 and "ffprobe" not in ((proc.stdout or "") + (proc.stderr or "")).lower():
        raise FFmpegError(f"`ffprobe -version` failed for {exe}")
