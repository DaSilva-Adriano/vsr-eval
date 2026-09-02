"""ffprobe metadata and pre-score validation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ffmpeg_check import FFmpegError
from .util import parse_fraction

PIX_FMT_REQUIRED = "yuv420p"
FPS_TOLERANCE = 0.05
HEVC_CODECS = frozenset({"hevc", "h265", "hev1", "hvc1"})


@dataclass
class VideoInfo:
    path: str
    exists: bool
    size_bytes: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    fps_raw: str = ""
    frame_count: int | None = None
    frame_count_estimated: bool = False
    width: int | None = None
    height: int | None = None
    pix_fmt: str = ""
    codec: str = ""
    color_range: str = ""
    color_space: str = ""
    bit_rate: str = ""
    error: str = ""

    @property
    def long_side(self) -> int:
        return max(self.width or 0, self.height or 0)

    def label(self) -> str:
        w = self.width or "?"
        h = self.height or "?"
        fps = f"{self.fps:.3f}" if self.fps else "?"
        n = self.frame_count if self.frame_count is not None else "?"
        est = "~" if self.frame_count_estimated else ""
        return f"{w}x{h}  {self.pix_fmt or '?'}  {self.codec or '?'}  {fps} fps  {est}{n} frames"


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    code: str
    message: str


@dataclass
class PairValidation:
    ref: VideoInfo
    dist: VideoInfo
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _ffprobe_json(ffprobe: Path, args: list[str], timeout: int = 60) -> dict:
    proc = subprocess.run(
        [str(ffprobe), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise FFmpegError(f"ffprobe failed for {args[-1] if args else ''}:\n{err}")
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON: {exc}") from exc


def probe_video(ffprobe: str | Path, path: str | Path) -> VideoInfo:
    p = Path(path)
    info = VideoInfo(path=str(p), exists=p.is_file())
    if not info.exists:
        info.error = f"File not found: {p}"
        return info
    try:
        info.size_bytes = p.stat().st_size
    except OSError:
        info.size_bytes = None

    exe = Path(ffprobe)
    try:
        data = _ffprobe_json(
            exe,
            [
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries",
                "stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,"
                "nb_frames,nb_read_packets,duration,color_range,color_space,"
                "color_transfer,color_primaries,bits_per_raw_sample",
                "-show_entries", "format=duration,size,bit_rate,filename",
                "-of", "json",
                str(p),
            ],
        )
    except (FFmpegError, subprocess.TimeoutExpired, OSError) as exc:
        info.error = str(exc)
        return info

    streams = data.get("stream") or data.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = data.get("format") or {}

    info.codec = str(stream.get("codec_name") or "")
    info.width = int(stream["width"]) if stream.get("width") else None
    info.height = int(stream["height"]) if stream.get("height") else None
    info.pix_fmt = str(stream.get("pix_fmt") or "")
    info.color_range = str(stream.get("color_range") or "")
    info.color_space = str(stream.get("color_space") or "")
    info.bit_rate = str(fmt.get("bit_rate") or stream.get("bit_rate") or "")

    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate"))
    info.fps = fps
    info.fps_raw = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")

    dur = stream.get("duration") or fmt.get("duration")
    try:
        info.duration_s = float(dur) if dur not in (None, "N/A") else None
    except (TypeError, ValueError):
        info.duration_s = None

    nb = stream.get("nb_frames")
    if nb not in (None, "N/A", "0", 0, "0"):
        try:
            info.frame_count = int(nb)
        except (TypeError, ValueError):
            info.frame_count = None

    if info.frame_count is None:
        # Packet count is cheaper than a full decode.
        try:
            counted = _ffprobe_json(
                exe,
                [
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-count_packets",
                    "-show_entries", "stream=nb_read_packets",
                    "-of", "json",
                    str(p),
                ],
                timeout=180,
            )
            st = (counted.get("streams") or counted.get("stream") or [{}])[0]
            nbp = st.get("nb_read_packets")
            if nbp not in (None, "N/A"):
                info.frame_count = int(nbp)
        except Exception:
            if info.duration_s and info.fps:
                info.frame_count = int(round(info.duration_s * info.fps))
                info.frame_count_estimated = True

    if info.frame_count is None and info.duration_s and info.fps:
        info.frame_count = int(round(info.duration_s * info.fps))
        info.frame_count_estimated = True

    return info


def validate_pair(
    ref: VideoInfo,
    dist: VideoInfo,
    *,
    scale_distorted_for_vmaf: bool = False,
    metrics: list[str] | None = None,
) -> PairValidation:
    result = PairValidation(ref=ref, dist=dist)
    metrics = list(metrics or [])

    if ref.error:
        result.issues.append(ValidationIssue("error", "ref_unreadable", ref.error))
    if dist.error:
        result.issues.append(ValidationIssue("error", "dist_unreadable", dist.error))
    if result.errors:
        return result

    if ref.pix_fmt and ref.pix_fmt != PIX_FMT_REQUIRED:
        result.issues.append(ValidationIssue(
            "error", "ref_pix_fmt",
            f"Reference pix_fmt is {ref.pix_fmt!r}, expected {PIX_FMT_REQUIRED} (8-bit 4:2:0).",
        ))
    if dist.pix_fmt and dist.pix_fmt != PIX_FMT_REQUIRED:
        result.issues.append(ValidationIssue(
            "error", "dist_pix_fmt",
            f"Distorted pix_fmt is {dist.pix_fmt!r}, expected {PIX_FMT_REQUIRED} (8-bit 4:2:0).",
        ))

    if ref.codec and ref.codec.lower() not in HEVC_CODECS:
        result.issues.append(ValidationIssue(
            "warning", "ref_codec",
            f"Reference codec is {ref.codec!r}; this tool expects libx265 / HEVC. Decode will still be attempted.",
        ))
    if dist.codec and dist.codec.lower() not in HEVC_CODECS:
        result.issues.append(ValidationIssue(
            "warning", "dist_codec",
            f"Distorted codec is {dist.codec!r}; this tool expects libx265 / HEVC. Decode will still be attempted.",
        ))

    if ref.fps and dist.fps and abs(ref.fps - dist.fps) > FPS_TOLERANCE:
        result.issues.append(ValidationIssue(
            "error", "fps_mismatch",
            f"Frame rates differ: ref {ref.fps:.4f} fps vs dist {dist.fps:.4f} fps "
            f"(tolerance {FPS_TOLERANCE}). Clips must already be aligned.",
        ))

    if (
        ref.frame_count is not None
        and dist.frame_count is not None
        and ref.frame_count != dist.frame_count
    ):
        result.issues.append(ValidationIssue(
            "error" if not (ref.frame_count_estimated or dist.frame_count_estimated) else "warning",
            "frame_count_mismatch",
            f"Frame counts differ: ref {ref.frame_count} vs dist {dist.frame_count}. "
            "Use start/max-frames only if you are sure the segments are aligned.",
        ))

    res_differ = (
        ref.width and dist.width and ref.height and dist.height
        and (ref.width != dist.width or ref.height != dist.height)
    )
    if res_differ:
        spatial = {"psnr", "ssim", "ms_ssim", "lpips", "erqa"}
        needs_match = spatial.intersection(metrics) if metrics else spatial
        if needs_match:
            result.issues.append(ValidationIssue(
                "error", "resolution_mismatch",
                f"Resolutions differ: ref {ref.width}x{ref.height} vs dist {dist.width}x{dist.height}. "
                "PSNR/SSIM/MS-SSIM/LPIPS/ERQA require identical frames. "
                "Do not stretch silently. For VMAF only, enable "
                "'scale distorted to reference (bicubic, VMAF recommended)'.",
            ))
        elif "vmaf" in metrics and not scale_distorted_for_vmaf:
            result.issues.append(ValidationIssue(
                "error", "vmaf_scale_required",
                f"Resolutions differ: ref {ref.width}x{ref.height} vs dist {dist.width}x{dist.height}. "
                "Enable 'scale distorted to reference (bicubic, VMAF recommended)' to run VMAF. "
                "The reference will never be scaled down.",
            ))
        elif "vmaf" in metrics and scale_distorted_for_vmaf:
            result.issues.append(ValidationIssue(
                "warning", "vmaf_scale",
                f"Distorted {dist.width}x{dist.height} will be scaled to {ref.width}x{ref.height} "
                "with bicubic for VMAF only. Reference is not scaled.",
            ))

    return result


def info_to_dict(info: VideoInfo) -> dict:
    return asdict(info)
