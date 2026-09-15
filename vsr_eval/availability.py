"""Detect which metrics can actually run on this machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig
from .ffmpeg_check import check_ffmpeg, check_ffprobe
from .paths import vmaf_model_file


@dataclass
class MetricAvailability:
    name: str
    available: bool
    reason: str = ""


@dataclass
class ToolStatus:
    ffmpeg_ok: bool = False
    ffmpeg_version: str = ""
    ffmpeg_error: str = ""
    ffprobe_ok: bool = False
    ffprobe_error: str = ""
    cuda_ok: bool = False
    cuda_device: str = ""
    torch_version: str = ""
    metrics: dict[str, MetricAvailability] = field(default_factory=dict)

    def enabled(self, name: str) -> bool:
        m = self.metrics.get(name)
        return bool(m and m.available)


def _lpips_status() -> MetricAvailability:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return MetricAvailability(
            "lpips", False,
            f"PyTorch is not installed ({exc}). Install the CUDA wheel first (see README).",
        )
    import torch
    tv = getattr(torch, "__version__", "?")
    if not torch.cuda.is_available():
        return MetricAvailability(
            "lpips", False,
            "LPIPS requires CUDA. torch.cuda.is_available() is False. "
            f"You likely installed a CPU wheel of PyTorch ({tv}). "
            "Uninstall torch/torchvision and install the CUDA build from pytorch.org "
            "(this PC: RTX 4080 Super).",
        )
    try:
        name = torch.cuda.get_device_name(0)
    except Exception as exc:
        return MetricAvailability("lpips", False, f"CUDA device query failed: {exc}")
    try:
        import lpips  # noqa: F401
    except Exception as exc:
        return MetricAvailability(
            "lpips", False,
            f"CUDA is available ({name}) but the official `lpips` package failed to import: {exc}",
        )
    return MetricAvailability("lpips", True, f"CUDA GPU: {name} | torch {tv}")


def _erqa_status() -> MetricAvailability:
    try:
        import erqa  # noqa: F401
    except Exception as exc:
        return MetricAvailability(
            "erqa", False,
            f"Official `erqa` package is not importable ({exc}). pip install erqa",
        )
    try:
        import cv2  # noqa: F401
    except Exception as exc:
        return MetricAvailability("erqa", False, f"ERQA needs OpenCV (`opencv-python`): {exc}")
    return MetricAvailability("erqa", True, "erqa + OpenCV available (CPU)")


def probe_tools(cfg: AppConfig) -> ToolStatus:
    status = ToolStatus()
    ff = check_ffmpeg(cfg.ffmpeg_path)
    status.ffmpeg_ok = ff.ok
    status.ffmpeg_version = ff.version_line
    status.ffmpeg_error = ff.error
    try:
        check_ffprobe(cfg.ffprobe_path)
        status.ffprobe_ok = True
    except Exception as exc:
        status.ffprobe_ok = False
        status.ffprobe_error = str(exc)

    lp = _lpips_status()
    status.cuda_ok = lp.available
    if lp.available:
        status.cuda_device = lp.reason
        try:
            import torch
            status.torch_version = torch.__version__
            status.cuda_device = torch.cuda.get_device_name(0)
        except Exception:
            pass

    ffmpeg_metrics_reason = ""
    if not ff.ok:
        ffmpeg_metrics_reason = ff.error or "FFmpeg check failed"
    status.metrics["psnr"] = MetricAvailability(
        "psnr", ff.ok and ff.filters_found.get("psnr", False),
        "FFmpeg psnr filter" if ff.ok else ffmpeg_metrics_reason,
    )
    status.metrics["ssim"] = MetricAvailability(
        "ssim", ff.ok and ff.filters_found.get("ssim", False),
        "FFmpeg ssim filter" if ff.ok else ffmpeg_metrics_reason,
    )
    vmaf_ok = ff.ok and ff.filters_found.get("libvmaf", False)
    model_ok = vmaf_model_file("vmaf_v0.6.1").is_file()
    v1_ok = vmaf_model_file("vmaf_v1.0.16_3d0h").is_file()
    vmaf_reason = "FFmpeg libvmaf + shipped v0/v1 JSON models"
    if vmaf_ok and not model_ok:
        vmaf_reason = (
            "libvmaf is present but .\\models\\vmaf_v0.6.1.json is missing. "
            "Built-in version=vmaf_v0.6.1 may still work; ship the JSON models."
        )
    elif vmaf_ok and not v1_ok:
        vmaf_reason = "FFmpeg libvmaf + v0 models (v1 JSON missing from .\\models\\)"
    elif not vmaf_ok:
        vmaf_reason = ffmpeg_metrics_reason or "libvmaf filter missing"
    status.metrics["vmaf"] = MetricAvailability("vmaf", vmaf_ok, vmaf_reason)
    status.metrics["ms_ssim"] = MetricAvailability(
        "ms_ssim", vmaf_ok,
        "MS-SSIM via libvmaf feature=name=float_ms_ssim" if vmaf_ok else vmaf_reason,
    )
    status.metrics["lpips"] = lp
    status.metrics["erqa"] = _erqa_status()
    return status
