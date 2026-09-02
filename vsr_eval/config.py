"""Persisted settings in %APPDATA%\\VSR-Eval\\config.json."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_FFMPEG = r"C:\VSR\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
DEFAULT_FFPROBE = r"C:\VSR\ffmpeg-9.0.1-full_build\bin\ffprobe.exe"

DEFAULT_METRICS = ["psnr", "ssim", "ms_ssim", "vmaf", "lpips", "erqa"]
VMAF_MODELS = ("vmaf_v0.6.1", "vmaf_4k_v0.6.1")
LPIPS_NETS = ("alex", "vgg")
ERQA_STRIDES = (1, 2, 4, 8)


def config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "VSR-Eval"


def config_path() -> Path:
    return config_dir() / "config.json"


@dataclass
class AppConfig:
    ffmpeg_path: str = DEFAULT_FFMPEG
    ffprobe_path: str = DEFAULT_FFPROBE
    default_metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    vmaf_model: str = "auto"
    lpips_net: str = "alex"
    erqa_stride: int = 1
    lpips_batch_size: int = 4
    scale_distorted_for_vmaf: bool = False
    last_outdir: str = ""

    def ffmpeg(self) -> Path:
        return Path(self.ffmpeg_path)

    def ffprobe(self) -> Path:
        return Path(self.ffprobe_path)


def load_config() -> AppConfig:
    path = config_path()
    if not path.is_file():
        return AppConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppConfig()
    cfg = AppConfig()
    for key in asdict(cfg):
        if key in data and data[key] is not None:
            setattr(cfg, key, data[key])
    if not isinstance(cfg.default_metrics, list) or not cfg.default_metrics:
        cfg.default_metrics = list(DEFAULT_METRICS)
    if cfg.vmaf_model not in ("auto", *VMAF_MODELS):
        cfg.vmaf_model = "auto"
    if cfg.lpips_net not in LPIPS_NETS:
        cfg.lpips_net = "alex"
    try:
        cfg.erqa_stride = int(cfg.erqa_stride)
    except (TypeError, ValueError):
        cfg.erqa_stride = 1
    if cfg.erqa_stride not in ERQA_STRIDES:
        cfg.erqa_stride = 1
    return cfg


def save_config(cfg: AppConfig) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
    return path
