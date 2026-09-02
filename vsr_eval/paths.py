"""Filesystem helpers, including FFmpeg filter-graph path escaping for Windows."""

from __future__ import annotations

import os
import re
from pathlib import Path

_FILTER_SPECIAL = set("\\:'[],;")


def app_root() -> Path:
    """Repository / install root (parent of the vsr_eval package)."""
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    return app_root() / "models"


def vmaf_model_file(name: str) -> Path:
    filename = name if name.endswith(".json") else f"{name}.json"
    return models_dir() / filename


def escape_filter_value(value: str) -> str:
    """Escape a value so FFmpeg will not treat `:`, `\\`, quotes, or brackets as syntax."""
    return "".join(("\\" + ch) if ch in _FILTER_SPECIAL else ch for ch in value)


def to_ffmpeg_path(path: str | Path) -> str:
    """Forward-slash absolute path suitable as a libvmaf/model/stats value (unescaped)."""
    return Path(path).resolve().as_posix()


def escaped_filter_path(path: str | Path) -> str:
    """Windows-safe escaped path for use *inside* an FFmpeg filter option value."""
    return escape_filter_value(to_ffmpeg_path(path))


def filter_option_path(path: str | Path, cwd: str | Path | None = None) -> str:
    """Path for an FFmpeg filter option, preferring a relative posix path with no drive colon.

    libvmaf on Windows treats the colon in ``C:/...`` as an option separator unless it is
    escaped *and* survives CreateProcess. A same-drive relative path (``../models/foo.json``)
    avoids that entirely.
    """
    p = Path(path).resolve()
    if cwd is not None:
        c = Path(cwd).resolve()
        try:
            if p.drive.lower() == c.drive.lower():
                rel = os.path.relpath(p, c)
                posix = Path(rel).as_posix()
                if ":" not in posix:
                    return posix
        except Exception:
            pass
    return to_ffmpeg_path(p)


def quote_filter_arg(value: str) -> str:
    """Wrap a filter option value in FFmpeg single quotes (colons stay literal)."""
    return "'" + str(value).replace("'", r"\'") + "'"


def safe_stem(path: str | Path) -> str:
    stem = Path(path).stem
    cleaned = re.sub(r"[^\w.\-]+", "_", stem, flags=re.UNICODE)
    return cleaned or "dist"


def unique_stem(path: str | Path, used: set[str]) -> str:
    base = safe_stem(path)
    name = base
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name
