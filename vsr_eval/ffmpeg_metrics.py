"""PSNR, SSIM, VMAF and MS-SSIM via FFmpeg libvmaf / psnr / ssim filters."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .ffmpeg_check import FFmpegError
from .ffmpeg_run import run_ffmpeg, segment_filters
from .paths import escape_filter_value, filter_option_path, vmaf_model_file
from .progress import RunProgress
from .util import harmonic_mean, pool_scores

_PSNR_LINE = re.compile(
    r"n:(?P<n>\d+)\s+"
    r"mse_avg:(?P<mse_avg>[\d.]+|inf)\s+"
    r"mse_y:(?P<mse_y>[\d.]+|inf)\s+"
    r"mse_u:(?P<mse_u>[\d.]+|inf)\s+"
    r"mse_v:(?P<mse_v>[\d.]+|inf)\s+"
    r"psnr_avg:(?P<psnr_avg>[\d.]+|inf)\s+"
    r"psnr_y:(?P<psnr_y>[\d.]+|inf)\s+"
    r"psnr_u:(?P<psnr_u>[\d.]+|inf)\s+"
    r"psnr_v:(?P<psnr_v>[\d.]+|inf)",
    re.IGNORECASE,
)
_PSNR_SUM = re.compile(
    r"PSNR\s+y:(?P<y>[\d.]+|inf)\s+u:(?P<u>[\d.]+|inf)\s+v:(?P<v>[\d.]+|inf)\s+"
    r"average:(?P<avg>[\d.]+|inf)",
    re.IGNORECASE,
)
_SSIM_LINE = re.compile(
    r"n:(?P<n>\d+)\s+Y:(?P<y>[\d.]+)\s+\([^)]+\)\s+U:(?P<u>[\d.]+)\s+\([^)]+\)\s+"
    r"V:(?P<v>[\d.]+)\s+\([^)]+\)\s+All:(?P<all>[\d.]+)",
    re.IGNORECASE,
)
_SSIM_LINE_ALT = re.compile(
    r"n:(?P<n>\d+)\s+Y:(?P<y>[\d.]+)\s+U:(?P<u>[\d.]+)\s+V:(?P<v>[\d.]+)\s+All:(?P<all>[\d.]+)",
    re.IGNORECASE,
)
_SSIM_SUM = re.compile(
    r"SSIM\s+Y:(?P<y>[\d.]+)\s+\([^)]+\)\s+U:(?P<u>[\d.]+)\s+\([^)]+\)\s+"
    r"V:(?P<v>[\d.]+)\s+\([^)]+\)\s+All:(?P<all>[\d.]+)",
    re.IGNORECASE,
)


def _f(val: str) -> float:
    v = val.strip().lower()
    if v in {"inf", "+inf", "infinity"}:
        return float("inf")
    if v in {"-inf", "-infinity"}:
        return float("-inf")
    return float(val)


def choose_vmaf_model(model: str, ref_long_side: int) -> tuple[str, str]:
    """Return (model_id, reason). model is 'auto' or a concrete id."""
    if model and model != "auto":
        return model, f"User-selected VMAF model: {model}"
    if ref_long_side >= 2560:
        return "vmaf_4k_v0.6.1", (
            f"Reference long side is {ref_long_side} px (>= 2560). "
            "Defaulting to vmaf_4k_v0.6.1."
        )
    return "vmaf_v0.6.1", (
        f"Reference long side is {ref_long_side} px (< 2560). "
        "Defaulting to vmaf_v0.6.1."
    )


def resolve_vmaf_model_path(model_id: str) -> Path:
    path = vmaf_model_file(model_id)
    if not path.is_file():
        raise FFmpegError(
            f"VMAF model JSON not found: {path}\n"
            "Place the official Netflix model next to the app in .\\models\\"
        )
    return path


def parse_psnr_stats(text: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for raw in text.splitlines():
        m = _PSNR_LINE.search(raw)
        if not m:
            continue
        rows.append({
            "n": int(m.group("n")),
            "mse_avg": _f(m.group("mse_avg")),
            "mse_y": _f(m.group("mse_y")),
            "mse_u": _f(m.group("mse_u")),
            "mse_v": _f(m.group("mse_v")),
            "psnr_avg": _f(m.group("psnr_avg")),
            "psnr_y": _f(m.group("psnr_y")),
            "psnr_u": _f(m.group("psnr_u")),
            "psnr_v": _f(m.group("psnr_v")),
        })
    return rows


def parse_ssim_stats(text: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for raw in text.splitlines():
        m = _SSIM_LINE.search(raw) or _SSIM_LINE_ALT.search(raw)
        if not m:
            continue
        rows.append({
            "n": int(m.group("n")),
            "ssim_y": float(m.group("y")),
            "ssim_u": float(m.group("u")),
            "ssim_v": float(m.group("v")),
            "ssim_all": float(m.group("all")),
        })
    return rows


def _pool_key(rows: list[dict], key: str) -> dict[str, float]:
    return pool_scores([float(r[key]) for r in rows if key in r])


def run_psnr_ssim(
    *,
    ffmpeg: str | Path,
    ref: str | Path,
    dist: str | Path,
    outdir: str | Path,
    stem: str,
    start_frame: int = 0,
    max_frames: int | None = None,
    ref_codec: str | None = None,
    dist_codec: str | None = None,
    want_psnr: bool = True,
    want_ssim: bool = True,
    total_frames: int | None = None,
    progress: RunProgress | None = None,
    log_fh=None,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    psnr_log = outdir / f"{stem}_psnr.log"
    ssim_log = outdir / f"{stem}_ssim.log"

    dist_f = segment_filters(start_frame, max_frames)
    ref_f = segment_filters(start_frame, max_frames)

    labels = []
    chains = [
        f"[0:v]{dist_f}[dist]",
        f"[1:v]{ref_f}[ref]",
    ]
    maps: list[str] = []
    if want_psnr and want_ssim:
        chains.append("[dist]split[distp][dists]")
        chains.append("[ref]split[refp][refs]")
        chains.append(
            f"[distp][refp]psnr=stats_file={escape_filter_value(psnr_log.name)}:eof_action=endall[ps]"
        )
        chains.append(
            f"[dists][refs]ssim=stats_file={escape_filter_value(ssim_log.name)}:eof_action=endall[ss]"
        )
        maps = ["[ps]", "[ss]"]
        labels = ["PSNR", "SSIM"]
    elif want_psnr:
        chains.append(
            f"[dist][ref]psnr=stats_file={escape_filter_value(psnr_log.name)}:eof_action=endall[ps]"
        )
        maps = ["[ps]"]
        labels = ["PSNR"]
    elif want_ssim:
        chains.append(
            f"[dist][ref]ssim=stats_file={escape_filter_value(ssim_log.name)}:eof_action=endall[ss]"
        )
        maps = ["[ss]"]
        labels = ["SSIM"]
    else:
        return {"psnr": None, "ssim": None}

    graph = ";".join(chains)
    args = ["-hwaccel", "none"]
    if dist_codec and dist_codec.lower() in {"hevc", "h265"}:
        args.extend(["-c:v", "hevc"])
    args.extend(["-i", str(dist)])
    args.extend(["-hwaccel", "none"])
    if ref_codec and ref_codec.lower() in {"hevc", "h265"}:
        args.extend(["-c:v", "hevc"])
    args.extend(["-i", str(ref)])
    args.extend(["-filter_complex", graph, "-an", "-sn", "-fps_mode", "passthrough"])
    for m in maps:
        args.extend(["-map", m])
    args.extend(["-f", "null", "-"])

    if progress:
        progress.update(metric="+".join(labels), frame=0, total=total_frames, message="FFmpeg PSNR/SSIM")

    stderr = run_ffmpeg(
        ffmpeg, args, cwd=outdir, progress=progress, metric="+".join(labels),
        total_frames=total_frames, log_fh=log_fh,
    )

    result: dict[str, Any] = {"psnr": None, "ssim": None, "stderr": stderr}

    if want_psnr:
        text = psnr_log.read_text(encoding="utf-8", errors="replace") if psnr_log.is_file() else ""
        frames = parse_psnr_stats(text)
        pooled = {
            "y": _pool_key(frames, "psnr_y") if frames else {},
            "u": _pool_key(frames, "psnr_u") if frames else {},
            "v": _pool_key(frames, "psnr_v") if frames else {},
            "avg": _pool_key(frames, "psnr_avg") if frames else {},
        }
        sm = _PSNR_SUM.search(stderr)
        if sm and not frames:
            pooled = {
                "y": {"mean": _f(sm.group("y"))},
                "u": {"mean": _f(sm.group("u"))},
                "v": {"mean": _f(sm.group("v"))},
                "avg": {"mean": _f(sm.group("avg"))},
            }
        elif sm:
            for k, g in (("y", "y"), ("u", "u"), ("v", "v"), ("avg", "avg")):
                pooled[k]["ffmpeg_summary"] = _f(sm.group(g))
        result["psnr"] = {"frames": frames, "pooled": pooled, "stats_file": str(psnr_log)}

    if want_ssim:
        text = ssim_log.read_text(encoding="utf-8", errors="replace") if ssim_log.is_file() else ""
        frames = parse_ssim_stats(text)
        pooled = {
            "y": _pool_key(frames, "ssim_y") if frames else {},
            "u": _pool_key(frames, "ssim_u") if frames else {},
            "v": _pool_key(frames, "ssim_v") if frames else {},
            "all": _pool_key(frames, "ssim_all") if frames else {},
        }
        sm = _SSIM_SUM.search(stderr)
        if sm and not frames:
            pooled = {
                "y": {"mean": float(sm.group("y"))},
                "u": {"mean": float(sm.group("u"))},
                "v": {"mean": float(sm.group("v"))},
                "all": {"mean": float(sm.group("all"))},
            }
        result["ssim"] = {"frames": frames, "pooled": pooled, "stats_file": str(ssim_log)}

    return result


def run_vmaf_msssim(
    *,
    ffmpeg: str | Path,
    ref: str | Path,
    dist: str | Path,
    outdir: str | Path,
    stem: str,
    model_id: str,
    start_frame: int = 0,
    max_frames: int | None = None,
    ref_codec: str | None = None,
    dist_codec: str | None = None,
    scale_distorted: bool = False,
    ref_width: int | None = None,
    ref_height: int | None = None,
    total_frames: int | None = None,
    progress: RunProgress | None = None,
    log_fh=None,
    n_threads: int | None = None,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_name = f"{stem}_vmaf.json"
    log_path = outdir / log_name
    model_path = resolve_vmaf_model_path(model_id)

    threads = n_threads or (os.cpu_count() or 8)
    dist_extra = None
    if scale_distorted:
        if not ref_width or not ref_height:
            raise FFmpegError("scale_distorted requires reference width/height")
        dist_extra = f"scale={int(ref_width)}:{int(ref_height)}:flags=bicubic"
    dist_f = segment_filters(start_frame, max_frames, extra=dist_extra)
    ref_f = segment_filters(start_frame, max_frames)

    # Prefer a same-drive relative path so libvmaf never sees a Windows "C:" colon.
    # If that is impossible (other drive), copy the JSON next to the log as a bare filename.
    model_rel = filter_option_path(model_path, outdir)
    if ":" in model_rel:
        local_model = outdir / f"_{model_id}.json"
        shutil.copy2(model_path, local_model)
        model_rel = local_model.name
    opts = {
        "model": f"path={model_rel}",
        "n_threads": str(int(threads)),
        "log_fmt": "json",
        "log_path": log_name,
        "feature": "name=float_ms_ssim",
        "eof_action": "endall",
        "n_subsample": "1",
        "pool": "harmonic_mean",
    }
    libvmaf = "libvmaf=" + ":".join(
        f"{k}={escape_filter_value(v)}" for k, v in opts.items()
    )
    graph = f"[0:v]{dist_f}[dist];[1:v]{ref_f}[ref];[dist][ref]{libvmaf}"

    args: list[str] = ["-hwaccel", "none"]
    if dist_codec and dist_codec.lower() in {"hevc", "h265"}:
        args.extend(["-c:v", "hevc"])
    args.extend(["-i", str(dist)])
    args.extend(["-hwaccel", "none"])
    if ref_codec and ref_codec.lower() in {"hevc", "h265"}:
        args.extend(["-c:v", "hevc"])
    args.extend(["-i", str(ref)])
    args.extend([
        "-filter_complex", graph,
        "-an", "-sn", "-fps_mode", "passthrough",
        "-f", "null", "-",
    ])

    if progress:
        progress.update(metric="VMAF+MS-SSIM", frame=0, total=total_frames, message=f"libvmaf {model_id}")

    try:
        stderr = run_ffmpeg(
            ffmpeg, args, cwd=outdir, progress=progress, metric="VMAF+MS-SSIM",
            total_frames=total_frames, log_fh=log_fh,
        )
    except FFmpegError as exc:
        # Fallback: Gyan/libvmaf built-in models, no filesystem path.
        fallback = dict(opts)
        fallback["model"] = f"version={model_id}"
        libvmaf_fb = "libvmaf=" + ":".join(
            f"{k}={escape_filter_value(v)}" for k, v in fallback.items()
        )
        graph_fb = f"[0:v]{dist_f}[dist];[1:v]{ref_f}[ref];[dist][ref]{libvmaf_fb}"
        args_fb = list(args)
        idx = args_fb.index("-filter_complex")
        args_fb[idx + 1] = graph_fb
        if log_fh is not None:
            log_fh.write(f"libvmaf path= failed ({exc}); retrying model=version={model_id}\n")
        stderr = run_ffmpeg(
            ffmpeg, args_fb, cwd=outdir, progress=progress, metric="VMAF+MS-SSIM",
            total_frames=total_frames, log_fh=log_fh,
        )

    if not log_path.is_file():
        raise FFmpegError(
            f"libvmaf did not write {log_path}.\n"
            f"stderr tail:\n" + "\n".join(stderr.splitlines()[-30:])
        )

    parsed = parse_vmaf_json(log_path)
    parsed["stderr"] = stderr
    parsed["model_id"] = model_id
    parsed["model_path"] = str(model_path)
    parsed["scaled_distorted"] = bool(scale_distorted)
    parsed["n_threads"] = threads
    parsed["log_path"] = str(log_path)
    return parsed


def parse_vmaf_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames_out: list[dict[str, Any]] = []
    vmaf_vals: list[float] = []
    ms_vals: list[float] = []
    for fr in data.get("frames") or []:
        metrics = fr.get("metrics") or {}
        vmaf = _metric_from(metrics, "vmaf")
        ms = _metric_from(metrics, "float_ms_ssim") or _metric_from(metrics, "ms_ssim")
        n = fr.get("frameNum", len(frames_out))
        row = {"frameNum": n, "vmaf": vmaf, "float_ms_ssim": ms, "metrics": metrics}
        frames_out.append(row)
        if vmaf is not None:
            vmaf_vals.append(float(vmaf))
        if ms is not None:
            ms_vals.append(float(ms))

    pooled_src = data.get("pooled_metrics") or {}
    vmaf_pooled = _pooled_block(pooled_src, "vmaf", vmaf_vals)
    ms_pooled = _pooled_block(pooled_src, "float_ms_ssim", ms_vals)
    if not ms_pooled.get("mean") and ms_vals:
        ms_pooled = {**pool_scores(ms_vals), "harmonic_mean": harmonic_mean(ms_vals)}
    if vmaf_vals and "harmonic_mean" not in vmaf_pooled:
        vmaf_pooled["harmonic_mean"] = harmonic_mean(vmaf_vals)

    return {
        "frames": frames_out,
        "pooled": {"vmaf": vmaf_pooled, "float_ms_ssim": ms_pooled},
        "raw_pooled_metrics": pooled_src,
        "version": data.get("version"),
    }


def _metric_from(metrics: dict, name: str) -> float | None:
    if name in metrics and metrics[name] is not None:
        return float(metrics[name])
    for k, v in metrics.items():
        if k.lower() == name.lower() or k.lower().endswith("." + name.lower()):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    # named model e.g. vmaf_v0.6.1
    if name == "vmaf":
        for k, v in metrics.items():
            kl = k.lower()
            if "vmaf" in kl and "neg" not in kl and v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
    return None


def _pooled_block(src: dict, name: str, values: list[float]) -> dict[str, float]:
    block = {}
    raw = src.get(name) or {}
    if not raw:
        for k, v in src.items():
            if name in k.lower():
                raw = v
                break
    if isinstance(raw, dict):
        for key in ("min", "max", "mean", "harmonic_mean"):
            if key in raw and raw[key] is not None:
                try:
                    block[key] = float(raw[key])
                except (TypeError, ValueError):
                    pass
    if values:
        computed = pool_scores(values)
        for k, v in computed.items():
            block.setdefault(k, v)
        block.setdefault("harmonic_mean", harmonic_mean(values))
    return block
