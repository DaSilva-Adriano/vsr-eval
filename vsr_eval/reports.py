"""Write summary/per-frame CSV+JSON, plots, and run_config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .util import dump_json, json_safe


SUMMARY_COLUMNS = [
    "distorted",
    "psnr_y",
    "ssim_y",
    "ms_ssim",
    "vmaf",
    "lpips",
    "erqa",
]


def summary_row(stem: str, result: dict[str, Any]) -> dict[str, Any]:
    psnr = (result.get("psnr") or {}).get("pooled") or {}
    ssim = (result.get("ssim") or {}).get("pooled") or {}
    vmaf = (result.get("vmaf") or {}).get("pooled") or {}
    lpips = (result.get("lpips") or {}).get("pooled") or {}
    erqa = (result.get("erqa") or {}).get("pooled") or {}

    vmaf_score = None
    vblock = vmaf.get("vmaf") if isinstance(vmaf, dict) else None
    if isinstance(vblock, dict):
        vmaf_score = vblock.get("harmonic_mean", vblock.get("mean"))

    ms = None
    msblock = vmaf.get("float_ms_ssim") if isinstance(vmaf, dict) else None
    if isinstance(msblock, dict):
        ms = msblock.get("mean")

    psnr_y = None
    if isinstance(psnr, dict) and isinstance(psnr.get("y"), dict):
        psnr_y = psnr["y"].get("mean")
    ssim_y = None
    if isinstance(ssim, dict) and isinstance(ssim.get("y"), dict):
        ssim_y = ssim["y"].get("mean")

    return {
        "distorted": stem,
        "path": result.get("path"),
        "psnr_y": psnr_y,
        "psnr_u": (psnr.get("u") or {}).get("mean") if isinstance(psnr.get("u"), dict) else None,
        "psnr_v": (psnr.get("v") or {}).get("mean") if isinstance(psnr.get("v"), dict) else None,
        "psnr_avg": (psnr.get("avg") or {}).get("mean") if isinstance(psnr.get("avg"), dict) else None,
        "ssim_y": ssim_y,
        "ssim_u": (ssim.get("u") or {}).get("mean") if isinstance(ssim.get("u"), dict) else None,
        "ssim_v": (ssim.get("v") or {}).get("mean") if isinstance(ssim.get("v"), dict) else None,
        "ssim_all": (ssim.get("all") or {}).get("mean") if isinstance(ssim.get("all"), dict) else None,
        "ms_ssim": ms,
        "vmaf": vmaf_score,
        "vmaf_mean": (vblock or {}).get("mean") if isinstance(vblock, dict) else None,
        "vmaf_harmonic_mean": (vblock or {}).get("harmonic_mean") if isinstance(vblock, dict) else None,
        "lpips": lpips.get("mean"),
        "lpips_min": lpips.get("min"),
        "lpips_max": lpips.get("max"),
        "lpips_p50": lpips.get("p50"),
        "lpips_p95": lpips.get("p95"),
        "lpips_label": (result.get("lpips") or {}).get("label"),
        "erqa": erqa.get("mean"),
        "erqa_min": erqa.get("min"),
        "erqa_max": erqa.get("max"),
        "erqa_p50": erqa.get("p50"),
        "erqa_p95": erqa.get("p95"),
        "erqa_label": (result.get("erqa") or {}).get("label"),
        "warnings": result.get("warnings") or [],
        "errors": result.get("errors") or [],
    }


def per_frame_table(result: dict[str, Any], start_frame: int = 0) -> pd.DataFrame:
    by_frame: dict[int, dict[str, Any]] = {}

    def row(frame: int) -> dict[str, Any]:
        r = by_frame.setdefault(int(frame), {"frame": int(frame)})
        return r

    psnr_frames = ((result.get("psnr") or {}).get("frames")) or []
    for fr in psnr_frames:
        src = start_frame + int(fr["n"]) - 1
        r = row(src)
        r["psnr_y"] = fr.get("psnr_y")
        r["psnr_u"] = fr.get("psnr_u")
        r["psnr_v"] = fr.get("psnr_v")
        r["psnr_avg"] = fr.get("psnr_avg")

    ssim_frames = ((result.get("ssim") or {}).get("frames")) or []
    for fr in ssim_frames:
        src = start_frame + int(fr["n"]) - 1
        r = row(src)
        r["ssim_y"] = fr.get("ssim_y")
        r["ssim_u"] = fr.get("ssim_u")
        r["ssim_v"] = fr.get("ssim_v")
        r["ssim_all"] = fr.get("ssim_all")

    vmaf_frames = ((result.get("vmaf") or {}).get("frames")) or []
    for fr in vmaf_frames:
        src = start_frame + int(fr.get("frameNum", 0))
        r = row(src)
        r["vmaf"] = fr.get("vmaf")
        r["ms_ssim"] = fr.get("float_ms_ssim")

    for fr in ((result.get("lpips") or {}).get("frames")) or []:
        r = row(int(fr["frame"]))
        r["lpips"] = fr.get("lpips")

    for fr in ((result.get("erqa") or {}).get("frames")) or []:
        r = row(int(fr["frame"]))
        r["erqa"] = fr.get("erqa")

    if not by_frame:
        return pd.DataFrame(columns=["frame"])
    df = pd.DataFrame([by_frame[k] for k in sorted(by_frame)])
    cols = [
        "frame", "psnr_y", "psnr_u", "psnr_v", "psnr_avg",
        "ssim_y", "ssim_u", "ssim_v", "ssim_all",
        "ms_ssim", "vmaf", "lpips", "erqa",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def write_dist_outputs(outdir: Path, stem: str, result: dict[str, Any], start_frame: int) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    df = per_frame_table(result, start_frame=start_frame)
    csv_path = outdir / f"{stem}_per_frame.csv"
    df.to_csv(csv_path, index=False)
    paths["per_frame_csv"] = str(csv_path)
    dump_json(outdir / f"{stem}_per_frame.json", df.to_dict(orient="records"))
    return paths


def write_summary(outdir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = outdir / "summary.csv"
    json_path = outdir / "summary.json"
    # Keep a compact view plus extras.
    front = [c for c in SUMMARY_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df[front + rest].to_csv(csv_path, index=False)
    dump_json(json_path, rows)
    return csv_path, json_path


def write_scores_plot(outdir: Path, dist_tables: dict[str, pd.DataFrame]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("psnr_y", "PSNR-Y (higher better)"),
        ("ssim_y", "SSIM-Y (higher better)"),
        ("ms_ssim", "MS-SSIM (higher better)"),
        ("vmaf", "VMAF (higher better)"),
        ("lpips", "LPIPS (lower better)"),
        ("erqa", "ERQA (higher better)"),
    ]
    present = [m for m, _ in metrics if any(m in df.columns and df[m].notna().any() for df in dist_tables.values())]
    n = max(1, len(present))
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    lookup = dict(metrics)
    for ax, key in zip(axes, present):
        for name, df in dist_tables.items():
            if key not in df.columns:
                continue
            sub = df.dropna(subset=[key])
            if sub.empty:
                continue
            ax.plot(sub["frame"], sub[key], label=name, linewidth=1.0)
        ax.set_ylabel(lookup.get(key, key))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    if present:
        axes[-1].set_xlabel("source frame")
    fig.suptitle("VSR-Eval per-frame scores")
    fig.tight_layout()
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / "scores.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plotly_figure(dist_tables: dict[str, pd.DataFrame], metric: str | None = None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    series = [
        ("psnr_y", "PSNR-Y ↑"),
        ("ssim_y", "SSIM-Y ↑"),
        ("ms_ssim", "MS-SSIM ↑"),
        ("vmaf", "VMAF ↑"),
        ("lpips", "LPIPS ↓"),
        ("erqa", "ERQA ↑"),
    ]
    if metric:
        series = [s for s in series if s[0] == metric] or series
    present = [s for s in series if any(s[0] in df.columns and df[s[0]].notna().any() for df in dist_tables.values())]
    if not present:
        fig = go.Figure()
        fig.update_layout(title="No per-frame scores yet")
        return fig
    fig = make_subplots(rows=len(present), cols=1, shared_xaxes=True, subplot_titles=[p[1] for p in present])
    for i, (key, _title) in enumerate(present, start=1):
        for name, df in dist_tables.items():
            if key not in df.columns:
                continue
            sub = df.dropna(subset=[key])
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(x=sub["frame"], y=sub[key], name=f"{name} {key}", mode="lines"),
                row=i, col=1,
            )
    fig.update_layout(height=220 * len(present), legend_tracegroupgap=40, title="Per-frame quality")
    fig.update_xaxes(title_text="source frame", row=len(present), col=1)
    return fig


DIRECTION_NOTE = (
    "**Directionality** — PSNR / SSIM / MS-SSIM / VMAF / ERQA: **higher is better**. "
    "LPIPS: **lower is better**."
)
