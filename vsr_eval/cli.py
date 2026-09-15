"""Command-line interface: python -m vsr_eval --ref REF.mp4 --dist DIST.mp4 --out OUTDIR"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ALL_METRICS, __version__
from .config import AppConfig, load_config, save_config
from .pipeline import EvalError, RunRequest, run_evaluation
from .progress import CancelledError, RunProgress, attach_console
from .recover import RecoverError, recover_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vsr_eval",
        description="VSR-Eval — full-reference quality metrics for video super-resolution / upscaling.",
    )
    p.add_argument("--gui", action="store_true", help="Launch the local Gradio UI on 127.0.0.1")
    p.add_argument("--ref", type=str, help="Reference HR MP4 (ground truth, libx265 yuv420p)")
    p.add_argument(
        "--dist", action="append", default=[], dest="dists",
        help="Distorted / upscaled MP4. Repeat for multiple files.",
    )
    p.add_argument(
        "--metrics",
        default="psnr,ssim,ms_ssim,vmaf,lpips,erqa",
        help="Comma-separated subset of: " + ",".join(ALL_METRICS),
    )
    p.add_argument("--out", type=str, help="Output directory")
    p.add_argument("--start-time", type=float, default=None, help="Segment start in seconds")
    p.add_argument("--start-frame", type=int, default=0, help="Segment start frame (overrides --start-time if > 0)")
    p.add_argument("--max-frames", type=int, default=None, help="Compare at most N frames")
    p.add_argument("--stride", type=int, default=1, help="Frame stride for LPIPS/ERQA (1, 2, 4, or 8)")
    p.add_argument("--vmaf-model", default="auto", choices=["auto", "vmaf_v0.6.1", "vmaf_4k_v0.6.1"])
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg"])
    p.add_argument(
        "--scale-distorted", action="store_true",
        help="Scale distorted to reference with bicubic before VMAF (never scale the reference down)",
    )
    p.add_argument("--erqa-vis", action="store_true", help="Export ERQA visualization masks for a frame range")
    p.add_argument("--erqa-vis-start", type=int, default=None)
    p.add_argument("--erqa-vis-end", type=int, default=None)
    p.add_argument("--erqa-vis-video", action="store_true")
    p.add_argument("--ffmpeg", type=str, default=None, help="Absolute ffmpeg.exe path (overrides settings)")
    p.add_argument("--ffprobe", type=str, default=None, help="Absolute ffprobe.exe path (overrides settings)")
    p.add_argument("--lpips-batch", type=int, default=4)
    p.add_argument("--self-test", action="store_true", help="Run the synthetic two-frame unit test and exit")
    p.add_argument(
        "--recover",
        action="store_true",
        help="Rebuild summary.csv/json from existing per-file outputs in --out. "
             "Skips the last treated file (it may be incomplete after an interruption).",
    )
    p.add_argument("--save-settings", action="store_true", help="Write current ffmpeg/metric defaults to %%APPDATA%%\\VSR-Eval\\config.json")
    p.add_argument("--version", action="version", version=f"VSR-Eval {__version__}")
    return p


def parse_metrics(raw: str) -> list[str]:
    parts = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    unknown = [p for p in parts if p not in ALL_METRICS]
    if unknown:
        raise SystemExit(f"Unknown metrics: {unknown}. Choose from {list(ALL_METRICS)}")
    if not parts:
        raise SystemExit("No metrics selected.")
    return parts


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        from .selftest import run_standalone
        return run_standalone()

    if args.recover:
        if not args.out:
            parser.error("--recover requires --out (the interrupted run's output directory).")
        try:
            result = recover_summary(Path(args.out))
            try:
                from .history import save_run
                save_run(result)
            except Exception:
                pass
        except RecoverError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 1
        print()
        print(f"Recovered: {result['outdir']}")
        print(result.get("vmaf_model_reason") or "")
        _print_result_table(result)
        return 0

    if args.gui or (not args.ref and not args.dists and not args.out):
        from .app import launch_gui
        launch_gui()
        return 0

    if not args.ref or not args.dists or not args.out:
        parser.error("CLI mode requires --ref, --dist (repeatable), and --out. Use --gui for the desktop UI.")

    cfg = load_config()
    if args.ffmpeg:
        cfg.ffmpeg_path = args.ffmpeg
    if args.ffprobe:
        cfg.ffprobe_path = args.ffprobe
    if args.save_settings:
        save_config(cfg)

    metrics = parse_metrics(args.metrics)
    progress = RunProgress()
    attach_console(progress)

    req = RunRequest(
        ref=Path(args.ref),
        dists=[Path(d) for d in args.dists],
        outdir=Path(args.out),
        metrics=metrics,
        start_time=args.start_time,
        start_frame=args.start_frame or 0,
        max_frames=args.max_frames,
        stride=args.stride,
        vmaf_model=args.vmaf_model,
        lpips_net=args.lpips_net,
        scale_distorted_for_vmaf=args.scale_distorted,
        erqa_vis=args.erqa_vis,
        erqa_vis_start=args.erqa_vis_start,
        erqa_vis_end=args.erqa_vis_end,
        erqa_vis_video=args.erqa_vis_video,
        lpips_batch_size=args.lpips_batch,
        ffmpeg=Path(cfg.ffmpeg_path),
        ffprobe=Path(cfg.ffprobe_path),
    )
    try:
        result = run_evaluation(req, cfg=cfg, progress=progress)
        try:
            from .history import save_run
            save_run(result)
        except Exception:
            pass
    except (EvalError, RecoverError, CancelledError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130

    print()
    if result.get("cancelled"):
        print("Cancelled — summary written for completed files (in-progress file excluded).")
    print(f"Output: {result['outdir']}")
    print(result.get("vmaf_model_reason") or "")
    _print_result_table(result)
    if result.get("cancelled"):
        return 130
    if any(row.get("errors") for row in result.get("summary") or []):
        return 1
    return 0


def _print_result_table(result: dict) -> None:
    print("Higher is better: PSNR / SSIM / MS-SSIM / VMAF / ERQA")
    print("Lower is better:  LPIPS")
    print()
    print(f"{'distorted':40} {'PSNR-Y':>8} {'SSIM-Y':>8} {'MS-SSIM':>8} {'VMAF':>8} {'LPIPS':>8} {'ERQA':>8}")
    for row in result.get("summary") or []:
        def fmt(v, w=8):
            if v is None:
                return f"{'—':>{w}}"
            if isinstance(v, float):
                return f"{v:{w}.4f}"
            return f"{v!s:>{w}}"
        print(
            f"{str(row.get('distorted'))[:40]:40} "
            f"{fmt(row.get('psnr_y'))} {fmt(row.get('ssim_y'))} {fmt(row.get('ms_ssim'))} "
            f"{fmt(row.get('vmaf'))} {fmt(row.get('lpips'))} {fmt(row.get('erqa'))}"
        )
        for err in row.get("errors") or []:
            print(f"  ERROR: {err}")
        for warn in row.get("warnings") or []:
            print(f"  warn: {warn}")
