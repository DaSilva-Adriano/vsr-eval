"""Orchestrate probe → FFmpeg metrics → LPIPS → ERQA → reports."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from . import __version__
from .availability import probe_tools
from .config import AppConfig
from .decode import PairedRGBDecoder
from .erqa_metrics import export_erqa_visualization, run_erqa
from .ffmpeg_check import check_ffmpeg, check_ffprobe
from .ffmpeg_metrics import choose_vmaf_model, run_psnr_ssim, run_vmaf_msssim
from .lpips_metrics import run_lpips
from .paths import unique_stem
from .probe import VideoInfo, info_to_dict, probe_video, validate_pair
from .progress import ERROR, CancelledError, RunProgress
from .reports import (
    DIRECTION_NOTE,
    per_frame_table,
    plotly_figure,
    summary_row,
    write_dist_outputs,
    write_scores_plot,
    write_summary,
)
from .util import dump_json


class EvalError(RuntimeError):
    pass


@dataclass
class RunRequest:
    ref: Path
    dists: list[Path]
    outdir: Path
    metrics: list[str]
    start_time: float | None = None
    start_frame: int = 0
    max_frames: int | None = None
    stride: int = 1
    vmaf_model: str = "auto"
    lpips_net: str = "alex"
    scale_distorted_for_vmaf: bool = False
    erqa_vis: bool = False
    erqa_vis_start: int | None = None
    erqa_vis_end: int | None = None
    erqa_vis_video: bool = False
    lpips_batch_size: int = 4
    ffmpeg: Path | None = None
    ffprobe: Path | None = None


def resolve_start_frame(start_time: float | None, start_frame: int, fps: float | None) -> int:
    if start_frame and int(start_frame) > 0:
        return int(start_frame)
    if start_time and float(start_time) > 0:
        if not fps:
            raise EvalError("Start time was set but the reference FPS is unknown.")
        return int(round(float(start_time) * float(fps)))
    return 0


def _open_log(outdir: Path) -> TextIO:
    outdir.mkdir(parents=True, exist_ok=True)
    return (outdir / "log.txt").open("a", encoding="utf-8")


def run_evaluation(req: RunRequest, cfg: AppConfig | None = None, progress: RunProgress | None = None) -> dict[str, Any]:
    cfg = cfg or AppConfig()
    ffmpeg = Path(req.ffmpeg or cfg.ffmpeg())
    ffprobe = Path(req.ffprobe or cfg.ffprobe())
    outdir = Path(req.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    progress = progress or RunProgress()
    log = _open_log(outdir)
    metrics = [m.strip().lower() for m in req.metrics if m.strip()]

    try:
        return _run(req, cfg, ffmpeg, ffprobe, outdir, metrics, progress, log)
    finally:
        log.close()


def _run(
    req: RunRequest,
    cfg: AppConfig,
    ffmpeg: Path,
    ffprobe: Path,
    outdir: Path,
    metrics: list[str],
    progress: RunProgress,
    log: TextIO,
) -> dict[str, Any]:
    progress.set_plan(req.dists, metrics)
    progress.begin_setup("ffmpeg", "Checking FFmpeg")
    try:
        ff_status = check_ffmpeg(ffmpeg)
        ff_status.require()
        check_ffprobe(ffprobe)
    except Exception as exc:
        progress.finish_setup("ffmpeg", ERROR, str(exc))
        raise
    progress.finish_setup("ffmpeg")
    log.write(ff_status.version_line + "\n")
    log.flush()

    tools = probe_tools(cfg)
    for m in metrics:
        avail = tools.metrics.get(m)
        if avail and not avail.available:
            raise EvalError(f"Metric {m!r} is not available: {avail.reason}")

    progress.begin_setup("probe", "Probing videos")
    ref_info = probe_video(ffprobe, req.ref)
    if ref_info.error:
        progress.finish_setup("probe", ERROR, ref_info.error)
        raise EvalError(ref_info.error)

    dist_infos: list[VideoInfo] = []
    for d in req.dists:
        info = probe_video(ffprobe, d)
        dist_infos.append(info)
    progress.finish_setup("probe")

    start_frame = resolve_start_frame(req.start_time, req.start_frame, ref_info.fps)
    model_id, model_reason = choose_vmaf_model(req.vmaf_model, ref_info.long_side)

    used_stems: set[str] = set()
    dist_results: list[dict[str, Any]] = []
    tables: dict[str, Any] = {}
    cancelled = False

    for dist_info in dist_infos:
        progress.check_cancel()
        stem = unique_stem(dist_info.path, used_stems)
        progress.begin_file(stem)
        progress.begin_step("validate", f"Validating {stem}")
        one: dict[str, Any] = {
            "stem": stem,
            "path": dist_info.path,
            "probe": info_to_dict(dist_info),
            "warnings": [],
            "errors": [],
        }
        if dist_info.error:
            one["errors"].append(dist_info.error)
            progress.finish_step("validate", ERROR, dist_info.error)
            progress.finish_file(stem, ERROR, dist_info.error)
            dist_results.append(one)
            continue

        pair = validate_pair(
            ref_info, dist_info,
            scale_distorted_for_vmaf=req.scale_distorted_for_vmaf,
            metrics=metrics,
        )
        one["warnings"].extend(i.message for i in pair.warnings)
        if pair.errors:
            msg = "; ".join(i.message for i in pair.errors)
            one["errors"].extend(i.message for i in pair.errors)
            progress.finish_step("validate", ERROR, msg)
            progress.finish_file(stem, ERROR, msg)
            dist_results.append(one)
            continue
        progress.finish_step("validate")

        total = req.max_frames
        if total is None and ref_info.frame_count is not None:
            total = max(0, ref_info.frame_count - start_frame)
        one["start_frame"] = start_frame
        one["max_frames"] = total

        try:
            _score_one(
                req=req,
                ffmpeg=ffmpeg,
                ref_info=ref_info,
                dist_info=dist_info,
                stem=stem,
                outdir=outdir,
                metrics=metrics,
                start_frame=start_frame,
                total=total,
                model_id=model_id,
                progress=progress,
                log=log,
                one=one,
            )
        except CancelledError:
            one["errors"].append("Cancelled")
            progress.finish_file(stem, ERROR, "Cancelled")
            # Do not append the in-progress file — it may be incomplete.
            if not dist_results:
                raise
            cancelled = True
            break
        except Exception as exc:
            one["errors"].append(str(exc))
            progress.finish_file(stem, ERROR, str(exc))
            log.write(traceback.format_exc() + "\n")
            log.flush()
        else:
            progress.finish_file(stem)

        tables[stem] = per_frame_table(one, start_frame=start_frame)
        write_dist_outputs(outdir, stem, one, start_frame=start_frame)
        dist_results.append(one)
        # Keep summary in sync after each completed file so a crash still
        # leaves a usable report of everything finished so far.
        _emit_summary(outdir, dist_results, tables, log)

    progress.begin_setup("reports", "Writing reports")
    summary_rows, plot_path = _emit_summary(outdir, dist_results, tables, log)

    run_config = {
        "vsr_eval": __version__,
        "ref": info_to_dict(ref_info),
        "dists": [r.get("path") for r in dist_results],
        "metrics": metrics,
        "start_frame": start_frame,
        "start_time": req.start_time,
        "max_frames": req.max_frames,
        "stride": req.stride,
        "vmaf_model": model_id,
        "vmaf_model_reason": model_reason,
        "lpips_net": req.lpips_net,
        "scale_distorted_for_vmaf": req.scale_distorted_for_vmaf,
        "ffmpeg": str(ffmpeg),
        "ffprobe": str(ffprobe),
        "ffmpeg_version": ff_status.version_line,
        "direction": DIRECTION_NOTE,
    }
    if cancelled:
        run_config["cancelled"] = True
    dump_json(outdir / "run_config.json", run_config)
    progress.finish_setup("reports")
    if cancelled:
        progress.complete("Cancelled — summary written for completed files")
    else:
        progress.complete("Done")

    return {
        "outdir": str(outdir),
        "ref": info_to_dict(ref_info),
        "vmaf_model": model_id,
        "vmaf_model_reason": model_reason,
        "start_frame": start_frame,
        "results": dist_results,
        "summary": summary_rows,
        "tables": tables,
        "plot": str(plot_path) if plot_path else None,
        "run_config": run_config,
        "plotly": plotly_figure(tables) if tables else None,
        "cancelled": cancelled,
    }


def _emit_summary(
    outdir: Path,
    dist_results: list[dict[str, Any]],
    tables: dict[str, Any],
    log: TextIO,
) -> tuple[list[dict[str, Any]], Path | None]:
    summary_rows = [summary_row(r["stem"], r) for r in dist_results]
    write_summary(outdir, summary_rows)
    plot_path = None
    if tables:
        try:
            plot_path = write_scores_plot(outdir, tables)
        except Exception as exc:
            log.write(f"plot failed: {exc}\n")
            log.flush()
    return summary_rows, plot_path


def _score_one(
    *,
    req: RunRequest,
    ffmpeg: Path,
    ref_info: VideoInfo,
    dist_info: VideoInfo,
    stem: str,
    outdir: Path,
    metrics: list[str],
    start_frame: int,
    total: int | None,
    model_id: str,
    progress: RunProgress,
    log: TextIO,
    one: dict[str, Any],
) -> None:
    want_psnr = "psnr" in metrics
    want_ssim = "ssim" in metrics
    want_vmaf = "vmaf" in metrics
    want_ms = "ms_ssim" in metrics
    want_lpips = "lpips" in metrics
    want_erqa = "erqa" in metrics

    res_differ = (
        ref_info.width and dist_info.width and ref_info.height and dist_info.height
        and (ref_info.width != dist_info.width or ref_info.height != dist_info.height)
    )

    if want_psnr or want_ssim:
        progress.begin_step("psnr_ssim", stem)
        progress.update(metric="PSNR/SSIM", frame=0, total=total, message=stem)
        ff = run_psnr_ssim(
            ffmpeg=ffmpeg,
            ref=ref_info.path,
            dist=dist_info.path,
            outdir=outdir,
            stem=stem,
            start_frame=start_frame,
            max_frames=req.max_frames,
            ref_codec=ref_info.codec,
            dist_codec=dist_info.codec,
            want_psnr=want_psnr,
            want_ssim=want_ssim,
            total_frames=total,
            progress=progress,
            log_fh=log,
        )
        one["psnr"] = ff.get("psnr")
        one["ssim"] = ff.get("ssim")
        progress.finish_step("psnr_ssim")

    if want_vmaf or want_ms:
        progress.begin_step("vmaf", model_id)
        progress.update(metric="VMAF", frame=0, total=total, message=model_id)
        scale = bool(req.scale_distorted_for_vmaf and res_differ)
        vmaf = run_vmaf_msssim(
            ffmpeg=ffmpeg,
            ref=ref_info.path,
            dist=dist_info.path,
            outdir=outdir,
            stem=stem,
            model_id=model_id,
            start_frame=start_frame,
            max_frames=req.max_frames,
            ref_codec=ref_info.codec,
            dist_codec=dist_info.codec,
            scale_distorted=scale,
            ref_width=ref_info.width,
            ref_height=ref_info.height,
            total_frames=total,
            progress=progress,
            log_fh=log,
        )
        one["vmaf"] = vmaf
        progress.finish_step("vmaf")

    if want_lpips or want_erqa:
        perc_key = "lpips_erqa" if want_lpips and want_erqa else ("lpips" if want_lpips else "erqa")
        progress.begin_step(perc_key, stem)
        if res_differ:
            raise EvalError(
                "LPIPS/ERQA require identical resolutions. "
                f"ref {ref_info.width}x{ref_info.height} vs dist {dist_info.width}x{dist_info.height}."
            )
        stride = max(1, int(req.stride or 1))
        vis_range = None
        if req.erqa_vis:
            a = req.erqa_vis_start if req.erqa_vis_start is not None else start_frame
            b = req.erqa_vis_end if req.erqa_vis_end is not None else a
            vis_range = (int(a), int(b))

        sampled_total = None
        if total is not None:
            sampled_total = max(1, (total + stride - 1) // stride)

        # One RGB decode pass for both perceptual metrics.
        rgb_w = int(ref_info.width or 0)
        rgb_h = int(ref_info.height or 0)
        vis_acc: dict[int, Any] = {}
        with PairedRGBDecoder(
            ffmpeg,
            ref_info.path,
            dist_info.path,
            width=rgb_w,
            height=rgb_h,
            start_frame=start_frame,
            max_frames=req.max_frames,
            stride=stride,
            ref_codec=ref_info.codec,
            dist_codec=dist_info.codec,
            progress=progress,
            log_fh=log,
        ) as decoder:
            if want_lpips and want_erqa:
                one["lpips"], one["erqa"] = _lpips_and_erqa_stream(
                    decoder=decoder,
                    net=req.lpips_net,
                    batch_size=req.lpips_batch_size,
                    stride=stride,
                    vis_range=vis_range,
                    vis_acc=vis_acc,
                    progress=progress,
                    total_frames=sampled_total,
                )
            elif want_lpips:
                one["lpips"] = run_lpips(
                    decoder=decoder,
                    net=req.lpips_net,
                    batch_size=req.lpips_batch_size,
                    progress=progress,
                    total_frames=sampled_total,
                )
            else:
                one["erqa"] = run_erqa(
                    decoder=decoder,
                    stride=stride,
                    progress=progress,
                    total_frames=sampled_total,
                    vis_range=vis_range,
                    vis_frames=vis_acc,
                )

        if want_erqa and vis_acc:
            one["erqa_vis"] = export_erqa_visualization(
                vis_frames=vis_acc,
                outdir=outdir,
                stem=stem,
                ffmpeg=ffmpeg,
                fps=ref_info.fps or 24.0,
                write_video=req.erqa_vis_video,
            )
        progress.finish_step(perc_key)


def _lpips_and_erqa_stream(
    *,
    decoder: PairedRGBDecoder,
    net: str,
    batch_size: int,
    stride: int,
    vis_range: tuple[int, int] | None,
    vis_acc: dict,
    progress: RunProgress,
    total_frames: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Single RGB pass: LPIPS on GPU in batches, ERQA on CPU per frame."""
    from .erqa_metrics import _metric, rgb_to_bgr
    from .lpips_metrics import require_cuda, _build_model, _to_lpips_tensor, _resize_pair
    from .util import pool_scores
    import numpy as np
    import torch

    device_name = require_cuda()
    model = _build_model(net, "cuda")
    erqa_m = _metric()
    bs = max(1, int(batch_size))
    lpips_scores: list[float] = []
    lpips_idx: list[int] = []
    erqa_scores: list[float] = []
    erqa_idx: list[int] = []
    resized = False
    resize_to = None

    idx_buf: list[int] = []
    ref_buf: list = []
    dist_buf: list = []

    def flush() -> None:
        nonlocal bs, resized, resize_to
        if not ref_buf:
            return
        refs = np.stack(ref_buf, axis=0)
        dists = np.stack(dist_buf, axis=0)
        while True:
            progress.check_cancel()
            try:
                with torch.no_grad():
                    t0 = _to_lpips_tensor(refs, "cuda")
                    t1 = _to_lpips_tensor(dists, "cuda")
                    out = model(t0, t1)
                    vals = out.detach().float().view(-1).cpu().numpy().tolist()
                del t0, t1, out
                torch.cuda.empty_cache()
                lpips_scores.extend(float(v) for v in vals)
                lpips_idx.extend(idx_buf)
                return
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(ref_buf) > 1:
                    bs = max(1, bs // 2)
                    mid = max(1, len(ref_buf) // 2)
                    saved = (list(idx_buf), list(ref_buf), list(dist_buf))
                    idx_buf[:] = saved[0][:mid]
                    ref_buf[:] = saved[1][:mid]
                    dist_buf[:] = saved[2][:mid]
                    flush()
                    idx_buf[:] = saved[0][mid:]
                    ref_buf[:] = saved[1][mid:]
                    dist_buf[:] = saved[2][mid:]
                    flush()
                    idx_buf.clear()
                    ref_buf.clear()
                    dist_buf.clear()
                    return
                max_side = max(256, max(refs.shape[1], refs.shape[2]) // 2)
                new_refs, new_dists = [], []
                for r, d in zip(refs, dists):
                    rr, dd, hw = _resize_pair(r, d, max_side)
                    resize_to = hw
                    new_refs.append(rr)
                    new_dists.append(dd)
                refs = np.stack(new_refs, axis=0)
                dists = np.stack(new_dists, axis=0)
                resized = True

    seen = 0
    progress.update(metric="LPIPS+ERQA", frame=0, total=total_frames, message=f"net={net} {device_name}")
    for idx, ref, dist in decoder.frames():
        progress.check_cancel()
        target = rgb_to_bgr(dist)
        gt = rgb_to_bgr(ref)
        want_vis = vis_range is not None and vis_range[0] <= idx <= vis_range[1]
        if want_vis:
            value, vis = erqa_m(target, gt, return_vis=True)
            vis_acc[idx] = np.clip(np.asarray(vis) * 255.0, 0, 255).astype(np.uint8)
        else:
            value = erqa_m(target, gt)
        erqa_scores.append(float(value))
        erqa_idx.append(int(idx))

        idx_buf.append(int(idx))
        ref_buf.append(ref)
        dist_buf.append(dist)
        if len(idx_buf) >= bs:
            flush()
            idx_buf.clear()
            ref_buf.clear()
            dist_buf.clear()
        seen += 1
        progress.update(frame=seen, total=total_frames)

    flush()
    lpips_label = "LPIPS (resized)" if resized else "LPIPS"
    erqa_label = "ERQA" if stride <= 1 else f"ERQA sampled every {stride} frames"
    lpips = {
        "net": net,
        "device": device_name,
        "label": lpips_label,
        "resized": resized,
        "resize_to": resize_to,
        "frames": [{"frame": i, "lpips": s} for i, s in zip(lpips_idx, lpips_scores)],
        "pooled": pool_scores(lpips_scores) if lpips_scores else {},
    }
    erqa = {
        "label": erqa_label,
        "stride": int(stride),
        "sampled": stride > 1,
        "frames": [{"frame": i, "erqa": s} for i, s in zip(erqa_idx, erqa_scores)],
        "pooled": pool_scores(erqa_scores) if erqa_scores else {},
        "vis_frames": vis_acc,
    }
    return lpips, erqa
