"""Local Gradio desktop UI (127.0.0.1). Native Windows file dialogs via tkinter."""

from __future__ import annotations

import math
import threading
from pathlib import Path

import pandas as pd

from . import ALL_METRICS
from .availability import ToolStatus, probe_tools
from .config import AppConfig, load_config, save_config
from .ffmpeg_metrics import choose_vmaf_model
from .pipeline import EvalError, RunRequest, run_evaluation
from .probe import probe_video, validate_pair
from .progress import CancelledError, RunProgress
from .reports import DIRECTION_NOTE, plotly_figure

_CANCEL = threading.Event()
_LAST_PROGRESS = RunProgress()


def _pick_file(multiple: bool = False, dirs: bool = False) -> list[str] | str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    kwargs = {
        "filetypes": [("MP4 video", "*.mp4"), ("All files", "*.*")],
        "title": "VSR-Eval",
    }
    try:
        if dirs:
            path = filedialog.askdirectory(title="VSR-Eval output folder")
            return path or ""
        if multiple:
            paths = filedialog.askopenfilenames(**kwargs)
            return list(paths) if paths else []
        path = filedialog.askopenfilename(**kwargs)
        return path or ""
    finally:
        root.destroy()


def _status_markdown(st: ToolStatus) -> str:
    lines = ["### Tool status"]
    if st.ffmpeg_ok:
        lines.append(f"- FFmpeg: **OK** — `{st.ffmpeg_version}`")
    else:
        lines.append(f"- FFmpeg: **MISSING** — {st.ffmpeg_error}")
    if st.ffprobe_ok:
        lines.append("- FFprobe: **OK**")
    else:
        lines.append(f"- FFprobe: **MISSING** — {st.ffprobe_error}")
    if st.cuda_ok:
        lines.append(f"- CUDA / LPIPS: **OK** — `{st.cuda_device}` (torch {st.torch_version})")
    else:
        lp = st.metrics.get("lpips")
        lines.append(f"- CUDA / LPIPS: **unavailable** — {lp.reason if lp else 'not checked'}")
    lines.append("")
    lines.append("| Metric | Status | Notes |")
    lines.append("|---|---|---|")
    for name in ALL_METRICS:
        m = st.metrics.get(name)
        if not m:
            continue
        mark = "ready" if m.available else "disabled"
        lines.append(f"| {name} | **{mark}** | {m.reason} |")
    return "\n".join(lines)


def _probe_markdown(ref_path: str, dist_text: str, ffprobe: str, scale: bool, metrics: list[str]) -> str:
    if not ref_path:
        return "Choose a reference MP4."
    dists = [ln.strip() for ln in (dist_text or "").splitlines() if ln.strip()]
    ref = probe_video(ffprobe, ref_path)
    lines = ["### Probe", f"**Reference** `{ref.path}`", "", f"- {ref.label()}", ""]
    if ref.size_bytes:
        lines.append(f"- size: {ref.size_bytes / (1024 * 1024):.1f} MiB")
    if ref.duration_s:
        lines.append(f"- duration: {ref.duration_s:.3f}s")
    if ref.color_range:
        lines.append(f"- color range: {ref.color_range}")
    if ref.error:
        lines.append(f"- **ERROR:** {ref.error}")
    model_id, reason = choose_vmaf_model("auto", ref.long_side)
    lines.append("")
    lines.append(f"**Auto VMAF model:** `{model_id}` — {reason}")
    lines.append("")
    for d in dists:
        info = probe_video(ffprobe, d)
        lines.append(f"**Distorted** `{info.path}`")
        lines.append(f"- {info.label()}")
        if info.error:
            lines.append(f"- **ERROR:** {info.error}")
        pair = validate_pair(ref, info, scale_distorted_for_vmaf=scale, metrics=metrics)
        for issue in pair.issues:
            tag = "ERROR" if issue.level == "error" else "warning"
            lines.append(f"- **{tag}:** {issue.message}")
        lines.append("")
    return "\n".join(lines)


def launch_gui(server_name: str = "127.0.0.1", server_port: int = 7860) -> None:
    try:
        import gradio as gr
    except Exception as exc:
        raise SystemExit(
            f"Gradio is not installed ({exc}). pip install gradio\n"
            "CLI still works: python -m vsr_eval --ref ... --dist ... --out ..."
        ) from exc

    cfg = load_config()
    tools = probe_tools(cfg)

    def metric_choices():
        return [
            (m, tools.metrics[m].available if m in tools.metrics else False)
            for m in ALL_METRICS
        ]

    default_metrics = [m for m in cfg.default_metrics if tools.enabled(m)]
    if not default_metrics:
        default_metrics = [m for m in ALL_METRICS if tools.enabled(m)]

    with gr.Blocks(title="VSR-Eval", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# VSR-Eval\n"
            "Local full-reference evaluation of upscaled / reconstructed MP4s against an HR reference.\n\n"
            + DIRECTION_NOTE
        )
        status_md = gr.Markdown(_status_markdown(tools))

        with gr.Accordion("FFmpeg / settings (saved to %APPDATA%\\VSR-Eval\\config.json)", open=False):
            ffmpeg_tb = gr.Textbox(label="ffmpeg.exe", value=cfg.ffmpeg_path)
            ffprobe_tb = gr.Textbox(label="ffprobe.exe", value=cfg.ffprobe_path)
            save_btn = gr.Button("Save settings")
            settings_msg = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=2):
                ref_tb = gr.Textbox(label="Reference HR MP4", placeholder=r"C:\VSR\ref.mp4")
                ref_btn = gr.Button("Browse reference")
                dist_tb = gr.Textbox(
                    label="Distorted / upscaled MP4s (one path per line)",
                    lines=4,
                    placeholder=r"C:\VSR\upscaled.mp4",
                )
                with gr.Row():
                    dist_btn = gr.Button("Browse distorted (multi-select)")
                    dist_clear = gr.Button("Clear list")
                out_tb = gr.Textbox(label="Output directory", value=cfg.last_outdir or r"C:\VSR\results")
                out_btn = gr.Button("Browse output folder")
            with gr.Column(scale=2):
                metric_boxes = {}
                with gr.Row():
                    for m in ALL_METRICS:
                        avail = tools.enabled(m)
                        metric_boxes[m] = gr.Checkbox(
                            label=m.upper() if m != "ms_ssim" else "MS-SSIM",
                            value=(m in default_metrics) and avail,
                            interactive=avail,
                            info=None if avail else (tools.metrics[m].reason if m in tools.metrics else "unavailable"),
                        )
                erqa_vis = gr.Checkbox(label="ERQA visualization (off by default)", value=False)
                erqa_vis_video = gr.Checkbox(label="ERQA vis sidecar video", value=False)
                with gr.Row():
                    start_time = gr.Number(label="Start time (s)", value=0, precision=3)
                    start_frame = gr.Number(label="Start frame", value=0, precision=0)
                    max_frames = gr.Number(label="Max frames (0 = all)", value=0, precision=0)
                    stride = gr.Dropdown(
                        label="LPIPS/ERQA stride",
                        choices=["1", "2", "4", "8"],
                        value=str(int(cfg.erqa_stride or 1)),
                    )
                with gr.Row():
                    vmaf_model = gr.Dropdown(
                        label="VMAF model",
                        choices=["auto", "vmaf_v0.6.1", "vmaf_4k_v0.6.1"],
                        value=cfg.vmaf_model or "auto",
                    )
                    lpips_net = gr.Dropdown(
                        label="LPIPS net",
                        choices=["alex", "vgg"],
                        value=cfg.lpips_net or "alex",
                    )
                scale_vmaf = gr.Checkbox(
                    label="Scale distorted to reference (bicubic, VMAF recommended)",
                    value=bool(cfg.scale_distorted_for_vmaf),
                )
                with gr.Row():
                    vis_start = gr.Number(label="ERQA vis start frame", value=0, precision=0)
                    vis_end = gr.Number(label="ERQA vis end frame", value=0, precision=0)

        probe_btn = gr.Button("Probe files")
        probe_md = gr.Markdown("")
        with gr.Row():
            run_btn = gr.Button("Run", variant="primary")
            cancel_btn = gr.Button("Cancel")
        progress_md = gr.Markdown("Idle.")
        results_df = gr.Dataframe(label="Pooled scores (one row per distorted file)", wrap=True)
        chart = gr.Plot(label="Per-frame scores")
        log_md = gr.Markdown("")

        def browse_ref():
            p = _pick_file(multiple=False)
            return p or gr.update()

        def browse_dist(current: str):
            picked = _pick_file(multiple=True)
            if not picked:
                return gr.update()
            existing = [ln.strip() for ln in (current or "").splitlines() if ln.strip()]
            for p in picked:
                if p not in existing:
                    existing.append(p)
            return "\n".join(existing)

        def browse_out():
            p = _pick_file(dirs=True)
            return p or gr.update()

        def save_settings(ffmpeg, ffprobe):
            c = load_config()
            c.ffmpeg_path = ffmpeg
            c.ffprobe_path = ffprobe
            save_config(c)
            st = probe_tools(c)
            return _status_markdown(st), "Saved."

        def do_probe(ref, dists, ffmpeg, ffprobe, scale, *metric_vals):
            c = load_config()
            c.ffmpeg_path = ffmpeg
            c.ffprobe_path = ffprobe
            selected = [m for m, on in zip(ALL_METRICS, metric_vals) if on]
            try:
                return _probe_markdown(ref, dists, ffprobe, scale, selected)
            except Exception as exc:
                return f"**Probe failed:** {exc}"

        def do_cancel():
            _CANCEL.set()
            _LAST_PROGRESS.cancel()
            return "Cancel requested…"

        def do_run(ref, dists, outdir, ffmpeg, ffprobe, start_t, start_f, max_f, stride_v,
                   vmaf_m, lpips_n, scale, vis, vis_v, vis_a, vis_b, *metric_vals):
            global _LAST_PROGRESS
            _CANCEL.clear()
            progress = RunProgress()
            _LAST_PROGRESS = progress

            selected = [m for m, on in zip(ALL_METRICS, metric_vals) if on]
            dist_list = [ln.strip() for ln in (dists or "").splitlines() if ln.strip()]
            if not ref:
                yield "Choose a reference MP4.", pd.DataFrame(), None, ""
                return
            if not dist_list:
                yield "Add at least one distorted MP4.", pd.DataFrame(), None, ""
                return
            if not outdir:
                yield "Choose an output directory.", pd.DataFrame(), None, ""
                return
            if not selected:
                yield "Select at least one available metric.", pd.DataFrame(), None, ""
                return

            c = load_config()
            c.ffmpeg_path = ffmpeg
            c.ffprobe_path = ffprobe
            c.vmaf_model = vmaf_m
            c.lpips_net = lpips_n
            c.erqa_stride = int(stride_v)
            c.scale_distorted_for_vmaf = bool(scale)
            c.last_outdir = outdir
            c.default_metrics = selected
            save_config(c)

            max_frames = int(max_f) if max_f else None
            if max_frames == 0:
                max_frames = None
            req = RunRequest(
                ref=Path(ref),
                dists=[Path(d) for d in dist_list],
                outdir=Path(outdir),
                metrics=selected,
                start_time=float(start_t) if start_t else None,
                start_frame=int(start_f) if start_f else 0,
                max_frames=max_frames,
                stride=int(stride_v),
                vmaf_model=vmaf_m,
                lpips_net=lpips_n,
                scale_distorted_for_vmaf=bool(scale),
                erqa_vis=bool(vis),
                erqa_vis_start=int(vis_a) if vis else None,
                erqa_vis_end=int(vis_b) if vis else None,
                erqa_vis_video=bool(vis_v),
                lpips_batch_size=c.lpips_batch_size,
                ffmpeg=Path(ffmpeg),
                ffprobe=Path(ffprobe),
            )

            yield progress.format_line() or "Starting…", pd.DataFrame(), None, ""

            holder: dict = {}
            err: list[str] = []

            def worker():
                try:
                    holder["result"] = run_evaluation(req, cfg=c, progress=progress)
                except CancelledError:
                    err.append("Cancelled.")
                except Exception as exc:
                    err.append(str(exc))

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            while t.is_alive():
                yield progress.format_line(), pd.DataFrame(), None, ""
                t.join(timeout=0.4)

            if err:
                yield err[0], pd.DataFrame(), None, err[0]
                return

            result = holder.get("result") or {}
            rows = result.get("summary") or []
            df = pd.DataFrame(rows)
            keep = [c for c in ["distorted", "psnr_y", "ssim_y", "ms_ssim", "vmaf", "lpips", "erqa"] if c in df.columns]
            extra = []
            if "lpips_label" in df.columns:
                extra.append("lpips_label")
            if "erqa_label" in df.columns:
                extra.append("erqa_label")
            view = df[keep + extra] if not df.empty else df

            def _cell(v):
                if isinstance(v, float):
                    if math.isinf(v):
                        return "inf" if v > 0 else "-inf"
                    if math.isnan(v):
                        return None
                    return round(v, 6)
                return v

            if not view.empty:
                view = view.map(_cell) if hasattr(view, "map") else view.applymap(_cell)
            fig = result.get("plotly")
            if fig is None and result.get("tables"):
                fig = plotly_figure(result["tables"])
            notes = [
                DIRECTION_NOTE,
                result.get("vmaf_model_reason") or "",
                f"Output: `{result.get('outdir')}`",
            ]
            for row in rows:
                for e in row.get("errors") or []:
                    notes.append(f"**ERROR ({row.get('distorted')}):** {e}")
                for w in row.get("warnings") or []:
                    notes.append(f"Warning ({row.get('distorted')}): {w}")
            yield progress.format_line() or "Done.", view, fig, "\n\n".join(n for n in notes if n)

        ref_btn.click(browse_ref, outputs=ref_tb)
        dist_btn.click(browse_dist, inputs=dist_tb, outputs=dist_tb)
        dist_clear.click(lambda: "", outputs=dist_tb)
        out_btn.click(browse_out, outputs=out_tb)
        save_btn.click(save_settings, inputs=[ffmpeg_tb, ffprobe_tb], outputs=[status_md, settings_msg])
        metric_inputs = [metric_boxes[m] for m in ALL_METRICS]
        probe_btn.click(
            do_probe,
            inputs=[ref_tb, dist_tb, ffmpeg_tb, ffprobe_tb, scale_vmaf, *metric_inputs],
            outputs=probe_md,
        )
        cancel_btn.click(do_cancel, outputs=progress_md)
        run_btn.click(
            do_run,
            inputs=[
                ref_tb, dist_tb, out_tb, ffmpeg_tb, ffprobe_tb,
                start_time, start_frame, max_frames, stride,
                vmaf_model, lpips_net, scale_vmaf, erqa_vis, erqa_vis_video,
                vis_start, vis_end, *metric_inputs,
            ],
            outputs=[progress_md, results_df, chart, log_md],
        )

    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, inbrowser=True, show_api=False)
