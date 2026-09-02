"""ERQA (Kirillova et al. 2022 / MSU Video Group) via the official `erqa` package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .decode import PairedRGBDecoder
from .ffmpeg_run import run_ffmpeg
from .progress import RunProgress
from .util import pool_scores


class ERQAError(RuntimeError):
    pass


def _metric():
    try:
        import erqa
    except Exception as exc:
        raise ERQAError(
            f"Official `erqa` package failed to import ({exc}). pip install erqa"
        ) from exc
    return erqa.ERQA()


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ERQAError(f"ERQA expects HxWx3, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb[:, :, ::-1].copy()


def run_erqa(
    *,
    decoder: PairedRGBDecoder | None = None,
    frames: Iterable[tuple[int, np.ndarray, np.ndarray]] | None = None,
    stride: int = 1,
    progress: RunProgress | None = None,
    total_frames: int | None = None,
    vis_range: tuple[int, int] | None = None,
    vis_frames: dict[int, np.ndarray] | None = None,
    on_frame: Callable[[int, float], None] | None = None,
) -> dict[str, Any]:
    metric = _metric()
    scores: list[float] = []
    indices: list[int] = []
    if vis_frames is None:
        vis_frames = {}

    iterator: Iterable[tuple[int, np.ndarray, np.ndarray]]
    if decoder is not None:
        iterator = decoder.frames()
    elif frames is not None:
        iterator = frames
    else:
        raise ERQAError("run_erqa requires decoder or frames")

    if progress:
        label = "ERQA" if stride <= 1 else f"ERQA sampled every {stride} frames"
        progress.update(metric=label, frame=0, total=total_frames, message=label)

    seen = 0
    for idx, ref_rgb, dist_rgb in iterator:
        if progress:
            progress.check_cancel()
        if ref_rgb.shape != dist_rgb.shape:
            raise ERQAError(f"ERQA frame {idx}: shape mismatch {ref_rgb.shape} vs {dist_rgb.shape}")
        target = rgb_to_bgr(dist_rgb)
        gt = rgb_to_bgr(ref_rgb)
        want_vis = vis_range is not None and vis_range[0] <= idx <= vis_range[1]
        if want_vis:
            value, vis = metric(target, gt, return_vis=True)
            vis_u8 = np.clip(np.asarray(vis) * 255.0, 0, 255).astype(np.uint8)
            vis_frames[idx] = vis_u8
        else:
            value = metric(target, gt)
        scores.append(float(value))
        indices.append(int(idx))
        seen += 1
        if on_frame:
            on_frame(int(idx), float(value))
        if progress:
            progress.update(frame=seen, total=total_frames)

    pooled = pool_scores(scores) if scores else {}
    sampled = stride > 1
    label = "ERQA" if not sampled else f"ERQA sampled every {stride} frames"
    return {
        "label": label,
        "stride": int(stride),
        "sampled": sampled,
        "frames": [{"frame": i, "erqa": s} for i, s in zip(indices, scores)],
        "pooled": pooled,
        "values": scores,
        "indices": indices,
        "vis_frames": vis_frames,
    }


def export_erqa_visualization(
    *,
    vis_frames: dict[int, np.ndarray],
    outdir: Path,
    stem: str,
    ffmpeg: str | Path | None = None,
    fps: float = 24.0,
    write_video: bool = False,
) -> dict[str, str]:
    """Write a contact sheet (and optional sidecar video) of ERQA masks.

    Mask colors (BGR as produced by official ERQA):
      blue  = missing detail (FN)
      red   = misplaced detail (FP)
      white = correct detail (TP)
      black = correct background (TN)
    """
    import cv2

    outdir = Path(outdir)
    vis_dir = outdir / f"{stem}_erqa_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {"dir": str(vis_dir)}
    if not vis_frames:
        return written

    ordered = sorted(vis_frames.items())
    pngs: list[Path] = []
    for idx, bgr in ordered:
        path = vis_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(path), bgr)
        pngs.append(path)

    # Contact sheet: up to 8 columns.
    cols = min(8, len(ordered))
    rows = int(np.ceil(len(ordered) / cols))
    h, w = ordered[0][1].shape[:2]
    # Scale tiles if huge.
    max_tile = 320
    scale = min(1.0, max_tile / max(h, w))
    tw, th = int(w * scale), int(h * scale)
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (idx, bgr) in enumerate(ordered):
        r, c = divmod(i, cols)
        tile = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        y, x = r * th, c * tw
        sheet[y : y + th, x : x + tw] = tile
        cv2.putText(sheet, str(idx), (x + 4, y + 16), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    sheet_path = outdir / f"{stem}_erqa_contact.png"
    cv2.imwrite(str(sheet_path), sheet)
    written["contact_sheet"] = str(sheet_path)

    if write_video and ffmpeg is not None and pngs:
        video_path = outdir / f"{stem}_erqa_vis.mp4"
        # PNG sequence — visualization sidecar, not a scored clip.
        args = [
            "-framerate", str(fps or 24),
            "-i", str(vis_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(video_path),
        ]
        try:
            run_ffmpeg(ffmpeg, args, cwd=vis_dir)
            written["video"] = str(video_path)
        except Exception:
            # Frame numbers may not be contiguous; fall back to concat via OpenCV writer.
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(video_path), fourcc, float(fps or 24), (w, h))
            for _, bgr in ordered:
                vw.write(bgr)
            vw.release()
            written["video"] = str(video_path)

    return written
