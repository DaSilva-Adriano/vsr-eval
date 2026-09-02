"""LPIPS (Zhang et al. 2018) via the official `lpips` package on CUDA."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from .decode import PairedRGBDecoder
from .progress import RunProgress
from .util import pool_scores

LPIPS_NETS = ("alex", "vgg")


class LPIPSError(RuntimeError):
    pass


def require_cuda() -> str:
    try:
        import torch
    except Exception as exc:
        raise LPIPSError(
            f"PyTorch is not installed ({exc}). Install the CUDA wheel — not the CPU wheel. See README."
        ) from exc
    if not torch.cuda.is_available():
        raise LPIPSError(
            "LPIPS requires CUDA. torch.cuda.is_available() is False.\n"
            f"torch version: {getattr(torch, '__version__', '?')}\n"
            "Uninstall CPU PyTorch and install a CUDA build, e.g.\n"
            "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
        )
    try:
        return torch.cuda.get_device_name(0)
    except Exception as exc:
        raise LPIPSError(f"CUDA is reported available but the device cannot be queried: {exc}") from exc


def _build_model(net: str, device: str):
    import lpips
    import torch

    net = (net or "alex").lower()
    if net not in LPIPS_NETS:
        raise LPIPSError(f"Unsupported LPIPS net {net!r}; use 'alex' or 'vgg'")
    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _to_lpips_tensor(batch_rgb: np.ndarray, device: str):
    """uint8 RGB NHWC 0–255 → float NCHW in [-1, 1] as the official package requires."""
    import torch

    arr = np.ascontiguousarray(batch_rgb)
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    t = t.permute(0, 3, 1, 2) / 255.0
    return t * 2.0 - 1.0


def _resize_pair(ref: np.ndarray, dist: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    import cv2

    h, w = ref.shape[:2]
    scale = max_side / float(max(h, w))
    nw = max(16, int(round(w * scale)) // 2 * 2)
    nh = max(16, int(round(h * scale)) // 2 * 2)
    ref_r = cv2.resize(ref, (nw, nh), interpolation=cv2.INTER_AREA)
    dist_r = cv2.resize(dist, (nw, nh), interpolation=cv2.INTER_AREA)
    return ref_r, dist_r, (nw, nh)


def run_lpips(
    *,
    decoder: PairedRGBDecoder | None = None,
    frames: Iterable[tuple[int, np.ndarray, np.ndarray]] | None = None,
    net: str = "alex",
    batch_size: int = 4,
    progress: RunProgress | None = None,
    total_frames: int | None = None,
    on_frame: Callable[[int, float], None] | None = None,
) -> dict[str, Any]:
    """Score LPIPS. Provide either a streaming `decoder` or an in-memory `frames` iterable."""
    device_name = require_cuda()
    import torch

    device = "cuda"
    model = _build_model(net, device)
    resized = False
    resize_to: tuple[int, int] | None = None
    bs = max(1, int(batch_size))

    scores: list[float] = []
    indices: list[int] = []

    def flush(idx_buf: list[int], ref_buf: list[np.ndarray], dist_buf: list[np.ndarray]) -> None:
        nonlocal bs, resized, resize_to
        if not ref_buf:
            return
        refs = np.stack(ref_buf, axis=0)
        dists = np.stack(dist_buf, axis=0)
        while True:
            if progress:
                progress.check_cancel()
            try:
                with torch.no_grad():
                    t0 = _to_lpips_tensor(refs, device)
                    t1 = _to_lpips_tensor(dists, device)
                    out = model(t0, t1)
                    vals = out.detach().float().view(-1).cpu().numpy().tolist()
                del t0, t1, out
                torch.cuda.empty_cache()
                for i, v in zip(idx_buf, vals):
                    scores.append(float(v))
                    indices.append(int(i))
                    if on_frame:
                        on_frame(int(i), float(v))
                return
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(ref_buf) > 1:
                    bs = max(1, bs // 2)
                    if progress:
                        progress.update(message=f"LPIPS OOM — reducing batch to {bs}")
                    mid = max(1, len(ref_buf) // 2)
                    flush(idx_buf[:mid], ref_buf[:mid], dist_buf[:mid])
                    flush(idx_buf[mid:], ref_buf[mid:], dist_buf[mid:])
                    return
                if refs.shape[1] <= 256 and refs.shape[2] <= 256:
                    raise LPIPSError(
                        "LPIPS ran out of GPU memory even after resizing. "
                        f"Device: {device_name}"
                    )
                max_side = max(256, max(refs.shape[1], refs.shape[2]) // 2)
                new_refs = []
                new_dists = []
                for r, d in zip(refs, dists):
                    rr, dd, hw = _resize_pair(r, d, max_side)
                    resize_to = hw
                    new_refs.append(rr)
                    new_dists.append(dd)
                refs = np.stack(new_refs, axis=0)
                dists = np.stack(new_dists, axis=0)
                resized = True
                if progress:
                    progress.update(message=f"LPIPS OOM — resizing to {resize_to[0]}x{resize_to[1]}")

    idx_buf: list[int] = []
    ref_buf: list[np.ndarray] = []
    dist_buf: list[np.ndarray] = []
    seen = 0

    def push(idx: int, ref: np.ndarray, dist: np.ndarray) -> None:
        nonlocal seen
        if ref.shape != dist.shape:
            raise LPIPSError(f"LPIPS frame {idx}: shape mismatch {ref.shape} vs {dist.shape}")
        idx_buf.append(idx)
        ref_buf.append(ref)
        dist_buf.append(dist)
        seen += 1
        if progress:
            progress.update(metric="LPIPS", frame=seen, total=total_frames)
        if len(idx_buf) >= bs:
            flush(idx_buf, ref_buf, dist_buf)
            idx_buf.clear()
            ref_buf.clear()
            dist_buf.clear()

    if progress:
        progress.update(metric="LPIPS", frame=0, total=total_frames, message=f"net={net} on {device_name}")

    iterator: Iterable[tuple[int, np.ndarray, np.ndarray]]
    if decoder is not None:
        iterator = decoder.frames()
    elif frames is not None:
        iterator = frames
    else:
        raise LPIPSError("run_lpips requires decoder or frames")

    for idx, ref, dist in iterator:
        push(idx, ref, dist)

    flush(idx_buf, ref_buf, dist_buf)

    pooled = pool_scores(scores) if scores else {}
    label = "LPIPS (resized)" if resized else "LPIPS"
    return {
        "net": net,
        "device": device_name,
        "label": label,
        "resized": resized,
        "resize_to": resize_to,
        "batch_size_final": bs,
        "frames": [{"frame": i, "lpips": s} for i, s in zip(indices, scores)],
        "pooled": pooled,
        "values": scores,
        "indices": indices,
    }
