"""Tiny synthetic checks: identical frames → high PSNR, LPIPS ~ 0, ERQA ~ 1."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from .config import DEFAULT_FFMPEG, load_config
from .ffmpeg_metrics import parse_psnr_stats, parse_ssim_stats
from .paths import models_dir


def _edge_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """BGR/RGB uint8 frame with strong edges so ERQA is defined (uniform frames score 0)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[8:56, 8:24] = (255, 255, 255)
    img[20:28, 28:56] = (200, 40, 40)
    img[40:52, 32:60] = (40, 180, 40)
    return img


def test_erqa_identical() -> float:
    from .erqa_metrics import _metric, rgb_to_bgr

    rgb = _edge_frame()
    bgr = rgb_to_bgr(rgb)
    score = float(_metric()(bgr, bgr.copy()))
    if score < 0.99:
        raise AssertionError(f"ERQA on identical edged frames expected ~1, got {score}")
    return score


def test_erqa_different() -> float:
    from .erqa_metrics import _metric, rgb_to_bgr

    a = _edge_frame()
    b = _edge_frame()
    b[8:56, 8:24] = (0, 0, 0)
    b[8:56, 40:56] = (255, 255, 255)
    score = float(_metric()(rgb_to_bgr(b), rgb_to_bgr(a)))
    if score >= 0.99:
        raise AssertionError(f"ERQA on shifted edges expected < 1, got {score}")
    return score


def test_lpips_identical() -> float:
    from .lpips_metrics import run_lpips

    rgb = _edge_frame()
    result = run_lpips(frames=[(0, rgb, rgb.copy())], net="alex", batch_size=1)
    score = float(result["values"][0])
    if score > 0.02:
        raise AssertionError(f"LPIPS on identical frames expected ~0, got {score}")
    return score


def test_psnr_parser_inf() -> None:
    text = (
        "n:1 mse_avg:0.000000 mse_y:0.000000 mse_u:0.000000 mse_v:0.000000 "
        "psnr_avg:inf psnr_y:inf psnr_u:inf psnr_v:inf\n"
        "n:2 mse_avg:0.000000 mse_y:0.000000 mse_u:0.000000 mse_v:0.000000 "
        "psnr_avg:inf psnr_y:inf psnr_u:inf psnr_v:inf\n"
    )
    rows = parse_psnr_stats(text)
    if len(rows) != 2 or rows[0]["psnr_y"] != float("inf"):
        raise AssertionError(f"PSNR parser failed on inf rows: {rows}")


def test_ssim_parser() -> None:
    text = "n:1 Y:0.990000 U:0.980000 V:0.970000 All:0.985000 (18.0)\n"
    rows = parse_ssim_stats(text)
    if not rows or abs(rows[0]["ssim_y"] - 0.99) > 1e-9:
        raise AssertionError(f"SSIM parser failed: {rows}")


def test_vmaf_models_present() -> None:
    from .config import VMAF_MODELS

    for name in VMAF_MODELS:
        path = models_dir() / f"{name}.json"
        if not path.is_file() or path.stat().st_size < 1000:
            raise AssertionError(f"Missing VMAF model {path}")


def test_ffmpeg_identical_clips() -> dict:
    """Two-frame identical HEVC clips compared to themselves → PSNR inf / SSIM ~ 1."""
    from .ffmpeg_metrics import run_psnr_ssim
    from .ffmpeg_run import run_ffmpeg

    cfg = load_config()
    ffmpeg = Path(cfg.ffmpeg_path if Path(cfg.ffmpeg_path).is_file() else DEFAULT_FFMPEG)
    if not ffmpeg.is_file():
        raise AssertionError(f"ffmpeg not found at {ffmpeg}")

    tmp = Path(tempfile.mkdtemp(prefix="vsr_eval_"))
    clip = tmp / "ident.mp4"
    args = [
        "-f", "lavfi",
        "-i", "testsrc=size=256x256:rate=24",
        "-frames:v", "2",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx265",
        "-x265-params", "log-level=error",
        "-an",
        str(clip),
    ]
    run_ffmpeg(ffmpeg, args, cwd=tmp)
    result = run_psnr_ssim(
        ffmpeg=ffmpeg,
        ref=clip,
        dist=clip,
        outdir=tmp,
        stem="ident",
        want_psnr=True,
        want_ssim=True,
        total_frames=2,
        ref_codec="hevc",
        dist_codec="hevc",
    )
    psnr_y = ((result["psnr"] or {}).get("pooled") or {}).get("y", {}).get("mean")
    ssim_y = ((result["ssim"] or {}).get("pooled") or {}).get("y", {}).get("mean")
    if psnr_y is None or not (psnr_y == float("inf") or psnr_y >= 80):
        raise AssertionError(f"PSNR-Y on identical clip expected inf/high, got {psnr_y}")
    if ssim_y is None or ssim_y < 0.99:
        raise AssertionError(f"SSIM-Y on identical clip expected ~1, got {ssim_y}")

    from .ffmpeg_metrics import run_vmaf_msssim

    vmaf = run_vmaf_msssim(
        ffmpeg=ffmpeg,
        ref=clip,
        dist=clip,
        outdir=tmp,
        stem="ident",
        model_id="vmaf_v0.6.1",
        ref_codec="hevc",
        dist_codec="hevc",
        total_frames=2,
    )
    vmean = ((vmaf.get("pooled") or {}).get("vmaf") or {}).get("mean")
    if vmean is None or vmean < 95:
        raise AssertionError(f"VMAF on identical clip expected ~100, got {vmean}")
    return {"psnr_y": psnr_y, "ssim_y": ssim_y, "vmaf": vmean, "dir": str(tmp)}


def run_standalone() -> int:
    print("VSR-Eval synthetic tests")
    failures = 0

    def check(name, fn):
        nonlocal failures
        try:
            val = fn()
            extra = f" → {val}" if val is not None and not isinstance(val, dict) else ""
            if isinstance(val, dict):
                extra = " → " + ", ".join(f"{k}={v}" for k, v in val.items() if k != "dir")
            print(f"  PASS  {name}{extra}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")

    check("VMAF models on disk", test_vmaf_models_present)
    check("PSNR parser (inf)", test_psnr_parser_inf)
    check("SSIM parser", test_ssim_parser)
    check("ERQA identical frames ~ 1", test_erqa_identical)
    check("ERQA different frames < 1", test_erqa_different)
    check("FFmpeg PSNR/SSIM identical clip", test_ffmpeg_identical_clips)
    try:
        from .lpips_metrics import require_cuda
        require_cuda()
        check("LPIPS identical frames ~ 0", test_lpips_identical)
    except Exception as exc:
        print(f"  SKIP  LPIPS identical frames ({exc})")

    if failures:
        print(f"{failures} failure(s)")
        return 1
    print("All synthetic tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_standalone())
