# VSR-Eval

Local Windows desktop + CLI tool that compares a high-resolution **reference** MP4 against one or more **upscaled / reconstructed** MP4s and reports full-reference quality metrics used for video super-resolution evaluation.

This is **not** an upscaler and **not** a web SaaS. It runs on this PC.

## Metrics

| Metric | Source | Better | Notes |
|---|---|---|---|
| **PSNR** (Y, U, V, average) | FFmpeg `psnr` | higher | Baseline |
| **SSIM** (Y, U, V, average) | FFmpeg `ssim` | higher | Baseline |
| **MS-SSIM** | FFmpeg `libvmaf` extra feature `float_ms_ssim` | higher | Same pass as VMAF |
| **VMAF** | FFmpeg `libvmaf` + official Netflix JSON models | higher (0–100) | Video-level harmonic mean + per-frame |
| **LPIPS** | Official `lpips` package, `net='alex'` (optional `vgg`) | **lower** | Frame-by-frame on CUDA, then mean / min / max / p50 / p95 |
| **ERQA** | Official `erqa` package (MSU Video Group) | higher (0–1) | Frame-by-frame on BGR uint8; optional vis mask |

Directionality is also shown in the GUI:

- PSNR / SSIM / MS-SSIM / VMAF / ERQA: **higher is better**
- LPIPS: **lower is better**

Inputs are assumed to already be temporally aligned. Optional start time / start frame + max frames compare a segment (for example the first 300 frames).

## This machine

- GPU: NVIDIA GeForce RTX 4080 Super (16 GB). LPIPS uses CUDA. FFmpeg metrics stay on the CPU (software HEVC decode, libvmaf `n_threads` = CPU cores).
- FFmpeg / FFprobe (Gyan full build 9.0.1, `--enable-libvmaf`):

```
C:\VSR\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe
C:\VSR\ffmpeg-9.0.1-full_build\bin\ffprobe.exe
```

Those paths are the defaults. Override them in the GUI, with `--ffmpeg` / `--ffprobe`, or in `%APPDATA%\VSR-Eval\config.json`.

On startup the app runs `ffmpeg -version` and `ffmpeg -filters` and **fails clearly** if `libvmaf`, `psnr`, or `ssim` are missing.

Videos are expected to be **MP4 / libx265 / yuv420p (8-bit 4:2:0)**. Decode is software-first (`-hwaccel none`, native `hevc` decoder) so scores never silently depend on NVDEC.

## Install (Windows, this PC)

Python 3.11+ is required. This repo is set up for **3.12** via `uv` (already used to create `.venv`).

Open PowerShell in `C:\Users\adri1\Grok\03-SREVAL`:

```powershell
# 1. Virtualenv (skip if .venv already exists)
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1

# 2. PyTorch WITH CUDA — not the CPU wheel
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. The rest of the stack + this package
uv pip install -r requirements.txt
uv pip install -e .

# 4. Confirm the 4080 Super is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True` and `NVIDIA GeForce RTX 4080 Super`. If `torch.cuda.is_available()` is False, LPIPS is disabled with an error telling you to uninstall the CPU wheel.

If the `cu126` index 404s on a newer PyTorch, use the current CUDA index from https://pytorch.org/get-started/locally/ (`cu128`, etc.). Driver here reports CUDA 13.x, which runs CUDA 12 wheels.

VMAF models ship in `.\models\`:

- `vmaf_v0.6.1.json`
- `vmaf_4k_v0.6.1.json`

from https://github.com/Netflix/vmaf/tree/master/model. If the reference long side is **≥ 2560 px**, the app defaults to `vmaf_4k_v0.6.1` and says so in the UI.

### Optional conda

`environment.yml` is provided, but on this PC the pip CUDA wheels above are the supported path.

## Run

### GUI (Gradio on 127.0.0.1)

Double-click `start-gui.bat` in this folder. It uses `.venv` and opens the local UI.

Or from PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m vsr_eval
# or
python -m vsr_eval --gui
```

Opens a local UI with:

- Native file pickers (1 reference + N distorted MP4s)
- Metric checkboxes (all six on by default if available; ERQA visualization off)
- Segment: start time / start frame, max frames, LPIPS/ERQA stride (1/2/4/8)
- VMAF model dropdown and LPIPS net dropdown
- Optional “scale distorted to reference (bicubic, VMAF recommended)” — **never** scales the reference down
- Run + Cancel, live progress (metric, frame i/N, elapsed)
- Results table and per-frame Plotly chart
- Previous runs kept under `%APPDATA%\VSR-Eval\runs` and reloadable in the UI (table, chart, notes)
- Download summary CSV and per-frame CSV from the browser
- JSON + CSV export into the output folder

### CLI

```powershell
python -m vsr_eval --ref C:\VSR\ref.mp4 --dist C:\VSR\upscaled.mp4 --out C:\VSR\results
```

Multiple distorted files:

```powershell
python -m vsr_eval `
  --ref C:\VSR\ref.mp4 `
  --dist C:\VSR\upscaled_a.mp4 `
  --dist C:\VSR\upscaled_b.mp4 `
  --metrics psnr,ssim,ms_ssim,vmaf,lpips,erqa `
  --out C:\VSR\results
```

Useful flags:

```
--start-frame 0 --max-frames 300
--stride 4
--vmaf-model auto|vmaf_v0.6.1|vmaf_4k_v0.6.1
--lpips-net alex|vgg
--scale-distorted
--erqa-vis --erqa-vis-start 0 --erqa-vis-end 24 --erqa-vis-video
--ffmpeg C:\VSR\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe
--ffprobe C:\VSR\ffmpeg-9.0.1-full_build\bin\ffprobe.exe
--self-test
```

## Preconditions

Before scoring, both files are probed (`ffprobe`) and shown: path, size, duration, fps, frame count, width, height, pix_fmt, codec, color range.

The run **errors** (or warns hard) if:

- frame counts differ
- fps differs by more than 0.05
- resolutions differ (except the explicit VMAF bicubic-scale checkbox)
- pix_fmt is not `yuv420p`

Inputs are **never re-encoded**. Decode only.

If resolutions differ and you only want VMAF, enable *scale distorted to reference (bicubic, VMAF recommended)*. PSNR/SSIM/MS-SSIM/LPIPS/ERQA still require matching resolution.

## How each metric is computed

1. **PSNR + SSIM** — one FFmpeg filtergraph (`psnr` + `ssim` via `split`), per-frame stats files, pooled mean.
2. **VMAF + MS-SSIM** — `libvmaf` with `feature=name=float_ms_ssim`, `n_threads` = CPU cores, JSON log. Windows model paths use escaped drive-letter form (`C\:/…`). Software decode only.
3. **LPIPS** — paired RGB24 pipe decode (0–255 → `[-1, 1]`), official `lpips.LPIPS(net='alex')` on CUDA, batch 4 (auto-reduces on OOM). Forced resize is labelled **LPIPS (resized)**.
4. **ERQA** — same RGB decode converted to BGR uint8 HxWx3, `erqa.ERQA()`. Stride 1/2/4/8 is labelled *ERQA sampled every N frames*. Optional vis: blue=FN, red=FP, white=TP, black=TN.

Order: FFmpeg metrics first, then LPIPS (GPU), then ERQA (CPU). Frames are not dumped as millions of PNGs; RGB is streamed through pipes. Temp files are cleaned up with the process.

## Output (per run)

```
OUTDIR/
  summary.csv
  summary.json
  <dist_name>_per_frame.csv
  <dist_name>_vmaf.json
  plots/scores.png
  run_config.json
  log.txt
```

Each completed run is also copied into `%APPDATA%\VSR-Eval\runs\<run-id>\` with the summary table, per-frame scores, and CSVs needed to show that run again in the GUI after a restart or after the output folder is overwritten. Use **Previous runs** to switch between them, and **Download summary CSV** / **Download per-frame CSV** to save the currently shown results from the browser.

## Settings

`%APPDATA%\VSR-Eval\config.json`

```json
{
  "ffmpeg_path": "C:\\VSR\\ffmpeg-9.0.1-full_build\\bin\\ffmpeg.exe",
  "ffprobe_path": "C:\\VSR\\ffmpeg-9.0.1-full_build\\bin\\ffprobe.exe",
  "default_metrics": ["psnr", "ssim", "ms_ssim", "vmaf", "lpips", "erqa"],
  "vmaf_model": "auto",
  "lpips_net": "alex",
  "erqa_stride": 1
}
```

## Self-test

```powershell
python -m vsr_eval --self-test
python tests\test_synthetic.py
```

Identical generated frames: PSNR is inf / very high, LPIPS ≈ 0, ERQA ≈ 1.

## Layout

```
vsr_eval/ffmpeg_metrics.py   PSNR, SSIM, VMAF, MS-SSIM
vsr_eval/lpips_metrics.py    official lpips on CUDA
vsr_eval/erqa_metrics.py     official erqa on BGR uint8
vsr_eval/pipeline.py         orchestration + reports
vsr_eval/history.py          previous-run store for the GUI
vsr_eval/app.py              Gradio UI
vsr_eval/cli.py              python -m vsr_eval
models/                      Netflix VMAF JSON
```
