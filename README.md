# VSR-Eval

Local Windows desktop + CLI tool that compares a high-resolution **reference** MP4 against one or more **upscaled / reconstructed** MP4s and reports full-reference quality metrics used for video super-resolution evaluation.

This is **not** an upscaler and **not** a web service. It runs locally (Gradio UI on `127.0.0.1`).

Developed on an NVIDIA GeForce RTX 4080 Super (16 GB). Any NVIDIA GPU with a CUDA PyTorch build should work for LPIPS; FFmpeg metrics stay on the CPU.

Licensed under [GNU GPL v3 or later](LICENSE). Third-party model licenses: [THIRD_PARTY.md](THIRD_PARTY.md).

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

## Requirements

- **Windows**
- **Python 3.11+** (3.12 is what the repo is tested with)
- **NVIDIA GPU + CUDA PyTorch** for LPIPS. PSNR / SSIM / MS-SSIM / VMAF / ERQA do not need a GPU.
- **FFmpeg + FFprobe**, full build with `--enable-libvmaf` (the [Gyan](https://www.gyan.dev/ffmpeg/builds/) `full_build` zip is the usual Windows choice)

On startup the app runs `ffmpeg -version` and `ffmpeg -filters` and **fails clearly** if `libvmaf`, `psnr`, or `ssim` are missing. Point it at the binaries in the GUI, with `--ffmpeg` / `--ffprobe`, or in `%APPDATA%\VSR-Eval\config.json`. PATH is not used.

Videos are expected to be **MP4 / libx265 / yuv420p (8-bit 4:2:0)**. Decode is software-first (`-hwaccel none`, native `hevc` decoder) so scores never silently depend on NVDEC.

LPIPS uses CUDA. FFmpeg metrics stay on the CPU (software HEVC decode, libvmaf `n_threads` = CPU cores).

## Install

From the repo root in PowerShell:

```powershell
# 1. Virtualenv
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1

# 2. PyTorch WITH CUDA — not the CPU wheel
#    Pick the CUDA index that matches your driver from
#    https://pytorch.org/get-started/locally/
#    cu126 is a typical CUDA 12.6 wheel (works on a 4080 Super).
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. The rest of the stack + this package
uv pip install -r requirements.txt
uv pip install -e .

# 4. Confirm CUDA is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want `True` and your GPU name. If `torch.cuda.is_available()` is False, LPIPS is disabled with an error telling you to uninstall the CPU wheel.

If the `cu126` index 404s on a newer PyTorch, use the current CUDA index from the PyTorch site (`cu128`, etc.). Newer NVIDIA drivers generally run CUDA 12 wheels.

### FFmpeg

1. Download a **full** Windows build that includes libvmaf (Gyan `full_build`).
2. Unpack it somewhere stable, for example `C:\ffmpeg\`.
3. Set `ffmpeg.exe` and `ffprobe.exe` in the GUI (saved to `%APPDATA%\VSR-Eval\config.json`), or pass `--ffmpeg` / `--ffprobe`.

### VMAF models

Official Netflix JSON models ship in `models\` from https://github.com/Netflix/vmaf/tree/master/model.

**v0** (original):

- `vmaf_v0.6.1.json` — 1080p living-room (3H)
- `vmaf_4k_v0.6.1.json` — 4K TV (1.5H)

**v1** (June 2026, more accurate; uses CAMBI + chroma speed):

- `vmaf_v1.0.16_3d0h.json` — 1080p @ 3H
- `vmaf_v1.0.16_1d5h_2160.json` — 4K @ 1.5H (v1 4K default)
- `vmaf_v1.0.16_5d0h.json` — phone @ 5H
- `vmaf_v1.0.16_3d0h_2160.json` — 4K @ 3H (score range 0–110)

Two autos pick HD vs 4K from the reference long side (**≥ 2560 px** → 4K model):

- `auto` / `auto-v0` → v0 `vmaf_v0.6.1` or `vmaf_4k_v0.6.1`
- `auto-v1` → v1 `vmaf_v1.0.16_3d0h` or `vmaf_v1.0.16_1d5h_2160`

The UI and probe print both auto choices. Saved settings still default to `auto` (v0).

### Optional conda

`environment.yml` is provided. The pip CUDA wheels above are the path this project is developed against.

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
- Run + Cancel, live progress board (each file, steps done, what is still queued, frame i/N, elapsed)
- Recover summary from an interrupted output folder (skips the last treated file)
- Results table and per-frame Plotly chart
- Previous runs kept under `%APPDATA%\VSR-Eval\runs` and reloadable in the UI (table, chart, notes)
- Download summary CSV and per-frame CSV from the browser
- JSON + CSV export into the output folder

### CLI

```powershell
python -m vsr_eval --ref C:\path\to\ref.mp4 --dist C:\path\to\upscaled.mp4 --out C:\path\to\results
```

Multiple distorted files:

```powershell
python -m vsr_eval `
  --ref C:\path\to\ref.mp4 `
  --dist C:\path\to\upscaled_a.mp4 `
  --dist C:\path\to\upscaled_b.mp4 `
  --metrics psnr,ssim,ms_ssim,vmaf,lpips,erqa `
  --out C:\path\to\results
```

Useful flags:

```
--start-frame 0 --max-frames 300
--stride 4
--vmaf-model auto|auto-v0|auto-v1|vmaf_v0.6.1|vmaf_4k_v0.6.1|vmaf_v1.0.16_3d0h|vmaf_v1.0.16_1d5h_2160|vmaf_v1.0.16_5d0h|vmaf_v1.0.16_3d0h_2160
--lpips-net alex|vgg
--scale-distorted
--erqa-vis --erqa-vis-start 0 --erqa-vis-end 24 --erqa-vis-video
--ffmpeg C:\ffmpeg\bin\ffmpeg.exe
--ffprobe C:\ffmpeg\bin\ffprobe.exe
--self-test
--recover --out C:\path\to\results
```

### Recover a summary after an interruption

If a run is killed, crashes, or is cancelled before it writes `summary.csv` / `summary.json`, the per-file outputs (`*_per_frame.json`, `*_psnr.log`, `*_vmaf.json`, …) are usually already on disk.

Rebuild the summary from that folder:

```powershell
python -m vsr_eval --recover --out C:\path\to\results
```

In the GUI, set **Output directory** to that folder and click **Recover summary**.

If the output folder still contains files from **earlier runs**, recovery keeps only the **last run**: the clip family of the newest file (for example every `clip_lossless-*` next to leftover files from other clips). When names do not share that pattern, it uses the most recent time cluster instead.

The **last treated file** of that run is never added to the recovered summary. That file is the one most likely to have been truncated mid-write. `recovery.json` records the family, the skipped stem, and which older files were ignored. Cancelled runs also write a summary of files that had already finished (the in-progress file is excluded).

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
  "ffmpeg_path": "C:\\ffmpeg\\bin\\ffmpeg.exe",
  "ffprobe_path": "C:\\ffmpeg\\bin\\ffprobe.exe",
  "default_metrics": ["psnr", "ssim", "ms_ssim", "vmaf", "lpips", "erqa"],
  "vmaf_model": "auto",
  "lpips_net": "alex",
  "erqa_stride": 1
}
```

Use the absolute paths to *your* FFmpeg full build.

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
vsr_eval/recover.py          rebuild summary from per-file outputs
vsr_eval/history.py          previous-run store for the GUI
vsr_eval/app.py              Gradio UI
vsr_eval/cli.py              python -m vsr_eval
models/                      Netflix VMAF JSON (BSD-2-Clause-Patent)
```

## License

VSR-Eval is free software: you can redistribute it and/or modify it under the
terms of the **GNU General Public License v3 or later**. See [`LICENSE`](LICENSE).

Third-party metric **models** keep their own licenses (they are not GPL):

| Model / metric | Shipped here? | License | Copyright |
|---|---|---|---|
| Netflix VMAF JSON (v0 and v1) | Yes, `models/` | BSD-2-Clause-Patent | Netflix, Inc. |
| LPIPS weights (`alex` / `vgg`) + ImageNet trunks | No (downloaded at runtime by `lpips` / `torchvision`) | BSD-2-Clause / BSD-3-Clause | Zhang et al.; TorchVision |
| ERQA | No (pip package; no separate NN weights) | MIT | Kirillova, Lyapustin |

Full notices: [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`models/LICENSE`](models/LICENSE).
