"""Rebuild summary.csv/json from per-file outputs after an interrupted run.

Output folders often keep artifacts from many previous runs. Recovery therefore
takes only the last run — the clip family of the most recently treated file
(for example ``s-racenight_lossless-*``), or the last time-contiguous cluster
when names do not share a family prefix.

The last treated file in that run is never included: it is the one most likely
to have been truncated when the process was killed, cancelled, or stopped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .ffmpeg_metrics import parse_psnr_stats, parse_ssim_stats, parse_vmaf_json
from .paths import unique_stem
from .reports import (
    DIRECTION_NOTE,
    per_frame_table,
    plotly_figure,
    summary_row_from_table,
    write_scores_plot,
    write_summary,
)
from .util import dump_json, pool_scores


class RecoverError(RuntimeError):
    pass


# Longest suffixes first so ``_per_frame.json`` is not mistaken for something else.
_ARTIFACT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_per_frame.json", "per_frame_json"),
    ("_per_frame.csv", "per_frame_csv"),
    ("_psnr.log", "psnr_log"),
    ("_ssim.log", "ssim_log"),
    ("_vmaf.json", "vmaf_json"),
)

_SKIP_REASON = (
    "The most recently treated file is excluded because it may be incomplete "
    "after an interruption."
)

# ``-{res}-{fps}-`` before the upscaler/method token (use the last occurrence).
_FAMILY_TAIL = re.compile(
    r"-(?:360p|480p|720p|1080p|1440p|2160p|4k)-\d+fps-",
    re.IGNORECASE,
)
# Last resolution token, if the stricter tail did not match.
_LAST_RES = re.compile(r"-(360p|480p|720p|1080p|1440p|2160p|4k)(?=-|$)", re.IGNORECASE)

# Files in one run are typically tens of minutes apart; a multi-hour idle
# gap is a different run. Used only when a family prefix cannot be derived.
_TIME_CLUSTER_GAP_NS = 4 * 60 * 60 * 1_000_000_000

_CMD_INPUTS = re.compile(r"-i\s+(\S+\.mp4)", re.IGNORECASE)


@dataclass
class DistArtifact:
    stem: str
    mtime_ns: int = 0
    files: dict[str, Path] = field(default_factory=dict)

    def note_file(self, kind: str, path: Path) -> None:
        self.files[kind] = path
        try:
            ns = path.stat().st_mtime_ns
        except OSError:
            return
        if ns > self.mtime_ns:
            self.mtime_ns = ns


def _revive(obj: Any) -> Any:
    if obj == "inf":
        return float("inf")
    if obj == "-inf":
        return float("-inf")
    if isinstance(obj, dict):
        return {k: _revive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_revive(x) for x in obj]
    return obj


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_artifacts(outdir: Path) -> list[DistArtifact]:
    """Group top-level metric artifacts in *outdir* by distorted-file stem."""
    found: dict[str, DistArtifact] = {}
    if not outdir.is_dir():
        return []
    for path in outdir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        for suffix, kind in _ARTIFACT_SUFFIXES:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                if not stem:
                    break
                art = found.get(stem)
                if art is None:
                    art = DistArtifact(stem=stem)
                    found[stem] = art
                art.note_file(kind, path)
                break
    return sorted(found.values(), key=lambda a: (a.mtime_ns, a.stem))


def clip_family(stem: str) -> str | None:
    """Source/family prefix of a distorted stem, e.g. ``s-racenight_lossless``.

    ``s-racenight_lossless-720p-24fps-lanczos`` → ``s-racenight_lossless``.
    ``s-kartingtime_lossless-1080p-24fps-vsr`` is a different family from
    ``s-kartingtime-720p-24fps-vsr``.
    """
    name = (stem or "").strip()
    if not name:
        return None
    matches = list(_FAMILY_TAIL.finditer(name))
    if matches and matches[-1].start() > 0:
        return name[: matches[-1].start()].lower()
    matches = list(_LAST_RES.finditer(name))
    if matches and matches[-1].start() > 0:
        return name[: matches[-1].start()].lower()
    return None


def last_time_cluster(
    artifacts: list[DistArtifact],
    *,
    gap_ns: int = _TIME_CLUSTER_GAP_NS,
) -> list[DistArtifact]:
    """Newest contiguous burst of artifacts, split on a long idle gap."""
    if not artifacts:
        return []
    ordered = sorted(artifacts, key=lambda a: (a.mtime_ns, a.stem))
    cluster = [ordered[-1]]
    for i in range(len(ordered) - 2, -1, -1):
        if ordered[i + 1].mtime_ns - ordered[i].mtime_ns > gap_ns:
            break
        cluster.append(ordered[i])
    cluster.reverse()
    return cluster


def select_last_run(artifacts: list[DistArtifact]) -> tuple[list[DistArtifact], str | None]:
    """Keep only the interrupted run; return ``(run_artifacts, family_or_none)``.

    Preference: same clip family as the newest stem. Fallback: last time cluster.
    """
    if not artifacts:
        return [], None
    ordered = sorted(artifacts, key=lambda a: (a.mtime_ns, a.stem))
    last = ordered[-1]
    family = clip_family(last.stem)
    if family:
        same = [a for a in ordered if clip_family(a.stem) == family]
        if same:
            return same, family
    return last_time_cluster(ordered), None


def _load_per_frame(art: DistArtifact) -> pd.DataFrame | None:
    json_path = art.files.get("per_frame_json")
    if json_path is not None and json_path.is_file():
        try:
            raw = _revive(_read_json(json_path))
            if isinstance(raw, list) and raw:
                df = pd.DataFrame(raw)
                if "frame" in df.columns:
                    return df
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            pass

    csv_path = art.files.get("per_frame_csv")
    if csv_path is not None and csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and "frame" in df.columns:
                return df
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            pass
    return None


def _pool_channel(frames: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals = [float(r[key]) for r in frames if key in r and r[key] is not None]
    return pool_scores(vals) if vals else {}


def _table_from_logs(art: DistArtifact, start_frame: int) -> pd.DataFrame | None:
    """Best-effort reconstruction when per-frame CSV/JSON was never written."""
    one: dict[str, Any] = {"warnings": [], "errors": []}
    has = False

    psnr_log = art.files.get("psnr_log")
    if psnr_log is not None and psnr_log.is_file():
        try:
            frames = parse_psnr_stats(psnr_log.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            frames = []
        if frames:
            one["psnr"] = {
                "frames": frames,
                "pooled": {
                    "y": _pool_channel(frames, "psnr_y"),
                    "u": _pool_channel(frames, "psnr_u"),
                    "v": _pool_channel(frames, "psnr_v"),
                    "avg": _pool_channel(frames, "psnr_avg"),
                },
            }
            has = True

    ssim_log = art.files.get("ssim_log")
    if ssim_log is not None and ssim_log.is_file():
        try:
            frames = parse_ssim_stats(ssim_log.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            frames = []
        if frames:
            one["ssim"] = {
                "frames": frames,
                "pooled": {
                    "y": _pool_channel(frames, "ssim_y"),
                    "u": _pool_channel(frames, "ssim_u"),
                    "v": _pool_channel(frames, "ssim_v"),
                    "all": _pool_channel(frames, "ssim_all"),
                },
            }
            has = True

    vmaf_json = art.files.get("vmaf_json")
    if vmaf_json is not None and vmaf_json.is_file():
        try:
            one["vmaf"] = parse_vmaf_json(vmaf_json)
            has = True
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            pass

    if not has:
        return None
    df = per_frame_table(one, start_frame=start_frame)
    if df is None or df.empty:
        return None
    return df


def _load_run_config(outdir: Path) -> dict[str, Any]:
    path = outdir / "run_config.json"
    if not path.is_file():
        return {}
    try:
        loaded = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _paths_by_stem(dists: list[Any]) -> dict[str, str]:
    used: set[str] = set()
    out: dict[str, str] = {}
    for raw in dists:
        if not raw:
            continue
        stem = unique_stem(raw, used)
        out[stem] = str(raw)
    return out


def _paths_from_log(outdir: Path, stems: set[str]) -> dict[str, str]:
    """Map distorted stems to MP4 paths parsed from the newest ffmpeg CMD lines."""
    log_path = outdir / "log.txt"
    if not log_path.is_file() or not stems:
        return {}
    found: dict[str, str] = {}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    for line in text.splitlines():
        if "CMD:" not in line and "DECODE DIST:" not in line:
            continue
        inputs = _CMD_INPUTS.findall(line)
        if not inputs:
            continue
        dist = inputs[0]
        stem = unique_stem(dist, set())
        if stem in stems:
            found[stem] = dist
    return found


def recover_summary(outdir: str | Path) -> dict[str, Any]:
    """Write summary + plot from present per-file data, excluding the last treated file."""
    outdir = Path(outdir)
    if not outdir.is_dir():
        raise RecoverError(f"Output directory does not exist: {outdir}")

    artifacts = discover_artifacts(outdir)
    if not artifacts:
        raise RecoverError(
            f"No processed video artifacts found in {outdir}. "
            "Expected *_per_frame.json/csv, *_psnr.log, *_ssim.log, or *_vmaf.json."
        )

    run_arts, family = select_last_run(artifacts)
    skipped = run_arts[-1]
    kept = run_arts[:-1]
    ignored_other = [a.stem for a in artifacts if a.stem not in {x.stem for x in run_arts}]
    if not kept:
        extra = f" family={family!r}" if family else ""
        raise RecoverError(
            f"Only one treated file was found in the last run ({skipped.stem!r}{extra}); "
            "it is skipped as potentially incomplete. Nothing left to summarize."
        )

    run_config = _load_run_config(outdir)
    start_frame = int(run_config.get("start_frame") or 0)
    kept_stems = {a.stem for a in kept}
    path_by_stem = {
        stem: path
        for stem, path in _paths_by_stem(list(run_config.get("dists") or [])).items()
        if stem in kept_stems
    }
    for stem, path in _paths_from_log(outdir, kept_stems | {skipped.stem}).items():
        path_by_stem.setdefault(stem, path)

    tables: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    unread: list[str] = []
    from_logs: list[str] = []

    for art in kept:
        df = _load_per_frame(art)
        source = "per_frame"
        if df is None:
            df = _table_from_logs(art, start_frame)
            source = "logs"
        if df is None or df.empty:
            unread.append(art.stem)
            continue
        if source == "logs":
            from_logs.append(art.stem)
        tables[art.stem] = df
        rows.append(
            summary_row_from_table(
                art.stem,
                df,
                path=path_by_stem.get(art.stem),
            )
        )

    if not rows:
        raise RecoverError(
            "No readable per-file data remained after skipping "
            f"{skipped.stem!r} (last treated). Unreadable: {unread or 'none'}."
        )

    write_summary(outdir, rows)
    plot_path = None
    if tables:
        try:
            plot_path = write_scores_plot(outdir, tables)
        except Exception:
            plot_path = None

    included_stems = [r["distorted"] for r in rows]
    recovered_cfg: dict[str, Any] = {
        "vsr_eval": run_config.get("vsr_eval") or __version__,
        "recovered": True,
        "recovered_family": family,
        "recovered_skipped": skipped.stem,
        "recovered_included": included_stems,
        "recovered_unreadable": unread,
        "recovered_from_logs": from_logs,
        "recovered_ignored_other_runs": ignored_other,
        "recovered_reason": _SKIP_REASON,
        "direction": run_config.get("direction") or DIRECTION_NOTE,
        "start_frame": start_frame,
        "metrics": run_config.get("metrics"),
        "vmaf_model": run_config.get("vmaf_model"),
        "dists": [path_by_stem.get(stem, stem) for stem in included_stems],
    }
    old_ref = run_config.get("ref") if isinstance(run_config.get("ref"), dict) else {}
    ref_path = str((old_ref or {}).get("path") or "")
    if family and family in ref_path.lower():
        recovered_cfg["ref"] = old_ref
    elif family:
        recovered_cfg["ref"] = {"path": None, "note": f"Recovered family {family}"}
    elif old_ref:
        recovered_cfg["ref"] = old_ref

    dump_json(outdir / "run_config.json", recovered_cfg)
    dump_json(
        outdir / "recovery.json",
        {
            "family": family,
            "skipped": skipped.stem,
            "included": included_stems,
            "unreadable": unread,
            "from_logs": from_logs,
            "ignored_other_runs": ignored_other,
            "reason": _SKIP_REASON,
        },
    )

    reason = f"Recovered from existing per-file outputs in {outdir}."
    if family:
        reason += f" Last run family: {family}."
    else:
        reason += " Last run selected by recent time cluster (no shared clip family in names)."
    reason += f" Skipped last treated file (possibly incomplete): {skipped.stem}."
    if ignored_other:
        reason += f" Ignored {len(ignored_other)} file(s) from earlier runs in the same folder."
    if unread:
        reason += f" Unreadable (also omitted): {', '.join(unread)}."
    if from_logs:
        reason += f" Rebuilt from FFmpeg logs (no per-frame table): {', '.join(from_logs)}."

    return {
        "outdir": str(outdir),
        "ref": recovered_cfg.get("ref") if isinstance(recovered_cfg.get("ref"), dict) else {},
        "vmaf_model": recovered_cfg.get("vmaf_model"),
        "vmaf_model_reason": reason,
        "start_frame": start_frame,
        "results": [],
        "summary": rows,
        "tables": tables,
        "plot": str(plot_path) if plot_path else None,
        "run_config": recovered_cfg,
        "plotly": plotly_figure(tables) if tables else None,
        "recovered": True,
        "skipped_last": skipped.stem,
        "family": family,
        "ignored_other_runs": ignored_other,
        "unreadable": unread,
    }
