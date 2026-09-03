"""Persist previous evaluation runs so the GUI can reload tables, charts, and CSVs."""

from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import config_dir
from .reports import DIRECTION_NOTE, write_summary
from .util import dump_json, json_safe

_RUN_ID_RE = re.compile(r"^[\w.\-]+$")
_TABLES_NAME = "tables.json"
_META_NAME = "meta.json"
_SUMMARY_JSON = "summary.json"
_SUMMARY_CSV = "summary.csv"
_PER_FRAME_CSV = "per_frame.csv"


def runs_dir() -> Path:
    return config_dir() / "runs"


def _safe_run_id(run_id: str) -> str:
    rid = (run_id or "").strip()
    if not _RUN_ID_RE.fullmatch(rid):
        raise ValueError(f"Invalid run id: {run_id!r}")
    return rid


def _run_path(run_id: str) -> Path:
    return runs_dir() / _safe_run_id(run_id)


def _new_run_id(when: datetime) -> str:
    return f"{when.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


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


def format_run_notes(result: dict[str, Any]) -> str:
    notes = [
        DIRECTION_NOTE,
        result.get("vmaf_model_reason") or "",
        f"Output: `{result.get('outdir')}`",
    ]
    for row in result.get("summary") or []:
        stem = row.get("distorted")
        for e in row.get("errors") or []:
            notes.append(f"**ERROR ({stem}):** {e}")
        for w in row.get("warnings") or []:
            notes.append(f"Warning ({stem}): {w}")
    return "\n\n".join(n for n in notes if n)


def _label(when: datetime, ref_path: str, dists: list[str]) -> str:
    ref_name = Path(ref_path).name or "reference"
    stamp = when.strftime("%Y-%m-%d %H:%M")
    if not dists:
        return f"{stamp} — {ref_name}"
    if len(dists) == 1:
        return f"{stamp} — {ref_name} vs {Path(dists[0]).name}"
    return f"{stamp} — {ref_name} vs {len(dists)} files"


def _tables_to_records(tables: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for stem, table in (tables or {}).items():
        if isinstance(table, pd.DataFrame):
            records = table.to_dict(orient="records")
        elif isinstance(table, list):
            records = table
        else:
            continue
        out[str(stem)] = json_safe(records)
    return out


def _tables_from_records(raw: Any) -> dict[str, pd.DataFrame]:
    if not isinstance(raw, dict):
        return {}
    tables: dict[str, pd.DataFrame] = {}
    for stem, records in raw.items():
        if not isinstance(records, list):
            continue
        df = pd.DataFrame(_revive(records))
        tables[str(stem)] = df
    return tables


def write_combined_per_frame_csv(path: Path, tables: dict[str, pd.DataFrame]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for stem, df in tables.items():
        if df is None or df.empty:
            continue
        chunk = df.copy()
        if "distorted" not in chunk.columns:
            chunk.insert(0, "distorted", stem)
        frames.append(chunk)
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["distorted", "frame"])
    out.to_csv(path, index=False)
    return path


@dataclass(frozen=True)
class RunInfo:
    id: str
    created_at: str
    label: str
    path: Path


@dataclass
class SavedRun:
    info: RunInfo
    meta: dict[str, Any]
    summary: list[dict[str, Any]]
    tables: dict[str, pd.DataFrame]
    summary_csv: Path
    per_frame_csv: Path

    @property
    def id(self) -> str:
        return self.info.id

    @property
    def label(self) -> str:
        return self.info.label

    @property
    def notes(self) -> str:
        return str(self.meta.get("notes") or "")


def _read_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _info_from_meta(meta: dict[str, Any], path: Path) -> RunInfo:
    return RunInfo(
        id=str(meta.get("id") or path.name),
        created_at=str(meta.get("created_at") or ""),
        label=str(meta.get("label") or path.name),
        path=path,
    )


def list_runs() -> list[RunInfo]:
    root = runs_dir()
    if not root.is_dir():
        return []
    infos: list[RunInfo] = []
    for child in root.iterdir():
        meta_path = child / _META_NAME
        if not child.is_dir() or not meta_path.is_file():
            continue
        try:
            meta = _read_json(meta_path)
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        infos.append(_info_from_meta(meta, child))
    infos.sort(key=lambda r: (r.created_at, r.id), reverse=True)
    return infos


def load_run(run_id: str) -> SavedRun | None:
    try:
        path = _run_path(run_id)
    except ValueError:
        return None
    meta_path = path / _META_NAME
    if not meta_path.is_file():
        return None
    try:
        meta = _read_json(meta_path)
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None

    summary: list[dict[str, Any]] = []
    summary_path = path / _SUMMARY_JSON
    if summary_path.is_file():
        try:
            loaded = _revive(_read_json(summary_path))
            if isinstance(loaded, list):
                summary = loaded
        except (OSError, ValueError):
            summary = []

    tables: dict[str, pd.DataFrame] = {}
    tables_path = path / _TABLES_NAME
    if tables_path.is_file():
        try:
            tables = _tables_from_records(_read_json(tables_path))
        except (OSError, ValueError):
            tables = {}

    return SavedRun(
        info=_info_from_meta(meta, path),
        meta=meta,
        summary=summary,
        tables=tables,
        summary_csv=path / _SUMMARY_CSV,
        per_frame_csv=path / _PER_FRAME_CSV,
    )


def delete_run(run_id: str) -> bool:
    try:
        path = _run_path(run_id)
    except ValueError:
        return False
    if not path.is_dir():
        return False
    shutil.rmtree(path, ignore_errors=False)
    return not path.exists()


def dropdown_choices(selected_id: str | None = None) -> tuple[list[tuple[str, str]], str | None]:
    infos = list_runs()
    choices = [(item.label, item.id) for item in infos]
    if selected_id and any(item.id == selected_id for item in infos):
        return choices, selected_id
    return choices, (infos[0].id if infos else None)


def save_run(result: dict[str, Any]) -> SavedRun:
    """Store the display payload for one completed evaluation (does not alter outdir files)."""
    when = datetime.now().astimezone()
    run_id = _new_run_id(when)
    path = _run_path(run_id)
    path.mkdir(parents=True, exist_ok=True)

    try:
        ref = result.get("ref") or {}
        ref_path = ref.get("path") if isinstance(ref, dict) else ""
        cfg = result.get("run_config") or {}
        dists = list(cfg.get("dists") or [])
        if not dists:
            dists = [row.get("path") for row in (result.get("summary") or []) if row.get("path")]
        metrics = list(cfg.get("metrics") or [])
        tables = _tables_from_records(_tables_to_records(result.get("tables")))
        summary = _revive(json_safe(list(result.get("summary") or [])))

        meta = {
            "id": run_id,
            "created_at": when.isoformat(),
            "label": _label(when, str(ref_path or ""), [str(d) for d in dists]),
            "ref": ref_path,
            "dists": dists,
            "metrics": metrics,
            "outdir": result.get("outdir"),
            "vmaf_model": result.get("vmaf_model") or cfg.get("vmaf_model"),
            "vmaf_model_reason": result.get("vmaf_model_reason") or cfg.get("vmaf_model_reason") or "",
            "start_frame": result.get("start_frame", cfg.get("start_frame", 0)),
            "start_time": cfg.get("start_time"),
            "max_frames": cfg.get("max_frames"),
            "stride": cfg.get("stride"),
            "notes": format_run_notes(result),
        }
        dump_json(path / _META_NAME, meta)
        dump_json(path / _TABLES_NAME, _tables_to_records(tables))
        write_summary(path, summary)
        write_combined_per_frame_csv(path / _PER_FRAME_CSV, tables)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)
        raise

    return SavedRun(
        info=_info_from_meta(meta, path),
        meta=meta,
        summary=summary if isinstance(summary, list) else [],
        tables=tables,
        summary_csv=path / _SUMMARY_CSV,
        per_frame_csv=path / _PER_FRAME_CSV,
    )
