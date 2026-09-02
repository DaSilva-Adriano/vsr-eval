"""Small numeric / JSON helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def pool_scores(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"),
                "p50": float("nan"), "p95": float("nan")}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        sample = float(arr[0])
        return {"mean": sample, "min": sample, "max": sample, "p50": sample, "p95": sample}
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
    }


def harmonic_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return float("nan")
    return float(finite.size / np.sum(1.0 / finite))


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    return str(obj)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2) + "\n", encoding="utf-8")


def parse_fraction(rate: str | None) -> float | None:
    if not rate or rate in ("0/0", "N/A"):
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            d = float(den)
            if d == 0:
                return None
            return float(num) / d
        return float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
