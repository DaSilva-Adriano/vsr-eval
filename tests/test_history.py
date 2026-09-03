"""Save / reload previous runs without touching FFmpeg or the GUI."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vsr_eval.history import (  # noqa: E402
    delete_run,
    dropdown_choices,
    format_run_notes,
    list_runs,
    load_run,
    save_run,
)


def _result() -> dict:
    return {
        "outdir": r"C:\VSR\results",
        "ref": {"path": r"C:\VSR\ref.mp4"},
        "vmaf_model": "vmaf_v0.6.1",
        "vmaf_model_reason": "Forced vmaf_v0.6.1.",
        "start_frame": 0,
        "summary": [
            {
                "distorted": "upscaled",
                "path": r"C:\VSR\upscaled.mp4",
                "psnr_y": 40.1234567,
                "ssim_y": 0.99,
                "ms_ssim": 0.98,
                "vmaf": 95.5,
                "lpips": 0.01,
                "erqa": 0.9,
                "lpips_label": "LPIPS",
                "erqa_label": "ERQA",
                "warnings": [],
                "errors": [],
            },
            {
                "distorted": "identical",
                "path": r"C:\VSR\identical.mp4",
                "psnr_y": float("inf"),
                "ssim_y": 1.0,
                "ms_ssim": 1.0,
                "vmaf": 100.0,
                "lpips": 0.0,
                "erqa": 1.0,
                "warnings": [],
                "errors": [],
            },
        ],
        "tables": {
            "upscaled": pd.DataFrame(
                {
                    "frame": [0, 1],
                    "psnr_y": [40.0, 40.2],
                    "vmaf": [95.0, 96.0],
                    "lpips": [0.011, 0.009],
                }
            ),
            "identical": pd.DataFrame(
                {
                    "frame": [0, 1],
                    "psnr_y": [float("inf"), float("inf")],
                    "vmaf": [100.0, 100.0],
                }
            ),
        },
        "run_config": {
            "dists": [r"C:\VSR\upscaled.mp4", r"C:\VSR\identical.mp4"],
            "metrics": ["psnr", "ssim", "ms_ssim", "vmaf", "lpips", "erqa"],
            "start_time": None,
            "max_frames": None,
            "stride": 1,
            "vmaf_model": "vmaf_v0.6.1",
        },
    }


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("APPDATA")
        os.environ["APPDATA"] = self.tmp.name

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = self._prev
        self.tmp.cleanup()

    def test_save_load_roundtrip(self):
        saved = save_run(_result())
        self.assertTrue(saved.summary_csv.is_file())
        self.assertTrue(saved.per_frame_csv.is_file())

        loaded = load_run(saved.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.summary), 2)
        self.assertEqual(loaded.summary[0]["distorted"], "upscaled")
        self.assertEqual(loaded.summary[1]["psnr_y"], float("inf"))
        self.assertIn("upscaled", loaded.tables)
        self.assertEqual(list(loaded.tables["upscaled"]["frame"]), [0, 1])
        self.assertTrue(math_is_inf(loaded.tables["identical"]["psnr_y"].iloc[0]))

        summary = pd.read_csv(loaded.summary_csv)
        self.assertEqual(list(summary["distorted"]), ["upscaled", "identical"])
        frames = pd.read_csv(loaded.per_frame_csv)
        self.assertIn("distorted", frames.columns)
        self.assertEqual(set(frames["distorted"]), {"upscaled", "identical"})

    def test_list_newest_first_and_delete(self):
        first = save_run(_result())
        second = save_run(_result())
        ids = [item.id for item in list_runs()]
        self.assertEqual(ids[0], second.id)
        self.assertIn(first.id, ids)

        choices, selected = dropdown_choices()
        self.assertEqual(selected, second.id)
        self.assertEqual(len(choices), 2)

        self.assertTrue(delete_run(second.id))
        remaining = list_runs()
        self.assertEqual([item.id for item in remaining], [first.id])
        self.assertIsNone(load_run(second.id))

    def test_invalid_id_is_safe(self):
        self.assertIsNone(load_run("../secret"))
        self.assertIsNone(load_run(""))
        self.assertFalse(delete_run(".."))

    def test_notes_include_direction_and_output(self):
        notes = format_run_notes(_result())
        self.assertIn("higher is better", notes.lower())
        self.assertIn("C:\\VSR\\results", notes)


def math_is_inf(value) -> bool:
    return value == float("inf")


if __name__ == "__main__":
    unittest.main()
