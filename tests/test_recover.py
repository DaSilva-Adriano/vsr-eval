"""Rebuild summary from per-file outputs, always dropping the last treated file."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vsr_eval.recover import (  # noqa: E402
    RecoverError,
    clip_family,
    discover_artifacts,
    recover_summary,
    select_last_run,
)
from vsr_eval.reports import summary_row_from_table  # noqa: E402
from vsr_eval.util import dump_json  # noqa: E402

import pandas as pd  # noqa: E402


def _frames(psnr: float, vmaf: float) -> list[dict]:
    return [
        {"frame": 0, "psnr_y": psnr, "vmaf": vmaf, "lpips": 0.1, "erqa": 0.8},
        {"frame": 1, "psnr_y": psnr + 2.0, "vmaf": vmaf + 4.0, "lpips": 0.2, "erqa": 0.9},
    ]


def _write_per_frame(outdir: Path, stem: str, rows: list[dict], mtime: int) -> Path:
    path = outdir / f"{stem}_per_frame.json"
    dump_json(path, rows)
    os.utime(path, (mtime, mtime))
    return path


class RecoverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.outdir = Path(self.tmp.name)
        self._t = time.time()

    def tearDown(self):
        self.tmp.cleanup()

    def _mtime(self, offset: int) -> int:
        return int(self._t + offset)

    def test_clip_family_strips_resolution_method(self):
        self.assertEqual(clip_family("s-racenight_lossless-720p-24fps-lanczos"), "s-racenight_lossless")
        self.assertEqual(clip_family("s-racenight_lossless-360p-24fps-FSRCNNX8_LAPTOP"), "s-racenight_lossless")
        self.assertEqual(clip_family("a-sollevante_lossless-1080p-24fps-vsr"), "a-sollevante_lossless")
        self.assertEqual(clip_family("s-kartingtime_lossless-1080p-24fps-vsr"), "s-kartingtime_lossless")
        self.assertNotEqual(clip_family("s-kartingtime-720p-24fps-vsr"), clip_family("s-kartingtime_lossless-720p-24fps-vsr"))
        self.assertEqual(clip_family("s-kartingtime-4k-24fps_crf-1080p-24fps-vsr"), "s-kartingtime-4k-24fps_crf")
        self.assertIsNone(clip_family("alpha"))
        self.assertIsNone(clip_family("ref"))

    def test_ignores_older_runs_in_the_same_folder(self):
        _write_per_frame(self.outdir, "a-sollevante_lossless-360p-24fps-vsr", _frames(20.0, 50.0), self._mtime(10))
        _write_per_frame(self.outdir, "a-sollevante_lossless-720p-24fps-bicubic", _frames(21.0, 51.0), self._mtime(20))
        _write_per_frame(self.outdir, "s-racenight_lossless-360p-24fps-bicubic", _frames(40.0, 90.0), self._mtime(30))
        _write_per_frame(self.outdir, "s-racenight_lossless-480p-24fps-vsr", _frames(41.0, 91.0), self._mtime(40))
        _write_per_frame(self.outdir, "s-racenight_lossless-720p-24fps-lanczos", _frames(1.0, 1.0), self._mtime(50))

        result = recover_summary(self.outdir)
        stems = [row["distorted"] for row in result["summary"]]
        self.assertEqual(stems, [
            "s-racenight_lossless-360p-24fps-bicubic",
            "s-racenight_lossless-480p-24fps-vsr",
        ])
        self.assertEqual(result["skipped_last"], "s-racenight_lossless-720p-24fps-lanczos")
        self.assertEqual(result["family"], "s-racenight_lossless")
        self.assertIn("a-sollevante_lossless-360p-24fps-vsr", result["ignored_other_runs"])
        self.assertIn("a-sollevante_lossless-720p-24fps-bicubic", result["ignored_other_runs"])
        meta = json.loads((self.outdir / "recovery.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["family"], "s-racenight_lossless")
        self.assertEqual(meta["skipped"], "s-racenight_lossless-720p-24fps-lanczos")

    def test_time_cluster_when_names_have_no_family(self):
        _write_per_frame(self.outdir, "old_a", _frames(10.0, 10.0), self._mtime(-20000))
        _write_per_frame(self.outdir, "old_b", _frames(11.0, 11.0), self._mtime(-19000))
        _write_per_frame(self.outdir, "new_a", _frames(40.0, 90.0), self._mtime(10))
        _write_per_frame(self.outdir, "new_b", _frames(41.0, 91.0), self._mtime(20))
        _write_per_frame(self.outdir, "new_c", _frames(1.0, 1.0), self._mtime(30))

        result = recover_summary(self.outdir)
        self.assertEqual([row["distorted"] for row in result["summary"]], ["new_a", "new_b"])
        self.assertEqual(result["skipped_last"], "new_c")
        self.assertIsNone(result["family"])
        self.assertIn("old_a", result["ignored_other_runs"])
        self.assertIn("old_b", result["ignored_other_runs"])

    def test_select_last_run_prefers_family_of_newest(self):
        _write_per_frame(self.outdir, "a-bbb-sunflower_lossless-360p-24fps-vsr", _frames(10.0, 10.0), self._mtime(10))
        _write_per_frame(self.outdir, "s-racenight_lossless-360p-24fps-vsr", _frames(40.0, 90.0), self._mtime(20))
        arts = discover_artifacts(self.outdir)
        run, family = select_last_run(arts)
        self.assertEqual(family, "s-racenight_lossless")
        self.assertEqual([a.stem for a in run], ["s-racenight_lossless-360p-24fps-vsr"])

    def test_skips_last_treated_and_writes_summary(self):
        _write_per_frame(self.outdir, "alpha", _frames(40.0, 90.0), self._mtime(10))
        _write_per_frame(self.outdir, "bravo", _frames(42.0, 92.0), self._mtime(20))
        _write_per_frame(self.outdir, "charlie", _frames(10.0, 10.0), self._mtime(30))

        result = recover_summary(self.outdir)
        stems = [row["distorted"] for row in result["summary"]]
        self.assertEqual(stems, ["alpha", "bravo"])
        self.assertEqual(result["skipped_last"], "charlie")
        self.assertTrue(result["recovered"])
        self.assertNotIn("charlie", result["tables"])

        summary_json = json.loads((self.outdir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual([row["distorted"] for row in summary_json], ["alpha", "bravo"])
        self.assertTrue((self.outdir / "summary.csv").is_file())
        self.assertTrue((self.outdir / "recovery.json").is_file())
        meta = json.loads((self.outdir / "recovery.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["skipped"], "charlie")
        self.assertEqual(meta["included"], ["alpha", "bravo"])

        self.assertAlmostEqual(result["summary"][0]["psnr_y"], 41.0)
        self.assertAlmostEqual(result["summary"][1]["psnr_y"], 43.0)

    def test_last_truncated_json_is_not_included(self):
        _write_per_frame(self.outdir, "good", _frames(40.0, 90.0), self._mtime(10))
        _write_per_frame(self.outdir, "ok", _frames(41.0, 91.0), self._mtime(20))
        bad = self.outdir / "damaged_per_frame.json"
        bad.write_text('[{"frame": 0, "psnr_y": 1.0,', encoding="utf-8")
        os.utime(bad, (self._mtime(30), self._mtime(30)))

        result = recover_summary(self.outdir)
        self.assertEqual(result["skipped_last"], "damaged")
        self.assertEqual([row["distorted"] for row in result["summary"]], ["good", "ok"])

    def test_in_progress_logs_are_the_last_treated(self):
        _write_per_frame(self.outdir, "done_a", _frames(40.0, 90.0), self._mtime(10))
        _write_per_frame(self.outdir, "done_b", _frames(41.0, 91.0), self._mtime(20))
        log = self.outdir / "partial_psnr.log"
        log.write_text("n:1 mse_avg:1 mse_y:1 mse_u:1 mse_v:1 psnr_avg:20 psnr_y:20 psnr_u:20 psnr_v:20\n")
        os.utime(log, (self._mtime(30), self._mtime(30)))

        result = recover_summary(self.outdir)
        self.assertEqual(result["skipped_last"], "partial")
        self.assertEqual([row["distorted"] for row in result["summary"]], ["done_a", "done_b"])

    def test_single_file_cannot_recover(self):
        _write_per_frame(self.outdir, "only", _frames(40.0, 90.0), self._mtime(10))
        with self.assertRaises(RecoverError) as ctx:
            recover_summary(self.outdir)
        self.assertIn("Only one treated file", str(ctx.exception))

    def test_empty_dir_errors(self):
        with self.assertRaises(RecoverError):
            recover_summary(self.outdir)

    def test_missing_dir_errors(self):
        with self.assertRaises(RecoverError):
            recover_summary(self.outdir / "nope")

    def test_unreadable_non_last_is_omitted(self):
        _write_per_frame(self.outdir, "good", _frames(40.0, 90.0), self._mtime(10))
        broken = self.outdir / "broken_per_frame.json"
        broken.write_text("not-json", encoding="utf-8")
        os.utime(broken, (self._mtime(15), self._mtime(15)))
        _write_per_frame(self.outdir, "last", _frames(1.0, 1.0), self._mtime(30))

        result = recover_summary(self.outdir)
        self.assertEqual(result["skipped_last"], "last")
        self.assertEqual([row["distorted"] for row in result["summary"]], ["good"])
        self.assertEqual(result["unreadable"], ["broken"])

    def test_fallback_from_ffmpeg_logs_for_non_last(self):
        psnr = self.outdir / "logged_psnr.log"
        psnr.write_text(
            "n:1 mse_avg:1 mse_y:1 mse_u:1 mse_v:1 psnr_avg:30 psnr_y:40 psnr_u:50 psnr_v:60\n"
            "n:2 mse_avg:1 mse_y:1 mse_u:1 mse_v:1 psnr_avg:32 psnr_y:42 psnr_u:52 psnr_v:62\n",
            encoding="utf-8",
        )
        os.utime(psnr, (self._mtime(10), self._mtime(10)))
        _write_per_frame(self.outdir, "newer_incomplete", _frames(1.0, 1.0), self._mtime(20))

        result = recover_summary(self.outdir)
        self.assertEqual(result["skipped_last"], "newer_incomplete")
        self.assertEqual([row["distorted"] for row in result["summary"]], ["logged"])
        self.assertAlmostEqual(result["summary"][0]["psnr_y"], 41.0)

    def test_run_config_paths_are_attached(self):
        dump_json(
            self.outdir / "run_config.json",
            {
                "dists": [r"C:\VSR\alpha.mp4", r"C:\VSR\bravo.mp4", r"C:\VSR\charlie.mp4"],
                "start_frame": 0,
                "metrics": ["psnr", "vmaf"],
                "ref": {"path": r"C:\VSR\ref.mp4"},
            },
        )
        _write_per_frame(self.outdir, "alpha", _frames(40.0, 90.0), self._mtime(10))
        _write_per_frame(self.outdir, "bravo", _frames(42.0, 92.0), self._mtime(20))
        _write_per_frame(self.outdir, "charlie", _frames(1.0, 1.0), self._mtime(30))

        result = recover_summary(self.outdir)
        by_stem = {row["distorted"]: row for row in result["summary"]}
        self.assertEqual(by_stem["alpha"]["path"], r"C:\VSR\alpha.mp4")
        self.assertEqual(by_stem["bravo"]["path"], r"C:\VSR\bravo.mp4")
        cfg = json.loads((self.outdir / "run_config.json").read_text(encoding="utf-8"))
        self.assertTrue(cfg["recovered"])
        self.assertEqual(cfg["recovered_skipped"], "charlie")

    def test_inf_psnr_survives_json(self):
        _write_per_frame(
            self.outdir,
            "ident",
            [{"frame": 0, "psnr_y": "inf"}, {"frame": 1, "psnr_y": "inf"}],
            self._mtime(10),
        )
        _write_per_frame(self.outdir, "last", _frames(1.0, 1.0), self._mtime(20))
        result = recover_summary(self.outdir)
        self.assertEqual(result["summary"][0]["psnr_y"], float("inf"))

    def test_discover_ignores_copied_vmaf_model(self):
        _write_per_frame(self.outdir, "clip", _frames(40.0, 90.0), self._mtime(10))
        (self.outdir / "_vmaf_v0.6.1.json").write_text("{}", encoding="utf-8")
        found = discover_artifacts(self.outdir)
        self.assertEqual([a.stem for a in found], ["clip"])

    def test_cli_recover_writes_summary(self):
        from vsr_eval.cli import main

        _write_per_frame(self.outdir, "alpha", _frames(40.0, 90.0), self._mtime(10))
        _write_per_frame(self.outdir, "bravo", _frames(42.0, 92.0), self._mtime(20))
        _write_per_frame(self.outdir, "charlie", _frames(10.0, 10.0), self._mtime(30))
        prev = os.environ.get("APPDATA")
        os.environ["APPDATA"] = str(self.outdir / "appdata")
        try:
            rc = main(["--recover", "--out", str(self.outdir)])
        finally:
            if prev is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = prev
        self.assertEqual(rc, 0)
        stems = [row["distorted"] for row in json.loads((self.outdir / "summary.json").read_text(encoding="utf-8"))]
        self.assertEqual(stems, ["alpha", "bravo"])

    def test_cli_recover_requires_out(self):
        from vsr_eval.cli import main

        with self.assertRaises(SystemExit):
            main(["--recover"])

    def test_summary_row_from_table_pools_vmaf_harmonic_mean(self):
        df = pd.DataFrame({"frame": [0, 1], "vmaf": [90.0, 94.0], "psnr_y": [40.0, 42.0]})
        row = summary_row_from_table("x", df)
        self.assertAlmostEqual(row["psnr_y"], 41.0)
        self.assertAlmostEqual(row["vmaf_mean"], 92.0)
        expected_h = 2.0 / (1.0 / 90.0 + 1.0 / 94.0)
        self.assertAlmostEqual(row["vmaf"], expected_h)
        self.assertAlmostEqual(row["vmaf_harmonic_mean"], expected_h)


if __name__ == "__main__":
    unittest.main()
