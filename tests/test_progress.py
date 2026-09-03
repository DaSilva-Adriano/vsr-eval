"""Checklist board: files treated, steps done, work still queued."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vsr_eval.progress import (  # noqa: E402
    DONE,
    ERROR,
    PENDING,
    RUNNING,
    SKIPPED,
    RunProgress,
    attach_console,
    steps_for_metrics,
)


class StepsForMetricsTests(unittest.TestCase):
    def test_all_metrics_are_grouped_into_pipeline_passes(self):
        keys = [s.key for s in steps_for_metrics(["psnr", "ssim", "ms_ssim", "vmaf", "lpips", "erqa"])]
        self.assertEqual(keys, ["validate", "psnr_ssim", "vmaf", "lpips_erqa"])

    def test_subset_uses_separate_perceptual_step(self):
        keys = [s.key for s in steps_for_metrics(["vmaf", "lpips"])]
        self.assertEqual(keys, ["validate", "vmaf", "lpips"])
        labels = [s.label for s in steps_for_metrics(["psnr", "erqa"])]
        self.assertEqual(labels, ["Validate", "PSNR", "ERQA"])


class RunProgressBoardTests(unittest.TestCase):
    def _progress(self) -> RunProgress:
        p = RunProgress()
        p.set_plan(
            [r"C:\VSR\upscaled_a.mp4", r"C:\VSR\upscaled_b.mp4"],
            ["psnr", "vmaf", "lpips"],
        )
        return p

    def test_plan_lists_every_file_and_queued_steps(self):
        p = self._progress()
        board = p.format_markdown()
        self.assertIn("#### Files treated", board)
        self.assertIn("#### Still to do", board)
        self.assertIn("upscaled_a.mp4", board)
        self.assertIn("upscaled_b.mp4", board)
        self.assertIn("Validate, PSNR, VMAF, LPIPS", board)
        self.assertIn("Check FFmpeg", board)
        self.assertIn("Write reports", board)
        self.assertIn("None yet", board)
        files_done, files_total, steps_done, steps_total, remaining = p.counts()
        self.assertEqual(files_total, 2)
        self.assertEqual(files_done, 0)
        self.assertEqual(steps_done, 0)
        self.assertEqual(steps_total, 3 + 2 * 4)  # setup + per-file
        self.assertEqual(remaining, steps_total)

    def test_running_file_shows_done_current_and_remaining_steps(self):
        p = self._progress()
        p.begin_setup("ffmpeg")
        p.finish_setup("ffmpeg")
        p.begin_setup("probe")
        p.finish_setup("probe")
        p.begin_file("upscaled_a")
        p.begin_step("validate")
        p.finish_step("validate")
        p.begin_step("psnr_ssim")
        p.update(frame=12, total=300, message="FFmpeg PSNR/SSIM")
        board = p.format_markdown()
        self.assertIn("Files treated", board)
        self.assertIn("upscaled_a.mp4", board)
        self.assertIn("Validate ✓", board)
        self.assertIn("**PSNR**", board)
        self.assertIn("12/300", board)
        self.assertIn("VMAF remaining", board)
        self.assertIn("LPIPS remaining", board)
        self.assertIn("Still to do", board)
        self.assertIn("`upscaled_a.mp4` — VMAF, LPIPS", board)
        self.assertIn("`upscaled_b.mp4` — Validate, PSNR, VMAF, LPIPS", board)
        self.assertIn("Write reports", board)
        self.assertNotIn("Check FFmpeg", board.split("#### Still to do")[-1])
        line = p.format_line()
        self.assertIn("file 1/2", line)
        self.assertIn("upscaled_a", line)
        self.assertIn("frame 12/300", line)

    def test_finished_file_moves_out_of_still_to_do(self):
        p = self._progress()
        p.begin_file("upscaled_a")
        for key in ("validate", "psnr_ssim", "vmaf", "lpips"):
            p.begin_step(key)
            p.finish_step(key)
        p.finish_file("upscaled_a")
        board = p.format_markdown()
        treated, todo = board.split("#### Still to do")
        self.assertIn("`upscaled_a.mp4`", treated)
        self.assertIn("Validate · PSNR · VMAF · LPIPS", treated)
        self.assertNotIn("upscaled_a.mp4", todo)
        self.assertIn("upscaled_b.mp4", todo)

    def test_failed_file_marks_leftover_steps_and_keeps_other_files_queued(self):
        p = self._progress()
        p.begin_file("upscaled_a")
        p.begin_step("validate")
        p.finish_step("validate", ERROR, "frame counts differ")
        p.finish_file("upscaled_a", ERROR, "frame counts differ")
        board = p.format_markdown()
        self.assertIn("✕", board)
        self.assertIn("frame counts differ", board)
        self.assertIn("Validate ✕", board)
        self.assertIn("PSNR skipped", board)
        todo = board.split("#### Still to do")[-1]
        self.assertNotIn("upscaled_a.mp4", todo)
        self.assertIn("upscaled_b.mp4", todo)

    def test_complete_clears_remaining_work(self):
        p = self._progress()
        p.begin_setup("ffmpeg")
        p.finish_setup("ffmpeg")
        p.begin_setup("probe")
        p.finish_setup("probe")
        for stem in ("upscaled_a", "upscaled_b"):
            p.begin_file(stem)
            for key in ("validate", "psnr_ssim", "vmaf", "lpips"):
                p.begin_step(key)
                p.finish_step(key)
            p.finish_file(stem)
        p.begin_setup("reports")
        p.finish_setup("reports")
        p.complete("Done")
        board = p.format_markdown()
        self.assertIn("### Finished", board)
        self.assertIn("Nothing remaining", board)
        _files_done, files_total, steps_done, steps_total, remaining = p.counts()
        self.assertEqual(files_total, 2)
        self.assertEqual(steps_done, steps_total)
        self.assertEqual(remaining, 0)

    def test_console_prints_a_line_when_a_file_finishes(self):
        buf = io.StringIO()
        p = self._progress()
        attach_console(p, buf)
        p.begin_file("upscaled_a")
        p.begin_step("validate")
        p.finish_step("validate")
        p.finish_file("upscaled_a")
        text = buf.getvalue()
        self.assertIn("upscaled_a", text)
        self.assertIn("Validate", text)


if __name__ == "__main__":
    unittest.main()
