"""unittest wrapper around vsr_eval.selftest (two generated frames)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vsr_eval.selftest import (  # noqa: E402
    run_standalone,
    test_erqa_different,
    test_erqa_identical,
    test_ffmpeg_identical_clips,
    test_lpips_identical,
    test_psnr_parser_inf,
    test_ssim_parser,
    test_vmaf_models_present,
)


class SyntheticTests(unittest.TestCase):
    def test_vmaf_models_present(self):
        test_vmaf_models_present()

    def test_psnr_parser_inf(self):
        test_psnr_parser_inf()

    def test_ssim_parser(self):
        test_ssim_parser()

    def test_erqa_identical(self):
        test_erqa_identical()

    def test_erqa_different(self):
        test_erqa_different()

    def test_ffmpeg_identical_clips(self):
        test_ffmpeg_identical_clips()

    def test_lpips_identical(self):
        try:
            from vsr_eval.lpips_metrics import require_cuda
            require_cuda()
        except Exception as exc:
            self.skipTest(str(exc))
        test_lpips_identical()


if __name__ == "__main__":
    raise SystemExit(run_standalone())
