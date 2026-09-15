"""VMAF model selector: two autos (v0 / v1) plus concrete ids."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vsr_eval.config import VMAF_MODELS, VMAF_SELECTORS  # noqa: E402
from vsr_eval.ffmpeg_metrics import choose_vmaf_model  # noqa: E402
from vsr_eval.paths import models_dir  # noqa: E402


class VmafModelSelectTests(unittest.TestCase):
    def test_auto_v0_hd_and_4k(self):
        hid, hreason = choose_vmaf_model("auto", 1920)
        self.assertEqual(hid, "vmaf_v0.6.1")
        self.assertIn("auto (v0)", hreason)
        kid, kreason = choose_vmaf_model("auto-v0", 3840)
        self.assertEqual(kid, "vmaf_4k_v0.6.1")
        self.assertIn("auto (v0)", kreason)

    def test_auto_v1_hd_and_4k(self):
        hid, hreason = choose_vmaf_model("auto-v1", 1920)
        self.assertEqual(hid, "vmaf_v1.0.16_3d0h")
        self.assertIn("auto (v1)", hreason)
        kid, kreason = choose_vmaf_model("auto-v1", 2560)
        self.assertEqual(kid, "vmaf_v1.0.16_1d5h_2160")
        self.assertIn("auto (v1)", kreason)

    def test_explicit_model_is_unchanged(self):
        mid, reason = choose_vmaf_model("vmaf_v1.0.16_5d0h", 3840)
        self.assertEqual(mid, "vmaf_v1.0.16_5d0h")
        self.assertIn("User-selected", reason)

    def test_all_concrete_models_are_on_disk(self):
        for name in VMAF_MODELS:
            path = models_dir() / f"{name}.json"
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 1000, path)

    def test_selectors_include_both_autos(self):
        self.assertIn("auto", VMAF_SELECTORS)
        self.assertIn("auto-v0", VMAF_SELECTORS)
        self.assertIn("auto-v1", VMAF_SELECTORS)
        self.assertIn("vmaf_v1.0.16_3d0h", VMAF_SELECTORS)


if __name__ == "__main__":
    unittest.main()
