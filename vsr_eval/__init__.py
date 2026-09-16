"""VSR-Eval: local full-reference video super-resolution quality evaluation.

Copyright (C) 2026  Adriano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

__version__ = "1.0.0"

ALL_METRICS = ("psnr", "ssim", "ms_ssim", "vmaf", "lpips", "erqa")
HIGHER_BETTER = frozenset({"psnr", "ssim", "ms_ssim", "vmaf", "erqa"})
LOWER_BETTER = frozenset({"lpips"})
