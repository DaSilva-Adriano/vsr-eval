"""VSR-Eval: local full-reference video super-resolution quality evaluation."""

__version__ = "1.0.0"

ALL_METRICS = ("psnr", "ssim", "ms_ssim", "vmaf", "lpips", "erqa")
HIGHER_BETTER = frozenset({"psnr", "ssim", "ms_ssim", "vmaf", "erqa"})
LOWER_BETTER = frozenset({"lpips"})
