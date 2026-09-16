# Third-party models and metrics

VSR-Eval source code is GNU GPL v3 or later (see `LICENSE`). The following
works are **not** GPL. They keep their original licenses. Permissive licenses
here (BSD, MIT) are compatible with distributing this program under GPL-3.

## Bundled: Netflix VMAF JSON models

**Shipped in this repo** under `models/*.json`.

| Item | License | Copyright | Source |
|---|---|---|---|
| VMAF v0 / v1 JSON models | [BSD-2-Clause-Patent](https://opensource.org/licenses/BSDplusPatent) | Copyright (c) 2020 Netflix, Inc. | https://github.com/Netflix/vmaf |

Files:

- `vmaf_v0.6.1.json`, `vmaf_4k_v0.6.1.json`
- `vmaf_v1.0.16_3d0h.json`, `vmaf_v1.0.16_1d5h_2160.json`
- `vmaf_v1.0.16_5d0h.json`, `vmaf_v1.0.16_3d0h_2160.json`

Full license text: [`models/LICENSE`](models/LICENSE) (copy of
https://github.com/Netflix/vmaf/blob/master/LICENSE).

These models are passed to FFmpeg `libvmaf`. FFmpeg itself is not shipped here.

## Runtime: LPIPS (not bundled)

Downloaded by the official `lpips` package the first time LPIPS is run.

| Item | License | Copyright | Source |
|---|---|---|---|
| LPIPS metric and linear calibration weights (`net='alex'` / `'vgg'`) | BSD-2-Clause | Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang | https://github.com/richzhang/PerceptualSimilarity |
| AlexNet / VGG-16 trunk (ImageNet-pretrained, via `torchvision`) | BSD-3-Clause (`torchvision`) | TorchVision contributors; networks trained on ImageNet | https://github.com/pytorch/vision |

LPIPS BSD-2-Clause notice:

```
Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Runtime: ERQA (not bundled)

The `erqa` package is a conventional (non-neural) edge-restoration metric.
It does not ship a separate pretrained weight file.

| Item | License | Copyright | Source |
|---|---|---|---|
| ERQA | MIT | Copyright (c) 2021 Anastasia Kirillova, Eugene Lyapustin | https://github.com/msu-video-group/ERQA |

MIT notice:

```
MIT License

Copyright (c) 2021 Anastasia Kirillova, Eugene Lyapustin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Other metrics

PSNR, SSIM, and MS-SSIM are computed with FFmpeg filters (`psnr`, `ssim`,
`libvmaf` extra feature `float_ms_ssim`). Install a full FFmpeg build with
`--enable-libvmaf` yourself; it is not redistributed in this repository.
