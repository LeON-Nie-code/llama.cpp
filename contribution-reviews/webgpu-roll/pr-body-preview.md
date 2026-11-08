## Overview

Add F32 `ROLL` support to WebGPU by reusing the circular `PAD` shader, including strided inputs and offset views.

LFM2.5 Audio uses `ROLL` in each of its 17 Conformer layers. Keeping it on GPU avoids repeated CPU fallback and transfers.

Full-model inference on NVIDIA L4, seconds (mean +/- SD over five processes after warm-up):

| Audio input | Before | After |
| --- | ---: | ---: |
| 3-second speech | 0.257 +/- 0.004 | 0.244 +/- 0.002 |
| 30-second repeated speech | 0.842 +/- 0.005 | 0.825 +/- 0.005 |

Audio embeddings and generated tokens are identical. CPU and WebGPU `ROLL`/`PAD` tests pass 48/48 on each backend; operator performance cases cover the observed model shapes.

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES.
