## Overview

Add F32 `ROLL` support to WebGPU by reusing the circular `PAD` shader, including strided inputs and offset views.

LFM2.5 Audio uses `ROLL` in each of its 17 Conformer layers. This keeps those operations on GPU and removes their CPU fallback.

Validation on NVIDIA L4:

- CPU and WebGPU `ROLL`/`PAD` tests: 48/48 passed on each backend.
- Full-model audio inference: identical audio embeddings and generated tokens for short and repeated 30-second speech inputs. End-to-end timings showed no consistent speedup under shared GPU load.
- Added operator performance cases for the observed model shapes.

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES.
