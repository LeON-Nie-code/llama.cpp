Suggested title: `webgpu : add COUNT_EQUAL support`

---

## Overview

Add WebGPU support for `COUNT_EQUAL` with contiguous I32 inputs and an I64 scalar result. `ggml-opt` uses this operation to count correct token predictions after `ARGMAX`; it currently falls back to CPU.

Inputs with 1-1024 elements use a single workgroup to reduce the comparisons and write the result. Larger inputs accumulate workgroup counts. The implementation reuses the existing buffer-binding and dispatch helpers and supports contiguous views, including overlapping inputs.

## Additional information

Tested on an NVIDIA L4 with native Dawn WebGPU.

- All 31 COUNT_EQUAL cases pass against the CPU reference, and all 27 ARGMAX/SUM/SUM_ROWS regression cases pass.
- Additional checks pass for exact I64 results, repeated graph execution with changing inputs, empty inputs, offset and overlapping views, and large 2D dispatches. Tests cover both sides of the single-workgroup cutoff.

COUNT_EQUAL takes about 8.78 us for 256 elements and 8.80 us for 1024 elements in `test-backend-ops perf`.

Full-model validation uses `llama_opt_epoch` on WikiText-2, including model forward computation, loss and accuracy collection. Each epoch predicts 4096 tokens and executes COUNT_EQUAL once per batch. Values below are mean +/- sample SD across five runs after warm-up:

| Model | Context / batch | Upstream (ms) | PR (ms) |
|---|---:|---:|---:|
| Stories260K F32 | 256 | 71.73 +/- 1.71 | 70.75 +/- 1.99 |
| SmolLM2-135M-Instruct Q4_K_M | 256 | 2215.11 +/- 53.89 | 2174.81 +/- 8.57 |
| SmolLM2-135M-Instruct Q4_K_M | 1024 | 1940.22 +/- 11.88 | 1935.61 +/- 13.08 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 256 | 5864.41 +/- 35.00 | 5778.36 +/- 17.46 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 1024 | 5615.17 +/- 137.61 | 5551.35 +/- 64.16 |

Backend traces confirm that COUNT_EQUAL runs on CPU upstream and on WebGPU with this PR. Predictions, loss and accuracy match across both versions and all repetitions.

Overall validation times remain similar. COUNT_EQUAL compares only 256 or 1024 token IDs per batch, so it accounts for a small fraction of the work. Model forward computation, CPU cross-entropy and transfers of large logits/label tensors dominate these runs; the count still has to be read back to the CPU. Moving this reduction to WebGPU removes its CPU fallback but has little effect on those larger costs.

## Requirements

- [x] I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES.
