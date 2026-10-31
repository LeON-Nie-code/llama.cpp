Suggested title: `webgpu : add COUNT_EQUAL support`

> Private review preview; not submitted. The contributor agreement remains unchecked pending your review. The contributing guidelines require contributors to write their own public PR text; this draft is material for that review and rewrite.

---

## Overview

Add WebGPU `COUNT_EQUAL` for same-shaped contiguous I32 inputs and an I64 scalar result. `ggml-opt` uses it to count correct token predictions after `ARGMAX`.

For 1-1024 elements, a single 256-thread workgroup reduces the comparisons and writes both words of the result, avoiding a separate clear dispatch. Larger inputs clear the result and atomically accumulate workgroup counts; empty inputs only clear. Reuse existing overlap-binding and 2D-dispatch helpers, and bound input sizes to keep indexing/counts within the supported 32-bit range. Noncontiguous inputs remain unsupported.

Includes direct-I32 correctness cases, cutoff boundaries, performance cases and operation-support documentation.

## Validation

NVIDIA L4, native Dawn v20260908.214631, Release/GCC 14.2. Compared with upstream `0ec4027a2a0183de87c007069adddc06e665f7de`.

- CPU and WebGPU COUNT_EQUAL: 31/31 each; WebGPU ARGMAX/SUM/SUM_ROWS: 27/27.
- Separate exact-I64 harness: poisoned output/high word, cached replays with changing inputs, empty/offset/overlapping views, cutoff boundaries and large 2D dispatches through 67,108,872 elements passed.
- Build, clang-format 22, EditorConfig 3.0.3 and patch whitespace checks passed. Browser WebGPU and other GPUs were not tested.

### Small-input operator performance

Original two-dispatch WebGPU implementation versus the single-workgroup path, same executable, three alternating repetitions. Mean +/- sample SD:

| Elements | Before (us) | Optimized (us) | Speedup |
|---:|---:|---:|---:|
| 256 | 15.50 +/- 0.87 | 8.24 +/- 0.25 | 1.882x |
| 1,024 | 15.36 +/- 0.78 | 8.04 +/- 0.41 | 1.910x |

These are operator microbenchmarks, not full-model speedups.

### Complete-model validation that executes COUNT_EQUAL

Full checkpoints evaluated through `llama_opt_epoch` on WikiText-2; 4096 predicted tokens per epoch. Context/batch/ubatch match; two CPU threads; flash attention disabled; no backward execution or updates. Five independent processes per variant, one warm-up and one measured epoch each. Epoch timings include model forward, graph preparation, transfers, loss and accuracy collection; exclude model loading/tokenization. Mean +/- sample SD:

| Model | Context / batch | Counts / epoch | Upstream CPU fallback (ms) | Original WebGPU (ms) | Optimized WebGPU (ms) |
|---|---:|---:|---:|---:|---:|
| Stories260K F32 | 256 | 16 | 70.51 +/- 0.78 | 70.25 +/- 0.77 | 70.01 +/- 0.21 |
| SmolLM2-135M-Instruct Q4_K_M | 256 | 16 | 2214.32 +/- 44.79 | 2179.26 +/- 34.17 | 2171.36 +/- 20.10 |
| SmolLM2-135M-Instruct Q4_K_M | 1024 | 4 | 1947.14 +/- 6.39 | 1937.68 +/- 7.81 | 1955.05 +/- 22.90 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 256 | 16 | 6135.14 +/- 268.13 | 6035.15 +/- 226.29 | 6063.84 +/- 315.98 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 1024 | 4 | 5699.77 +/- 74.27 | 5615.99 +/- 87.70 | 5735.82 +/- 101.03 |

All predictions, reported loss and accuracy match across variants and repetitions. Separate backend traces confirm actual COUNT_EQUAL execution: CPU upstream, WebGPU in both implementations. The 75 measured epochs cover 840 COUNT_EQUAL calls.

The fast path improves small-input operator latency by about 1.9x, but these complete-model results do not establish a consistent end-to-end speedup. The surrounding loss/label-transfer/readback costs remain. Only the selected WebGPU backend differs; all model variants use the same driver-module pinning workaround for local teardown stability. The ARANGE change is absent.

[Reproduction, pinned inputs, raw timings and execution traces](https://github.com/LeON-Nie-code/llama.cpp/tree/preview/webgpu-count-equal-results/contribution-reviews/webgpu-count-equal/small-input-optimization).

## Requirements

- [ ] I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES. AI assisted with implementation, test cases, local validation, performance analysis and review preparation.
