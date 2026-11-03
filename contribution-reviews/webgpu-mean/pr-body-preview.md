## Overview

Add WebGPU support for F32 `MEAN` with contiguous rows, including offset views and strided higher dimensions. Reuse the row-sum shader with a separate mean pipeline.

Use linear output indexing and skip padded workgroups so the shared reduction shader handles 2D dispatch correctly. Includes regression tests for `MEAN` and `SUM_ROWS`, and updated operation support documentation.

Related to #14909.

## Additional information

Tested on NVIDIA L4 with native WebGPU:

- All 17 `MEAN` cases passed against the CPU reference. All 38 supported `MEAN`, `SUM_ROWS` and `SUM` cases passed together.
- Covers single-element rows, reduction boundaries, sliced/permuted tensors, model-derived shapes and 65,537-row dispatch.
- The new large-row `SUM_ROWS` regression fails on upstream and passes with this change.
- Release build, local regression tests, clang-format 22 and EditorConfig checks passed.

```sh
test-backend-ops test -b WebGPU -o MEAN,SUM_ROWS,SUM
test-backend-ops perf -b WebGPU -o MEAN
```

Full-model `llama-bench` results for Gemma3n E2B/E4B Q4_K_M. Five runs per version, alternating execution order, with the standard in-process warmup. Same weights and settings: WebGPU, two CPU threads, batch/microbatch 512, flash attention off. Model loading and warmup are excluded. Values are seconds, mean +/- SD.

| Model | Test | Upstream (s) | PR (s) | Time reduction |
|---|---|---:|---:|---:|
| Gemma3n E2B | pp128 | 0.265 +/- 0.003 | 0.229 +/- 0.003 | 13.7% |
| Gemma3n E2B | pp512 | 0.744 +/- 0.004 | 0.674 +/- 0.003 | 9.4% |
| Gemma3n E2B | tg128 | 4.201 +/- 0.112 | 3.669 +/- 0.248 | 12.7% |
| Gemma3n E4B | pp128 | 0.386 +/- 0.002 | 0.333 +/- 0.002 | 13.7% |
| Gemma3n E4B | pp512 | 1.137 +/- 0.005 | 0.975 +/- 0.005 | 14.3% |
| Gemma3n E4B | tg128 | 4.769 +/- 0.168 | 3.992 +/- 0.250 | 16.3% |

Both models execute `MEAN` in the first 10 FFN layers: 1,280 calls for 128 decode steps. Traces confirm the calls move from CPU to WebGPU. Row widths are 8,192 for E2B and 16,384 for E4B.

A separate natural-text check validated every executed `MEAN` against a double-precision reference (maximum absolute error below `6e-7`). On 32 WikiText chunks at context 512, perplexity was 29.6885 -> 29.6811 for E2B and 27.7562 -> 27.7657 for E4B. E4B generated the same 128 greedy tokens; E2B diverged at a near-tied choice at token 93 (baseline top-two logit gap 0.026).

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES.
