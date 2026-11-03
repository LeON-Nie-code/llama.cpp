## Overview

Add WebGPU support for F32 `MEAN` with contiguous rows, including offset views and strided higher dimensions. Reuse the row-sum shader and correct its output indexing for 2D dispatch.

Includes correctness and performance tests, plus updated operation support documentation. Related to #14909.

## Additional information

Tested on NVIDIA L4 with native WebGPU.

```sh
test-backend-ops test -b WebGPU -o MEAN,SUM_ROWS,SUM
test-backend-ops perf -b WebGPU -o MEAN
```

- All 17 `MEAN` cases passed against the CPU reference; all 38 supported reduction cases passed together.
- Covers sliced/permuted inputs, reduction boundaries and 65,537-row dispatch. The large-row `SUM_ROWS` regression fails on upstream and passes with this change.
- Release build, local regression tests, clang-format 22 and EditorConfig checks passed.
- WikiText perplexity, 32 chunks at context 512: E2B 29.6885 -> 29.6811; E4B 27.7562 -> 27.7657.

Full-model `llama-bench` results with Gemma3n E2B/E4B Q4_K_M. Five runs per version, alternating execution order, with in-process warmup. Both versions used the same weights, two CPU threads, batch/microbatch 512 and flash attention off. Model loading and warmup are excluded. Times are mean +/- SD for separate prompt-processing and decoding workloads.

| Model | Test | Upstream (s) | PR (s) | Time reduction |
|---|---|---:|---:|---:|
| Gemma3n E2B | pp512 | 0.744 +/- 0.004 | 0.674 +/- 0.003 | 9.4% |
| Gemma3n E2B | tg128 | 4.201 +/- 0.112 | 3.669 +/- 0.248 | 12.7% |
| Gemma3n E4B | pp512 | 1.137 +/- 0.005 | 0.975 +/- 0.005 | 14.3% |
| Gemma3n E4B | tg128 | 4.769 +/- 0.168 | 3.992 +/- 0.250 | 16.3% |

Both models execute `MEAN` in the first 10 FFN layers. Traces confirm all 1,280 calls in 128 decode steps move from CPU to WebGPU.

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES.
