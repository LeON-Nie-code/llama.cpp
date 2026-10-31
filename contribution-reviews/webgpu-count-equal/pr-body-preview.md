Suggested title: `webgpu : add COUNT_EQUAL support`

> Private review preview, not submitted. The repository requires contributors to write their own public PR text. Please use the evidence below to rewrite the description in your own words after reviewing the implementation. The contributor agreement remains unchecked until you confirm it.

---

## Overview

Add WebGPU support for `COUNT_EQUAL` with contiguous I32 inputs and an I64 scalar result. This operation is used by `ggml-opt` to count correct predictions after `ARGMAX`.

Clear the result before each evaluation, reduce integer comparisons within each workgroup, and atomically accumulate the partial counts. Reuse the existing memset pipeline, merged input bindings, and 2D dispatch helpers. The input size is bounded so the count fits in the low 32 bits; the high word is cleared as part of the I64 result.

Includes backend correctness and performance cases, plus updated operation support documentation. Noncontiguous inputs remain unsupported.

## Additional information

Tested on an NVIDIA L4 with native Dawn WebGPU, Release build, GCC 14.2. Baseline: `0ec4027a2a0183de87c007069adddc06e665f7de`.

```sh
test-backend-ops test -b WebGPU -o COUNT_EQUAL
test-backend-ops test -b CPU -o COUNT_EQUAL
test-backend-ops test -b WebGPU -o ARGMAX,SUM,SUM_ROWS
test-backend-ops perf -b WebGPU -o COUNT_EQUAL
```

- All 23 COUNT_EQUAL cases passed against the CPU reference, including the two existing ARGMAX graphs and 21 new direct-I32 cases. Coverage includes matching and differing adjacent extreme integers, workgroup boundaries, offset views, and identical inputs.
- All 27 ARGMAX/SUM/SUM_ROWS regression cases passed.
- A separate local harness checked exact I64 results, including the high word, after poisoning the output and repeatedly evaluating two COUNT_EQUAL nodes with changed inputs. Empty inputs, 4D inputs, overlapping views, and 16,777,217- and 67,108,872-element inputs passed; the large cases exercise 2D dispatch. Unsupported types, strides, and sizes were rejected.
- Local build, clang-format 22, EditorConfig 3.0.3, and patch checks passed.

Operator timings from `test-backend-ops perf` on the L4:

| Input elements | WebGPU (us/run) |
|---:|---:|
| 256 | 14.60 |
| 4,096 | 14.32 |
| 65,536 | 14.31 |
| 1,048,576 | 23.12 |

These are operator microbenchmarks, not end-to-end model speedups. Browser WebGPU was not tested.

### Full-model validation that uses COUNT_EQUAL

Measured the full SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct and Stories260K checkpoints through the existing `llama_opt_epoch` forward validation path on WikiText-2. Every validation batch executes COUNT_EQUAL to collect token accuracy. Separate traces confirm CPU execution before the change and WebGPU execution after it, with one fewer executed backend subgraph per batch.

NVIDIA L4, 4096 token predictions per epoch, context/batch/ubatch 256 or 1024, two CPU threads, flash attention disabled, no backward execution or weight updates. Five alternating process pairs per configuration; one warm-up plus two measured epochs per process. Values are mean +/- sample SD of the five process means, including model forward computation, graph preparation, transfers, loss and accuracy collection, excluding loading and tokenization.

| Model | Context / batch | COUNT_EQUAL per epoch | Baseline (ms) | Candidate (ms) | Baseline / candidate |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct Q4_K_M | 256 | 16 | 2168.12 +/- 30.84 | 2165.56 +/- 37.33 | 1.0012x |
| SmolLM2-135M-Instruct Q4_K_M | 1024 | 4 | 1930.24 +/- 9.04 | 1940.47 +/- 17.87 | 0.9947x |
| Qwen2.5-0.5B-Instruct Q4_K_M | 256 | 16 | 5753.11 +/- 38.27 | 5770.50 +/- 32.97 | 0.9970x |
| Qwen2.5-0.5B-Instruct Q4_K_M | 1024 | 4 | 5563.95 +/- 41.72 | 5561.65 +/- 34.01 | 1.0004x |
| Stories260K F32 | 256 | 16 | 73.08 +/- 3.09 | 71.74 +/- 3.41 | 1.0187x |

All predictions, reported loss and accuracy matched across baseline/candidate and repetitions for each configuration. The measured epochs cover 1120 COUNT_EQUAL executions. The additional Stories260K case invokes COUNT_EQUAL about 220 times per second, checking the impact in a faster full-model workload.

SmolLM2 and Qwen2.5 validation times are approximately unchanged. Stories260K has a nominal 1.9% improvement, smaller than the run-to-run standard deviation. These results do not establish a meaningful full-model speedup.

Both variants use the same application and unchanged libraries, with the exact baseline/candidate WebGPU backend selected at runtime. Both use the same local NVIDIA driver-module pinning workaround to avoid a teardown crash; the workaround is outside this change. The ARANGE PR is absent in both variants.

[Reproduction scripts, pinned model inputs, actual COUNT_EQUAL traces and raw timings](https://github.com/LeON-Nie-code/llama.cpp/tree/preview/webgpu-count-equal-results/contribution-reviews/webgpu-count-equal/frequent-models).

## Requirements

- [ ] I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES. AI assisted with implementation, test cases, local validation, and review preparation.
