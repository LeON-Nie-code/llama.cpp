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

### Full-model inference

NVIDIA L4, same checkpoints, prompt, speaker reference, seed and device settings. One unmeasured warm-up per variant, then five alternating runs per variant. Generate 16 codec frames (1.28 seconds of audio). Times are mean +/- sample SD of prompt evaluation + generation + waveform decoder, excluding model loading.

```sh
llama-tts -m MODEL -mm MMPROJ --tts-speaker-file reference.wav \
  -p PROMPT -n 16 -ngl 99 -dev WebGPU -mmdev WebGPU -fa off \
  -t 2 -tb 2 -c 1024 -b 512 -ub 512 -s 1234 --temp 0 --perf
```

| Model | Baseline (s) | COUNT_EQUAL branch (s) | PCM identical | COUNT_EQUAL calls |
|---|---:|---:|:---:|---:|
| PocketTTS English (BF16) | 0.408 +/- 0.004 | 0.404 +/- 0.005 | yes | 0 |
| Qwen3-TTS 0.6B (Q4_K_M) | 10.440 +/- 0.107 | 10.266 +/- 0.078 | yes | 0 |
| Qwen3-TTS 1.7B (Q4_K_M) | 10.640 +/- 0.343 | 10.434 +/- 0.149 | yes | 0 |

Diagnostic graph traces found no COUNT_EQUAL construction or execution in these generation workloads. The small timing differences cannot be attributed to accelerating COUNT_EQUAL; this change adds backend coverage without demonstrating a full-model generation speedup. Every baseline/candidate output was byte-identical at the PCM level.

### Full-model forward validation

The existing `llama_opt_epoch` validation path actually uses COUNT_EQUAL for token accuracy. Tested the full Stories260K and Stories15M checkpoints with context/batch/ubatch 256, four contexts per epoch (1,024 tokens), two CPU threads, no backward execution or weight updates. The workload repeats a fixed short story, so the accuracy values are consistency checks, not model-quality scores.

Five alternating process pairs per model; one warm-up epoch and six measured epochs per process. The table reports mean +/- sample SD of the five process means, including model forward computation, graph preparation, loss and accuracy collection, and excluding model loading. Diagnostic traces were collected separately from timings.

| Model | Baseline (ms/epoch) | COUNT_EQUAL branch (ms/epoch) | Predictions / loss / accuracy |
|---|---:|---:|---|
| stories260K (F32) | 17.72 +/- 0.16 | 17.85 +/- 0.50 | identical |
| stories15M (F32) | 275.85 +/- 2.44 | 274.71 +/- 0.91 | identical |

Traces confirmed that COUNT_EQUAL moved from CPU to WebGPU, with one fewer executed backend subgraph per validation batch. Overall validation time was approximately unchanged; these measurements do not establish a meaningful end-to-end speedup.

Both variants used the same executable and unchanged libraries with the exact baseline/candidate WebGPU backend selected at runtime. Both also used the same local NVIDIA driver-module pinning workaround as the previous ARANGE measurements to avoid a process-teardown crash. This workaround is not part of the change. The ARANGE PR is absent in both variants.

Reproduction scripts, exact commands, traces and raw timings: [benchmark evidence](https://github.com/LeON-Nie-code/llama.cpp/tree/preview/webgpu-count-equal-results/contribution-reviews/webgpu-count-equal).

## Requirements

- [ ] I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES. AI assisted with implementation, test cases, local validation, and review preparation.
