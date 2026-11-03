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

Full TTS inference, using the same text, reference audio, seed and device settings. Five measured runs per version, alternating execution order; mean +/- SD. Includes reference-speaker encoding, prompt evaluation, generation and waveform decoding; excludes model loading.

| Model | Frames | Upstream (s) | PR (s) |
|---|---:|---:|---:|
| PocketTTS | 64 | 1.068 +/- 0.029 | 1.035 +/- 0.033 |
| Qwen3-TTS 0.6B (Q4_K_M) | 16 | 9.634 +/- 0.147 | 9.685 +/- 0.112 |

Execution traces confirm that all 256 `MEAN` calls in PocketTTS and all 7 in Qwen3-TTS move from CPU to WebGPU. Overall timing changes are small: PocketTTS reduces 512 values per call, while Qwen3-TTS uses `MEAN` only in speaker encoding; other operations dominate the full runs. The Qwen3-TTS timing difference is smaller than run-to-run variation.

PocketTTS produced byte-identical audio. Qwen3-TTS speaker embeddings had normalized mean squared error of `3.8e-8`; its sampled audio was not bit-identical.

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES.
