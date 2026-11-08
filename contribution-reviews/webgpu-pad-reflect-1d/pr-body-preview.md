## Overview

Add F32 `PAD_REFLECT_1D` support to WebGPU by reusing the existing padding shader. This operation is used by the Qwen3-TTS speaker encoder and Chatterbox vocoder. Related to #14909.

Extend the existing tests to cover all four dimensions, asymmetric and zero padding, offset views and 2D dispatch. Update operation support documentation.

## Additional information

Tested on NVIDIA L4 with native WebGPU. All 44 `PAD_REFLECT_1D` and existing `PAD` cases passed against the CPU reference. Build, clang-format 22 and EditorConfig checks passed.

Full-model inference time, excluding model loading; five alternating runs per version with identical inputs and settings (mean ± SD):

| Model / request | Upstream (s) | PR (s) |
|---|---:|---:|
| Qwen3-TTS 0.6B, 16 frames | 9.637 ± 0.069 | 9.663 ± 0.078 |
| Chatterbox Turbo, short | 1.783 ± 0.071 | 1.763 ± 0.044 |
| Chatterbox Turbo, long | 4.409 ± 0.177 | 4.417 ± 0.143 |

Audio outputs were byte-identical. End-to-end timing remains within run-to-run variation: reflection is a small part of Qwen3-TTS reference-speaker encoding and runs only once per Chatterbox request. Traces confirm 21 of 22 Qwen3-TTS reflection operations move to WebGPU; the remaining input-side operation stays in a CPU subgraph.

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES.
