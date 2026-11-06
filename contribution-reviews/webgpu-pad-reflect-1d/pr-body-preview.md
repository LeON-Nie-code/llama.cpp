## Overview

Add WebGPU support for F32 `PAD_REFLECT_1D`, reusing the existing padding shader with a separate reflection variant. Used by the Qwen3-TTS speaker encoder and Chatterbox vocoder.

Extend the existing tests with asymmetric and zero padding, offset views and 2D dispatch. Correct the test tensor construction to exercise all four dimensions, and update operation support documentation. Related to #14909.

## Additional information

Tested on NVIDIA L4 with native WebGPU.

```sh
test-backend-ops test -b WebGPU -o PAD_REFLECT_1D,PAD
test-backend-ops perf -b WebGPU -o PAD_REFLECT_1D
```

All 44 reflection and existing padding cases passed against the CPU reference, including circular padding and 67,108,868-element dispatch. Build, clang-format 22 and EditorConfig checks passed.

Full Qwen3-TTS 0.6B Q4_K_M inference, 16 frames, with matching Q8_0 mmproj. Five runs per version, alternating execution order, with the same prompt, reference audio, seed and device settings. Includes reference-speaker encoding, prompt processing, generation and waveform decoding; excludes model loading. Mean +/- SD.

| Model | Upstream (s) | PR (s) |
|---|---:|---:|
| Qwen3-TTS 0.6B | 9.637 +/- 0.069 | 9.663 +/- 0.078 |

Both versions produced byte-identical audio. Traces confirm 21 of 22 reflection operations move to WebGPU; the input-side operation remains in a CPU subgraph. The timing difference is within run-to-run variation: these operations are a small part of reference-speaker encoding, rather than the main generation loop.

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES.
