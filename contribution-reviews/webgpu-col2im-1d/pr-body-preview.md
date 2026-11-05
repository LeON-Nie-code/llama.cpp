## Overview

Add WebGPU support for F32 `COL2IM_1D`, used by the Qwen3-TTS and PocketTTS waveform decoders. Each output gathers overlapping input columns without floating-point atomics.

Includes correctness and performance tests, offset-view and 2D-dispatch coverage, and updated operation support documentation. Related to #14909.

## Additional information

Tested on NVIDIA L4 with native WebGPU.

```sh
test-backend-ops test -b WebGPU -o COL2IM_1D
test-backend-ops perf -b WebGPU -o COL2IM_1D
```

- All 16 supported F32 cases passed against the CPU reference, covering overlap, cropping, stride gaps, offset views and 16,777,217-element dispatch.
- Release build of the test and TTS targets, targeted regression tests, clang-format 22 and EditorConfig checks passed.

Full TTS inference, 64 frames. Five runs per version, alternating execution order, with identical text, reference audio, seed and device settings. Times include reference-speaker encoding, prompt evaluation, generation and waveform decoding; model loading is excluded. Mean +/- SD.

| Model | Upstream (s) | PR (s) | Speedup |
|---|---:|---:|---:|
| PocketTTS | 1.158 +/- 0.035 | 0.838 +/- 0.075 | 1.38x |
| Qwen3-TTS 0.6B (Q4_K_M) | 33.299 +/- 0.361 | 5.676 +/- 0.202 | 5.87x |

Both versions produced byte-identical audio for each model. Traces confirm all `COL2IM_1D` executions moved from CPU to WebGPU: 32 for PocketTTS and 390 for Qwen3-TTS across each complete diagnostic run.

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES.
