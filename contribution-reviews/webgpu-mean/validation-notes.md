# WebGPU MEAN validation

Code branch: `preview/webgpu-mean`
Base: `b74298c109c107494b792c30af4bb8fa0ce1eb2f`
Candidate: `afcd18a` (code and documentation only)

## Scope and implementation review

- F32 input/output, contiguous elements within each row. Higher dimensions may be strided/permuted.
- Existing row-reduction shader and binding infrastructure reused. Mean and sum pipeline variants are cached separately.
- Added flattened dispatch output indexing and an exact row bound. The shared shader must handle padded 2D dispatch for MEAN; SUM_ROWS receives the same correction.
- Uniform data layout, offset units, bounds and dispatch-uniform early return reviewed.
- No public API, dependency, other backend, model graph or sampler changes.

## Validation

- Full default Release build: passed (including server, CLI, TTS and test targets).
- `test-backend-ops test -b WebGPU -o MEAN,SUM_ROWS,SUM`: 38/38 supported cases passed; MEAN 17/17, SUM_ROWS 11/11, SUM 10/10.
- CPU counterpart: 40/40 passed. Two additional SUM layouts are not advertised as supported on WebGPU.
- 49 selected local CTest cases passed. The first run passed 47; Jinja Python and FP8 conversion initially lacked system-Python dependencies, then both passed using isolated dependencies and an existing Python environment. Tests requiring model downloads/generation and unrelated performance/model suites were excluded; this is not the full 66-case CTest inventory.
- Full WebGPU C/C++ clang-format 22 check, changed-file EditorConfig, and git whitespace checks passed.
- The added 65,537-row SUM_ROWS regression fails with the upstream WebGPU library and passes with the candidate.
- Independent checks passed for 4-byte input offsets, padded row strides, repeated graph execution, mixed MEAN/SUM_ROWS pipelines, and capability rejection of F16/strided-element inputs.

## Model measurements

NVIDIA L4, native Dawn WebGPU. Both variants use the same freshly built executable and core libraries; only the ggml backend/base libraries are switched to the saved, unmodified upstream build. Models and assets already present locally were reused.

Five measured runs per model/version, alternating execution order, after an excluded diagnostic/warm-up pair. PocketTTS: 64 frames; Qwen3-TTS 0.6B Q4_K_M with Q8_0 mmproj: 16 frames. Seed 1234, two CPU threads, WebGPU for backbone and mmproj, flash attention off. Identical prompt/reference voice and settings per comparison.

Timing uses a local copy of llama-tts with microsecond-resolution timing and an added timer around set_input. The original CLI starts its reported total after set_input, omitting Qwen speaker encoding; the reported full-inference values here include that stage. Model loading and input file loading are excluded. No instrumentation is included in the proposed code diff.

PocketTTS: 1.068 +/- 0.029 s upstream; 1.035 +/- 0.033 s candidate (3.2% lower mean, modest relative to variability). Qwen: 9.634 +/- 0.147 s upstream; 9.685 +/- 0.112 s candidate (0.5% higher mean, smaller than run-to-run variability). These measurements do not establish a statistically reliable end-to-end speedup.

Traces show 256 PocketTTS MEAN executions moving CPU -> WebGPU and 7 Qwen executions moving CPU -> WebGPU. The trace also counts scheduler-level nodes; those are not additional executions.

PocketTTS audio is byte-identical across variants/runs. Qwen audio is repeatable within each variant but differs between them. A diagnostic callback checked every executed mean against an independent double-precision reference: Qwen maximum absolute error 7.34e-7, maximum normalized mean squared error 2.74e-15. Final speaker-embedding NMSE 3.80e-8, cosine similarity 0.999999981. PocketTTS means were also checked against a double-precision reference. No bit-identical Qwen output claim is made.

## PR search

At final audit on 2026-11-03, no existing PR with WebGPU MEAN in its title was found. The earlier broader keyword search and potentially overlapping open WebGPU PR patch audit also found no implementation. Maintainer-requested missing-op umbrella: https://github.com/ggml-org/llama.cpp/issues/14909 . No upstream PR or message was created for MEAN.
