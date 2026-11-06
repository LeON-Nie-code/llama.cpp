# Chatterbox Turbo full-model comparison

Same base/candidate libraries as Qwen tests, native Dawn on L4. Official ggml-org weights pinned in model-manifest.json. Same fixed prompt/settings, Q8_0 backbone/F16 mmproj, 2 CPU threads, seed1234, temperature0, FA off, WebGPU model and mmproj. Five measured fresh processes per version alternating order after excluded diagnostic pair. Includes input preparation/prompt/generation/vocoder, excludes model loading. Diagnostic tracing not active in measured runs.

| Request | Upstream seconds | Candidate seconds |
|---|---:|---:|
| Short (64 generated frame limit, 2.68s audio) | 1.783 +/- 0.071 | 1.763 +/- 0.044 |
| Long (256 generated frame limit, 10.36s audio) | 4.409 +/- 0.177 | 4.417 +/- 0.143 |

Audio byte-identical across variants and repetitions per request. One PAD_REFLECT_1D call moves CPU->GPU per utterance. Short input shape [16080,64] (~3.93 MiB); long input [62160,64] (~15.18 MiB). The pipeline runs one whole-utterance vocoder invocation rather than per-token reflection. CONV_TRANSPOSE_1D (3 executions) and LEAKY_RELU (5) remain CPU in the diagnostic run; no change to those ops is included. Backend boundaries therefore remain nearby.

No clear end-to-end performance benefit was observed. The short case's ~1.1% lower mean is smaller than its variability, and the long case is essentially unchanged. This is not evidence of a reliable speedup or regression. No attempt was made to select only favorable runs or broaden the operator implementation to improve benchmark results.
