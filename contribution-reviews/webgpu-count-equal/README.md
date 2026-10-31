# WebGPU COUNT_EQUAL contribution review

- [Implementation diff: six changed files](https://github.com/LeON-Nie-code/llama.cpp/compare/0ec4027a2a0183de87c007069adddc06e665f7de...preview/webgpu-count-equal)
- [PR body preview](pr-body-preview.md)
- [Current full-model benchmarks: three models that actually run COUNT_EQUAL](frequent-models/README.md)
- [Core implementation validation](core-validation.md)

The current report uses SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct and Stories260K on WikiText validation. Operator traces establish real COUNT_EQUAL execution in every configuration. Full-model timing is approximately unchanged; the evidence does not establish a substantial speedup.

The previous TTS measurements did not execute COUNT_EQUAL and have been removed from the PR preview. Their raw files and the earlier short-story validation data remain here as experimental history, not as evidence of COUNT_EQUAL acceleration.

Nothing has been submitted to ggml-org/llama.cpp. This separate branch contains personal-fork review material; the implementation branch retains only the six code/test/documentation changes.
