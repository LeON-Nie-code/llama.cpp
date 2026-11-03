# WebGPU COUNT_EQUAL contribution review

- [Current implementation diff](https://github.com/LeON-Nie-code/llama.cpp/compare/ca8daad199a9c12c5b54643b974c640939a0f7bd...preview/webgpu-count-equal-ready)
- [Updated PR body preview](pr-body-preview.md)
- [Current-base validation, fresh model/operator timings and duplicate audit](current-preview/README.md)
- [Previous small-input optimization measurements](small-input-optimization/README.md)
- [Earlier diagnostic breakdown](analysis/README.md)

The code is rebased onto current upstream after ARANGE and the graph-reservation fix merged. The refreshed diff contains only COUNT_EQUAL support, tests and support documentation. Old branches and measurements remain for history; the PR preview uses the fresh current-base measurements.

The complete-model measurements exercise COUNT_EQUAL for validation accuracy and do not establish a consistent end-to-end speedup. Nothing has been submitted to ggml-org/llama.cpp; these are personal-fork previews for human review.
