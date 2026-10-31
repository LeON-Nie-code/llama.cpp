# Small-input COUNT_EQUAL optimization: validation and benchmarks

Implementation: aacd845, on top of the original COUNT_EQUAL implementation e3ded50f4b2dea3e50d5f6f3f7ce594f5047096f. Upstream reference: 0ec4027a2a0183de87c007069adddc06e665f7de.

## Implementation scope

For 1 through 1024 contiguous I32 elements, dispatch one 256-thread workgroup. Each invocation compares at most four pairs. A workgroup reduction produces the count; invocation zero writes both 32-bit words of the I64 result, including a zero high word. This overwrites poisoned or previous results without a separate clear dispatch.

Empty inputs still clear the output. Larger inputs retain the multi-workgroup comparison and partial-count accumulation path, with an initial clear. The large-input shader's comparison control flow is preserved with conditional compilation; only the small-input variant loops over several elements per invocation. Both paths keep existing input-overlap bindings, alignment handling and size gating.

Boundary tests include 1023/1024/1025 elements and identical-input variants on both sides of the cutoff. The separate exact-I64 harness poisons outputs with -1, changes inputs between repeated graph executions, checks high words, offset views and overlapping views, and checks large 2D grids. It tests empty through 67,108,872-element inputs.

## Correctness and style

- CPU COUNT_EQUAL: 31/31 cases passed.
- L4 COUNT_EQUAL plus ARGMAX/SUM/SUM_ROWS: 58/58 passed (31 count cases and 27 neighbors).
- Exact I64 / cached replay / empty / offset / overlap / cutoff / large 2D checks passed.
- clang-format 22.1.0, EditorConfig 3.0.3 and git diff --check passed.
- Native Release build, GCC 14.2, Dawn v20260908.214631, NVIDIA L4 with driver 550.163.01.
- Browser WebGPU and other GPUs were not tested.

## Operator benchmarks

Use the same freshly built test-backend-ops executable for both implementations, so the data and cases are identical. LD_LIBRARY_PATH selects either the saved pre-optimization WebGPU library or the current optimized library. The original library was copied before editing. Three alternating repetitions per variant across 256, 1024, 1025, 4096, 65536 and 1048576 elements. Report mean and sample SD of the three tool-reported latencies.

The initial grid-stride implementation also changed large-input control flow and showed small latency differences outside the target range. It was replaced with conditional compilation that preserves the original large-input path. Initial measurements are retained in initial-grid-stride-perf as development history; all operator-results/operator-summary files at this directory's top level refer to the final implementation.

## Full-model three-way comparison

Use exactly the previous complete COUNT_EQUAL-using workloads: Stories260K F32 at context/batch/ubatch 256; SmolLM2-135M-Instruct Q4_K_M and Qwen2.5-0.5B-Instruct Q4_K_M at 256 and 1024. Same pinned checkpoints and WikiText-2 validation split as ../frequent-models/inputs.json. Every epoch predicts the first 4096 token positions, requiring 16 count operations at batch 256 or four at batch 1024.

Compare three variants in the same experiment:

1. upstream: native upstream WebGPU backend, COUNT_EQUAL falls back to CPU;
2. before: original two-dispatch WebGPU COUNT_EQUAL implementation e3ded50;
3. optimized: single-workgroup fast path aacd845.

Same evaluation executable and all unchanged llama/CPU/common code. The only implementation difference selected at runtime is the WebGPU backend. Library resolution is recorded in the three *-libraries.txt files. Both WebGPU COUNT_EQUAL variants are confirmed by separately traced diagnostic runs; the upstream variant must execute counts on CPU. Traces include one warm-up plus one diagnostic epoch and are excluded from timing statistics.

Five independent process groups per configuration, rotating/reversing variant order. Every process executes one unmeasured warm-up epoch and one measured epoch. Report mean and sample SD of the five independent measured epochs. There are 75 measured processes across the five configurations and three variants. This uses one measured epoch per process rather than the earlier two, keeping fresh three-way comparisons practical while retaining five independent samples per variant.

Epoch time includes the existing llama_opt_epoch forward validation path: graph preparation, model computation, transfers, cross-entropy, predictions and accuracy collection. Excludes model loading, corpus tokenization, initial dataset construction and prediction-file writing. There is no backward execution or optimizer update. Two CPU threads; flash attention disabled. Exact predictions, reported loss and accuracy must agree across all three variants and all trials. All accepted processes must exit successfully.

GPU jobs run sequentially. The existing driver-pin.so workaround is applied equally to all model variants; diagnostic tracing is only enabled in the separate diagnostic processes. Neither the earlier ARANGE change nor optimizations to loss/labels/readbacks are included.

## Reproduction

Build the current branch and prepare the baseline and e3ded50 WebGPU libraries with matching Release/Dawn flags. The runner expects the pre-optimization libggml-webgpu.so.0 at /tmp/llama-count-equal-original-backend, the baseline at /tmp/llama-count-equal-baseline-build/bin and the current build at /tmp/llama-count-equal-build/bin. Check the recorded command JSONs for actual paths and environment.

Model input download/checksum script and manifest are in ../frequent-models. Helper sources trace-ops.cpp and driver-pin.cpp are in the parent report directory. Compile their shared libraries there for a standalone report checkout; scripts also accept the existing local ../fullmodel helper directory.

```sh
c++ -O2 -std=c++17 eval-llm.cpp \
  -I/tmp/llama-count-equal-work/include -I/tmp/llama-count-equal-work/ggml/include \
  -L/tmp/llama-count-equal-build/bin -Wl,-rpath,/tmp/llama-count-equal-build/bin \
  -lllama -lggml -lggml-base -o eval-llm
python3 run-perf.py
python3 run-models.py
```

Source and raw logs are review evidence only. Model weights, dataset text and binary prediction files are not added to the fork. This branch does not submit anything to ggml-org/llama.cpp.
