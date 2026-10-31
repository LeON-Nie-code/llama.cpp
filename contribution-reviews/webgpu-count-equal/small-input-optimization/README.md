# Small-input COUNT_EQUAL optimization

Current implementation: `aacd8456dec97e2d7646d8f7d1a437c01cf2ece2`; original COUNT_EQUAL implementation: `e3ded50f4b2dea3e50d5f6f3f7ce594f5047096f`; upstream baseline: `0ec4027a2a0183de87c007069adddc06e665f7de`.

For 1–1024 elements, one 256-thread workgroup compares and reduces the inputs, then overwrites both words of the I64 result. This removes the separate clear dispatch and inter-workgroup atomic accumulation for small inputs. Empty inputs still clear; larger inputs retain the original multi-workgroup path. The cutoff is conservative and was tested on the L4; it is not claimed to be optimal on every adapter.

## Correctness

31/31 COUNT_EQUAL cases pass on both CPU and WebGPU. All 27 ARGMAX/SUM/SUM_ROWS regression cases pass. Exact-I64 checks cover poisoned outputs, repeated execution with changing inputs, offsets, overlapping bindings, empty inputs, 1023/1024/1025 boundaries, and large 2D dispatches through 67,108,872 elements. See [test log](webgpu-test.log), [CPU log](cpu-test.log), [exact check log](exact-replay.log) and [harness source](validate-count-equal.cpp). Formatting, EditorConfig and patch whitespace checks passed.

## Operator performance

NVIDIA L4, native Dawn, Release build. Three alternating repetitions; mean ± sample SD of reported microbenchmark latency. Before and after use the same executable and select only the backend library at runtime.

| Elements | Before (us) | Optimized (us) | Speedup |
|---:|---:|---:|---:|
| 256 | 15.50 ± 0.87 | 8.24 ± 0.25 | 1.882x |
| 1,024 | 15.36 ± 0.78 | 8.04 ± 0.41 | 1.910x |
| 1,025 | 15.34 ± 0.84 | 14.63 ± 0.94 | 1.048x |
| 4,096 | 15.35 ± 0.86 | 14.55 ± 0.87 | 1.055x |
| 65,536 | 15.08 ± 0.80 | 14.68 ± 0.87 | 1.027x |
| 1,048,576 | 23.36 ± 0.31 | 23.17 ± 0.01 | 1.008x |

Small-input latency falls about 47–48%, corresponding to about 1.9x throughput in this operator benchmark. Differences above the cutoff are small and are not claimed as benefits of the fast path. Browser WebGPU and other GPUs have not been tested.

## Complete-model validation performance

Each configuration predicts 4096 WikiText-2 tokens through the full `llama_opt_epoch` forward validation path, including loss, transfers and accuracy collection. No backward execution or weight updates. Five independent processes per variant; each performs one warm-up and one measured epoch. Order rotates/reverses across repetitions. Mean ± sample SD follows.

| Model | Context / batch | Counts / epoch | Upstream CPU fallback (ms) | Original WebGPU (ms) | Optimized WebGPU (ms) |
|---|---:|---:|---:|---:|---:|
| Stories260K F32 | 256 | 16 | 70.51 ± 0.78 | 70.25 ± 0.77 | 70.01 ± 0.21 |
| SmolLM2-135M-Instruct Q4_K_M | 256 | 16 | 2214.32 ± 44.79 | 2179.26 ± 34.17 | 2171.36 ± 20.10 |
| SmolLM2-135M-Instruct Q4_K_M | 1024 | 4 | 1947.14 ± 6.39 | 1937.68 ± 7.81 | 1955.05 ± 22.90 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 256 | 16 | 6135.14 ± 268.13 | 6035.15 ± 226.29 | 6063.84 ± 315.98 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 1024 | 4 | 5699.77 ± 74.27 | 5615.99 ± 87.70 | 5735.82 ± 101.03 |

All 75 timed processes succeeded, covering 840 measured COUNT_EQUAL executions; all predictions, reported loss and accuracy match across the three implementations and repetitions within each configuration. Separate diagnostic runs confirm CPU fallback upstream and actual GPU COUNT_EQUAL execution in both WebGPU implementations. These diagnostics are excluded from timing statistics.

These measurements do not establish a consistent end-to-end speedup. Small improvements and regressions vary across workloads, and the absolute operator savings are tiny beside model evaluation, CPU cross-entropy and dense-label transfers. The fast path reduces the targeted dispatch overhead; it does not remove those surrounding costs or the scalar readback.

## Evidence and reproduction

- [Methodology, environment and build/run instructions](METHODOLOGY.md)
- [Operator summary](operator-summary.json), [operator raw measurements](operator-results.json), [runner](run-perf.py)
- [Model summary](model-summary.json), [model raw measurements and traces](model-results.json), [runner](run-models.py), [evaluation source](eval-llm.cpp)
- [Pinned model/corpus inputs](../frequent-models/inputs.json)

Individual `.log` and `.command.json` files record every run. Library-resolution records and SHA-256 checksums identify the selected backend binaries. Initial grid-stride measurements are retained only as development history; top-level results describe the final implementation.
