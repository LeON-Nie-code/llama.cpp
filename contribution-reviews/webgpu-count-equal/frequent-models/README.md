# Complete models that actually execute COUNT_EQUAL

This report focuses only on full-model workflows that run COUNT_EQUAL. All three selected models perform token accuracy calculation through the existing llama_opt_epoch forward validation path. No ordinary generation timings are used as evidence for COUNT_EQUAL performance.

- SmolLM2-135M-Instruct Q4_K_M, pinned bartowski checkpoint.
- Qwen2.5-0.5B-Instruct Q4_K_M, pinned official Qwen checkpoint.
- Stories260K F32, ggml-org checkpoint: a fast full-model case that invokes COUNT_EQUAL more frequently per second.

## Results on NVIDIA L4

WikiText-2 raw validation data; 4096 next-token predictions per epoch. Full model forward pass, graph preparation, transfers, cross-entropy loss, predictions and accuracy collection are timed. Loading and tokenization are excluded. Context, batch and ubatch have the same size. Two CPU threads, flash attention disabled, all supported layers on WebGPU. No backward execution or weight update.

Five alternating process pairs per configuration. Each process runs one warm-up epoch and two measured epochs. Values are mean +/- sample SD of the five process means. There are 50 timed processes and 100 measured epochs across the five configurations. COUNT_EQUAL executes once per batch, for 1120 executions in the measured epochs (560 per variant).

| Model | Context / batch | COUNT_EQUAL per epoch | Baseline (ms) | Candidate (ms) | Baseline / candidate |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct Q4_K_M | 256 | 16 | 2168.12 +/- 30.84 | 2165.56 +/- 37.33 | 1.0012x |
| SmolLM2-135M-Instruct Q4_K_M | 1024 | 4 | 1930.24 +/- 9.04 | 1940.47 +/- 17.87 | 0.9947x |
| Qwen2.5-0.5B-Instruct Q4_K_M | 256 | 16 | 5753.11 +/- 38.27 | 5770.50 +/- 32.97 | 0.9970x |
| Qwen2.5-0.5B-Instruct Q4_K_M | 1024 | 4 | 5563.95 +/- 41.72 | 5561.65 +/- 34.01 | 1.0004x |
| Stories260K F32 | 256 | 16 | 73.08 +/- 3.09 | 71.74 +/- 3.41 | 1.0187x |

## Confirmed operation usage

Each configuration has separate baseline and candidate diagnostic runs with operator traces, excluded from timing statistics. They confirm COUNT_EQUAL executes on CPU in the baseline and WebGPU in the candidate. For context 256, one warm-up plus one diagnostic epoch executes 32 COUNT_EQUAL nodes per variant; for context 1024, it executes 8. Scheduler graph scans are not mistaken for operation executions. Before the matrix, both larger models also passed an independent pilot gate.

- Every predicted token is identical across variants and repetitions for the same configuration.
- Reported loss and accuracy are identical across variants and repetitions.
- One fewer backend subgraph executes per validation batch after the change.
- The Stories260K candidate executes 16 counts in roughly 72 ms, about 220 COUNT_EQUAL calls per second, so the additional small-model case tests higher call frequency.

## Interpretation

SmolLM2 and Qwen2.5 show approximately unchanged full-model validation time at both batch sizes. Stories260K has a nominal 1.9% improvement, but the difference is smaller than the run-to-run standard deviation; it does not establish a stable speedup. These measurements support backend correctness and reduced CPU fallback, not a claim of substantial full-model acceleration.

COUNT_EQUAL usage is determined by the accuracy-statistics workflow. The current llama.cpp generation paths do not call it; there is no model architecture here that invokes it in every transformer layer. This experiment exercises the actual supported call path instead of adding artificial COUNT_EQUAL nodes to a generation graph.

## Evidence and reproduction

- [Exact methodology](METHODOLOGY.md)
- [Combined summary](combined-summary.json)
- [Larger-model raw results](results.json), [fast-model raw results](small-results.json)
- [Independent pilot gate](pilot-results.json)
- [Complete model evaluation harness](eval-llm.cpp)
- [Input download and checksum verification](download-inputs.py), [pinned input manifest](inputs.json)
- [Two-model runner](run-matrix.py), [fast-model runner](run-small.py), [pilot runner](pilot.py)
- `*.command.json` records exact argv and relevant environment; matching `*.log` files contain model logs, raw per-epoch times and diagnostic placement counters.
- Operator tracer and driver workaround sources remain in the parent directory. Both variants use identical driver pinning and unchanged application code; only the WebGPU implementation differs.

[Implementation diff](https://github.com/LeON-Nie-code/llama.cpp/compare/0ec4027a2a0183de87c007069adddc06e665f7de...preview/webgpu-count-equal). This evidence is published only to the personal fork; no upstream PR or messages have been submitted.
