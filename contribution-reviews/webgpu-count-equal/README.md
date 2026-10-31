# WebGPU COUNT_EQUAL contribution review

Implementation diff: [view the six changed files](https://github.com/LeON-Nie-code/llama.cpp/compare/0ec4027a2a0183de87c007069adddc06e665f7de...preview/webgpu-count-equal).

[Private PR body preview](pr-body-preview.md). Nothing has been submitted to the official ggml-org repository. This evidence branch is only in the personal fork and is separate from the six-file implementation branch.

## What the measurements show

COUNT_EQUAL is used by ggml-opt for accuracy statistics, not ordinary autoregressive generation. Three complete speech models were measured and runtime-traced; none created or executed COUNT_EQUAL. Two complete language models were also tested through forward validation, where the operator did move from CPU to WebGPU. Those validation timings were approximately unchanged. We do not claim a meaningful full-model speedup.

## Full generation inference

Five measurements per variant, mean +/- sample SD; 16 codec frames; 1.28 seconds of PCM audio. Excludes model loading. Full prompt processing, autoregressive generation and waveform decoding execute in every run.

| Model | Baseline (s) | COUNT_EQUAL branch (s) | PCM identical | COUNT_EQUAL calls |
|---|---:|---:|:---:|---:|
| PocketTTS English (BF16) | 0.408 +/- 0.004 | 0.404 +/- 0.005 | yes | 0 |
| Qwen3-TTS 0.6B (Q4_K_M) | 10.440 +/- 0.107 | 10.266 +/- 0.078 | yes | 0 |
| Qwen3-TTS 1.7B (Q4_K_M) | 10.640 +/- 0.343 | 10.434 +/- 0.149 | yes | 0 |

## Full LLM forward validation

Four contexts of 256 tokens per epoch. Five process pairs per model, each with one warm-up and six measured epochs; SD is across process means. Includes graph preparation, model forward computation, cross-entropy loss and token accuracy. No backward execution or weight updates. Dataset is a fixed short story repeated into contexts and is a performance fixture, not an accuracy benchmark.

| Model | Baseline (ms/epoch) | COUNT_EQUAL branch (ms/epoch) | Predictions / loss / accuracy |
|---|---:|---:|---|
| stories260K (F32) | 17.72 +/- 0.16 | 17.85 +/- 0.50 | identical |
| stories15M (F32) | 275.85 +/- 2.44 | 274.71 +/- 0.91 | identical |

The diagnostic runs execute eight validation batches each. Baseline: eight COUNT_EQUAL executions on CPU. Candidate: eight on WebGPU. Executed backend subgraphs decreased by one per batch. Both checkpoints are public [ggml-org TinyStories models](https://huggingface.co/ggml-org/tiny-llamas/tree/main); checksums are recorded in model-checksums.json.

## Reproduction and evidence

- [Methodology and environment](METHODOLOGY.md)
- [Generation summary](summary.json), [raw measurements](results.json)
- [Forward validation summary](eval-summary.json), [raw measurements](eval-results.json)
- [Generation runner](run-fullmodel.py), [validation runner](run-eval.py), [full-model validation harness](eval-llm.cpp)
- [Operator tracer](trace-ops.cpp), [positive control](trace-positive-control.log)
- `*.command.json` records the actual argv and relevant environment for each subprocess. Matching `*.log` files contain raw model timing and diagnostic output.
- [Core implementation validation](core-validation.md), `webgpu-test.log`, `cpu-test.log`, `exact-replay-test.log`, `reduction-regression.log`, `webgpu-perf.log`.

Both variants use the same freshly built application and unchanged llama/mtmd/CPU code. The WebGPU backend is selected with LD_LIBRARY_PATH from exact baseline and candidate builds. Driver module pinning is applied equally, following the existing ARANGE experiment setup. Initial unsuccessful harness setup runs are retained separately and excluded from results. Generation was rerun from scratch after the driver workaround was applied. There are no dropped runs in the final repeated measurement series.

This branch contains review material, not additional changes intended for an upstream PR.
