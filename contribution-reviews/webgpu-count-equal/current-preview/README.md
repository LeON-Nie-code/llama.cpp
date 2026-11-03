# Current COUNT_EQUAL preview

The implementation is rebased onto upstream `ca8daad199a9c12c5b54643b974c640939a0f7bd` at `83dbf9b9556c51adc875809d4b110d7e3368301e`. The kernel and support logic are unchanged from the earlier reviewed implementation; the base now includes ARANGE and the upstream WebGPU graph-reservation fix. The implementation diff remains six files, 245 insertions and four deletions.

- [Code diff](https://github.com/LeON-Nie-code/llama.cpp/compare/ca8daad199a9c12c5b54643b974c640939a0f7bd...preview/webgpu-count-equal-ready)
- [PR body preview](../pr-body-preview.md)
- [Build and validation results](checks.json)
- [Fresh complete-model results](model-summary.json)
- [Fresh operator timings](perf-summary.json)
- [Duplicate inspection of 16 open WebGPU PRs](overlap-audit.json)

## Validation

Full native Release build passed. CPU COUNT_EQUAL 31/31; WebGPU COUNT_EQUAL 31/31 and ARGMAX/SUM/SUM_ROWS 27/27 with two workers; exact-I64 high-word, poison/replay, empty/offset/overlap/cutoff/large-dispatch harness passed. CTest test-opt and test-backend-tensor passed. clang-format 22 across the complete WebGPU directory, repository-wide EditorConfig, regenerated operation docs and git diff --check passed. The full upstream CI hardware matrix was not run in this refresh.

## Fresh model methodology

Both variants use upstream `ca8daad199a9c12c5b54643b974c640939a0f7bd` application/CPU/common code. The only selected difference is the WebGPU backend, with or without COUNT_EQUAL support. A copy of the exact upstream backend is selected using LD_LIBRARY_PATH; the PR backend is built from the refreshed branch. Binary hashes are recorded in metadata.json.

Same three pinned models and WikiText-2 inputs as [the input manifest](../frequent-models/inputs.json). Stories260K uses context/batch/ubatch 256; SmolLM2-135M and Qwen2.5-0.5B use 256 and 1024. Each epoch predicts 4096 tokens. Five alternating independent process pairs per configuration; one warm-up and one measured forward validation epoch per process, with no backward execution or optimizer updates. Timings include graph preparation, model forward, transfers, loss and accuracy collection; exclude model loading/tokenization and prediction-file writing. Two CPU threads, flash attention disabled, F32 KV. No concurrent GPU jobs.

Separate traced processes confirm CPU COUNT_EQUAL upstream and actual GPU execution in the PR. Traces are excluded from timings. Predictions, reported loss and accuracy match across both variants and all repetitions. There are 50 measured processes and 560 measured COUNT_EQUAL calls. The same driver-module pinning workaround is applied to both variants. The new measurements replace the old-base timings in the PR preview.

Evidence storage filled after a successfully completed process. The task artifacts were moved to /tmp, and the runner resumed from completed logs with successful-exit checks. No completed timing samples were discarded. The original interrupted progress log and resumed progress log are both retained.

| Model | Context / batch | Upstream (ms) | PR (ms) |
|---|---:|---:|---:|
| Stories260K F32 | 256 | 71.73 +/- 1.71 | 70.75 +/- 1.99 |
| SmolLM2-135M-Instruct Q4_K_M | 256 | 2215.11 +/- 53.89 | 2174.81 +/- 8.57 |
| SmolLM2-135M-Instruct Q4_K_M | 1024 | 1940.22 +/- 11.88 | 1935.61 +/- 13.08 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 256 | 5864.41 +/- 35.00 | 5778.36 +/- 17.46 |
| Qwen2.5-0.5B-Instruct Q4_K_M | 1024 | 5615.17 +/- 137.61 | 5551.35 +/- 64.16 |

Mean +/- sample SD across five independent samples. These measurements do not establish a consistent full-model speedup. COUNT_EQUAL's per-batch reduction is small relative to the forward, cross-entropy, label/logit transfers and readback costs.

## Reproduction

Build both revisions with the same Release flags and Dawn release. Compile eval-llm.cpp and validate-count-equal.cpp against the PR build. run-models.py records every command and checks successful exit, exact predictions, reported loss/accuracy and traced backend placement. The inputs and helper sources are shared with the preceding reports. For a report-only checkout, run-models.py also accepts helper shared libraries in the parent directory.

Only text evidence is published here; model weights, corpus text, shared libraries, binaries and prediction files are excluded. No upstream PR or comment has been created.
