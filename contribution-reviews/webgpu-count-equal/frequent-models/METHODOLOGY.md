# COUNT_EQUAL-using complete-model workloads

This experiment only times model workflows that execute COUNT_EQUAL. The prior TTS generation measurements are excluded from the PR preview because those graphs do not use this operator.

## Models and dataset

- SmolLM2-135M-Instruct Q4_K_M: bartowski/SmolLM2-135M-Instruct-GGUF, revision 09816acd5d99df7be770d85ea30822623dab342c.
- Qwen2.5-0.5B-Instruct Q4_K_M: official Qwen/Qwen2.5-0.5B-Instruct-GGUF, revision 9217f5db79a29953eb74d5343926648285ec7e67.
- Stories260K F32: ggml-org/tiny-llamas, revision def3e2dd70df35ecbf6403ea347de4c5977220c1. This is an additional fast full-model case to test more COUNT_EQUAL calls per second; only context 256 is used, within its training context.
- WikiText-2 raw validation split, downloaded from the same ggml-org/ci archive used by llama.cpp/scripts/get-wikitext-2.sh. Checksums and model revisions are in inputs.json. Do not repeat the old short story fixture.

The first 4097 tokens of the actual validation corpus provide 4096 next-token input/label pairs. Tokenization differs by model. Context boundaries occur every 256 or 1024 tokens. Both baseline and candidate use the identical token windows for each configuration. Accuracy is checked for equality across baseline/candidate, not compared between different context lengths or models.

## Execution and COUNT_EQUAL placement

Use the repository's complete llama_opt_epoch forward validation path with idata_split=0. Each batch performs the full model forward pass, cross-entropy loss, predictions, label ARGMAX and COUNT_EQUAL token accuracy. There is one COUNT_EQUAL per batch, comparing one predicted token against its label for every token position. These are full model validation workloads, not ordinary chat generation and not stand-alone operator graphs.

Before timing, pilot.py checks each model at context/batch/ubatch 256 with 8 batches. One warm-up plus one diagnostic epoch must execute 16 COUNT_EQUAL nodes: all CPU in baseline and all WebGPU in candidate. It also checks every prediction for exact equality. Both pilots passed.

The matrix separately confirms placement for each model and each batch size before its timed series. Context, batch and ubatch are the same size. There are 4096 total token predictions per epoch: 16 batches at 256, or 4 batches at 1024. Diagnostic graphs are inspected with trace-ops.so, separately from timed runs. Graph scans at the scheduler level are not counted as executed operations; only backend-specific COUNT_EQUAL counts establish CPU/GPU placement.

## Timings and correctness

Five independent process pairs per configuration, alternating baseline/candidate order. Each process loads the model, executes one unmeasured warm-up epoch and two measured validation epochs. Summary mean and sample standard deviation are calculated over the five process means, without treating within-process epochs as independent samples. Record all timings; do not selectively drop slow runs.

Epoch time starts before llama_opt_epoch and ends after it, so it includes input/label preparation inside that API, graph preparation, all model computation, CPU/GPU transfers, loss, accuracy and result collection. Excludes initial model loading, tokenization, dataset construction and prediction-file writing. Two CPU threads, flash attention disabled, all supported layers offloaded to the NVIDIA L4. No backward pass or optimizer update executes. The API builds backward metadata and requires one selected parameter; select the F32 output norm parameter solely to satisfy this requirement.

Check SHA-256 of every predicted token across the whole process and exact reported loss/accuracy for both measured epochs, all trials, baseline and candidate. Results are only accepted if the process exits successfully and these checks pass.

## A/B isolation

Baseline: 0ec4027a2a0183de87c007069adddc06e665f7de. Candidate: e3ded50f4b2dea3e50d5f6f3f7ce594f5047096f. Same freshly built executable and unchanged llama/CPU/common code; LD_LIBRARY_PATH selects the exact baseline or candidate WebGPU backend. The earlier ARANGE PR is absent in both. Driver 550.163.01, native Dawn v20260908.214631, GCC 14.2, Release.

Use the same existing driver-pin.so RTLD_NODELETE workaround on both variants to avoid NVIDIA driver teardown crashes. This does not change llama.cpp, the model or COUNT_EQUAL. Reproduction sources for the pinning helper and operator tracer are in ../ alongside the previous benchmark support files. GPU workloads run sequentially.

## Reproduction

Run download-inputs.py to fetch the pinned checkpoints and validate all checksums. Compile eval-llm.cpp against the same current implementation build:

```sh
python3 download-inputs.py
# For a standalone copy of this report, build the helpers in the parent directory:
c++ -shared -fPIC -O2 ../driver-pin.cpp -ldl -o ../driver-pin.so
c++ -shared -fPIC -O2 -std=c++17 ../trace-ops.cpp \
  -I/tmp/llama-count-equal-work/ggml/include -L/tmp/llama-count-equal-build/bin \
  -lggml-base -ldl -o ../trace-ops.so
c++ -O2 -std=c++17 eval-llm.cpp \
  -I/tmp/llama-count-equal-work/include -I/tmp/llama-count-equal-work/ggml/include \
  -L/tmp/llama-count-equal-build/bin -Wl,-rpath,/tmp/llama-count-equal-build/bin \
  -lllama -lggml -lggml-base -o eval-llm
python3 pilot.py
python3 run-matrix.py
python3 run-small.py
```

After the two-model matrix, run-small.py tests the complete Stories260K model at context 256 using exactly the same WikiText token count, warm-up, paired process count and measured epoch count. This additional case is reported regardless of whether it improves performance.

Every *.command.json records the exact arguments and relevant environment used. Matching *.log files contain raw model logs, per-epoch measurements and placement traces for diagnostic runs. Model weights, dataset text and binary prediction outputs are not added to the fork.
