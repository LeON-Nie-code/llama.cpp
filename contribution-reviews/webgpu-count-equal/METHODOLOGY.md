# Full-model validation methodology

Baseline is upstream 0ec4027a2a0183de87c007069adddc06e665f7de. Candidate is e3ded50. The only code delta is WebGPU COUNT_EQUAL plus its tests/documentation.

Both variants use the same newly built llama-tts executable, libllama, libmtmd, CPU backend and common code. LD_LIBRARY_PATH selects the baseline or candidate WebGPU backend (and the identical, unchanged ggml-base source). The baseline backend is separately built from the exact baseline commit. The previous ARANGE PR is absent in both variants.

Hardware: NVIDIA L4, driver 550.163.01, native Dawn v20260908.214631, GCC 14.2, Release. All GPU workloads are run sequentially.

Models: PocketTTS English BF16; Qwen3-TTS 12Hz Base 0.6B and 1.7B, both Q4_K_M with matching Q8_0 mmproj. Use the same cached checkpoints, speaker reference, prompt and seed for baseline/candidate. Generate 16 codec frames (1.28 seconds of audio), greedy sampling, seed 1234, 2 CPU threads, context 1024, batch/ubatch 512, all supported layers on WebGPU, flash attention off. Actual argv and environment are saved for every process.

One diagnostic candidate run per model uses trace-ops.so. It counts COUNT_EQUAL construction and scans scheduled/executed graphs. Diagnostic runs are excluded from timing statistics. A CPU COUNT_EQUAL backend test is the positive control for the tracer.

Then one unmeasured warm-up per variant, followed by five measured runs per variant, alternating which variant runs first. Report mean and sample standard deviation of prompt evaluation + generation + waveform decoder time, excluding loading, from llama-tts logs. Also save whole-process wall time separately. Compare SHA-256 of decoded PCM audio, not WAV headers.

The machine needs driver-pin.so, the same local RTLD_NODELETE driver-module workaround used in the previous ARANGE experiments. It is applied equally to both variants, is outside the repository change and is not a model-performance optimization. The initial unpinned run finished inference and saved its WAV, then segfaulted during teardown; that run is excluded and its log is preserved. Subsequent runs must exit successfully to be accepted.

Ordinary generation has no COUNT_EQUAL call site in llama.cpp. These runs therefore measure full-model regression behavior, not expected acceleration. Full LLM forward validation through llama_opt_epoch, if measured, is reported separately and explicitly includes loss/accuracy statistics rather than autoregressive generation.

## Full LLM forward validation

Use Stories260K and Stories15M F32 checkpoints from ggml-org/tiny-llamas. llama_opt_epoch uses idata_split=0, so only its forward validation loop executes. The API still builds backward metadata and requires one selected F32 parameter; the harness selects output_norm.weight but executes no backward pass and no optimizer update. Context/batch/ubatch are 256, and dataset shape comes from actual llama_n_ctx to account for context padding. Four contexts (1024 tokens) per epoch, two CPU threads, flash attention disabled. The fixed story repeats across contexts.

One diagnostic process per variant uses the tracer and one warm-up plus one measured epoch. Five subsequent alternating process pairs per model use one warm-up plus six measured epochs each. Statistics use the five process means, rather than treating all within-process epochs as independent samples. Every predicted token, loss and accuracy is checked for equality across baseline/candidate and repetitions.

The initial validation harness attempts used no selected parameter, then a context smaller than the actual padded context, and hit API assertions. Those setup failures are retained in initial-eval-setup-failure. The corrected harness was used unchanged for both variants; no llama.cpp implementation was modified to make the benchmark work.
