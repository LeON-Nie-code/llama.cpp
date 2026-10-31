# Why COUNT_EQUAL did not noticeably accelerate full-model validation

The relevant bottleneck is the fraction of end-to-end work that changes. COUNT_EQUAL compares one predicted token ID against one label ID per token position. At batch 256 it compares 256 I32 pairs, reading 2048 bytes. It does not process every vocabulary logit or run in every transformer layer.

## Separate diagnostic run

Added host-side interception around graph allocation/evaluation, backend graph execution, tensor copies, label initialization and result reads. Ran the same three complete WikiText validation models, batch/context 256, 4096 tokens, one warm-up and one diagnostic epoch per variant. Predictions match exactly across baseline/candidate. This is a diagnostic breakdown, not a replacement for the earlier five-process repeated performance measurements.

CPU backend graph execution is synchronous, so the CPU graph timing includes the actual CPU computation and its graph/thread-pool overhead. GPU encode/submit time is not GPU kernel time. Tensor copy/read/synchronize wall time may include waiting for preceding GPU work. Several raw buckets are nested within opt_eval and other operations: do not sum all raw buckets or interpret every wait as pure copy bandwidth.

## Qwen2.5-0.5B example

| Item | Baseline | Candidate |
|---|---:|---:|
| Whole validation epoch | 5722.819 ms | 5776.155 ms |
| CPU cross-entropy graph | 1394.472 ms | 1397.949 ms |
| Large device-to-host copies | 653.657 ms | 661.821 ms |
| Host-to-device copies | 1925.368 ms | 1955.789 ms |
| Dense label zeroing | 336.945 ms | 340.113 ms |
| CPU COUNT_EQUAL graph | 1.066 ms | 0.000 ms |
| Prediction tensor_get calls, inclusive | 254.744 ms | 261.270 ms |
| Correct-count tensor_get calls | 0.002 ms | 1.277 ms |

The baseline CPU COUNT_EQUAL graphs total 1.066 ms out of 5722.819 ms, about 0.019%. The operator is genuinely used, but removing that CPU work alone is too small to visibly change whole-model time. This does not count possible synchronization/transfer savings; those must be analyzed separately.

Cross-entropy still executes on CPU: all 16 CROSS_ENTROPY_LOSS nodes run on CPU in both variants. COUNT_EQUAL has moved, but this much larger fallback has not.

Runtime transfer counters show 2,489,319,424 bytes of large device-to-host copies and 2,537,619,456 bytes of host-to-device copies per epoch, unchanged after the change: about 5.03 GB total. Their host-side calls take about 2.58 seconds in the baseline, including relevant waits.

The tensor sizes explain the scale. Qwen has 151936 vocabulary entries. A F32 [151936,256] tensor occupies 155582464 bytes. Across 4096 token positions, one such logit/label stream contains 2489319424 bytes. Source builds labels as a dense one-hot tensor matching the logits, zeros it, sets the label entries, computes CPU cross-entropy, and ARGMAXes both logits and labels for accuracy. Almost all of this work is unaffected by COUNT_EQUAL support. The original sparse labels are already available in llama-context.cpp; bypassing dense labels would be a separate architectural/API change.

## Fewer copies and subgraphs do not mean all GPU waits disappeared

Before the change, the CPU count graph needs its two I32 input vectors. The diagnostic sees 32 small device-to-host copy calls totaling 32768 bytes across 16 batches. These scheduler copies disappear after the change.

However, ggml_opt_eval still unconditionally reads predictions and the I64 correct-count result back to host to populate ggml_opt_result. The GPU result is not consumed entirely on GPU. The baseline's prediction tensor_get calls total about 255 ms, including the scheduler's early read; after the change the result-stage prediction reads total about 261 ms. Work that previously blocked an early read can instead block the later read, so disappearing early-copy time is not automatically a latency saving.

The correct-count result was already in CPU memory before the change (about 0.002 ms total to retrieve). It now needs 16 GPU readbacks of 8 bytes, taking about 1.277 ms total in this diagnostic, including transfer/map/synchronization overhead. Fewer bytes do not remove the fixed readback cost.

## Small-input GPU overhead

The current implementation encodes a memset dispatch to clear the I64 output, followed by the comparison/reduction dispatch. These operations are needed for correct reset/replay in the current multi-workgroup implementation, but add launch and synchronization costs for small input vectors. A GPU is not automatically faster at comparing just 256 or 1024 integers.

A single-workgroup small-input path could reduce locally and write both words of the I64 result without the separate clear dispatch. That is an implementation optimization to evaluate, not evidence that whole-model time would improve substantially. It does not address the large tensors, CPU loss or result-readback path.

## Why the fast small model looks somewhat more sensitive

Stories260K's validation epoch is about 70-75 ms, while Qwen's is around 5.7 seconds. A millisecond of changed synchronization or scheduling can matter more for the smaller model. The earlier repeated benchmark's nominal 1.9% Stories260K improvement was smaller than its run-to-run standard deviation, so a stable speedup has not been demonstrated.

## Implication for the contribution

The supported claim is correct WebGPU operator coverage and one fewer backend subgraph per validation batch. A substantial end-to-end speedup is not supported by these workloads. For meaningful validation throughput gains, the measured priorities are the cross-entropy CPU fallback and the dense label/logit transfer path. Those are separate changes and have not been implemented here.

Source evidence in the implementation branch:

- [Accuracy graph construction](https://github.com/LeON-Nie-code/llama.cpp/blob/preview/webgpu-count-equal/ggml/src/ggml-opt.cpp#L436)
- [CPU-side result collection](https://github.com/LeON-Nie-code/llama.cpp/blob/preview/webgpu-count-equal/ggml/src/ggml-opt.cpp#L850)
- [Dense label initialization](https://github.com/LeON-Nie-code/llama.cpp/blob/preview/webgpu-count-equal/src/llama-context.cpp#L3970)
- [WebGPU COUNT_EQUAL dispatches](https://github.com/LeON-Nie-code/llama.cpp/blob/preview/webgpu-count-equal/ggml/src/ggml-webgpu/ggml-webgpu.cpp#L3050)

Raw observations: results.json, matching model-variant.log files, and the actual command/environment JSONs. Instrumentation source: phases.cpp and eval-phases.cpp. No implementation change was made for this diagnosis.
