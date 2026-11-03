# High-frequency MEAN model measurements

Code commit: `afcd18a53bd0516f80e88d1bacc11fa60cc24e31`; upstream baseline: `b74298c109c107494b792c30af4bb8fa0ce1eb2f`. No implementation changes were made for these measurements.

## Workloads and method

- Real Gemma3n E2B/E4B Q4_K_M weights; fixed Hugging Face revisions are recorded in the model JSON files. Both models use activation sparsity in the first 10 FFN layers.
- L4 native Dawn WebGPU; same benchmark executable/core libraries with upstream vs candidate ggml libraries selected through LD_LIBRARY_PATH. Two CPU threads, all model layers offloaded, batch/microbatch 512, flash attention off.
- Five separate processes per variant, alternating baseline/candidate order. Each process runs pp128, pp512 and tg128 once; llama-bench performs its standard warmup within the process before each measured workload. Model loading and warmup are outside the timer.
- llama-bench runs complete real model graphs on synthetic token streams. PP and TG are separate workloads, not a timed natural-language conversation. The additional natural-text generation is a separate correctness check.
- Diagnostic runs use only tg128 with one warmup token. Traces contain 1,290 backend MEAN executions: 10 warmup + 1,280 measured executions. Scheduler-node counts are not additional operator executions. E2B rows contain 8,192 elements; E4B rows contain 16,384.
- Every one of the five paired comparisons was faster for the candidate, for each of the three workloads and both models.

## Results

| Model | Workload | Upstream seconds | Candidate seconds | Time reduction |
|---|---|---:|---:|---:|
| E2B | pp128 | 0.265 +/- 0.003 | 0.229 +/- 0.003 | 13.7% |
| E2B | pp512 | 0.744 +/- 0.004 | 0.674 +/- 0.003 | 9.4% |
| E2B | tg128 | 4.201 +/- 0.112 | 3.669 +/- 0.248 | 12.7% |
| E4B | pp128 | 0.386 +/- 0.002 | 0.333 +/- 0.002 | 13.7% |
| E4B | pp512 | 1.137 +/- 0.005 | 0.975 +/- 0.005 | 14.3% |
| E4B | tg128 | 4.769 +/- 0.168 | 3.992 +/- 0.250 | 16.3% |

Elapsed time is now the primary presentation for consistency with earlier previews. Raw JSON retains both elapsed time and throughput. Per run, throughput = tokens / elapsed seconds; an average of reciprocals is not exactly the reciprocal of the average. Throughput gains and latency reductions are different percentages.

The earlier TTS experiments timed a complete post-loading speech request, including reference-speaker encoding. These Gemma3n experiments time separate warmed PP and TG workloads. They do not measure the same request boundary, and pp512 + tg128 should not be presented as a single 512-prompt-token/128-output-token request.

## Correctness observations

- A fixed natural-language prompt (37 tokens) followed by 128 greedy decode steps was tested on both variants. E4B token IDs match; E2B first differs at zero-based index 92 (token 93), after which free-running contexts differ.
- Comparing logits after that free-running divergence is not a same-input comparison. We therefore reran candidate generation using the baseline token sequence as teacher-forced inputs. Next-token top-1 agreement: E2B 127/128, E4B 128/128. The teacher-check JSON records same-prefix logit NMSE at the first and last positions.
- A separate diagnostic callback compared all 1,280 executed candidate MEAN nodes (1,640 output rows including the prompt batch) against independent double-precision means of their actual inputs. E2B max absolute error 5.71e-7, aggregate NMSE 1.51e-15; E4B max absolute error 5.12e-7, aggregate NMSE 1.74e-15. This validates the operator computation; it does not imply bit-identical full-model output.
- No model or sampler changes, extra precision algorithms or benchmark-specific fast paths were added to the contribution.

## Interpretation

These workloads exercise MEAN every decode step, on rows 16-32 times wider than PocketTTS. Keeping those reductions on the GPU also removes repeated CPU fallback boundaries. This is a much better workload for evaluating the backend addition than the previous small-input TTS examples. The experiment measures the total effect of placement and arithmetic; it does not separately attribute milliseconds to copies, synchronization or kernel execution.

The earlier PocketTTS/Qwen3-TTS measurements remain in the parent directory. The PR preview now uses the more relevant Gemma3n results. No upstream PR or comment was posted.
