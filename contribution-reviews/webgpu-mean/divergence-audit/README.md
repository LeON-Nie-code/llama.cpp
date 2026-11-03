# Timing presentation and E2B divergence audit

## Timing

The contribution code remains `afcd18a`; no code changes were made during this audit. The PR preview now presents the original raw elapsed times in seconds for consistency with the earlier TTS preview. No performance experiment was rerun or replaced to change those numbers. JSON stores both elapsed time and tokens/s.

Per run, tokens/s = token count / elapsed seconds. A percentage throughput increase differs from a percentage elapsed-time reduction. For E4B tg128, the observed mean elapsed time was 4.769 -> 3.992 s (16.3% reduction), while mean throughput was 26.87 -> 32.17 tokens/s (19.7% increase). Averaging reciprocal quantities introduces a further small distinction.

The experiment boundary also differs: the earlier TTS experiment measured a whole post-load speech request, whereas llama-bench measures separate warmed prompt-processing and token-generation workloads. A tg128 case is not a measurement of 128 output tokens following the pp512 case. These should not be combined or compared as identical request-latency workloads.

## First differing token

A rerun used the same 37-token prompt and the baseline-generated tokens as teacher-forced inputs, so both models see identical prefixes. At zero-based decode position 92 (the 93rd output token):

| Candidate token ID | Baseline logit | Candidate logit |
|---|---:|---:|
| 669 | 24.2335262 | 24.1760597 |
| 138 | 24.2070732 | 24.2128448 |

The baseline top-two gap was only 0.0264530. The order flips in the candidate. These are greedy decisions, not random sampling. Softmax probabilities for IDs 669 / 138 changed from 0.466900 / 0.454711 to 0.452723 / 0.469687. Full next-token-distribution KL(baseline || candidate) at this position was 4.8425e-4, with total variation 0.01511.

The first changed input after this choice leads to different subsequent contexts. Comparing free-running logits after that point does not measure same-input arithmetic error. With identical teacher-forced prefixes, E2B top-1 agreement is 127/128; E4B is 128/128.

Backend-node tracing found that only MEAN execution placement changed in this E2B request: 1,280 CPU executions became 1,280 WebGPU executions. Other operator placement counts matched. This does not prove identical kernels, memory layouts, graph segmentation or floating-point evaluation order downstream.

## Operator computation

Previous independent callback checks covered all 1,280 executed MEAN nodes per model: maximum absolute errors were 5.71e-7 (E2B) and 5.12e-7 (E4B), aggregate normalized mean squared errors 1.51e-15 / 1.74e-15. All rows passed the diagnostic absolute-plus-relative error check. The repository's default operator NMSE threshold is 1e-7; this operator-level check is not a universal tolerance for full-model logits or text generation.

The findings support a near-tie rank flip caused by accumulated numerical differences. They do not establish an exact layer-by-layer causal attribution of every full-model logit difference, or bit-identical model equivalence.

## Fixed-text perplexity

Both models were also evaluated on the same WikiText validation input, context 512, batch/microbatch 512, two CPU threads, flash attention off, all layers on WebGPU. The first exploratory run used 4 chunks; the follow-up used 32 chunks. PPL is lower-is-better.

| Model | Upstream PPL | Candidate PPL | Relative change |
|---|---:|---:|---:|
| E2B | 29.6885 | 29.6811 | -0.025% |
| E4B | 27.7562 | 27.7657 | +0.034% |

No material quality deterioration was observed in this 32-chunk sample. This is a bounded validation sample, not proof of equal quality on every prompt. Reported PPL sampling uncertainty is preserved in the raw summary but is not used as a paired significance test.

Overall: no MEAN implementation error has been found. The evidence supports interpreting the E2B sequence divergence as numerical sensitivity at a near-tied greedy choice. A single generated-string comparison alone would not justify that conclusion.
