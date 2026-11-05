# Performance and review audit

Qwen3-TTS 64-frame phase means (seconds): input 0.697 -> 0.677; prompt 0.147 -> 0.148; generation 31.349 -> 4.588; waveform flush 1.107 -> 0.263. Thus the difference is in the repeated waveform generation and flush path, not model loading or prompt timing.

Independent 16-frame rerun: full inference 9.660 -> 2.185 seconds, same output audio. 102 CPU graph executions containing COL2IM_1D totaled 304.4 ms. The ~7.48 second overall difference therefore cannot be explained solely by CPU arithmetic. Removed backend boundaries, copies and waits are a plausible major source; no precise per-category attribution is claimed. A copy_async interposition attempt observed no calls, so it cannot be used as transfer-volume evidence.

Compared variants share the executable, core libraries, model files and settings; only ggml libraries are switched. Saved baseline reports COL2IM_1D unsupported, candidate supports F32. MEAN stays on CPU on both variants. Fresh-process first-use graph/pipeline costs remain in inference timing. Five runs per version alternate order, with diagnostic runs excluded. All recorded outputs match byte-for-byte per model. 16-frame rerun is diagnostic, not mixed into the 64-frame five-run table.

Maintainer review applied: COUNT_EQUAL #31578 asked for simpler execution paths and reuse of existing test classes. COL2IM_1D has one gather path and extends test_col2im_1d. MEAN #31591 was approved by yomaytk, including its shared shader bug fix. Similar #22406 feedback cautions against unsupported numerical complexity; no compensated summation or floating atomics are introduced here. #28883 asks for real model callers; these are traced and benchmarked. No upstream replies or changes were made during this audit.

The code commit and results branch were recreated without assistant attribution trailers. AI usage disclosure in the PR preview is retained. Historical unrelated upstream commits are not rewritten.
