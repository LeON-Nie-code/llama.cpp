# COL2IM_1D validation

Base: 5851a8e505f4d5fbb183f2847698eb6155d176b7. F32-only, contiguous inputs/output; offset contiguous views supported. Unsigned gather bounds avoid negative division and floating-point atomics. Input/output bindings use the existing alignment-offset helpers. Source and output buffer-size limits are inherited from WebGPU supports_op. No other backend/model/core API changes.

CPU: 38/38 cases passed. WebGPU: 16/16 supported F32 cases passed; F16/BF16 intentionally unsupported. Extra small-shape exhaustive gather-index comparison passed. clang-format 22 and changed-file EditorConfig passed. Release test-backend-ops and llama-tts targets built; CTest test-opt, test-backend-tensor, test-mtmd-impl passed 3/3. This is targeted local validation, not all-platform CI.

Models use existing pinned local PocketTTS and Qwen3-TTS 0.6B Q4_K_M + Q8_0 mmproj assets. Same executable/core libraries, baseline saved upstream ggml libraries versus candidate libraries. Five measured fresh-process runs alternating order after an excluded diagnostic pair. Two CPU threads, seed1234, temperature0, flash attention off, backbone/mmproj WebGPU, 64 frames (5.12s audio). A local copy of llama-tts adds microsecond timers around set_input, omitted by the upstream CLI timing. Model loading excluded; per-process graph/shader initialization inside inference remains included. No tracing in timed runs. Output PCM is identical across variants and repetitions for each model.

Backend traces confirm 32/390 COL2IM_1D executions moved CPU->GPU in full diagnostic requests; includes warmup/initialization/flush and is not a per-frame steady-state count. Existing MEAN remains CPU on both sides, so this comparison isolates the new COL2IM_1D contribution rather than stacking the pending MEAN PR. Large end-to-end savings include avoided CPU fallback boundaries and transfers, not just arithmetic speed. No separate transfer-time attribution was measured.

No upstream PR or comment created. Detailed raw commands, logs, timings and audio hashes are kept locally in /tmp/llama-col2im-evidence.
