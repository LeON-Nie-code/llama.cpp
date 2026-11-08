# Review guidance for the WebGPU ROLL change

Read-only review history collected before implementation, principally from yomaytk's comments on our existing PRs and related WebGPU work. These are recurring review concerns, not predictions of approval.

| Observed concern | Evidence | How the ROLL change addresses it |
| --- | --- | --- |
| Prefer one simple path for a lightweight operator; justify special paths with performance evidence | [COUNT_EQUAL review](https://github.com/ggml-org/llama.cpp/pull/31578#discussion_r4465982811) | Reuses the existing circular PAD pipeline; no new shader, special workgroup path or cache. |
| Extend existing tests rather than adding another input/test mechanism | [Existing test class](https://github.com/ggml-org/llama.cpp/pull/31578#discussion_r4466006927), [remove redundant direct-I32 test mode](https://github.com/ggml-org/llama.cpp/pull/31578#discussion_r4490141390) | Adds shape/view parameters to `test_roll`; preserves its two existing cases. |
| Use device workgroup limits unless a custom size measurably helps | [Workgroup-size review](https://github.com/ggml-org/llama.cpp/pull/31578#discussion_r4490061800) | Keeps existing PAD dispatch sizing and 2D workgroup handling. |
| Keep support predicates precise and avoid unjustified restrictions or dead shader variants | [GROUP_NORM support checks](https://github.com/ggml-org/llama.cpp/pull/31490#discussion_r4465544888), [unreachable F16 path](https://github.com/ggml-org/llama.cpp/pull/31490#discussion_r4465521770) | F32-only, matching the reused shader. Source byte strides and aligned view offsets are already handled by PAD; no added contiguous-only restriction. |
| Memory layout is a correctness condition | [SUM contiguity discussion](https://github.com/ggml-org/llama.cpp/pull/28851#issuecomment-6246989872) | Tests permuted inputs, offset/padded-row views, their combination, and a large 2D dispatch. |
| Check benchmark tables carefully and avoid copied/mislabeled results | [ARANGE review](https://github.com/ggml-org/llama.cpp/pull/31455#pullrequestreview-5732386136) | Fixed model/assets, same executable/core libraries, only the WebGPU backend replaced; retained per-run logs and hashes. Clearly labels the 30-second input as repeated speech. |
| Backend acceptance is distinct from shared test review | [ARANGE follow-up](https://github.com/ggml-org/llama.cpp/pull/31455#issuecomment-6595312637) | Keeps test changes small, explains each additional coverage case, and makes no claim of guaranteed merge. |

## Private implementation review

- Scope: one existing operator, one backend; no model code, public API, shader language extension, dependency, lock, allocation subsystem or graph rewrite.
- Semantics: a roll of output index `x` reads `(x - shift) mod n`. With equal input/output dimensions, the existing circular PAD shader does exactly this when its left padding is the normalized shift.
- Negative shifts become `shift + n`, which lies in `[0, n)` under the public operation's existing bounds. No additional modulo on the host is necessary.
- Source strides and view offsets use the existing PAD binding path; output is the contiguous tensor created by `ggml_roll`.
- F32 type checks match the reused shader, and the existing device-wide binding-size check still applies. The operator's core implementation enforces its first-axis stride and shift bounds.
- PAD parameter ordering and pipeline keys are unchanged for actual PAD nodes. Regression tests exercise both zero and circular PAD.
- The ROLL result is not in-place; input and output storage do not overlap by operation construction and scheduler rules.
- ROLL has no floating-point arithmetic or accumulation. Exact audio embedding equality is checked separately from generated token equality.
- Large test tensors follow the common test framework's memory/device support handling. The new case reaches 65,537 workgroups on a 1024-invocation device and verifies the existing 2D dispatch mapping.
- Code style: backend files checked with clang-format 22; changed files checked with editorconfig and `git diff --check`.
- Operator microbenchmarks are CPU-versus-GPU measurements, not baseline-versus-candidate full-model measurements. CPU arithmetic can be faster for these small shapes; the full-model motivation is removing backend crossings.
