#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char * msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}

static void run(ggml_backend_t backend, int64_t n) {
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * a_base = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n + 1);
    ggml_tensor * b_base = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n + 3);
    ggml_tensor * a = n == 3084 ? ggml_view_4d(ctx, a_base, 257, 3, 2, 2, 257*4, 257*3*4, 257*3*2*4, 4) : ggml_view_1d(ctx, a_base, n, 4);
    ggml_tensor * b = n == 3084 ? ggml_view_4d(ctx, b_base, 257, 3, 2, 2, 257*4, 257*3*4, 257*3*2*4, 12) : ggml_view_1d(ctx, b_base, n, 12);
    ggml_tensor * out = ggml_count_equal(ctx, a, b);
    ggml_tensor * out2 = ggml_count_equal(ctx, a, b);
    require(ggml_backend_supports_op(backend, out), "supports contiguous offset views");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_build_forward_expand(graph, out2);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "buffer allocation");
    std::vector<int32_t> data(n, INT32_MIN);
    ggml_backend_tensor_set(a, data.data(), 0, n * sizeof(int32_t));
    const int32_t left = 0x12345678;
    const int32_t right = 0x76543210;
    ggml_backend_tensor_set(a_base, &left, 0, sizeof(left));
    ggml_backend_tensor_set(b_base, &right, 0, sizeof(right));
    // Same graph and output buffers, changed inputs: verify reset and cached replay.
    for (int phase = 0; phase < 4; ++phase) {
        const int every = phase == 0 || phase == 3 ? 1 : phase == 1 ? 0 : 3;
        int64_t expected = 0;
        for (int64_t i = 0; i < n; ++i) {
            const bool equal = every != 0 && i % every == 0;
            data[i] = equal ? INT32_MIN : INT32_MIN + 1;
            expected += equal;
        }
        ggml_backend_tensor_set(b, data.data(), 0, n * sizeof(int32_t));
        const int64_t poison = -1;
        ggml_backend_tensor_set(out, &poison, 0, sizeof(poison));
        ggml_backend_tensor_set(out2, &poison, 0, sizeof(poison));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");
        for (ggml_tensor * result : {out, out2}) {
            int64_t actual = -1;
            ggml_backend_tensor_get(result, &actual, 0, sizeof(actual));
            require(actual == expected, "exact I64 result including high word");
        }
        int32_t guard = 0;
        ggml_backend_tensor_get(a_base, &guard, 0, sizeof(guard));
        require(guard == left, "a prefix sentinel");
        ggml_backend_tensor_get(b_base, &guard, 0, sizeof(guard));
        require(guard == right, "b prefix sentinel");
        std::printf("N=%lld phase=%d exact_I64=%lld two_nodes=OK prefixes=OK\n",
                    (long long) n, phase, (long long) expected);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void run_overlap(ggml_backend_t backend, bool identical) {
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    const int64_t n = 1025;
    ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n + 3);
    ggml_tensor * a = ggml_view_1d(ctx, base, n, 4);
    ggml_tensor * b = identical ? a : ggml_view_1d(ctx, base, n, 12);
    ggml_tensor * out = ggml_count_equal(ctx, a, b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "overlap allocation");
    std::vector<int32_t> data(n + 3);
    for (int i = 0; i < n + 3; ++i) { data[i] = i % 3 == 0 ? INT32_MAX : INT32_MIN; }
    ggml_backend_tensor_set(base, data.data(), 0, data.size() * sizeof(int32_t));
    int64_t expected = 0;
    for (int i = 0; i < n; ++i) { expected += data[i+1] == data[i + (identical ? 1 : 3)]; }
    for (int replay = 0; replay < 3; ++replay) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "overlap compute");
        int64_t actual = -1;
        ggml_backend_tensor_get(out, &actual, 0, sizeof(actual));
        require(actual == expected, "overlap exact I64");
    }
    std::printf("overlap identical=%d exact_I64=%lld replays=3 OK\n", identical, (long long) expected);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    ggml_backend_t backend = ggml_backend_webgpu_init();
    require(backend != nullptr, "physical WebGPU backend");
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 34, 4);
    ggml_tensor * strided = ggml_view_2d(ctx, base, 17, 4, base->nb[1], 0);
    require(!ggml_backend_supports_op(backend, ggml_count_equal(ctx, strided, strided)), "reject strided inputs");
    ggml_tensor * wrong_type = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    require(!ggml_backend_supports_op(backend, ggml_count_equal(ctx, wrong_type, wrong_type)), "reject F32 inputs");
    ggml_tensor * oversized = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, int64_t(UINT32_MAX) / 4 + 1);
    require(!ggml_backend_supports_op(backend, ggml_count_equal(ctx, oversized, oversized)), "reject oversized input before allocation");
    std::puts("capability rejection checks: OK");
    ggml_free(ctx);
    for (int64_t n : {0LL, 1LL, 257LL, 3084LL, 16777217LL, 67108872LL}) { run(backend, n); }
    run_overlap(backend, true);
    run_overlap(backend, false);
    ggml_backend_free(backend);
    std::puts("ALL EXACT I64 / REPLAY / 2D DISPATCH CHECKS PASSED");
}
