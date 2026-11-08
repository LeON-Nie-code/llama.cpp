#include "ggml.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <atomic>

static std::atomic<unsigned long long> cpu[GGML_OP_COUNT]{}, gpu[GGML_OP_COUNT]{};
extern "C" ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, ggml_cgraph * graph) {
    using fn = ggml_status (*)(ggml_backend_t, ggml_cgraph *);
    static auto real = reinterpret_cast<fn>(dlsym(RTLD_NEXT, "ggml_backend_graph_compute_async"));
    const char * name = ggml_backend_name(backend);
    bool pool = false;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * t = ggml_graph_node(graph, i);
        if (name[0] == 'C') cpu[t->op]++;
        if (name[0] == 'W') gpu[t->op]++;
        if (t->op == GGML_OP_ROLL) {
            pool = true;
            std::fprintf(stderr, "ROLL_NODE backend=%s src_name=%s dst_name=%s src=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n", name, t->src[0]->name, t->name,
                (long long)t->src[0]->ne[0], (long long)t->src[0]->ne[1], (long long)t->src[0]->ne[2], (long long)t->src[0]->ne[3],
                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
        }
    }
    const auto start = std::chrono::steady_clock::now();
    const auto status = real(backend, graph);
    if (pool && name[0] == 'C') {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::fprintf(stderr, "ROLL_CPU_GRAPH elapsed_ms=%.6f ops=", ms);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) std::fprintf(stderr, "%s,", ggml_op_name(ggml_graph_node(graph, i)->op));
        std::fprintf(stderr, "\n");
    }
    return status;
}
extern "C" void ggml_backend_tensor_copy(const ggml_tensor * src, ggml_tensor * dst) {
    using fn = void (*)(const ggml_tensor *, ggml_tensor *);
    static auto real = reinterpret_cast<fn>(dlsym(RTLD_NEXT, "ggml_backend_tensor_copy"));
    const bool sh = src->buffer && ggml_backend_buffer_is_host(src->buffer);
    const bool dh = dst->buffer && ggml_backend_buffer_is_host(dst->buffer);
    const auto start = std::chrono::steady_clock::now();
    real(src, dst);
    if (sh != dh) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::fprintf(stderr, "COPY_DETAIL direction=%s src=%s dst=%s bytes=%zu elapsed_ms=%.6f\n", sh ? "H2D" : "D2H", src->name, dst->name, ggml_nbytes(src), ms);
    }
}
__attribute__((destructor)) static void report() {
    for (int i = 0; i < GGML_OP_COUNT; ++i) if (cpu[i] || gpu[i])
        std::fprintf(stderr, "BACKEND_OP op=%s cpu=%llu gpu=%llu\n", ggml_op_name((ggml_op)i), cpu[i].load(), gpu[i].load());
}
