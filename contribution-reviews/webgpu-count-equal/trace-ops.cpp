#include "ggml.h"
#include "ggml-backend.h"
#include <atomic>
#include <cstdio>
#include <dlfcn.h>

static std::atomic<unsigned long long> graphs{0}, nodes{0}, counts{0}, creations{0}, webgpu_counts{0}, cpu_counts{0};
static void inspect(ggml_cgraph * graph, ggml_backend_t backend = nullptr) {
    graphs++;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * t = ggml_graph_node(graph, i);
        nodes++;
        if (t->op == GGML_OP_COUNT_EQUAL) {
            counts++;
            if (backend) {
                const char * name = ggml_backend_name(backend);
                if (name[0] == 'W') { webgpu_counts++; }
                if (name[0] == 'C') { cpu_counts++; }
            }
        }
    }
}
extern "C" ggml_tensor * ggml_count_equal(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    using fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *);
    static auto real = reinterpret_cast<fn>(dlsym(RTLD_NEXT, "ggml_count_equal"));
    creations++;
    return real(ctx, a, b);
}
extern "C" ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, ggml_cgraph * graph) {
    using fn = ggml_status (*)(ggml_backend_t, ggml_cgraph *);
    static auto real = reinterpret_cast<fn>(dlsym(RTLD_NEXT, "ggml_backend_graph_compute_async"));
    inspect(graph, backend);
    return real(backend, graph);
}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, ggml_cgraph * graph) {
    using fn = ggml_status (*)(ggml_backend_sched_t, ggml_cgraph *);
    static auto real = reinterpret_cast<fn>(dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute_async"));
    inspect(graph);
    return real(sched, graph);
}
__attribute__((destructor)) static void report() {
    std::fprintf(stderr, "OP_TRACE graphs=%llu nodes=%llu count_equal_nodes=%llu count_equal_creations=%llu webgpu_count_equal=%llu cpu_count_equal=%llu\n",
        graphs.load(), nodes.load(), counts.load(), creations.load(), webgpu_counts.load(), cpu_counts.load());
}
