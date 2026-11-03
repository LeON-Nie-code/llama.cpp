#include "ggml.h"
#include "ggml-backend.h"
#include <atomic>
#include <cstdio>
#include <dlfcn.h>

static std::atomic<unsigned long long> graphs{0}, nodes{0}, counts{0}, creations{0}, webgpu_counts{0}, cpu_counts{0};
static std::atomic<unsigned long long> cpu_ops[GGML_OP_COUNT]{}, gpu_ops[GGML_OP_COUNT]{};
static void inspect(ggml_cgraph * graph, ggml_backend_t backend = nullptr) {
    graphs++;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * t = ggml_graph_node(graph, i);
        nodes++;
        if (backend) {
            const char * name = ggml_backend_name(backend);
            if (name[0] == 'C') { cpu_ops[t->op]++; }
            if (name[0] == 'W') { gpu_ops[t->op]++; }
        }
        if (t->op == GGML_OP_MEAN) {
            counts++;
            static std::atomic<int> shapes{0};
            if (shapes++ < 12) { std::fprintf(stderr, "MEAN_NODE backend=%s shape=[%lld,%lld,%lld,%lld]\n", backend ? ggml_backend_name(backend) : "scheduler", (long long)t->src[0]->ne[0], (long long)t->src[0]->ne[1], (long long)t->src[0]->ne[2], (long long)t->src[0]->ne[3]); }
            if (backend) {
                const char * name = ggml_backend_name(backend);
                if (name[0] == 'W') { webgpu_counts++; }
                if (name[0] == 'C') { cpu_counts++; }
            }
        }
    }
}
extern "C" ggml_tensor * ggml_mean(ggml_context * ctx, ggml_tensor * a) {
    using fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *);
    static auto real = reinterpret_cast<fn>(dlsym(RTLD_NEXT, "ggml_mean"));
    creations++;
    return real(ctx, a);
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
    for (int op=0;op<GGML_OP_COUNT;op++) {
        if (cpu_ops[op] || gpu_ops[op]) { std::fprintf(stderr,"BACKEND_OP op=%s cpu=%llu gpu=%llu\n",ggml_op_name((ggml_op)op),cpu_ops[op].load(),gpu_ops[op].load()); }
    }
    std::fprintf(stderr, "OP_TRACE graphs=%llu nodes=%llu mean_nodes=%llu mean_creations=%llu webgpu_mean=%llu cpu_mean=%llu\n",
        graphs.load(), nodes.load(), counts.load(), creations.load(), webgpu_counts.load(), cpu_counts.load());
}
