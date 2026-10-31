#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>

enum Bucket { ALLOC, EVAL, LABEL_ZERO, LABEL_SET, CPU_CE, CPU_COUNT, CPU_OTHER, GPU_ENCODE,
              COPY_D2H_LARGE, COPY_D2H_IDS, COPY_H2D, GET_PRED, GET_COUNT, GPU_SYNC, NUM_BUCKETS };
static const char * names[] = {"opt_alloc","opt_eval","label_zero","label_set","cpu_cross_entropy_graph",
    "cpu_count_equal_graph","cpu_other_graph","gpu_encode_submit","copy_d2h_large","copy_d2h_ids",
    "copy_h2d","result_get_pred","result_get_ncorrect","gpu_synchronize"};
struct Stat { std::atomic<uint64_t> ns{0}, calls{0}, bytes{0}; };
static Stat stats[NUM_BUCKETS];
static std::atomic<uint64_t> cpu_counts{0}, gpu_counts{0}, cpu_ce{0}, gpu_ce{0};
static std::atomic<uint64_t> vocab{0}, batch{0};
static uint64_t now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void add(Bucket b, uint64_t start, uint64_t bytes=0) { stats[b].ns+=now()-start; stats[b].calls++; stats[b].bytes+=bytes; }
template<typename F> static F next(const char * name) { return reinterpret_cast<F>(dlsym(RTLD_NEXT,name)); }
extern "C" void phase_reset() {
    for (auto & s:stats) { s.ns=0; s.calls=0; s.bytes=0; }
    cpu_counts=0;gpu_counts=0;cpu_ce=0;gpu_ce=0;vocab=0;batch=0;
}
extern "C" void phase_report() {
    std::fprintf(stderr,"PHASE_META vocab=%llu batch=%llu cpu_count=%llu gpu_count=%llu cpu_ce=%llu gpu_ce=%llu\n",
        (unsigned long long)vocab.load(),(unsigned long long)batch.load(),(unsigned long long)cpu_counts.load(),
        (unsigned long long)gpu_counts.load(),(unsigned long long)cpu_ce.load(),(unsigned long long)gpu_ce.load());
    for(int i=0;i<NUM_BUCKETS;i++) std::fprintf(stderr,"PHASE name=%s ms=%.6f calls=%llu bytes=%llu\n",names[i],
        stats[i].ns.load()/1e6,(unsigned long long)stats[i].calls.load(),(unsigned long long)stats[i].bytes.load());
}
extern "C" void ggml_opt_alloc(ggml_opt_context_t ctx,bool backward) {
    using F=void(*)(ggml_opt_context_t,bool);static F real=next<F>("ggml_opt_alloc");auto t=now();real(ctx,backward);add(ALLOC,t);
}
extern "C" void ggml_opt_eval(ggml_opt_context_t ctx,ggml_opt_result_t result) {
    using F=void(*)(ggml_opt_context_t,ggml_opt_result_t);static F real=next<F>("ggml_opt_eval");auto t=now();real(ctx,result);add(EVAL,t);
}
extern "C" ggml_tensor * ggml_set_zero(ggml_tensor * tensor) {
    using F=ggml_tensor *(*)(ggml_tensor *);static F real=next<F>("ggml_set_zero");auto t=now();auto r=real(tensor);
    if(std::strcmp(tensor->name,"labels")==0)add(LABEL_ZERO,t,ggml_nbytes(tensor));return r;
}
extern "C" void ggml_backend_tensor_set(ggml_tensor * tensor,const void * data,size_t offset,size_t size) {
    using F=void(*)(ggml_tensor *,const void *,size_t,size_t);static F real=next<F>("ggml_backend_tensor_set");auto t=now();real(tensor,data,offset,size);
    if(std::strcmp(tensor->name,"labels")==0)add(LABEL_SET,t,size);
}
extern "C" ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend,ggml_cgraph * graph) {
    using F=ggml_status(*)(ggml_backend_t,ggml_cgraph *);static F real=next<F>("ggml_backend_graph_compute_async");
    bool cpu=ggml_backend_name(backend)[0]=='C',ce=false,count=false;
    for(int i=0;i<ggml_graph_n_nodes(graph);i++) {
        auto * node=ggml_graph_node(graph,i);
        if(node->op==GGML_OP_COUNT_EQUAL) { count=true;if(cpu)cpu_counts++;else gpu_counts++; }
        if(node->op==GGML_OP_CROSS_ENTROPY_LOSS) {ce=true;if(cpu)cpu_ce++;else gpu_ce++;vocab=node->src[0]->ne[0];batch=node->src[0]->ne[1];}
    }
    auto t=now();auto result=real(backend,graph);add(cpu?(ce?CPU_CE:count?CPU_COUNT:CPU_OTHER):GPU_ENCODE,t);return result;
}
extern "C" void ggml_backend_tensor_copy(const ggml_tensor * src,ggml_tensor * dst) {
    using F=void(*)(const ggml_tensor *,ggml_tensor *);static F real=next<F>("ggml_backend_tensor_copy");
    bool sh=src->buffer&&ggml_backend_buffer_is_host(src->buffer),dh=dst->buffer&&ggml_backend_buffer_is_host(dst->buffer);
    auto t=now();real(src,dst);
    if(!sh&&dh)add(ggml_nbytes(src)>1024*1024?COPY_D2H_LARGE:COPY_D2H_IDS,t,ggml_nbytes(src));
    if(sh&&!dh)add(COPY_H2D,t,ggml_nbytes(src));
}
extern "C" void ggml_backend_tensor_get(const ggml_tensor * tensor,void * data,size_t offset,size_t size) {
    using F=void(*)(const ggml_tensor *,void *,size_t,size_t);static F real=next<F>("ggml_backend_tensor_get");auto t=now();real(tensor,data,offset,size);
    if(std::strcmp(tensor->name,"pred")==0)add(GET_PRED,t,size);
    if(std::strcmp(tensor->name,"ncorrect")==0)add(GET_COUNT,t,size);
}
extern "C" void ggml_backend_synchronize(ggml_backend_t backend) {
    using F=void(*)(ggml_backend_t);static F real=next<F>("ggml_backend_synchronize");auto t=now();real(backend);
    if(ggml_backend_name(backend)[0]=='W')add(GPU_SYNC,t);
}
