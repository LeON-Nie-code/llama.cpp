#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static void require(bool ok, const char * s) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", s); std::exit(1); } }

static double mean_error2 = 0, mean_norm2 = 0, max_mean_abs = 0;
static long long mean_rows = 0, mean_nodes = 0;
static bool check_mean(ggml_tensor * t, bool ask, void *) {
    if (t->op != GGML_OP_MEAN) { return false; }
    if (ask) { return true; }
    auto * a = t->src[0]; require(a->type == GGML_TYPE_F32, "mean input F32");
    std::vector<float> src((ggml_nbytes(a)+3)/4), dst(ggml_nelements(t));
    ggml_backend_tensor_get(a, src.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
    int64_t row = 0;
    for (int64_t i3=0;i3<a->ne[3];++i3) for (int64_t i2=0;i2<a->ne[2];++i2) for (int64_t i1=0;i1<a->ne[1];++i1,++row) {
        double ref=0;
        for (int64_t i0=0;i0<a->ne[0];++i0) { ref += src[(i3*a->nb[3]+i2*a->nb[2]+i1*a->nb[1]+i0*a->nb[0])/4]; }
        ref/=a->ne[0];
        double d=dst[row]-ref;
        max_mean_abs=std::max(max_mean_abs,std::abs(d)); mean_error2+=d*d; mean_norm2+=ref*ref; ++mean_rows;
        require(std::isfinite(dst[row]) && std::abs(d) <= 1e-5+1e-5*std::abs(ref), "real-model mean matches independent reference");
    }
    ++mean_nodes;
    return true;
}

int main(int argc, char ** argv) {
    require(argc == 3 || argc == 4, "usage: model.gguf output-prefix");
    std::vector<llama_token> forced(128);
    if (argc == 4) {
        std::ifstream f(argv[3], std::ios::binary);
        f.read(reinterpret_cast<char *>(forced.data()), forced.size() * sizeof(llama_token));
        require(f.gcount() == 128 * sizeof(llama_token), "forced token sequence");
    }
    llama_backend_init();
    ggml_backend_dev_t devs[] = {ggml_backend_dev_by_name("WebGPU"), nullptr};
    require(devs[0] != nullptr, "WebGPU device");
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99; mp.devices = devs;
    auto * model = llama_model_load_from_file(argv[1], mp); require(model != nullptr, "model");
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = check_mean;
    auto * ctx = llama_init_from_model(model, cp); require(ctx != nullptr, "context");
    const char * prompt = "<start_of_turn>user\nExplain how a computer executes a neural network, from loading the weights through generating the output. Give a detailed explanation with several concrete examples.\n<end_of_turn>\n<start_of_turn>model\n";
    const auto * vocab = llama_model_get_vocab(model);
    int count = -llama_tokenize(vocab, prompt, std::strlen(prompt), nullptr, 0, true, true);
    require(count > 0, "prompt size");
    std::vector<llama_token> tokens(count);
    require(llama_tokenize(vocab, prompt, std::strlen(prompt), tokens.data(), tokens.size(), true, true) == count, "tokenization");
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    require(llama_decode(ctx, batch) == 0, "prompt decode");
    std::ofstream logits_file(std::string(argv[2]) + ".logits.bin", std::ios::binary);
    std::ofstream token_file(std::string(argv[2]) + ".tokens.bin", std::ios::binary);
    const int nv = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < 128; ++i) {
        float * logits = llama_get_logits_ith(ctx, -1); require(logits != nullptr, "logits");
        for (int j = 0; j < nv; ++j) { require(std::isfinite(logits[j]) || logits[j] == -INFINITY, "finite logits or masked -infinity"); }
        if (i == 0 || i == 127) { logits_file.write(reinterpret_cast<const char *>(logits), nv * sizeof(float)); }
        llama_token token = std::max_element(logits, logits + nv) - logits;
        token_file.write(reinterpret_cast<const char *>(&token), sizeof(token));
        if (i == 127) { break; }
        if (argc == 4) { token = forced[i]; }
        batch = llama_batch_get_one(&token, 1);
        require(llama_decode(ctx, batch) == 0, "generation decode");
    }
    std::printf("CHECK model=%s prompt_tokens=%d generated_tokens=128 vocabulary=%d\n", argv[1], count, nv);
    std::printf("MEAN_REFERENCE nodes=%lld rows=%lld max_abs=%.12g nmse=%.12g\n",mean_nodes,mean_rows,max_mean_abs,mean_error2/std::max(mean_norm2,1e-30));
    llama_free(ctx); llama_model_free(model); llama_backend_free();
}
