#include "llama.h"
#include "ggml-opt.h"
#include "ggml-backend.h"
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <dlfcn.h>
#include <vector>

static void require(bool ok, const char * msg) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", msg); std::exit(1); } }
static int selected_params = 0;
static bool norm_params(const ggml_tensor * tensor, void *) {
    const bool selected = tensor->type == GGML_TYPE_F32 && std::strcmp(tensor->name, "output_norm.weight") == 0;
    selected_params += selected;
    return selected;
}
int main(int argc, char ** argv) {
    require(argc >= 7, "usage: eval-llm model.gguf predictions.bin epochs context contexts dataset.txt");
    const int requested_ctx = std::atoi(argv[4]);
    const int ndata = std::atoi(argv[5]);
    require(requested_ctx >= 256 && ndata > 0, "context and dataset dimensions");
    const int epochs = argc >= 4 ? std::atoi(argv[3]) : 6;
    llama_backend_init();
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("WebGPU"), nullptr};
    require(devices[0] != nullptr, "WebGPU device");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    auto * model = llama_model_load_from_file(argv[1], mp);
    require(model != nullptr, "model");
    auto cp = llama_context_default_params();
    cp.n_ctx = requested_ctx; cp.n_batch = requested_ctx; cp.n_ubatch = requested_ctx;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    auto * ctx = llama_init_from_model(model, cp);
    require(ctx != nullptr, "context");
    llama_opt_params op{};
    op.n_ctx_train = llama_n_ctx(ctx);
    op.param_filter = norm_params;
    op.get_opt_pars = ggml_opt_get_default_optimizer_params;
    op.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    // The API builds backward metadata even for forward validation; select one F32 parameter.
    // idata_split=0 below still prevents backward execution and optimizer updates.
    llama_opt_init(ctx, model, op);
    require(selected_params == 1, "one selected output norm parameter");
    std::ifstream text_file(argv[6]);
    require(bool(text_file), "dataset file");
    const std::string text((std::istreambuf_iterator<char>(text_file)), std::istreambuf_iterator<char>());
    std::vector<llama_token> tokens(text.size() + 16);
    const auto * vocab = llama_model_get_vocab(model);
    int nt = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
    const int nctx = llama_n_ctx(ctx);
    require(nctx == requested_ctx, "actual context matches request");
    require(nt > nctx * ndata, "enough corpus tokens");
    auto dataset = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, nctx, nctx, ndata, 1);
    auto * inputs = static_cast<int32_t *>(ggml_get_data(ggml_opt_dataset_data(dataset)));
    auto * labels = static_cast<int32_t *>(ggml_get_data(ggml_opt_dataset_labels(dataset)));
    for (int i = 0; i < nctx*ndata; ++i) { inputs[i] = tokens[i]; labels[i] = tokens[i+1]; }
    std::printf("WORKLOAD context=%d batches_per_epoch=%d tokens_per_epoch=%d count_equal_per_epoch=%d\n", nctx, ndata, nctx*ndata, ndata);
    auto train = ggml_opt_result_init();
    auto eval = ggml_opt_result_init();
    std::ofstream pred_file(argv[2], std::ios::binary);
    require(bool(pred_file), "prediction file");
    for (int epoch = -1; epoch < epochs; ++epoch) {
        ggml_opt_result_reset(train); ggml_opt_result_reset(eval);
        using hook = void(*)();
        auto reset = reinterpret_cast<hook>(dlsym(RTLD_DEFAULT,"phase_reset"));
        auto report = reinterpret_cast<hook>(dlsym(RTLD_DEFAULT,"phase_report"));
        require(reset && report,"diagnostic hooks");
        reset();
        auto start = std::chrono::steady_clock::now();
        // idata_split=0 selects forward validation only: no optimizer steps.
        llama_opt_epoch(ctx, dataset, train, eval, 0, nullptr, nullptr);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        if(epoch>=0)report();
        double loss=0, accuracy=0;
        int64_t n=0;
        ggml_opt_result_ndata(eval, &n);
        ggml_opt_result_loss(eval, &loss, nullptr);
        ggml_opt_result_accuracy(eval, &accuracy, nullptr);
        require(n==nctx*ndata, "evaluated token count");
        std::vector<int32_t> predictions(n);
        ggml_opt_result_pred(eval, predictions.data());
        pred_file.write(reinterpret_cast<const char *>(predictions.data()), n*sizeof(int32_t));
        std::printf("EVAL epoch=%d tokens=%lld ms=%.6f loss=%.12g accuracy=%.12g\n", epoch, (long long)n, ms, loss, accuracy);
        std::fflush(stdout);
    }
    ggml_opt_result_free(train); ggml_opt_result_free(eval); ggml_opt_dataset_free(dataset);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
}
