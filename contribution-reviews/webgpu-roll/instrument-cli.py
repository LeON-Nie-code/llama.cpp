from pathlib import Path

source = Path('/mnt/data2/rsi/llama-webgpu-roll-work/tools/mtmd/mtmd-cli.cpp').read_text()
source = source.replace('    std::string generated_text = common_detokenize', '''    std::fprintf(stderr, "BENCH_TOKENS");
    for (llama_token token : generated_tokens) {
        std::fprintf(stderr, " %d", token);
    }
    std::fprintf(stderr, "\\n");
    std::string generated_text = common_detokenize''')
source = source.replace('''                int64_t time_start = ggml_time_ms();''', '''                int64_t time_start = ggml_time_ms();
                int64_t bench_audio_start = ggml_time_us();''')
source = source.replace('''                embd = mtmd_batch_get_output_embd(mbatch.get(), chunk);
            }

            GGML_ASSERT(embd != nullptr);''', '''                std::fprintf(stderr, "BENCH_AUDIO encoding_s=%.9f\\n", (ggml_time_us() - bench_audio_start) / 1e6);
                embd = mtmd_batch_get_output_embd(mbatch.get(), chunk);
            }

            GGML_ASSERT(embd != nullptr);
            if (const char * dump = std::getenv("ROLL_BENCH_EMBEDDINGS")) {
                static int dump_index = 0;
                const std::string path = std::string(dump) + "-" + std::to_string(dump_index++) + ".bin";
                FILE * file = std::fopen(path.c_str(), "wb");
                GGML_ASSERT(file != nullptr);
                const size_t count = mtmd_input_chunk_get_n_tokens(chunk) * llama_model_n_embd(llama_get_model(ctx.lctx));
                GGML_ASSERT(std::fwrite(embd, sizeof(float), count, file) == count);
                std::fclose(file);
            }''')
start = source.index('    if (is_single_turn) {')
end = source.index('    } else {', start)
branch = source[start:end]
needle = '''        common_chat_msg msg;
        msg.role = "user";'''
assert needle in branch
branch = branch.replace(needle, '''        for (int bench_trial = -2; bench_trial < 5; bench_trial++) {
        llama_memory_clear(llama_get_memory(ctx.lctx), true);
        ctx.n_past = 0;
        ctx.chat_history.clear();
        common_sampler_reset(ctx.smpl);
        g_is_generating = true;
        const int64_t bench_start = ggml_time_us();
        if (eval_system_prompt_if_present()) {
            return 1;
        }
        common_chat_msg msg;
        msg.role = "user";''')
branch += '''        llama_synchronize(ctx.lctx);
        std::fprintf(stderr, "BENCH_INFERENCE trial=%d total_s=%.9f\\n", bench_trial, (ggml_time_us() - bench_start) / 1e6);
        }

'''
source = source[:start] + branch + source[end:]
out = Path('/home/leon/code/tmp/penguin/roll-evidence')
out.mkdir(exist_ok=True)
(out / 'mtmd-bench.cpp').write_text(source)
print('Instrumented copied CLI; repository source unchanged.')
