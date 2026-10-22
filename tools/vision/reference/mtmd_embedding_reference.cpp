// Independent mtmd API driver, linked against the owner's ggml-org checkout.
// No Strata encoder, preprocessing, model-header helper or embedding code is used.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) try {
    if (argc != 8) throw std::runtime_error("usage: mtmd-embedding-reference MODEL MMPROJ IMAGE OUT THREADS MAX_TOKENS MIN_TOKENS");
    llama_backend_init();
    auto lp = llama_model_default_params();
    lp.vocab_only = true;
    lp.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(argv[1], lp);
    if (!model) throw std::runtime_error("text vocabulary load failed");
    auto cp = mtmd_context_params_default();
    cp.use_gpu = false;
    cp.warmup = false;
    cp.print_timings = false;
    cp.n_threads = std::atoi(argv[5]);
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.image_max_tokens = std::atoi(argv[6]);
    cp.image_min_tokens = std::atoi(argv[7]);
    auto* ctx = mtmd_init_from_file(argv[2], model, cp);
    if (!ctx) throw std::runtime_error("mtmd load failed");
    gguf_init_params gp{true, nullptr};
    auto* gg = gguf_init_from_file(argv[2], gp);
    const auto key = gg ? gguf_find_key(gg, "clip.vision.projection_dim") : -1;
    if (key < 0) throw std::runtime_error("missing projection dimension");
    const int32_t width = gguf_get_val_u32(gg, key);
    gguf_free(gg);
    auto bitmap = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false, mtmd_helper_init_opt_default());
    if (!bitmap.bitmap) throw std::runtime_error("image load failed");
    auto* chunks = mtmd_input_chunks_init();
    const char* marker = mtmd_default_marker();
    mtmd_input_text text{marker, std::strlen(marker), false, true};
    const mtmd_bitmap* images[] = {bitmap.bitmap};
    if (mtmd_tokenize(ctx, chunks, &text, images, 1)) throw std::runtime_error("mtmd tokenize failed");
    FILE* out = std::fopen(argv[4], "wb");
    if (!out) throw std::runtime_error("output open failed");
    int count = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const auto* chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        if (mtmd_encode_chunk(ctx, chunk)) throw std::runtime_error("mtmd encode failed");
        const auto n = mtmd_input_chunk_get_n_tokens(chunk);
        const auto* grid = mtmd_input_chunk_get_tokens_image(chunk);
        const auto last = mtmd_image_tokens_get_decoder_pos(grid, 0, n - 1);
        const int32_t nx = last.x + 1, ny = last.y + 1;
        if (size_t(nx) * ny != n) throw std::runtime_error("nonrectangular grid");
        // Check every row, not merely the last: Strata's SVE grid must encode the
        // same t/h/w order as mtmd's decoder positions, including nonsquare images.
        for (size_t row = 0; row < n; ++row) {
            auto p = mtmd_image_tokens_get_decoder_pos(grid, 0, row);
            if (p.t != 0 || p.x != int(row % nx) || p.y != int(row / nx))
                throw std::runtime_error("mtmd positions disagree with SVE row order");
        }
        int32_t header[] = {0x31455653, int32_t(n), nx, ny, width};
        if (std::fwrite(header, sizeof(header), 1, out) != 1 ||
            std::fwrite(mtmd_get_output_embd(ctx), sizeof(float) * width, n, out) != n)
            throw std::runtime_error("output write failed");
        ++count;
    }
    if (std::fclose(out) || count != 1) throw std::runtime_error("expected one image chunk");
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap.bitmap);
    mtmd_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "reference: %s\n", e.what());
    return 1;
}
