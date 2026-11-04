// Opt-in residual trace. Included after CK and the CUDA/HIP compatibility headers.
// All integers/floats are little endian, with no C struct padding:
//   file: "Q36LYR1\0", u32 endian marker 0x01020304, n_embd, main_layers
//   record: u32 layer, stage (0 mixer / 1 MoE), rows, flags (1 verify, 2 MTP, 4 batch),
//           u64 forward_call; i64 p0; i32 token[rows], cell[rows], slot[rows]; f32 x[rows][n_embd].
// Warmup is excluded. TP records rank zero's replicated residual, after each join AND residual add.
#pragma once
namespace q36dump {
class Layers {
    FILE* file_ = nullptr;
    float* pinned_ = nullptr;
    size_t capacity_ = 0, count_ = 0;
    uint32_t rows_ = 0, flags_ = 0;
    uint64_t call_ = 0;
    int64_t p0_ = 0;
    bool header_ = false, active_ = false;
    std::vector<int32_t> tokens_, cells_, slots_;
    void write(const void* p, size_t bytes) {
        if (std::fwrite(p, 1, bytes, file_) != bytes) fail("write failed");
    }
    [[noreturn]] static void fail(const char* why) {
        std::fprintf(stderr, "strata-q36: STRATA_Q36_DUMP_LAYERS: %s\n", why);
        std::exit(1);
    }
public:
    Layers() {
        const char* path = std::getenv("STRATA_Q36_DUMP_LAYERS");
        if (!path || !*path) return;
        const uint32_t endian = 1;
        if (*(const uint8_t*)&endian != 1 || sizeof(float) != 4) fail("requires little-endian float32");
        file_ = std::fopen(path, "wb");
        if (!file_) fail("cannot open output file");
        std::fprintf(stderr, "strata-q36: STRATA_Q36_DUMP_LAYERS=%s: full residuals after mixer/MoE, warmup excluded; "
                             "graphs disabled, synchronous diagnostic readback\n", path);
    }
    ~Layers() {
        if (file_ && std::fclose(file_)) fail("close failed");
        if (pinned_) CK(cudaFreeHost(pinned_));
    }
    bool enabled() const { return file_ != nullptr; }
    bool active() const { return active_; }
    void begin(int n, int layers, int rows, int64_t p0, bool verify, bool mtp, int slot, bool warming,
               const int32_t* tokens, const int32_t* cells, const int32_t* slots, cudaStream_t stream) {
        active_ = enabled() && !warming;
        if (!active_) return;
        if (!header_) {
            const uint32_t h[] = {0x01020304, (uint32_t)n, (uint32_t)layers};
            write("Q36LYR1\0", 8); write(h, sizeof(h)); header_ = true;
        }
        count_ = (size_t)rows * n;
        if (2 * count_ > capacity_) {
            if (pinned_) CK(cudaFreeHost(pinned_));
            capacity_ = 2 * count_;
            CK(cudaHostAlloc((void**)&pinned_, capacity_ * sizeof(float), cudaHostAllocPortable));
        }
        rows_ = rows; p0_ = p0; flags_ = (verify ? 1u : 0u) | (mtp ? 2u : 0u) | (cells ? 4u : 0u);
        ++call_;
        tokens_.resize(rows); cells_.resize(rows); slots_.assign(rows, slot);
        for (int i = 0; i < rows; ++i) cells_[i] = (int32_t)(p0 + i);
        CK(cudaMemcpyAsync(tokens_.data(), tokens, rows * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        if (cells) CK(cudaMemcpyAsync(cells_.data(), cells, rows * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        if (slots) CK(cudaMemcpyAsync(slots_.data(), slots, rows * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream)); // before either rank launches a rendezvous for this forward
    }
    void mixer(const float* x, cudaStream_t stream) {
        if (active_) CK(cudaMemcpyAsync(pinned_, x, count_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    void finish(size_t layer, const float* x, cudaStream_t stream) {
        if (!active_) return;
        CK(cudaMemcpyAsync(pinned_ + count_, x, count_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream)); // TP calls this only AFTER submitting both ranks
        for (uint32_t stage = 0; stage < 2; ++stage) {
            const uint32_t h[] = {(uint32_t)layer, stage, rows_, flags_};
            write(h, sizeof(h)); write(&call_, sizeof(call_)); write(&p0_, sizeof(p0_));
            for (const auto* v : {&tokens_, &cells_, &slots_}) write(v->data(), rows_ * sizeof(int32_t));
            write(pinned_ + stage * count_, count_ * sizeof(float));
        }
        if (std::fflush(file_)) fail("flush failed");
    }
};
} // namespace q36dump
