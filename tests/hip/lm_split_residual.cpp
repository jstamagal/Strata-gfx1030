// Synthetic layer-major receiving stage: per-layer host residuals versus one boundary upload.
// Calls the production upload helper and HC write kernel. No model or model weights.
#include "strata/prefill/lm_residual.hpp"
#include "strata/prefill/kernels.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, T = 2048, Layers = 24;
void ck(cudaError_t e, const char* op) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", op, cudaGetErrorString(e));
        std::exit(1);
    }
}
template<class P> void alloc(P*& p, size_t n) { ck(cudaMalloc(&p, n * sizeof(P)), "allocate"); }

double elapsed(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
bool check(int dev, int rows, int generation) {
    ck(cudaSetDevice(dev), "device");
    const size_t words = (size_t) rows * D, bytes = words * sizeof(float);
    float* host = nullptr;
    ck(cudaHostAlloc((void**) &host, bytes, cudaHostAllocDefault), "pinned hand-off");
    auto initial = [&] {
        for (size_t i = 0; i < words; ++i) host[i] = (float) ((int) ((i + generation * 17) % 257) - 128) / 32.0f;
    };
    float *resident = nullptr, *scratch[2]{}, *bo = nullptr, *inj = nullptr;
    alloc(resident, words);
    for (auto& p : scratch) alloc(p, (size_t) T * D);
    alloc(bo, (size_t) T * N); alloc(inj, (size_t) T * HC);
    std::vector<float> b((size_t) T * N), in((size_t) T * HC);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (float) ((int) (i % 31) - 15) / 64.0f;
    for (size_t i = 0; i < in.size(); ++i) in[i] = (float) ((int) (i % 7) - 3) / 8.0f;
    ck(cudaMemcpy(bo, b.data(), b.size() * 4, cudaMemcpyHostToDevice), "bo");
    ck(cudaMemcpy(inj, in.data(), in.size() * 4, cudaMemcpyHostToDevice), "inj");
    cudaStream_t copy{}, compute{};
    ck(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking), "copy stream");
    ck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), "compute stream");
    cudaEvent_t ready{}, done[2]{}, landed[2]{};
    ck(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "ready");
    for (int j = 0; j < 2; ++j) {
        ck(cudaEventCreateWithFlags(&done[j], cudaEventDisableTiming), "done");
        ck(cudaEventCreateWithFlags(&landed[j], cudaEventDisableTiming), "landed");
    }
    // The default-stream weight uploads must finish before either nonblocking test stream reads them.
    ck(cudaDeviceSynchronize(), "fixture inputs ready");
    // Same two parity buffers and release events as the host-residual path, no timing sleeps.
    initial();
    const auto h0 = std::chrono::steady_clock::now();
    for (int l = 0; l < Layers; ++l) {
        int c = 0;
        for (int r = 0; r < rows; r += T, ++c) {
            const int bank = c & 1, nr = std::min(T, rows - r);
            ck(cudaStreamWaitEvent(copy, landed[bank], 0), "scratch reusable");
            ck(cudaMemcpyAsync(scratch[bank], host + (size_t) r * D, (size_t) nr * D * 4,
                               cudaMemcpyHostToDevice, copy), "host residual upload");
            ck(cudaEventRecord(ready, copy), "upload ready");
            ck(cudaStreamWaitEvent(compute, ready, 0), "input wait");
            strata::prefill::gr_write(scratch[bank], bo, inj, HC, nr, compute);
            ck(cudaEventRecord(done[bank], compute), "write done");
            ck(cudaStreamWaitEvent(copy, done[bank], 0), "download wait");
            ck(cudaMemcpyAsync(host + (size_t) r * D, scratch[bank], (size_t) nr * D * 4,
                               cudaMemcpyDeviceToHost, copy), "host residual download");
            ck(cudaEventRecord(landed[bank], copy), "download landed");
        }
        ck(cudaStreamSynchronize(copy), "host layer complete");
    }
    const double host_ms = elapsed(h0);
    std::vector<float> ref(host, host + words), got(words);
    initial();
    // Poison the target, then use the production cross-stream boundary helper. Without the wait the first
    // consumer may read poison. A different generation also catches stale residuals between super-chunks.
    // Poison on the copy stream too: cudaMemset on the legacy stream can finish AFTER a nonblocking upload.
    ck(cudaMemsetAsync(resident, 0xff, bytes, copy), "poison target");
    const auto d0 = std::chrono::steady_clock::now();
    ck(strata::prefill::detail::lm_stage_upload(resident, host, bytes, copy, compute, ready), "production boundary upload");
    for (int l = 0; l < Layers; ++l)
        for (int r = 0; r < rows; r += T)
            strata::prefill::gr_write(resident + (size_t) r * D, bo, inj, HC, std::min(T, rows - r), compute);
    ck(cudaStreamSynchronize(compute), "resident layers complete");
    const double resident_ms = elapsed(d0);
    ck(cudaMemcpy(got.data(), resident, bytes, cudaMemcpyDeviceToHost), "reference readback");
    const bool equal = std::memcmp(ref.data(), got.data(), bytes) == 0;
    size_t different = 0;
    for (size_t i = 0; i < words; ++i) if (std::memcmp(&ref[i], &got[i], 4) != 0) {
        if (different < 4)
            std::printf("difference GPU%d word %zu: host %.9g, resident %.9g\n", dev, i, ref[i], got[i]);
        ++different;
    }
    std::printf("GPU%d rows=%d generation=%d layers=%d: host %.3f ms, resident %.3f ms, "
                "host traffic %.1f MB, resident input %.1f MB, %zu differing words: %s\n",
                dev, rows, generation, Layers, host_ms, resident_ms,
                (double) bytes * 2 * Layers / 1e6, (double) bytes / 1e6, different, equal ? "PASS" : "FAIL");
    ck(cudaEventDestroy(ready), "destroy ready");
    for (int j = 0; j < 2; ++j) { ck(cudaEventDestroy(done[j]), "destroy done"); ck(cudaEventDestroy(landed[j]), "destroy landed"); }
    ck(cudaStreamDestroy(compute), "destroy compute"); ck(cudaStreamDestroy(copy), "destroy copy");
    ck(cudaFree(resident), "free resident");
    for (auto p : scratch) ck(cudaFree(p), "free scratch");
    ck(cudaFree(bo), "free bo"); ck(cudaFree(inj), "free inj"); ck(cudaFreeHost(host), "free host");
    return equal;
}
}

int main() {
    using strata::prefill::detail::lm_split_r_vram_from;
    if (lm_split_r_vram_from(nullptr) || lm_split_r_vram_from("") || lm_split_r_vram_from("0") ||
        lm_split_r_vram_from("10") || !lm_split_r_vram_from("1")) return 1;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    bool ok = true;
    for (int d = 0; d < count; ++d)
        for (auto shape : {std::pair<int, int>{4096, 0}, {4353, 1}, {2048, 2}, {17, 3}})
            ok = check(d, shape.first, shape.second) && ok;
    std::puts(ok ? "PASS: residual transfer policy and HC output are bit-identical" : "FAIL");
    return ok ? 0 : 1;
}
