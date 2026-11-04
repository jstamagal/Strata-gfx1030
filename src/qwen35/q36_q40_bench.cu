// Synthetic TP half-expert and dense decode benchmark; no model file needed.
// build/q36-q40-bench [device=0] [reps=100]
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "q36_experts.hpp"
#include "q36_kernels.cuh"
#include "q36_q4_dense.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>
namespace K = strata::kernels;
namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
template<class T> T* dev(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T)), "alloc"); return p; }
double timed(cudaStream_t st, int reps, const std::function<void(int)>& run) {
    cudaEvent_t a = nullptr, b = nullptr;
    ck(cudaEventCreate(&a), "event"); ck(cudaEventCreate(&b), "event");
    for (int i = 0; i < 3; ++i) run(i);
    ck(cudaStreamSynchronize(st), "warmup");
    ck(cudaEventRecord(a, st), "record");
    for (int i = 0; i < reps; ++i) run(i);
    ck(cudaEventRecord(b, st), "record"); ck(cudaEventSynchronize(b), "sync");
    float ms = 0; ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    ck(cudaEventDestroy(a), "destroy"); ck(cudaEventDestroy(b), "destroy");
    return 1000.0 * ms / reps;
}
void weights(std::vector<uint8_t>& b, int type, size_t blocks) {
    const size_t bs = type == 2 ? 18 : type == 3 ? 20 : 0;
    std::mt19937 rng(19 + type);
    for (size_t i = 0; i < blocks; ++i) {
        uint8_t* p = b.data() + i * bs;
        p[0] = 0; p[1] = 0x3c; // finite scale 1
        if (type == 3) { p[2] = 0; p[3] = 0; }
        for (size_t j = type == 3 ? 4 : 2; j < bs; ++j) p[j] = (uint8_t)rng();
    }
}
void expert(int gu, int dn, int T, int reps, cudaStream_t st) {
    constexpr int N = 3072, FF = 512, top = 8;
    const int entries = T * top, groups = entries;
    const K::NativeExpertLayout L = K::native_expert_layout(gu, dn, N, FF);
    const size_t stride = (L.bytes + 15) & ~(size_t)15;
    const int copies = std::max(1, std::min(16, (int)((96ull << 20) / ((size_t)groups * stride))));
    const size_t pool_bytes = (size_t)copies * groups * stride;
    std::vector<uint8_t> blob(pool_bytes, 0);
    for (int c = 0; c < copies * groups; ++c) {
        uint8_t* p = blob.data() + (size_t)c * stride;
        std::vector<uint8_t> g((size_t)2 * FF * L.gu_row), d((size_t)N * L.d_row);
        weights(g, gu, (size_t)2 * FF * N / 32);
        weights(d, dn, (size_t)N * FF / 32);
        std::memcpy(p, g.data(), g.size()); std::memcpy(p + L.down_off, d.data(), d.size());
    }
    uint8_t* dw = dev<uint8_t>(pool_bytes); ck(cudaMemcpy(dw, blob.data(), pool_bytes, cudaMemcpyHostToDevice), "weights");
    std::vector<unsigned long long*> ptr(copies);
    for (int c = 0; c < copies; ++c) {
        std::vector<unsigned long long> hp((size_t)groups + 1);
        for (int g = 0; g < groups; ++g) hp[(size_t)g] = (unsigned long long)(dw + ((size_t)c * groups + g) * stride);
        ptr[c] = dev<unsigned long long>(hp.size());
        ck(cudaMemcpy(ptr[c], hp.data(), hp.size() * 8, cudaMemcpyHostToDevice), "ptr");
    }
    std::vector<int32_t> start((size_t)groups + 1), dst(entries), tok(entries);
    for (int e = 0; e < entries; ++e) { start[(size_t)e] = e; dst[(size_t)e] = e; tok[(size_t)e] = e / top; }
    start.back() = entries;
    int32_t *ds = dev<int32_t>(start.size()), *dd = dev<int32_t>(entries), *dt = dev<int32_t>(entries), *dg = dev<int32_t>(1);
    ck(cudaMemcpy(ds, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "start");
    ck(cudaMemcpy(dd, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(dt, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "tok");
    ck(cudaMemcpy(dg, &groups, 4, cudaMemcpyHostToDevice), "groups");
    std::vector<float> hx((size_t)T * N);
    std::mt19937 rng(42); for (float& v : hx) v = (float)((int)(rng() % 257) - 128) / 32.f;
    float* dx = dev<float>(hx.size()); ck(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
    void* xq = dev<uint8_t>(K::native_q8_1_bytes(N, T));
    K::quantize_q8_1_rows(dx, T, N, xq, st);
    void* scratch = dev<uint8_t>(K::native_expert_scratch_bytes(entries, FF));
    float* y = dev<float>((size_t)entries * N), *ref = dev<float>((size_t)entries * N);
    auto old = [&](int i) { K::native_expert_grouped(L, ptr[(size_t)i % copies], ds, dg, dd, dt,
                                    groups, entries, xq, scratch, ref, st); };
    auto neu = [&](int i) {
        if (!K::q36_expert_decode(L, ptr[(size_t)i % copies], ds, dg, dd, dt,
                                  groups, entries, xq, scratch, y, st, true, true)) {
            std::fprintf(stderr, "Q4 expert dispatch refused %d/%d\n", gu, dn); std::exit(1);
        }
    };
    old(0); neu(0); ck(cudaStreamSynchronize(st), "expert parity");
    std::vector<float> a((size_t)entries * N), b(a.size());
    ck(cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost), "new");
    ck(cudaMemcpy(b.data(), ref, b.size() * 4, cudaMemcpyDeviceToHost), "ref");
    const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    const double us_ref = timed(st, reps, old), us_new = timed(st, reps, neu);
    const double gb = (double)groups * L.bytes / 1e9;
    std::printf("expert %d/%d T=%d groups=%d old %.2f us %.1f GB/s new %.2f us %.1f GB/s %s\n",
                gu, dn, T, groups, us_ref, gb * 1e6 / us_ref, us_new, gb * 1e6 / us_new,
                same ? "bitwise" : "MISMATCH");
    if (!same) std::exit(2);
    ck(cudaFree(dw), "free"); for (auto p : ptr) ck(cudaFree(p), "free");
    for (void* p : {(void*)ds, (void*)dd, (void*)dt, (void*)dg, (void*)dx, xq, scratch, (void*)y, (void*)ref}) ck(cudaFree(p), "free");
}
void dense(int type, int T, int reps, cudaStream_t st) {
    constexpr int N = 3072, rows = 512;
    const size_t one = K::native_mmvq_weight_bytes(type, N, rows);
    const int copies = std::max(1, std::min(64, (int)((96ull << 20) / one)));
    uint8_t* w = dev<uint8_t>((size_t)copies * one);
    if (type == 2) {
        std::vector<uint8_t> hw((size_t)copies * one);
        weights(hw, type, hw.size() / 18);
        ck(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "dense weights");
    } else ck(cudaMemset(w, 0, (size_t)copies * one), "dense weights");
    std::vector<float> hx((size_t)T * N);
    std::mt19937 rng(13); for (float& v : hx) v = (float)((int)(rng() % 257) - 128) / 32.f;
    float* dx = dev<float>(hx.size()); ck(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "dense x");
    void* xq = dev<uint8_t>(K::native_q8_1_bytes(N, T));
    K::quantize_q8_1_rows(dx, T, N, xq, st);
    float *y = dev<float>((size_t)T * rows), *ref = dev<float>((size_t)T * rows);
    auto old = [&](int i) { K::native_mmvq(type, w + (size_t)(i % copies) * one, xq, ref, N, rows, T, st); };
    auto neu = [&](int i) { q36q4::launch(w + (size_t)(i % copies) * one, xq, y, N, rows, T, st); };
    bool same = true;
    if (type == 2) {
        old(0); neu(0); ck(cudaStreamSynchronize(st), "dense parity");
        std::vector<float> a((size_t)T * rows), b(a.size());
        ck(cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost), "dense new");
        ck(cudaMemcpy(b.data(), ref, b.size() * 4, cudaMemcpyDeviceToHost), "dense ref");
        same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    }
    const double us_ref = timed(st, reps, old);
    std::printf("dense type=%d T=%d K=%d rows=%d native %.2f us %.1f GB/s", type, T, N, rows,
                us_ref, (double)one / 1000.0 / us_ref);
    if (type == 2) {
        const double us_new = timed(st, reps, neu);
        std::printf(" q4-row %.2f us %.1f GB/s %s", us_new, (double)one / 1000.0 / us_new,
                    same ? "bitwise" : "MISMATCH");
    }
    std::printf("\n");
    if (!same) std::exit(2);
    for (void* p : {(void*)w, (void*)dx, xq, (void*)y, (void*)ref}) ck(cudaFree(p), "free");
}
}
int main(int argc, char** argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const int reps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 100;
    ck(cudaSetDevice(device), "device");
    cudaStream_t st = nullptr; ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    std::printf("q36 TP Q4 benchmark device=%d reps=%d FF=512 K=3072 top=8; GB/s uses GGUF weight bytes\n", device, reps);
    for (int T = 1; T <= 4; ++T) {
        for (int dn : {2, 3}) expert(2, dn, T, reps, st);
        expert(3, 3, T, reps, st);
        for (int type : {2, 6, 14, 8}) dense(type, T, reps, st);
    }
    ck(cudaStreamDestroy(st), "destroy stream");
}
