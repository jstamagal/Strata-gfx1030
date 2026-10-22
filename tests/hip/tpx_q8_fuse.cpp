// T=1 TP Q8 projection pair: separate native MMVQ versus the RDNA2 fused
// projection used by STRATA_Q36_TPX_Q8_T1_FUSE. Random GGUF Q8 blocks, exact
// float output comparison, rotating weights beyond the V620 cache.
#include <hip/hip_runtime.h>
#include "strata/kernels/native_mmvq.hpp"
#include "q36_decode.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { auto e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); std::exit(2); } } while (0)

int main(int argc, char** argv) {
    const int K = argc > 1 ? std::atoi(argv[1]) : 3072;
    const int A = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int B = argc > 3 ? std::atoi(argv[3]) : 1024;
    if (K < 32 || K % 32 || A < 1 || B < 1) return 2;
    const size_t wa = (size_t) K / 32 * 34 * A, wb = (size_t) K / 32 * 34 * B;
    const bool use_packed = std::getenv("LAB_PACKED") && std::strcmp(std::getenv("LAB_PACKED"), "1") == 0;
    const int copies = std::max(2, std::min(16, (int) (((size_t) 256 << 20) / (wa + wb))));
    std::mt19937 rng(20261021);
    auto weights = [&](size_t bytes) {
        std::vector<uint8_t> v(bytes);
        for (size_t i = 0; i < bytes; i += 34) {
            v[i] = 0x00; v[i + 1] = 0x38; // finite f16 scale 0.5
            for (int j = 2; j < 34; ++j) v[i + j] = (uint8_t) rng();
        }
        return v;
    };
    const auto ha = weights(wa), hb = weights(wb);
    std::vector<const uint8_t*> da(copies), db(copies);
    std::vector<const uint8_t*> pa(copies, nullptr), pb(copies, nullptr);
    auto packed = [&](const std::vector<uint8_t>& src, int rows) {
        std::vector<uint8_t> dst(src.size());
        strata::kernels::native_q8_0_pack_host(src.data(), dst.data(), K, rows);
        return dst;
    };
    const bool pack_a = use_packed && strata::kernels::native_q8_0_packed_eligible(K, A);
    const bool pack_b = use_packed && strata::kernels::native_q8_0_packed_eligible(K, B);
    const auto hpa = pack_a ? packed(ha, A) : std::vector<uint8_t>{};
    const auto hpb = pack_b ? packed(hb, B) : std::vector<uint8_t>{};
    for (int i = 0; i < copies; ++i) {
        uint8_t *a = nullptr, *b = nullptr;
        CK(hipMalloc(&a, wa)); CK(hipMalloc(&b, wb));
        CK(hipMemcpy(a, ha.data(), wa, hipMemcpyHostToDevice));
        CK(hipMemcpy(b, hb.data(), wb, hipMemcpyHostToDevice));
        da[i] = a; db[i] = b;
        if (pack_a) {
            uint8_t* p = nullptr; CK(hipMalloc(&p, wa)); CK(hipMemcpy(p, hpa.data(), wa, hipMemcpyHostToDevice));
            pa[i] = p; strata::kernels::native_q8_0_packed_register(a, p, K, A);
        }
        if (pack_b) {
            uint8_t* p = nullptr; CK(hipMalloc(&p, wb)); CK(hipMemcpy(p, hpb.data(), wb, hipMemcpyHostToDevice));
            pb[i] = p; strata::kernels::native_q8_0_packed_register(b, p, K, B);
        }
    }
    std::vector<float> hx(K);
    std::normal_distribution<float> normal(0.f, 1.f);
    for (float& v : hx) v = normal(rng);
    float* dx = nullptr;
    void *xq = nullptr;
    float *old_a = nullptr, *old_b = nullptr, *new_a = nullptr, *new_b = nullptr;
    CK(hipMalloc(&dx, (size_t) K * 4));
    CK(hipMalloc(&xq, strata::kernels::native_q8_1_bytes(K, 1)));
    CK(hipMalloc(&old_a, (size_t) A * 4)); CK(hipMalloc(&old_b, (size_t) B * 4));
    CK(hipMalloc(&new_a, (size_t) A * 4)); CK(hipMalloc(&new_b, (size_t) B * 4));
    CK(hipMemcpy(dx, hx.data(), (size_t) K * 4, hipMemcpyHostToDevice));
    hipStream_t stream;
    CK(hipStreamCreate(&stream));
    strata::kernels::native_quantize_q8_1(dx, xq, K, 1, stream);
    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    auto fused = [&](int i) {
        q36decode::Q8Mat a{pa[i] ? pa[i] : da[i], pa[i] ? (const half*) (pa[i] + (size_t) K * A) : nullptr,
                            new_a, K / 32, A, std::min((A + 3) / 4, prop.multiProcessorCount * 8)};
        q36decode::Q8Mat b{pb[i] ? pb[i] : db[i], pb[i] ? (const half*) (pb[i] + (size_t) K * B) : nullptr,
                            new_b, K / 32, B, std::min((B + 3) / 4, prop.multiProcessorCount * 8)};
        q36decode::project(a, b, {}, xq, {}, 1, stream);
    };
    auto old = [&](int i) {
        strata::kernels::native_mmvq(8, da[i], xq, old_a, K, A, 1, stream);
        strata::kernels::native_mmvq(8, db[i], xq, old_b, K, B, 1, stream);
    };
    old(0); fused(0); CK(hipStreamSynchronize(stream));
    std::vector<uint8_t> oa((size_t) A * 4), ob((size_t) B * 4), na(oa.size()), nb(ob.size());
    CK(hipMemcpy(oa.data(), old_a, oa.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(ob.data(), old_b, ob.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(na.data(), new_a, na.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(nb.data(), new_b, nb.size(), hipMemcpyDeviceToHost));
    const bool same = oa == na && ob == nb;
    auto time_us = [&](bool use_fused) {
        hipGraph_t graph;
        hipGraphExec_t exec;
        CK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        for (int i = 0; i < 32; ++i) { if (use_fused) fused(i % copies); else old(i % copies); }
        CK(hipStreamEndCapture(stream, &graph));
        CK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        double best = 1e30;
        for (int rep = 0; rep < 8; ++rep) {
            CK(hipEventRecord(e0, stream)); CK(hipGraphLaunch(exec, stream)); CK(hipEventRecord(e1, stream));
            CK(hipEventSynchronize(e1));
            float ms = 0;
            CK(hipEventElapsedTime(&ms, e0, e1));
            if (rep >= 2) best = std::min(best, (double) ms * 1000.0 / 32.0);
        }
        CK(hipEventDestroy(e0)); CK(hipEventDestroy(e1));
        CK(hipGraphExecDestroy(exec)); CK(hipGraphDestroy(graph));
        return best;
    };
    const double t0 = time_us(false), t1 = time_us(true);
    std::printf("Q8 pair K=%d M=%d+%d packed=%d/%d weights=%.2f MB old %.1f us %.0f GB/s -> fused %.1f us %.0f GB/s bits=%s\n",
                K, A, B, pack_a, pack_b, (wa + wb) / 1e6, t0, (wa + wb) / t0 / 1000., t1, (wa + wb) / t1 / 1000., same ? "OK" : "FAIL");
    for (auto p : da) if (pack_a) strata::kernels::native_q8_0_packed_unregister(p);
    for (auto p : db) if (pack_b) strata::kernels::native_q8_0_packed_unregister(p);
    for (auto p : pa) if (p) CK(hipFree((void*) p));
    for (auto p : pb) if (p) CK(hipFree((void*) p));
    for (auto p : da) CK(hipFree((void*) p));
    for (auto p : db) CK(hipFree((void*) p));
    for (auto p : {static_cast<void*>(dx), xq, static_cast<void*>(old_a), static_cast<void*>(old_b),
                   static_cast<void*>(new_a), static_cast<void*>(new_b)}) CK(hipFree(p));
    CK(hipStreamDestroy(stream));
    return same ? 0 : 1;
}
