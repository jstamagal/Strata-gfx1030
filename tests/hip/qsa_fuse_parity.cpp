// tests/hip/qsa_fuse_parity.cpp - two fused launches of a QSA layer's decode window against the pairs of launches they
// replace, bit for bit: STRATA_QSA_GATE_Q8 (the output gate and the q8_1 quantization of its result:
// native_gate_quantize_q8_1 against native_qsa_gate_apply + native_quantize_q8_1) and STRATA_IDX_PAIR (two BF16
// projections of the same rows: bf16_gemv_fp32_mmvf_multi_pair against two bf16_gemv_fp32_mmvf_multi calls, which at
// one token are bf16_gemv_fp32_mmvf's), for 1 to 8 tokens.  GPU, synthetic, no model.
#include <hip/hip_runtime.h>
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

namespace {
std::mt19937 rng(20261013);
void* to_device(const void* h, size_t bytes) {
    void* d = nullptr;
    CHECK(hipMalloc(&d, bytes));
    CHECK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice));
    return d;
}
std::vector<uint8_t> back(const void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    CHECK(hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost));
    return h;
}
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    using namespace strata::kernels;
    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    std::normal_distribution<float> nd(0.0f, 2.0f);
    int bad = 0;

    // ---- the gate and its q8_1 image
    int gate_cases = 0, gate_bad = 0;
    for (int hd : {256, 128})
        for (int heads : {24, 48, 72, 96, 144, 192, 5}) {
            std::vector<float> attn((size_t) heads * hd), qf((size_t) heads * 2 * hd);
            for (auto& v : attn) v = nd(rng);
            for (auto& v : qf) v = nd(rng) * 3.0f;
            if (heads == 5) for (int i = 0; i < 64; ++i) attn[(size_t) i] = 0.0f;   // two all-zero blocks
            auto* d_attn = (float*) to_device(attn.data(), attn.size() * 4);
            auto* d_qf = (float*) to_device(qf.data(), qf.size() * 4);
            const size_t ob = attn.size() * 4, qb = native_q8_1_bytes(heads * hd, 1);
            float *o1, *o2;
            void *q1, *q2;
            CHECK(hipMalloc((void**) &o1, ob)); CHECK(hipMalloc((void**) &o2, ob));
            CHECK(hipMalloc(&q1, qb)); CHECK(hipMalloc(&q2, qb));
            CHECK(hipMemsetAsync(o1, 0x5a, ob, stream)); CHECK(hipMemsetAsync(o2, 0x5a, ob, stream));
            CHECK(hipMemsetAsync(q1, 0x5a, qb, stream)); CHECK(hipMemsetAsync(q2, 0x5a, qb, stream));
            native_qsa_gate_apply(d_attn, d_qf, o1, heads, hd, stream);
            native_quantize_q8_1(o1, q1, heads * hd, 1, stream);
            const bool took = native_gate_quantize_q8_1(d_attn, d_qf, o2, q2, heads, hd, stream);
            CHECK(hipStreamSynchronize(stream));
            ++gate_cases;
            if (!took || back(o1, ob) != back(o2, ob) || back(q1, qb) != back(q2, qb)) {
                if (gate_bad++ < 6) std::printf("  gate + q8_1: %d heads x %d: %s\n", heads, hd, took ? "DIFFERENT" : "refused");
            }
            hipFree(d_attn); hipFree(d_qf); hipFree(o1); hipFree(o2); hipFree(q1); hipFree(q2);
        }
    std::printf("gate + q8_1 in one launch: %d shapes, %s\n", gate_cases, gate_bad ? "DIFFERENT" : "bitwise equal (floats and q8_1 bytes)");
    bad += gate_bad;

    // ---- two BF16 projections of the same rows
    int pair_cases = 0, pair_bad = 0;
    const int n_in = 2560;
    for (const auto& outs : {std::pair<int, int>{128, 512}, {512, 128}, {77, 1}, {1, 300}}) {
        const int no1 = outs.first, no2 = outs.second;
        std::vector<uint16_t> w1((size_t) no1 * n_in), w2((size_t) no2 * n_in);
        std::normal_distribution<float> wd(0.0f, 0.05f);
        for (auto* w : {&w1, &w2})
            for (auto& v : *w) { const float f = wd(rng); uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t) (u >> 16); }
        auto* dw1 = (const uint16_t*) to_device(w1.data(), w1.size() * 2);
        auto* dw2 = (const uint16_t*) to_device(w2.data(), w2.size() * 2);
        const int64_t ldx = n_in + 64, ld1 = no1 + 3, ld2 = no2;
        std::vector<float> x((size_t) 8 * ldx);
        for (auto& v : x) v = nd(rng);
        auto* dx = (const float*) to_device(x.data(), x.size() * 4);
        const size_t b1 = (size_t) 8 * ld1 * 4, b2 = (size_t) 8 * ld2 * 4;
        float *r1, *r2, *p1, *p2;
        CHECK(hipMalloc((void**) &r1, b1)); CHECK(hipMalloc((void**) &p1, b1));
        CHECK(hipMalloc((void**) &r2, b2)); CHECK(hipMalloc((void**) &p2, b2));
        for (int nt = 1; nt <= 8; ++nt) {   // one row: STRATA_T1_DIET (4)'s one-row QSA layer
            for (float* p : {r1, p1}) CHECK(hipMemsetAsync(p, 0x5a, b1, stream));
            for (float* p : {r2, p2}) CHECK(hipMemsetAsync(p, 0x5a, b2, stream));
            bf16_gemv_fp32_mmvf_multi(dx, ldx, dw1, r1, ld1, n_in, no1, nt, stream);
            bf16_gemv_fp32_mmvf_multi(dx, ldx, dw2, r2, ld2, n_in, no2, nt, stream);
            const bool took = bf16_gemv_fp32_mmvf_multi_pair(dx, ldx, dw1, p1, ld1, no1, dw2, p2, ld2, no2, n_in, nt, stream);
            CHECK(hipStreamSynchronize(stream));
            ++pair_cases;
            if (!took || back(r1, b1) != back(p1, b1) || back(r2, b2) != back(p2, b2)) {
                if (pair_bad++ < 6) std::printf("  BF16 pair %d + %d rows, %d tokens: %s\n", no1, no2, nt, took ? "DIFFERENT" : "refused");
            }
        }
        hipFree((void*) dw1); hipFree((void*) dw2); hipFree((void*) dx); hipFree(r1); hipFree(r2); hipFree(p1); hipFree(p2);
    }
    std::printf("two BF16 projections in one launch: %d cases of 1-8 tokens, %s\n", pair_cases, pair_bad ? "DIFFERENT" : "bitwise equal");
    bad += pair_bad;
    std::printf("%s\n", bad ? "FAILED" : "PASSED");
    return bad ? 1 : 0;
}
