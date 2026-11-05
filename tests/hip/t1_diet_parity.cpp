// tests/hip/t1_diet_parity.cpp - STRATA_T1_DIET (2): the hyper-connection read whose wide down launch also makes the
// split register norm (fused_gr_read_multi with a call site's nrm_sync) against the same read without it, bit for bit.
// GPU, synthetic, no model.
//  1. Direct launches: with and without the inject rows (a layer's read / the head's), the pending write, the MoE
//     combine in the norm, the q8_1 image; one ticket block per configuration taken by 60 reads of 1-8 tokens in random
//     order (the fold runs at 1-4, its ticket count is checked; 5-8 leave it alone): bo, R, lo, rs, inject, mixed, xn
//     and q8_1 every read.
//  2. Captured graphs: 24 chained reads (each read's mixed and inject feed the next one's pending write, one ticket block
//     per read as the verifier owns them), 1-4 tokens, with and without the combine; both graphs replayed the same
//     number of times from the same state, then every buffer compared.  Microseconds per read both ways (median of 21
//     replays, cold weights: every read its own 13 MB).
// The read's switches default to the live config's (STRATA_GR_NORM_FAST=1 STRATA_HC_WIDE=3 STRATA_HC_DOWN2=1
// STRATA_HC_WIDE8=1 STRATA_HC_UP_TSUM=1) where the environment does not name them.
#include <hip/hip_runtime.h>
#include "strata/kernels/fused_gr.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

namespace {
using strata::kernels::FusedGrArgs;
using strata::kernels::kFusedGrMaxT;
using strata::kernels::kFusedGrNrmSyncBytes;
constexpr int N = 2560, HC = 4, LR = 320, D = N * HC, K = 10, Q8B = (N / 32) * 36;
std::mt19937 rng(20261105);
template <class T> T* dev(const std::vector<T>& h) {
    void* d = nullptr;
    CHECK(hipMalloc(&d, h.size() * sizeof(T)));
    CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return (T*) d;
}
void* dev_fill(size_t bytes, int byte) {
    void* d = nullptr;
    CHECK(hipMalloc(&d, bytes));
    CHECK(hipMemset(d, byte, bytes));
    CHECK(hipDeviceSynchronize());   // the fill is on the null stream, which the test's non-blocking stream does not wait for
    return d;
}
std::vector<uint8_t> back(const void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    CHECK(hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost));
    return h;
}
uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
std::vector<float> normal(size_t n, float sd, float mean = 0.0f) {
    std::normal_distribution<float> nd(mean, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}
std::vector<uint16_t> normal_bf16(size_t n, float sd) {
    std::normal_distribution<float> nd(0.0f, sd);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = bf16(nd(rng));
    return v;
}
unsigned long long tickets(const void* sync) {
    unsigned long long t = 0;
    CHECK(hipMemcpy(&t, sync, 8, hipMemcpyDeviceToHost));
    return t;
}
struct Weights {
    float* norm; uint16_t *down, *up, *inj;
    void make() {
        norm = dev(normal(D, 0.3f, 1.0f));
        down = dev(normal_bf16((size_t) LR * D, 0.02f));
        up = dev(normal_bf16((size_t) D * LR, 0.05f));
        inj = dev(normal_bf16((size_t) HC * D, 0.02f));
    }
    void drop() { (void) hipFree(norm); (void) hipFree(down); (void) hipFree(up); (void) hipFree(inj); }
};
// one arm's buffers for up to 8 tokens: two of mixed / inject so a chain of reads can alternate them
struct Arm {
    float *R, *bo, *ip, *lo, *rs, *io[2], *mixed[2], *xn; uint8_t* q8; unsigned* cnt;
    void make() {
        R = (float*) dev_fill((size_t) 8 * D * 4, 0); bo = (float*) dev_fill((size_t) 8 * N * 4, 0x5a);
        ip = (float*) dev_fill(8 * HC * 4, 0);
        lo = (float*) dev_fill(8 * LR * 4, 0); rs = (float*) dev_fill(8 * HC * 4, 0);
        for (int i = 0; i < 2; ++i) { io[i] = (float*) dev_fill(8 * HC * 4, 0); mixed[i] = (float*) dev_fill((size_t) 8 * N * 4, 0); }
        xn = (float*) dev_fill((size_t) 8 * D * 4, 0); q8 = (uint8_t*) dev_fill((size_t) 8 * Q8B, 0x33);
        cnt = (unsigned*) dev_fill((N / 32) * 4, 0);
    }
    void drop() {
        for (void* p : {(void*) R, (void*) bo, (void*) ip, (void*) lo, (void*) rs, (void*) io[0], (void*) io[1], (void*) mixed[0],
                        (void*) mixed[1], (void*) xn, (void*) q8, (void*) cnt})
            (void) hipFree(p);
    }
    // every output buffer's bytes for T tokens, in one vector per buffer
    std::vector<std::pair<const char*, std::vector<uint8_t>>> snap(int T) const {
        return {{"R", back(R, (size_t) T * D * 4)}, {"bo", back(bo, (size_t) T * N * 4)}, {"lo", back(lo, (size_t) T * LR * 4)},
                {"rs", back(rs, (size_t) T * HC * 4)}, {"inject", back(io[0], (size_t) T * HC * 4)},
                {"inject'", back(io[1], (size_t) T * HC * 4)}, {"mixed", back(mixed[0], (size_t) T * N * 4)},
                {"mixed'", back(mixed[1], (size_t) T * N * 4)}, {"xn", back(xn, (size_t) T * D * 4)},
                {"q8_1", back(q8, (size_t) T * Q8B)}};
    }
};
// the read's per-token arguments: pending write from `bo_prev` / `inj_prev`, outputs to mixed[o] / io[o]
void args(FusedGrArgs* fa, const Arm& a, int T, const Weights& w, bool inject, bool apply, const float* bo_prev,
          const float* inj_prev, int o, const float* parts, const float* cw, const float* sh, bool q8) {
    for (int t = 0; t < T; ++t) {
        FusedGrArgs& x = fa[t];
        x = FusedGrArgs{};
        x.R = a.R + (size_t) t * D; x.R_out = a.R + (size_t) t * D; x.apply = apply;
        if (apply) { x.bo_prev = bo_prev + (size_t) t * N; x.inj_prev = inj_prev + t * HC; }
        x.w_norm = w.norm; x.w_down = w.down; x.w_up = w.up; x.w_inject = inject ? w.inj : nullptr;
        x.eps = 1e-6f; x.lo = a.lo + t * LR; x.rs = a.rs + t * HC;
        x.inject_out = inject ? a.io[o] + t * HC : nullptr;
        x.mixed = a.mixed[o] + (size_t) t * N;
        if (q8) { x.q8_cnt = a.cnt; x.q8_mixed = a.q8 + (size_t) t * Q8B; }
        if (parts != nullptr) { x.cmb_parts = parts + (size_t) t * K * N; x.cmb_w = cw + t * K; x.cmb_shared = sh + (size_t) t * N; }
    }
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    for (const char* kv : {"STRATA_GR_NORM_FAST=1", "STRATA_HC_WIDE=3", "STRATA_HC_DOWN2=1", "STRATA_HC_WIDE8=1", "STRATA_HC_UP_TSUM=1"}) {
        const std::string s(kv);
        const auto eq = s.find('=');
        setenv(s.substr(0, eq).c_str(), s.substr(eq + 1).c_str(), 0);
    }
    const bool no_bench = argc > 1 && std::strcmp(argv[1], "--no-bench") == 0;
    using namespace strata::kernels;
    hipStream_t st;
    CHECK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    fused_gr_check();
    std::printf("t1_diet_parity: read variant %d; STRATA_GR_NORM_FAST=%s STRATA_HC_WIDE=%s STRATA_HC_DOWN2=%s STRATA_HC_WIDE8=%s\n",
                fused_gr_variant(), std::getenv("STRATA_GR_NORM_FAST"), std::getenv("STRATA_HC_WIDE"), std::getenv("STRATA_HC_DOWN2"),
                std::getenv("STRATA_HC_WIDE8"));
    int bad = 0;
    Weights w;
    w.make();

    // ---- 1. direct launches, one ticket block per configuration
    {
        int cases = 0, nbad = 0, folded = 0, want_folded = 0;
        Arm arm[2];
        for (auto& a : arm) a.make();
        float* parts = (float*) dev_fill((size_t) 8 * K * N * 4, 0);
        float* cw = (float*) dev_fill(8 * K * 4, 0);
        float* sh = (float*) dev_fill((size_t) 8 * N * 4, 0);
        for (int inject = 1; inject >= 0; --inject)
            for (int apply = 1; apply >= 0; --apply)
                for (int cmb = 0; cmb <= apply; ++cmb)
                    for (int q8 = 1; q8 >= 0; --q8) {
                        void* sync = dev_fill(kFusedGrNrmSyncBytes, 0);
                        const unsigned long long rows = LR + (inject ? HC : 0);
                        unsigned long long want = 0;
                        int cbad = 0;
                        for (int step = 0; step < 60; ++step) {
                            const int T = step < 8 ? 1 + step % 4 : (rng() % 3 != 0 ? 1 + (int) (rng() % 4) : 1 + (int) (rng() % 8));
                            const auto R = normal((size_t) T * D, 1.0f), bo = normal((size_t) T * N, 0.5f), ip = normal((size_t) T * HC, 1.0f);
                            if (cmb) {
                                const auto p = normal((size_t) T * K * N, 0.5f), c = normal((size_t) T * K, 0.2f, 0.1f), s = normal((size_t) T * N, 0.5f);
                                CHECK(hipMemcpy(parts, p.data(), p.size() * 4, hipMemcpyHostToDevice));
                                CHECK(hipMemcpy(cw, c.data(), c.size() * 4, hipMemcpyHostToDevice));
                                CHECK(hipMemcpy(sh, s.data(), s.size() * 4, hipMemcpyHostToDevice));
                            }
                            for (int k = 0; k < 2; ++k) {
                                Arm& a = arm[k];
                                CHECK(hipMemcpy(a.R, R.data(), R.size() * 4, hipMemcpyHostToDevice));
                                CHECK(hipMemcpy(a.bo, bo.data(), bo.size() * 4, hipMemcpyHostToDevice));
                                CHECK(hipMemcpy(a.ip, ip.data(), ip.size() * 4, hipMemcpyHostToDevice));
                                CHECK(hipMemsetAsync(a.cnt, 0, (N / 32) * 4, st));
                                FusedGrArgs fa[kFusedGrMaxT];
                                args(fa, a, T, w, inject, apply, a.bo, a.ip, 0, cmb ? parts : nullptr, cw, sh, q8);
                                fused_gr_read_multi(fa, T, a.xn, st, nullptr, 0, k == 1 ? sync : nullptr);
                            }
                            CHECK(hipStreamSynchronize(st));
                            ++cases;
                            if (T <= 4) { want += rows; ++want_folded; }
                            const unsigned long long got = tickets(sync);
                            if (got == want && T <= 4) ++folded;
                            const auto s0 = arm[0].snap(T), s1 = arm[1].snap(T);
                            const char* what = got != want ? "ticket count (the fold did not run as expected)" : nullptr;
                            for (size_t i = 0; what == nullptr && i < s0.size(); ++i)
                                if (s0[i].second != s1[i].second) what = s0[i].first;
                            if (what) {
                                want = got;   // resynchronize the expectation
                                if (cbad++ < 4)
                                    std::printf("  inject %d apply %d combine %d q8_1 %d, step %d, %d tokens: %s DIFFERENT (tickets %llu)\n",
                                                inject, apply, cmb, q8, step, T, what, got);
                            }
                        }
                        nbad += cbad;
                        (void) hipFree(sync);
                    }
        for (auto& a : arm) a.drop();
        (void) hipFree(parts); (void) hipFree(cw); (void) hipFree(sh);
        std::printf("norm in the down launch, direct: %d reads (%d of 1-4 tokens, %d folded), %s\n", cases, want_folded, folded,
                    nbad ? "DIFFERENT" : "bitwise equal");
        if (folded == 0) std::printf("  the fold never ran: this card / environment does not take the wide down's exact-T form\n");
        bad += nbad + (folded == 0);
    }

    // ---- 2. captured graphs: 24 chained reads a graph
    if (!no_bench) {
        const int L = 24;
        std::vector<Weights> lw(L);
        for (auto& x : lw) x.make();
        float* parts = dev(normal((size_t) 8 * K * N, 0.5f));
        float* cw = dev(normal(8 * K, 0.2f, 0.1f));
        float* sh = dev(normal((size_t) 8 * N, 0.5f));
        hipEvent_t e0, e1;
        CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        int gbad = 0;
        for (int cmb = 0; cmb <= 1; ++cmb)
            for (int T = 1; T <= 4; ++T) {
                Arm arm[2];
                for (auto& a : arm) a.make();
                std::vector<void*> syncs(L);
                for (auto& s : syncs) s = dev_fill(kFusedGrNrmSyncBytes, 0);
                const auto R = normal((size_t) T * D, 1.0f), m0 = normal((size_t) T * N, 0.5f), i0 = normal((size_t) T * HC, 1.0f);
                for (auto& a : arm) {
                    CHECK(hipMemcpy(a.R, R.data(), R.size() * 4, hipMemcpyHostToDevice));
                    CHECK(hipMemcpy(a.mixed[1], m0.data(), m0.size() * 4, hipMemcpyHostToDevice));
                    CHECK(hipMemcpy(a.io[1], i0.data(), i0.size() * 4, hipMemcpyHostToDevice));
                }
                hipGraphExec_t ge[2];
                for (int k = 0; k < 2; ++k) {
                    Arm& a = arm[k];
                    hipGraph_t g;
                    CHECK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
                    for (int l = 0; l < L; ++l) {
                        // the previous read's mixed and inject are this read's pending write (with the combine, its
                        // bo is made into the previous mixed buffer, which nothing reads after this read)
                        const int o = l & 1, p = o ^ 1;
                        FusedGrArgs fa[kFusedGrMaxT];
                        args(fa, a, T, lw[l], true, true, a.mixed[p], a.io[p], o, cmb ? parts : nullptr, cw, sh, true);
                        fused_gr_read_multi(fa, T, a.xn, st, nullptr, 0, k == 1 ? syncs[l] : nullptr);
                    }
                    CHECK(hipStreamEndCapture(st, &g));
                    CHECK(hipGraphInstantiate(&ge[k], g, nullptr, nullptr, 0));
                    CHECK(hipGraphDestroy(g));
                }
                double us[2];
                std::vector<float> ms[2];
                for (int i = 0; i < 3; ++i) for (int k = 0; k < 2; ++k) CHECK(hipGraphLaunch(ge[k], st));
                CHECK(hipStreamSynchronize(st));
                for (int r = 0; r < 21; ++r)
                    for (int k = 0; k < 2; ++k) {   // interleaved: both see the same clocks and the same cold caches
                        CHECK(hipEventRecord(e0, st)); CHECK(hipGraphLaunch(ge[k], st)); CHECK(hipEventRecord(e1, st));
                        CHECK(hipEventSynchronize(e1));
                        float m = 0; CHECK(hipEventElapsedTime(&m, e0, e1)); ms[k].push_back(m);
                    }
                for (int k = 0; k < 2; ++k) { std::sort(ms[k].begin(), ms[k].end()); us[k] = 1e3 * ms[k][ms[k].size() / 2] / L; }
                const auto s0 = arm[0].snap(T), s1 = arm[1].snap(T);
                const char* what = nullptr;
                for (size_t i = 0; what == nullptr && i < s0.size(); ++i)
                    if (s0[i].second != s1[i].second) what = s0[i].first;
                unsigned long long tk = 0;
                for (auto* s : syncs) tk += tickets(s);
                const unsigned long long want_tk = (unsigned long long) L * 24 * (LR + HC);
                if (tk != want_tk) what = what ? what : "ticket count";
                std::printf("BENCH T %d combine %d: read %6.2f us | norm in the down launch %6.2f us (%+.2f us) | 24 replays of 24 chained reads: %s%s\n",
                            T, cmb, us[0], us[1], us[1] - us[0], what ? what : "bitwise equal", what ? " DIFFERENT" : "");
                gbad += what != nullptr;
                for (auto& x : ge) (void) hipGraphExecDestroy(x);
                for (auto* s : syncs) (void) hipFree(s);
                for (auto& a : arm) a.drop();
            }
        for (auto& x : lw) x.drop();
        (void) hipFree(parts); (void) hipFree(cw); (void) hipFree(sh);
        std::printf("norm in the down launch, captured graphs: %s\n", gbad ? "DIFFERENT" : "bitwise equal");
        bad += gbad;
    }
    w.drop();
    std::printf("%s\n", bad ? "FAILED" : "PASSED");
    return bad ? 1 : 0;
}
