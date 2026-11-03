// tests/hip/node_fuse_parity.cpp - STRATA_NODE_FUSE's fused launches against the launches they replace, bit for bit,
// 1 to 8 tokens.  GPU, synthetic, no model.
//  1. the MoE combine inside the next hyper-connection read: native_moe_combine_multi + fused_gr_read_multi against
//     fused_gr_read_multi with cmb_* set (bo, R, lo, rs, inject, mixed and its q8_1 image).  The combine sits in the
//     norm kernel only with STRATA_GR_NORM_FAST=1 (else the read launches it first): run it with and without.
//  2. three per-head RMSNorm + RoPE calls (k, q out of q|gate rows, indexer q) against native_qsa_rms_norm_rope_multi,
//     for every rotation form, without and with YaRN.
//  3. gdn_conv_l2_multi + gdn_ab_multi + gdn_step_norm_multi (with and without the q8_1 image) against gdn_fused_step
//     (h, gate, beta, y, q8_1, the state left as it was).
//  4. copy_from_mapped (the indexer tail's snapshot) + native_qsa_indexer_append_steps against the append with `snap`
//     (the snapshot, tail, dead, pooled keys and block position), windows of 1-8 steps from several positions.
//  5. STRATA_HC_WIDE8: the hyper-connection read with STRATA_HC_WIDE=3 (7-8 tokens: the row kernel's down in one
//     launch) against the default read, 1-8 tokens, with and without the pending write (every output); and
//     STRATA_HC_UP_TSUM (the up kernel's per-token sums as one transposed butterfly) and STRATA_HC_DOWN2 (two rows a
//     block in the down) against that wide read.
#include <hip/hip_runtime.h>
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

namespace {
std::mt19937 rng(20261020);
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
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (std::getenv("STRATA_NORM_ROPE") == nullptr) setenv("STRATA_NORM_ROPE", "2", 1);   // the HIP opt-in of the fused norm + RoPE
    using namespace strata::kernels;
    hipStream_t st;
    CHECK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    int bad = 0;

    // ---- 1. the combine inside the hyper-connection read
    {
        fused_gr_check();
        native_moe_combine_set_enabled(true);
        const int N = 2560, HC = 4, LR = 320, D = N * HC, K = 10;
        auto w_norm = normal(D, 0.3f, 1.0f);
        auto w_down = normal_bf16((size_t) LR * D, 0.02f), w_up = normal_bf16((size_t) D * LR, 0.05f),
             w_inj = normal_bf16((size_t) HC * D, 0.02f);
        float* d_norm = dev(w_norm);
        uint16_t *d_down = dev(w_down), *d_up = dev(w_up), *d_inj = dev(w_inj);
        int cases = 0, cbad = 0;
        for (int T = 1; T <= kFusedGrMaxT; ++T) {
            auto R = normal((size_t) T * D, 1.0f), parts = normal((size_t) T * K * N, 0.5f), sh = normal((size_t) T * N, 0.5f);
            auto w = normal((size_t) T * K, 0.2f, 0.1f), inj = normal((size_t) T * HC, 1.0f);
            float* d_parts = dev(parts);
            float* d_sh = dev(sh);
            float* d_w = dev(w);
            float* d_inj_prev = dev(inj);
            struct Arm { float *R, *bo, *lo, *rs, *io, *mixed, *xn; uint8_t* q8; unsigned* cnt; } arm[2];
            for (auto& a : arm) {
                a.R = dev(R);
                a.bo = (float*) dev_fill((size_t) T * N * 4, 0x5a);
                a.lo = (float*) dev_fill((size_t) T * LR * 4, 0);
                a.rs = (float*) dev_fill((size_t) T * HC * 4, 0);
                a.io = (float*) dev_fill((size_t) T * HC * 4, 0);
                a.mixed = (float*) dev_fill((size_t) T * N * 4, 0);
                a.xn = (float*) dev_fill((size_t) T * D * 4, 0);
                a.q8 = (uint8_t*) dev_fill((size_t) T * (N / 32) * 36, 0x33);
                a.cnt = (unsigned*) dev_fill((size_t) (N / 32) * 4, 0);
            }
            for (int k = 0; k < 2; ++k) {
                Arm& a = arm[k];
                FusedGrArgs fa[kFusedGrMaxT];
                for (int t = 0; t < T; ++t) {
                    FusedGrArgs& x = fa[t];
                    x.R = a.R + (size_t) t * D; x.R_out = a.R + (size_t) t * D; x.apply = true;
                    x.bo_prev = a.bo + (size_t) t * N; x.inj_prev = d_inj_prev + t * HC;
                    x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up; x.w_inject = d_inj;
                    x.eps = 1e-6f; x.lo = a.lo + t * LR; x.rs = a.rs + t * HC; x.inject_out = a.io + t * HC;
                    x.mixed = a.mixed + (size_t) t * N;
                    x.q8_cnt = a.cnt; x.q8_mixed = a.q8 + (size_t) t * (N / 32) * 36;
                    if (k == 1) { x.cmb_parts = d_parts + (size_t) t * K * N; x.cmb_w = d_w + t * K; x.cmb_shared = d_sh + (size_t) t * N; }
                }
                if (k == 0) native_moe_combine_multi(d_parts, d_w, d_sh, a.bo, N, K, T, st);
                fused_gr_read_multi(fa, T, a.xn, st);
            }
            CHECK(hipStreamSynchronize(st));
            ++cases;
            const char* what = nullptr;
            if (back(arm[0].bo, (size_t) T * N * 4) != back(arm[1].bo, (size_t) T * N * 4)) what = "bo";
            else if (back(arm[0].R, (size_t) T * D * 4) != back(arm[1].R, (size_t) T * D * 4)) what = "R";
            else if (back(arm[0].lo, (size_t) T * LR * 4) != back(arm[1].lo, (size_t) T * LR * 4)) what = "lo";
            else if (back(arm[0].rs, (size_t) T * HC * 4) != back(arm[1].rs, (size_t) T * HC * 4)) what = "rs";
            else if (back(arm[0].io, (size_t) T * HC * 4) != back(arm[1].io, (size_t) T * HC * 4)) what = "inject";
            else if (back(arm[0].mixed, (size_t) T * N * 4) != back(arm[1].mixed, (size_t) T * N * 4)) what = "mixed";
            else if (back(arm[0].q8, (size_t) T * (N / 32) * 36) != back(arm[1].q8, (size_t) T * (N / 32) * 36)) what = "q8_1";
            if (what) { ++cbad; std::printf("  combine in the read, %d tokens: %s DIFFERENT\n", T, what); }
            for (auto& a : arm) { hipFree(a.R); hipFree(a.bo); hipFree(a.lo); hipFree(a.rs); hipFree(a.io); hipFree(a.mixed); hipFree(a.xn); hipFree(a.q8); hipFree(a.cnt); }
            hipFree(d_parts); hipFree(d_sh); hipFree(d_w); hipFree(d_inj_prev);
        }
        hipFree(d_norm); hipFree(d_down); hipFree(d_up); hipFree(d_inj);
        std::printf("combine in the hyper-connection read (variant %d, STRATA_GR_NORM_FAST=%s): %d token counts, %s\n",
                    fused_gr_variant(), std::getenv("STRATA_GR_NORM_FAST") ? std::getenv("STRATA_GR_NORM_FAST") : "unset",
                    cases, cbad ? "DIFFERENT" : "bitwise equal");
        bad += cbad;
    }

    // ---- 2. k, q and the indexer query's norm + RoPE in one launch
    {
        const int NKV = 2, NH = 24, HD = 256, IQ = 4, ID = 128, n_rot = 64;
        RopeScaling none, yarn2;
        yarn2.type = RopeScalingType::YaRN; yarn2.factor = 2.0; yarn2.ext_factor = 1.0;
        int cases = 0, nbad = 0;
        for (int form = 0; form < 3; ++form)
            for (const RopeScaling* sc : {&none, &yarn2})
                for (int T = 1; T <= 8; ++T) {
                    auto k = normal((size_t) T * NKV * HD, 1.5f), qf = normal((size_t) T * NH * 2 * HD, 1.5f),
                         qi = normal((size_t) T * IQ * ID, 1.5f);
                    auto gk = normal(HD, 0.3f, 1.0f), gq = normal(HD, 0.3f, 1.0f), gi = normal(ID, 0.3f, 1.0f);
                    std::vector<int> pk((size_t) T * NKV), pq((size_t) T * NH), pi((size_t) T * IQ);
                    std::uniform_int_distribution<int> pd(0, 400000);
                    const int base = pd(rng);
                    for (int t = 0; t < T; ++t) {
                        for (int h = 0; h < NKV; ++h) pk[(size_t) t * NKV + h] = base + t;
                        for (int h = 0; h < NH; ++h) pq[(size_t) t * NH + h] = base + t;
                        for (int h = 0; h < IQ; ++h) pi[(size_t) t * IQ + h] = base + t;
                    }
                    float *dgk = dev(gk), *dgq = dev(gq), *dgi = dev(gi), *dqf = dev(qf);
                    int *dpk = dev(pk), *dpq = dev(pq), *dpi = dev(pi);
                    float *k1 = dev(k), *k2 = dev(k), *i1 = dev(qi), *i2 = dev(qi);
                    float* q1 = (float*) dev_fill((size_t) T * NH * HD * 4, 0x11);
                    float* q2 = (float*) dev_fill((size_t) T * NH * HD * 4, 0x11);
                    native_norm_rope_tune(form);
                    native_qsa_rms_norm_rope(k1, HD, dgk, k1, T * NKV, HD, n_rot, 1e-6f, *sc, dpk, st);
                    native_qsa_rms_norm_rope(dqf, 2 * HD, dgq, q1, T * NH, HD, n_rot, 1e-6f, *sc, dpq, st);
                    native_qsa_rms_norm_rope(i1, ID, dgi, i1, T * IQ, ID, n_rot, 1e-6f, *sc, dpi, st);
                    NormRopeItems its;
                    its.it[0] = {k2, HD, dgk, k2, T * NKV, HD, dpk};
                    its.it[1] = {dqf, 2 * HD, dgq, q2, T * NH, HD, dpq};
                    its.it[2] = {i2, ID, dgi, i2, T * IQ, ID, dpi};
                    its.n = 3;
                    const bool took = native_qsa_rms_norm_rope_multi(its, n_rot, 1e-6f, *sc, st);
                    CHECK(hipStreamSynchronize(st));
                    ++cases;
                    const char* what = !took ? "refused"
                                     : back(k1, k.size() * 4) != back(k2, k.size() * 4) ? "k"
                                     : back(q1, (size_t) T * NH * HD * 4) != back(q2, (size_t) T * NH * HD * 4) ? "q"
                                     : back(i1, qi.size() * 4) != back(i2, qi.size() * 4) ? "indexer q" : nullptr;
                    if (what && nbad++ < 8) std::printf("  norm + RoPE x3, form %d, %s, %d tokens: %s\n", form,
                                                        sc == &none ? "no scaling" : "YaRN 2", T, what);
                    for (void* p : {(void*) dgk, (void*) dgq, (void*) dgi, (void*) dqf, (void*) dpk, (void*) dpq, (void*) dpi,
                                    (void*) k1, (void*) k2, (void*) i1, (void*) i2, (void*) q1, (void*) q2})
                        hipFree(p);
                }
        native_norm_rope_tune(-1);
        std::printf("k, q and indexer-q norm + RoPE in one launch: %d cases, %s\n", cases, nbad ? "DIFFERENT" : "bitwise equal");
        bad += nbad;
    }

    // ---- 3. the GDN conv, gates and recurrence in one launch
    {
        const int S = 128, HK = 16, HV = 48, C = S * (2 * HK + HV), N = 2560;
        auto state = normal((size_t) S * HV * S, 0.05f), hist = normal((size_t) C * 3, 1.0f), cw = normal((size_t) C * 4, 0.4f);
        auto wa = normal_bf16((size_t) HV * N, 0.02f), wb = normal_bf16((size_t) HV * N, 0.02f);
        auto dt = normal(HV, 0.5f), sa = normal(HV, 0.5f, -1.0f), gamma = normal(S, 0.2f, 1.0f);
        float *d_state = dev(state), *d_hist = dev(hist), *d_cw = dev(cw), *d_dt = dev(dt), *d_sa = dev(sa), *d_gamma = dev(gamma);
        uint16_t *d_wa = dev(wa), *d_wb = dev(wb);
        int cases = 0, gbad = 0;
        const int reps = std::getenv("NODE_FUSE_PARITY_REPS") ? std::atoi(std::getenv("NODE_FUSE_PARITY_REPS")) : 1;
        for (int rep = 0; rep < reps; ++rep)
        for (int q8 = 0; q8 < 2; ++q8)
            for (int T = 1; T <= kVerifyNarrowT; ++T) {
                auto qkv = normal((size_t) T * C, 1.0f), x = normal((size_t) T * N, 1.0f), z = normal((size_t) T * HV * S, 1.5f);
                float *d_qkv = dev(qkv), *d_x = dev(x), *d_z = dev(z);
                const size_t hb = (size_t) T * C * 4, gb = (size_t) T * HV * 4, yb = (size_t) T * HV * S * 4,
                             qb = (size_t) T * HV * S / 32 * 36;
                float *h[2], *g[2], *b[2], *y[2];
                void* xq[2];
                for (int k = 0; k < 2; ++k) {
                    h[k] = (float*) dev_fill(hb, 0x7e); g[k] = (float*) dev_fill(gb, 0x7e); b[k] = (float*) dev_fill(gb, 0x7e);
                    y[k] = (float*) dev_fill(yb, 0x7e); xq[k] = dev_fill(qb, 0x7e);
                }
                gdn_conv_l2_multi(d_hist, d_qkv, d_cw, h[0], C, 2 * HK, 1e-6f, T, st, 0, false);
                gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_sa, g[0], b[0], N, HV, T, st);
                gdn_step_norm_multi(d_state, h[0], C, g[0], b[0], d_z, d_gamma, 1e-6f, y[0], HK, HV, T, nullptr, st, 0,
                                    q8 ? xq[0] : nullptr);
                const bool took = gdn_fused_step(d_state, d_hist, d_qkv, d_cw, h[1], C, 1e-6f, d_x, d_wa, d_wb, d_dt, d_sa, g[1],
                                                 b[1], N, d_z, d_gamma, 1e-6f, y[1], HK, HV, T, st, q8 ? xq[1] : nullptr);
                CHECK(hipStreamSynchronize(st));
                ++cases;
                const char* what = !took ? "refused"
                                 : back(h[0], hb) != back(h[1], hb) ? "h"
                                 : back(g[0], gb) != back(g[1], gb) ? "gate"
                                 : back(b[0], gb) != back(b[1], gb) ? "beta"
                                 : back(y[0], yb) != back(y[1], yb) ? "y"
                                 : back(xq[0], qb) != back(xq[1], qb) ? "q8_1"
                                 : back(d_state, state.size() * 4) != std::vector<uint8_t>((const uint8_t*) state.data(),
                                                                                           (const uint8_t*) state.data() + state.size() * 4) ? "state"
                                 : back(d_hist, hist.size() * 4) != std::vector<uint8_t>((const uint8_t*) hist.data(),
                                                                                         (const uint8_t*) hist.data() + hist.size() * 4) ? "history" : nullptr;
                if (what && gbad++ < 8) {
                    std::printf("  GDN conv + gates + recurrence, %d tokens%s: %s\n", T, q8 ? " (q8_1)" : "", what);
                    if (what[0] == 'h' && what[1] == '\0') {   // where: the first few differing (token, channel)
                        const auto a = back(h[0], hb), b2 = back(h[1], hb);
                        const float* fa = (const float*) a.data();
                        const float* fb = (const float*) b2.data();
                        for (size_t i = 0, shown = 0; i < (size_t) T * C && shown < 6; ++i)
                            if (std::memcmp(fa + i, fb + i, 4) != 0) {
                                std::printf("    token %zu channel %zu: %.9g vs %.9g\n", i / C, i % C, fa[i], fb[i]);
                                ++shown;
                            }
                    }
                }
                for (int k = 0; k < 2; ++k) { hipFree(h[k]); hipFree(g[k]); hipFree(b[k]); hipFree(y[k]); hipFree(xq[k]); }
                hipFree(d_qkv); hipFree(d_x); hipFree(d_z);
            }
        for (void* p : {(void*) d_state, (void*) d_hist, (void*) d_cw, (void*) d_dt, (void*) d_sa, (void*) d_gamma, (void*) d_wa, (void*) d_wb})
            hipFree(p);
        std::printf("GDN conv + gates + recurrence in one launch: %d cases, %s\n", cases, gbad ? "DIFFERENT" : "bitwise equal");
        bad += gbad;
    }
    // ---- 4. the indexer tail's snapshot made by the append
    {
        const QsaShapes sh = qsa_real_shapes();
        const int D = (int) sh.idx_dim, R = (int) sh.idx_block, TS = (R - 1) * D;
        const int64_t max_cells = 4096;
        RopeScaling yarn2;
        yarn2.type = RopeScalingType::YaRN; yarn2.factor = 2.0; yarn2.ext_factor = 1.0;
        auto gamma = normal(D, 0.2f, 1.0f);
        float* d_gamma = dev(gamma);
        int cases = 0, sbad = 0;
        for (int p0 : {0, 1, 2, 3, 5, 102, 1023})
            for (int n = 1; n <= 8; ++n) {
                auto raw = normal((size_t) n * D, 1.0f), tail = normal((size_t) TS, 1.0f), dead = normal(D, 1.0f),
                     pooled = normal((size_t) (max_cells / R + 1) * D, 1.0f);
                const int stride = 3;   // the window's step rows: position at [t * stride]
                std::vector<int32_t> pos((size_t) n * stride, -7);
                for (int t = 0; t < n; ++t) pos[(size_t) t * stride] = p0 + t;
                float* d_raw = dev(raw);
                int32_t* d_pos = dev(pos);
                float *tl[2], *dd[2], *pl[2], *sn[2];
                int32_t* bp[2];
                for (int k = 0; k < 2; ++k) {
                    tl[k] = dev(tail); dd[k] = dev(dead); pl[k] = dev(pooled);
                    sn[k] = (float*) dev_fill((size_t) TS * 4, 0x44);
                    bp[k] = (int32_t*) dev_fill(4, 0x44);
                }
                QsaIndexerBuffers b0{tl[0], dd[0], pl[0], bp[0]}, b1{tl[1], dd[1], pl[1], bp[1]};
                copy_from_mapped(sn[0], tl[0], TS, st);
                native_qsa_indexer_append_steps(d_raw, d_pos, stride, n, 0, d_gamma, 1e-6f, b0, sh, max_cells, yarn2, st);
                native_qsa_indexer_append_steps_snap(d_raw, d_pos, stride, n, 0, d_gamma, 1e-6f, b1, sh, max_cells, yarn2, st, sn[1]);
                CHECK(hipStreamSynchronize(st));
                ++cases;
                const size_t pb = pooled.size() * 4;
                const char* what = back(sn[0], TS * 4) != back(sn[1], TS * 4) ? "snapshot"
                                 : back(tl[0], TS * 4) != back(tl[1], TS * 4) ? "tail"
                                 : back(dd[0], D * 4) != back(dd[1], D * 4) ? "dead"
                                 : back(pl[0], pb) != back(pl[1], pb) ? "pooled"
                                 : back(bp[0], 4) != back(bp[1], 4) ? "block position" : nullptr;
                if (what && sbad++ < 8) std::printf("  indexer append + snapshot, from %d, %d steps: %s\n", p0, n, what);
                for (int k = 0; k < 2; ++k) { hipFree(tl[k]); hipFree(dd[k]); hipFree(pl[k]); hipFree(sn[k]); hipFree(bp[k]); }
                hipFree(d_raw); hipFree(d_pos);
            }
        hipFree(d_gamma);
        std::printf("indexer tail snapshot inside the append: %d cases, %s\n", cases, sbad ? "DIFFERENT" : "bitwise equal");
        bad += sbad;
    }
    // ---- 5. STRATA_HC_WIDE8: 7-8 token reads with the row kernel's down in one launch, against the default read
    // (STRATA_HC_WIDE off: the staged down in chunks)
    {
        const int N = 2560, HC = 4, LR = 320, D = N * HC;
        auto w_norm = normal(D, 0.3f, 1.0f);
        auto w_down = normal_bf16((size_t) LR * D, 0.02f), w_up = normal_bf16((size_t) D * LR, 0.05f),
             w_inj = normal_bf16((size_t) HC * D, 0.02f);
        float* d_norm = dev(w_norm);
        uint16_t *d_down = dev(w_down), *d_up = dev(w_up), *d_inj = dev(w_inj);
        for (int cmp = 0; cmp < 3; ++cmp) {   // 0: STRATA_HC_WIDE8 against the default read; 1: STRATA_HC_UP_TSUM, 2: STRATA_HC_DOWN2 against the wide read
        int cases = 0, wbad = 0;
        for (int rep = 0; rep < 3; ++rep)
            for (int T = 1; T <= kFusedGrMaxT; ++T)
                for (int apply = 0; apply < 2; ++apply) {
                    auto R = normal((size_t) T * D, 1.0f), bo = normal((size_t) T * N, 0.5f), inj = normal((size_t) T * HC, 1.0f);
                    float* d_bo = dev(bo);
                    float* d_inj_prev = dev(inj);
                    struct Arm { float *R, *lo, *rs, *io, *mixed, *xn; uint8_t* q8; unsigned* cnt; } arm[2];
                    for (auto& a : arm) {
                        a.R = dev(R);
                        a.lo = (float*) dev_fill((size_t) T * LR * 4, 0);
                        a.rs = (float*) dev_fill((size_t) T * HC * 4, 0);
                        a.io = (float*) dev_fill((size_t) T * HC * 4, 0);
                        a.mixed = (float*) dev_fill((size_t) T * N * 4, 0);
                        a.xn = (float*) dev_fill((size_t) T * D * 4, 0);
                        a.q8 = (uint8_t*) dev_fill((size_t) T * (N / 32) * 36, 0x33);
                        a.cnt = (unsigned*) dev_fill((size_t) (N / 32) * 4, 0);
                    }
                    for (int k = 0; k < 2; ++k) {
                        Arm& a = arm[k];
                        FusedGrArgs fa[kFusedGrMaxT];
                        for (int t = 0; t < T; ++t) {
                            FusedGrArgs& x = fa[t];
                            x.R = a.R + (size_t) t * D; x.R_out = a.R + (size_t) t * D; x.apply = apply != 0;
                            x.bo_prev = d_bo + (size_t) t * N; x.inj_prev = d_inj_prev + t * HC;
                            x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up; x.w_inject = d_inj;
                            x.eps = 1e-6f; x.lo = a.lo + t * LR; x.rs = a.rs + t * HC; x.inject_out = a.io + t * HC;
                            x.mixed = a.mixed + (size_t) t * N;
                            x.q8_cnt = a.cnt; x.q8_mixed = a.q8 + (size_t) t * (N / 32) * 36;
                        }
                        if (cmp == 0) { fused_gr_set_wide(k == 0 ? 0 : 3); fused_gr_set_wide8(k); fused_gr_set_up_tsum(0); fused_gr_set_down2(0); }
                        else if (cmp == 1) { fused_gr_set_wide(3); fused_gr_set_wide8(1); fused_gr_set_up_tsum(k); fused_gr_set_down2(0); }
                        else { fused_gr_set_wide(3); fused_gr_set_wide8(1); fused_gr_set_up_tsum(0); fused_gr_set_down2(k); }
                        fused_gr_read_multi(fa, T, a.xn, st);
                        CHECK(hipStreamSynchronize(st));
                    }
                    ++cases;
                    const char* what = back(arm[0].lo, (size_t) T * LR * 4) != back(arm[1].lo, (size_t) T * LR * 4) ? "lo"
                                     : back(arm[0].io, (size_t) T * HC * 4) != back(arm[1].io, (size_t) T * HC * 4) ? "inject"
                                     : back(arm[0].R, (size_t) T * D * 4) != back(arm[1].R, (size_t) T * D * 4) ? "R"
                                     : back(arm[0].rs, (size_t) T * HC * 4) != back(arm[1].rs, (size_t) T * HC * 4) ? "rs"
                                     : back(arm[0].mixed, (size_t) T * N * 4) != back(arm[1].mixed, (size_t) T * N * 4) ? "mixed"
                                     : back(arm[0].q8, (size_t) T * (N / 32) * 36) != back(arm[1].q8, (size_t) T * (N / 32) * 36) ? "q8_1" : nullptr;
                    if (what && wbad++ < 8) std::printf("  wide read, %d tokens%s: %s DIFFERENT\n", T, apply ? " (pending write)" : "", what);
                    for (auto& a : arm) { hipFree(a.R); hipFree(a.lo); hipFree(a.rs); hipFree(a.io); hipFree(a.mixed); hipFree(a.xn); hipFree(a.q8); hipFree(a.cnt); }
                    hipFree(d_bo); hipFree(d_inj_prev);
                }
        std::printf("hyper-connection read, %s: %d cases, %s\n",
                    cmp == 0 ? "STRATA_HC_WIDE=3 + STRATA_HC_WIDE8 against the default read"
                    : cmp == 1 ? "STRATA_HC_UP_TSUM against STRATA_HC_WIDE=3 + STRATA_HC_WIDE8"
                               : "STRATA_HC_DOWN2 against STRATA_HC_WIDE=3 + STRATA_HC_WIDE8",
                    cases, wbad ? "DIFFERENT" : "bitwise equal");
        bad += wbad;
        }
        fused_gr_set_wide(-1);
        fused_gr_set_wide8(-1);
        fused_gr_set_up_tsum(-1);
        fused_gr_set_down2(-1);
        hipFree(d_norm); hipFree(d_down); hipFree(d_up); hipFree(d_inj);
    }
    std::printf(bad ? "FAILED\n" : "all bitwise equal\n");
    return bad ? 1 : 0;
}
