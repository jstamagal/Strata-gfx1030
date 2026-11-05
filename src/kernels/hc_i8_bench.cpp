// src/kernels/hc_i8_bench.cpp - STRATA_HC_REQ8's int8 hyper-connection read (int8 codes + one fp32 scale per 32 values,
// hc_req8_quantize: the engine's codes) in the wide kernels (STRATA_HC_WIDE=3 + WIDE8 + UP_TSUM, DOWN2 off and on), against
// the int8 multi kernels it replaces and against the bf16 read, through fused_gr_read_multi on the current card:
//
//   PARITY  int8 wide vs int8 multi: lo, rs, inject, mixed, the in-place R and the q8_1 image of mixed, bit for bit, 1-8
//           tokens, with and without the pending write, the inject rows and the q8_1 image, DOWN2 off and on.  Not bitwise:
//           the max |difference| / rms of the output is printed and the run fails.
//   ERROR   int8 wide vs bf16 wide (the same tokens): relative RMS, mean and max relative error per output.  Relative error
//           of an element = |int8 - bf16| / max(|bf16|, 1% of the output's RMS) (the floor keeps near-zero values honest).
//   TIME    cold weights (the read rotates over `sets` copies, as the engine reads another layer every call: 32 x 7.4 MB
//           int8 / 13.2 MB bf16 is past the 128 MB Infinity Cache): the read (events) and the down / up stages (median
//           GPU stamps, each with one ~1-2 us stamp kernel), us and GB/s of the stage's weight bytes, per arm and T.
//
//     build-hip/hc_i8_bench [iters 300] [sets 32] [T list 1,2,3,8] [--pack DIR --layer L]
// With --pack the weights are layer L's hc_attn projections and norm from a Strata pack (its raw bf16 rows), else
// N(0, s) bf16 (s = 0.02 down / inject, 0.05 up).  Exit 1 when a parity check differs.
#include "strata/core/hc_req8.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, TM = 8;
constexpr size_t Q8ROW = (size_t) (N / 32) * 36;   // a token's q8_1 image of `mixed`

#define CK(call) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "hc_i8_bench: %s: %s\n", #call, cudaGetErrorString(e_)); std::exit(2); } } while (0)

uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
template <class T> T* dev_copy(const std::vector<T>& v) {
    T* p = nullptr;
    CK(cudaMalloc((void**) &p, v.size() * sizeof(T)));
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

struct Host {   // one layer's weights on the host
    std::vector<uint16_t> wd, wu, wi;   // bf16 [LR][D], [D][LR], [HC][D]
    std::vector<float> wn;              // f32 [D]
    std::vector<int8_t> qd, qu;         // int8 codes, as the engine makes them
    std::vector<float> sd, su;          // their scales
};
struct Dev {
    uint16_t *wd, *wu, *wi;
    float* wn;
    int8_t *qd, *qu;
    float *sd, *su;
};
Dev upload(const Host& h) {
    return {dev_copy(h.wd), dev_copy(h.wu), dev_copy(h.wi), dev_copy(h.wn), dev_copy(h.qd), dev_copy(h.qu), dev_copy(h.sd), dev_copy(h.su)};
}
void quantize(Host& h) {
    auto one = [](const std::vector<uint16_t>& w, std::vector<int8_t>& q, std::vector<float>& s) {
        std::vector<float> f(w.size());
        for (size_t i = 0; i < w.size(); ++i) f[i] = f32(w[i]);
        q.resize(w.size());
        s.resize(w.size() / 32);
        strata::core::hc_req8_quantize(f.data(), f.size(), q.data(), s.data());
    };
    one(h.wd, h.qd, h.sd);
    one(h.wu, h.qu, h.su);
}
// layer L's blk.L.hc_attn_{down,up,inject,norm}.weight from <pack>/index.txt + dense.bin (raw 16-bit bf16 rows, f32 norm)
bool load_pack(const std::string& pack, int layer, Host& h) {
    std::FILE* ix = std::fopen((pack + "/index.txt").c_str(), "rb");
    if (!ix) { std::fprintf(stderr, "hc_i8_bench: cannot open %s/index.txt\n", pack.c_str()); return false; }
    std::FILE* bin = std::fopen((pack + "/dense.bin").c_str(), "rb");
    if (!bin) { std::fprintf(stderr, "hc_i8_bench: cannot open %s/dense.bin\n", pack.c_str()); std::fclose(ix); return false; }
    const std::string pre = "blk." + std::to_string(layer) + ".hc_attn_";
    char line[1024], name[256];
    int found = 0;
    while (std::fgets(line, sizeof line, ix)) {
        int file = 0, kind = 0;
        unsigned long long off = 0, bytes = 0;
        if (line[0] == '#' || std::sscanf(line, "%255s %d %d %llu %llu", name, &file, &kind, &off, &bytes) != 5) continue;
        const std::string n = name;
        if (n.rfind(pre, 0) != 0 || file != 0) continue;
        const std::string m = n.substr(pre.size());
        void* dst = nullptr;
        size_t want = 0;
        if (m == "down.weight") { h.wd.resize((size_t) LR * D); dst = h.wd.data(); want = h.wd.size() * 2; }
        else if (m == "up.weight") { h.wu.resize((size_t) D * LR); dst = h.wu.data(); want = h.wu.size() * 2; }
        else if (m == "inject.weight") { h.wi.resize((size_t) HC * D); dst = h.wi.data(); want = h.wi.size() * 2; }
        else if (m == "norm.weight") { h.wn.resize(D); dst = h.wn.data(); want = h.wn.size() * 4; }
        else continue;
        if (bytes != want) {
            std::fprintf(stderr, "hc_i8_bench: %s holds %llu bytes, not the %zu of raw bf16 / f32 rows\n", name, bytes, want);
            std::fclose(ix); std::fclose(bin); return false;
        }
#if defined(_WIN32)
        const bool sought = _fseeki64(bin, (long long) off, SEEK_SET) == 0;
#else
        const bool sought = fseeko(bin, (off_t) off, SEEK_SET) == 0;   // 64-bit: dense.bin passes 2 GiB
#endif
        if (!sought || std::fread(dst, 1, want, bin) != want) {
            std::fprintf(stderr, "hc_i8_bench: reading %s failed\n", name);
            std::fclose(ix); std::fclose(bin); return false;
        }
        ++found;
    }
    std::fclose(ix);
    std::fclose(bin);
    if (found != 4) { std::fprintf(stderr, "hc_i8_bench: %d of the 4 tensors of layer %d found\n", found, layer); return false; }
    return true;
}

struct Bufs {
    float *R, *bo, *inj, *xn, *lo, *rs, *io, *mix;
    uint8_t* q8;
    unsigned* cnt;
};
void make_args(K::FusedGrArgs* a, int T, const Bufs& b, const Dev& w, bool i8, bool apply, bool inject, bool qfuse) {
    for (int t = 0; t < T; ++t) {
        K::FusedGrArgs& x = a[t];
        x = K::FusedGrArgs();
        x.R = b.R + (size_t) t * D; x.R_out = b.R + (size_t) t * D; x.apply = apply;
        x.bo_prev = b.bo + (size_t) t * N; x.inj_prev = b.inj + (size_t) t * HC;
        x.w_norm = w.wn;
        x.w_down = i8 ? (const uint16_t*) w.qd : w.wd;
        x.w_up = i8 ? (const uint16_t*) w.qu : w.wu;
        x.w_inject = inject ? w.wi : nullptr;
        if (i8) { x.s_down = w.sd; x.s_up = w.su; }
        x.lo = b.lo + (size_t) t * LR; x.rs = b.rs + (size_t) t * HC;
        x.inject_out = b.io + (size_t) t * HC; x.mixed = b.mix + (size_t) t * N;
        if (qfuse) { x.q8_mixed = b.q8 + (size_t) t * Q8ROW; x.q8_cnt = b.cnt; }
    }
}
// the kernels' configuration: 0 = the multi kernels (STRATA_HC_WIDE=0), 1 = the live wide set (WIDE=3, WIDE8, UP_TSUM)
void configure(int wide, int down2) {
    K::fused_gr_set_wide(wide ? 3 : 0);
    K::fused_gr_set_wide8(wide ? 1 : 0);
    K::fused_gr_set_up_tsum(wide ? 1 : 0);
    K::fused_gr_set_down2(down2);
}
struct Out {   // one read's outputs: R, lo, rs, inject, mixed (floats), then the q8_1 image (bytes)
    std::vector<float> f;
    std::vector<uint8_t> q;
    size_t o_lo, o_rs, o_io, o_mix;
};
Out run_once(const K::FusedGrArgs* a, int T, const Bufs& b, const std::vector<float>& R0, cudaStream_t s) {
    CK(cudaMemcpy(b.R, R0.data(), R0.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(b.lo, 0xff, (size_t) TM * LR * 4));
    CK(cudaMemset(b.rs, 0xff, (size_t) TM * HC * 4));
    CK(cudaMemset(b.io, 0xff, (size_t) TM * HC * 4));
    CK(cudaMemset(b.mix, 0xff, (size_t) TM * N * 4));
    CK(cudaMemset(b.q8, 0xff, (size_t) TM * Q8ROW));
    K::fused_gr_read_multi(a, T, b.xn, s);
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());
    Out o;
    o.o_lo = (size_t) T * D; o.o_rs = o.o_lo + (size_t) T * LR; o.o_io = o.o_rs + (size_t) T * HC; o.o_mix = o.o_io + (size_t) T * HC;
    o.f.resize(o.o_mix + (size_t) T * N);
    CK(cudaMemcpy(o.f.data(), b.R, (size_t) T * D * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.f.data() + o.o_lo, b.lo, (size_t) T * LR * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.f.data() + o.o_rs, b.rs, (size_t) T * HC * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.f.data() + o.o_io, b.io, (size_t) T * HC * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o.f.data() + o.o_mix, b.mix, (size_t) T * N * 4, cudaMemcpyDeviceToHost));
    o.q.resize((size_t) T * Q8ROW);
    CK(cudaMemcpy(o.q.data(), b.q8, o.q.size(), cudaMemcpyDeviceToHost));
    return o;
}
struct Err { double rel_rms, mean_rel, max_rel, max_abs; };
Err err_of(const float* x, const float* r, size_t n) {
    double ss = 0, sd = 0;
    for (size_t i = 0; i < n; ++i) { ss += (double) r[i] * r[i]; sd += std::pow((double) x[i] - r[i], 2); }
    const double rms = std::sqrt(ss / (double) std::max<size_t>(n, 1)), fl = 0.01 * rms;
    Err e{ss > 0 ? std::sqrt(sd / ss) : 0.0, 0, 0, 0};
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) x[i] - r[i]), rel = d / std::max(std::fabs((double) r[i]), fl > 0 ? fl : 1e-30);
        e.mean_rel += rel; e.max_rel = std::max(e.max_rel, rel); e.max_abs = std::max(e.max_abs, d);
    }
    e.mean_rel /= (double) std::max<size_t>(n, 1);
    return e;
}
double med(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
    if (std::getenv("STRATA_GR_V3") == nullptr) _putenv_s("STRATA_GR_V3", "0");
#else
    if (std::getenv("STRATA_GR_V3") == nullptr) setenv("STRATA_GR_V3", "0", 1);   // the engine's default read on gfx1030
#endif
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> pos;
    std::string pack;
    int layer = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pack" && i + 1 < argc) pack = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoi(argv[++i]);
        else pos.push_back(a);
    }
    const int iters = pos.size() > 0 ? std::max(10, std::atoi(pos[0].c_str())) : 300;
    const int sets = pos.size() > 1 ? std::max(1, std::atoi(pos[1].c_str())) : 32;
    std::vector<int> Ts;
    {
        const std::string l = pos.size() > 2 ? pos[2] : "1,2,3,8";
        for (size_t p = 0; p < l.size();) {
            const size_t q = l.find(',', p);
            const int t = std::atoi(l.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str());
            if (t >= 1 && t <= TM) Ts.push_back(t);
            if (q == std::string::npos) break;
            p = q + 1;
        }
    }
    int dev = 0;
    cudaDeviceProp prop{};
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("hc_i8_bench: device %d %s, iters %d, %d cold sets, weights %s\n", dev, prop.name, iters, sets,
                pack.empty() ? "random N(0, s) bf16" : (pack + " layer " + std::to_string(layer) + " hc_attn").c_str());
    K::fused_gr_check();   // the engine's per-card variant choice (Verifier::init runs it)
    std::printf("hc_i8_bench: fused_gr_variant %d\n", K::fused_gr_variant());

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    Host h;
    if (!pack.empty()) {
        if (!load_pack(pack, layer, h)) return 2;
    } else {
        h.wd.resize((size_t) LR * D); h.wu.resize((size_t) D * LR); h.wi.resize((size_t) HC * D); h.wn.resize(D);
        for (auto& x : h.wd) x = bf16(0.02f * nd(rng));
        for (auto& x : h.wu) x = bf16(0.05f * nd(rng));
        for (auto& x : h.wi) x = bf16(0.02f * nd(rng));
        for (auto& x : h.wn) x = 1.0f + 0.3f * nd(rng);
    }
    quantize(h);
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC);
    for (auto& x : R) x = nd(rng);
    for (auto& x : bo) x = 0.5f * nd(rng);
    for (auto& x : inj) x = nd(rng);
    Bufs b{};
    CK(cudaMalloc((void**) &b.R, R.size() * 4));
    b.bo = dev_copy(bo);
    b.inj = dev_copy(inj);
    CK(cudaMalloc((void**) &b.xn, (size_t) TM * D * 4));
    CK(cudaMalloc((void**) &b.lo, (size_t) TM * LR * 4));
    CK(cudaMalloc((void**) &b.rs, (size_t) TM * HC * 4));
    CK(cudaMalloc((void**) &b.io, (size_t) TM * HC * 4));
    CK(cudaMalloc((void**) &b.mix, (size_t) TM * N * 4));
    CK(cudaMalloc((void**) &b.q8, (size_t) TM * Q8ROW));
    CK(cudaMalloc((void**) &b.cnt, (size_t) (N / 32) * sizeof(unsigned)));
    CK(cudaMemset(b.cnt, 0, (size_t) (N / 32) * sizeof(unsigned)));
    const Dev w0 = upload(h);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));

    // ---- PARITY: int8 wide vs int8 multi; ERROR: int8 wide vs bf16 wide
    int failures = 0, cases = 0;
    double worst_rel_rms[4] = {0, 0, 0, 0}, worst_max_rel[4] = {0, 0, 0, 0}, sum_mean_rel[4] = {0, 0, 0, 0};
    int n_err = 0;
    for (int T = 1; T <= TM; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject)
                for (int qfuse = 0; qfuse < 2; ++qfuse)
                    for (int down2 = 0; down2 < 2; ++down2) {
                        if (down2 && T < 5) continue;   // DOWN2 runs from 5 tokens
                        K::FusedGrArgs a[TM];
                        configure(0, 0);
                        make_args(a, T, b, w0, true, apply, inject, qfuse);
                        const Out multi = run_once(a, T, b, R, s);
                        configure(1, down2);
                        const Out wide = run_once(a, T, b, R, s);
                        ++cases;
                        const bool fsame = std::memcmp(multi.f.data(), wide.f.data(), multi.f.size() * 4) == 0;
                        const bool qsame = !qfuse || multi.q == wide.q;
                        if (!fsame || !qsame) {
                            ++failures;
                            const Err e = err_of(wide.f.data(), multi.f.data(), multi.f.size());
                            size_t qd = 0;
                            for (size_t i = 0; qfuse && i < multi.q.size(); ++i) qd += multi.q[i] != wide.q[i];
                            std::printf("PARITY DIFFERS int8 wide vs int8 multi: T %d apply %d inject %d qfuse %d down2 %d | floats %s "
                                        "(max |d| %.3g, rel rms %.3g) | q8_1 image %zu bytes differ\n", T, apply, inject, qfuse, down2,
                                        fsame ? "same" : "differ", e.max_abs, e.rel_rms, qd);
                        }
                        if (qfuse || down2) continue;   // the error table: one row per (T, apply, inject)
                        make_args(a, T, b, w0, false, apply, inject, false);
                        const Out ref = run_once(a, T, b, R, s);   // bf16, the same wide kernels
                        const Err el = err_of(wide.f.data() + wide.o_lo, ref.f.data() + ref.o_lo, (size_t) T * LR);
                        const Err em = err_of(wide.f.data() + wide.o_mix, ref.f.data() + ref.o_mix, (size_t) T * N);
                        const Err er = err_of(wide.f.data(), ref.f.data(), (size_t) T * D);
                        const Err ei = err_of(wide.f.data() + wide.o_io, ref.f.data() + ref.o_io, inject ? (size_t) T * HC : 0);
                        const Err all[4] = {el, em, er, ei};
                        for (int k = 0; k < 4; ++k) {
                            worst_rel_rms[k] = std::max(worst_rel_rms[k], all[k].rel_rms);
                            worst_max_rel[k] = std::max(worst_max_rel[k], all[k].max_rel);
                            sum_mean_rel[k] += all[k].mean_rel;
                        }
                        ++n_err;
                        std::printf("ERROR int8 vs bf16 T %d apply %d inject %d | lo rel rms %.4f mean %.4f max %.3f | mixed rel rms %.4f "
                                    "mean %.4f max %.3f | R max |d| %.3g | inject max |d| %.3g\n", T, apply, inject, el.rel_rms,
                                    el.mean_rel, el.max_rel, em.rel_rms, em.mean_rel, em.max_rel, er.max_abs, ei.max_abs);
                    }
    std::printf("PARITY int8 wide vs int8 multi: %d of %d cases bitwise equal (lo, rs, inject, mixed, R, q8_1 image)\n",
                cases - failures, cases);
    if (n_err)
        std::printf("ERROR SUMMARY int8 vs bf16 over %d cases: lo rel rms <= %.4f, mean rel %.4f, max rel %.3f | mixed rel rms <= %.4f, "
                    "mean rel %.4f, max rel %.3f | R (the pending write, weight-free) max rel %.3g | inject (bf16 rows) max rel %.3g\n",
                    n_err, worst_rel_rms[0], sum_mean_rel[0] / n_err, worst_max_rel[0], worst_rel_rms[1], sum_mean_rel[1] / n_err,
                    worst_max_rel[1], worst_max_rel[2], worst_max_rel[3]);

    // ---- TIME: cold weights, per arm and T
    std::vector<Dev> cold((size_t) sets);
    cold[0] = w0;
    for (int c = 1; c < sets; ++c) cold[(size_t) c] = upload(h);
    unsigned long long* dst = nullptr;
    CK(cudaMalloc((void**) &dst, (size_t) iters * 4 * 8));
    std::vector<unsigned long long> hst((size_t) iters * 4);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const double inj_b = (double) HC * D * 2;
    const double down_b[2] = {(double) LR * D * 2 + inj_b, (double) LR * D + (double) LR * D / 32 * 4 + inj_b};
    const double up_b[2] = {(double) D * LR * 2, (double) D * LR + (double) D * LR / 32 * 4};
    struct ArmDef { const char* name; bool i8; int wide; int down2; };
    const ArmDef arms[] = {{"bf16-wide", false, 1, 0}, {"bf16-wide+down2", false, 1, 1}, {"int8-wide", true, 1, 0},
                           {"int8-wide+down2", true, 1, 1}, {"int8-multi", true, 0, 0}};
    std::vector<K::FusedGrArgs> as((size_t) sets * TM);
    for (const int T : Ts)
        for (const ArmDef& arm : arms) {
            if (arm.down2 && T < 5) continue;
            configure(arm.wide, arm.down2);
            for (int c = 0; c < sets; ++c) make_args(&as[(size_t) c * TM], T, b, cold[(size_t) c], arm.i8, true, true, true);
            CK(cudaMemcpy(b.R, R.data(), R.size() * 4, cudaMemcpyHostToDevice));
            for (int rep = 0; rep < 2; ++rep) {
                for (int i = 0; i < iters; ++i) {
                    K::gpu_stamp(dst, i * 4, s);
                    K::fused_gr_read_multi(&as[(size_t) (i % sets) * TM], T, b.xn, s, dst, i * 4 + 1);
                    K::gpu_stamp(dst, i * 4 + 3, s);
                }
                CK(cudaStreamSynchronize(s));
            }
            CK(cudaMemcpy(hst.data(), dst, hst.size() * 8, cudaMemcpyDeviceToHost));
            CK(cudaEventRecord(e0, s));
            for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(&as[(size_t) (i % sets) * TM], T, b.xn, s);
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            CK(cudaGetLastError());
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            std::vector<double> nv, dv, uv;
            for (int i = 1; i < iters; ++i) {
                const unsigned long long* p = &hst[(size_t) i * 4];
                nv.push_back((double) (p[1] - p[0]) / 1e3); dv.push_back((double) (p[2] - p[1]) / 1e3); uv.push_back((double) (p[3] - p[2]) / 1e3);
            }
            const double dn = med(dv), up = med(uv);
            std::printf("TIME T %d %-16s | read %6.1f us | norm %5.1f  down %5.1f us %5.0f GB/s  up %5.1f us %5.0f GB/s (stamps, median)\n",
                        T, arm.name, 1e3 * ms / iters, med(nv), dn, down_b[arm.i8] / (dn * 1e3), up, up_b[arm.i8] / (up * 1e3));
        }
    std::printf("hc_i8_bench: weight bytes a layer half, down + inject / up: bf16 %.2f / %.2f MB, int8 %.2f / %.2f MB\n",
                down_b[0] / 1e6, up_b[0] / 1e6, down_b[1] / 1e6, up_b[1] / 1e6);
    std::printf("hc_i8_bench: %s\n", failures ? "PARITY FAILED" : "PARITY OK");
    return failures ? 1 : 0;
}
