// Bitwise router parity and graph microbench, synthetic Ornith geometry; never loads a model.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>
#include "strata/hip_compat/cuda_runtime.h"
#include "strata/kernels/verify_kernels.hpp"
#include "q36_kernels.cuh"
#include "q36_tp_collective.cuh"
#include "q36_tp_fuse2.cuh"
#include "q36_tp_router_pair.cuh"
#include "router_reference.cuh"

#define CK(x) do { auto e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); std::exit(2); } } while (0)
constexpr int N = 3072, NE = 256, MT = 16, KK = 8;
static_assert(MT >= strata::kernels::kVerifyMaxT);
struct Output {
    float logits[MT * (NE + 1)];
    int32_t ids[MT * KK];
    float weights[MT * KK];
    q36k::QB q[MT * (N / 32)];
    unsigned long long ptr[MT * KK];
    int32_t start[MT * KK + 1], ng, dst[MT * KK], tok[MT * KK];
};
#define BLOBS reinterpret_cast<const uint8_t*>(0x100000000ull)
constexpr unsigned long long blob = 65536;
static hipStream_t stream;
#define ARGS(o, T, ld, map) (o)->logits, (o)->ids, (o)->weights, T, ld, map, BLOBS, blob, (o)->ptr, (o)->start, &(o)->ng, (o)->dst, (o)->tok
#define PROJ_ARGS(o) W, x, (o)->logits, N, rows, T, rows, (o)->ids, (o)->weights, BLOBS, blob, (o)->ptr, (o)->start, &(o)->ng, (o)->dst, (o)->tok, done, (o)->q

template <bool HY, bool Old>
__global__ void grouping(Output* o, int T, int ld, const int32_t* map) {
    if constexpr (Old) router_reference::route_group_body<HY>(ARGS(o, T, ld, map));
    else q36tp_fuse2::route_group_body<HY>(ARGS(o, T, ld, map));
}

enum Mode { Old, Default, MaskOnly, UnrollOnly, Split, Projection, Fallback, Pair };
template <bool LF, int Warps, bool Q>
static void project(Mode mode, const float* W, const float* x, Output* o, int rows, int T, unsigned* done) {
    const int grid = ((rows + Warps - 1) / Warps) * T;
    auto tuned = [&](auto unroll) {
        constexpr int U = decltype(unroll)::value;
        if (mode == UnrollOnly)
            q36tp_fuse2::router_group<LF, Warps, Q, false, U><<<grid, Warps * 32, 0, stream>>>(PROJ_ARGS(o));
        else
            q36tp_fuse2::router_group<LF, Warps, Q, true, U, false><<<grid, Warps * 32, 0, stream>>>(PROJ_ARGS(o));
    };
    if (mode == Old)
        router_reference::router_group<LF, Warps, Q><<<grid, Warps * 32, 0, stream>>>(PROJ_ARGS(o));
    else if (mode == MaskOnly)
        q36tp_fuse2::router_group<LF, Warps, Q, true, 1><<<grid, Warps * 32, 0, stream>>>(PROJ_ARGS(o));
    else if (mode == Default || mode == Fallback)
        q36tp_fuse2::launch_router_group<LF, Warps, Q>(PROJ_ARGS(o), stream, mode != Fallback);
    else if (mode == Pair)
        q36tp_router_pair::launch_router_group(W, x, o->logits, N, rows, T, rows, o->ids, o->weights,
            BLOBS, blob, o->ptr, o->start, &o->ng, o->dst, o->tok, done, stream);
    else {
        if (T == 1) tuned(std::integral_constant<int, 4>{});
        else tuned(std::integral_constant<int, 8>{});
        if (mode == Split) grouping<false, false><<<1, Warps * 32, 0, stream>>>(o, T, rows, nullptr);
    }
    CK(hipGetLastError());
}
static void project(int variant, Mode mode, const float* W, const float* x, Output* o, int rows, int T, unsigned* done) {
    switch (variant) {
#define CASE(i, lf, w, q) case i: project<lf, w, q>(mode, W, x, o, rows, T, done); break
        CASE(0, false, 8, true); CASE(1, true, 8, true);
        CASE(2, false, 8, false); CASE(3, true, 8, false);
        CASE(4, false, 4, true); CASE(5, true, 4, true);
        CASE(6, false, 4, false); CASE(7, true, 4, false);
#undef CASE
    }
}
static int cases = 0;
static void equal(Output* a, Output* b, const char* test, int T, int variant, int trial) {
    Output ha, hb;
    CK(hipStreamSynchronize(stream));
    CK(hipMemcpy(&ha, a, sizeof(ha), hipMemcpyDeviceToHost));
    CK(hipMemcpy(&hb, b, sizeof(hb), hipMemcpyDeviceToHost));
    ++cases;
    if (std::memcmp(&ha, &hb, sizeof(ha))) {
        const auto* pa = reinterpret_cast<const unsigned char*>(&ha);
        const auto* pb = reinterpret_cast<const unsigned char*>(&hb);
        size_t off = 0;
        while (pa[off] == pb[off]) ++off;
        std::fprintf(stderr, "FAIL %s T=%d variant=%d trial=%d byte=%zu %02x/%02x\n", test, T, variant, trial, off, pa[off], pb[off]);
        std::exit(1);
    }
}
static void fill(Output* o) { CK(hipMemsetAsync(o, 0x5a, sizeof(Output), stream)); }
static void counter_zero(unsigned* done) {
    unsigned h;
    CK(hipMemcpy(&h, done, sizeof(h), hipMemcpyDeviceToHost));
    if (h) { std::fprintf(stderr, "counter not reset: %u\n", h); std::exit(1); }
}
static hipGraphExec_t capture(const std::function<void(int)>& call, int layers) {
    hipGraph_t graph;
    hipGraphExec_t exec;
    CK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (int l = 0; l < layers; ++l) call(l);
    CK(hipStreamEndCapture(stream, &graph));
    CK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CK(hipGraphDestroy(graph));
    return exec;
}
static std::vector<double> time_graphs(const std::vector<std::function<void(int)>>& calls, int layers = 48) {
    std::vector<hipGraphExec_t> execs;
    for (const auto& call : calls) execs.push_back(capture(call, layers));
    for (int r = 0; r < 8; ++r) for (auto exec : execs) CK(hipGraphLaunch(exec, stream));
    CK(hipStreamSynchronize(stream));
    hipEvent_t start, end;
    CK(hipEventCreate(&start)); CK(hipEventCreate(&end));
    std::vector<std::vector<float>> times(execs.size());
    for (int r = 0; r < 31; ++r) for (size_t i = 0; i < execs.size(); ++i) {
        const size_t j = r % 2 ? execs.size() - 1 - i : i;
        CK(hipEventRecord(start, stream)); CK(hipGraphLaunch(execs[j], stream)); CK(hipEventRecord(end, stream));
        CK(hipEventSynchronize(end));
        float ms;
        CK(hipEventElapsedTime(&ms, start, end));
        times[j].push_back(ms * 1000.0f / layers);
    }
    std::vector<double> us;
    for (auto& t : times) { std::sort(t.begin(), t.end()); us.push_back(t[t.size() / 2]); }
    CK(hipEventDestroy(start)); CK(hipEventDestroy(end));
    for (auto exec : execs) CK(hipGraphExecDestroy(exec));
    return us;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const bool profile = argc > 1 && std::strcmp(argv[1], "--profile") == 0;
    const bool variants = argc > 1 && std::strcmp(argv[1], "--variants") == 0;
    const bool only_bench = profile || variants || (argc > 1 && std::strcmp(argv[1], "--bench-only") == 0);
    const int trials = argc > 2 ? std::atoi(argv[2]) : 24;
    const char* visible = std::getenv("HIP_VISIBLE_DEVICES");
    if (!visible || std::strcmp(visible, "1")) { std::fprintf(stderr, "Use HIP_VISIBLE_DEVICES=1\n"); return 2; }
    CK(hipSetDevice(0));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    if (std::strncmp(prop.gcnArchName, "gfx1030", 7)) return 77;
    std::printf("device 1: %s %s; N=%d experts=%d K=%d verify_max=%d\n", prop.name, prop.gcnArchName, N, NE, KK, strata::kernels::kVerifyMaxT);
    CK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    Output *a, *b;
    float *W, *x;
    int32_t* map;
    unsigned* done;
    CK(hipMalloc(&a, sizeof(Output))); CK(hipMalloc(&b, sizeof(Output)));
    CK(hipMalloc(&W, (NE + 1) * N * sizeof(float))); CK(hipMalloc(&x, MT * N * sizeof(float)));
    CK(hipMalloc(&map, NE * sizeof(int32_t))); CK(hipMalloc(&done, sizeof(unsigned)));
    CK(hipMemset(done, 0, sizeof(unsigned)));
    std::mt19937 rng(271828);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hw((NE + 1) * N), hx(MT * N), hl(MT * (NE + 1));
    std::vector<int32_t> hm(NE);
    if (!only_bench) {
        for (int trial = 0; trial < trials; ++trial) {
            for (float& v : hw) v = nd(rng) * 0.05f;
            if (trial % 3 == 1) for (int e = 1; e < NE; e += 3) std::copy_n(hw.data(), N, hw.data() + e * N);
            CK(hipMemcpy(W, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
            for (float& v : hx) v = nd(rng);
            if (trial % 4 == 0) std::fill(hx.begin(), hx.end(), 0.f);
            if (trial % 4 == 1) for (int t = 1; t < MT; ++t) std::copy_n(hx.data(), N, hx.data() + t * N);
            if (trial % 11 == 2) for (float& v : hx) v *= 1.e-7f;
            if (trial % 11 == 3) for (float& v : hx) v *= 1.e4f;
            CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
            for (int rows : {256, 257}) for (int T = 1; T <= MT; ++T) for (int v = 0; v < 8; ++v) {
                fill(a);
                project(v, Old, W, x, a, rows, T, done);
                for (Mode mode : {Default, MaskOnly, Split, Fallback}) {
                    if (mode == Fallback && T > (v >= 4 ? 4 : 8)) continue;
                    fill(b);
                    project(v, mode, W, x, b, rows, T, done);
                    equal(a, b, "projection", T, v * 10 + mode, trial);
                    counter_zero(done);
                }
                if (v == 2 && T <= 8) {
                    fill(b); project(v, Pair, W, x, b, rows, T, done);
                    equal(a, b, "pair", T, v, trial); counter_zero(done);
                }
            }
            for (float& v : hl) v = nd(rng) * (trial % 5 == 0 ? 40.f : 2.f);
            if (trial % 3 == 0) for (float& v : hl) v = std::round(v * 2.f) / 2.f;
            if (trial % 7 == 0) std::fill(hl.begin(), hl.end(), 0.f);
            if (trial % 7 == 1) for (int t = 1; t < MT; ++t) std::copy_n(hl.data(), NE + 1, hl.data() + t * (NE + 1));
            if (trial % 7 == 2) hl[77] = NAN;
            if (trial % 13 == 3) std::fill(hl.begin(), hl.begin() + NE, -INFINITY);
            if (trial % 13 == 4) hl[55] = INFINITY;
            for (int e = 0; e < NE; ++e) hm[e] = trial % 4 == 0 || (trial % 4 == 1 && e % 3) ? -1 : (e * 73) % NE;
            CK(hipMemcpy(map, hm.data(), hm.size() * 4, hipMemcpyHostToDevice));
            for (int T = 1; T <= MT; ++T) for (int threads : {128, 256}) for (int hy = 0; hy < 2; ++hy) {
                fill(a); fill(b);
                CK(hipMemcpyAsync(a->logits, hl.data(), hl.size() * 4, hipMemcpyHostToDevice, stream));
                CK(hipMemcpyAsync(b->logits, hl.data(), hl.size() * 4, hipMemcpyHostToDevice, stream));
                if (hy) {
                    grouping<true, true><<<1, threads, 0, stream>>>(a, T, NE + 1, map);
                    grouping<true, false><<<1, threads, 0, stream>>>(b, T, NE + 1, map);
                } else {
                    grouping<false, true><<<1, threads, 0, stream>>>(a, T, NE + 1, nullptr);
                    grouping<false, false><<<1, threads, 0, stream>>>(b, T, NE + 1, nullptr);
                }
                equal(a, b, "grouping", T, threads + hy, trial);
                if (threads == 256) {
                    for (int mask = 0; mask < 2; ++mask) {
                        if (!mask && T > 8) continue;
                        fill(b);
                        CK(hipMemcpyAsync(b->logits, hl.data(), hl.size() * 4, hipMemcpyHostToDevice, stream));
#define ROUTE(hy, mask) q36k::k_route_group<hy, mask><<<1, dim3(32, 8), 0, stream>>>(ARGS(b, T, NE + 1, map))
                        if (hy) { if (mask) ROUTE(true, true); else ROUTE(true, false); }
                        else { if (mask) ROUTE(false, true); else ROUTE(false, false); }
#undef ROUTE
                        equal(a, b, "standalone", T, hy * 10 + mask, trial);
                    }
                }
            }
        }
    }
    const int layers = bench || only_bench ? 48 : 8;
    std::vector<float*> matrices(layers);
    for (auto& p : matrices) {
        for (float& v : hw) v = nd(rng) * 0.05f;
        CK(hipMalloc(&p, hw.size() * 4)); CK(hipMemcpy(p, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
    }
    if (!only_bench) {
        for (int T : {1, 2, 3, 4, 8}) for (int v = 0; v < 8; ++v) {
            if (v >= 4 && T > 4) continue;
            fill(a); fill(b);
            const auto old = capture([&](int l) { project(v, Old, matrices[l], x, a, 257, T, done); }, 8);
            const auto now = capture([&](int l) { project(v, Default, matrices[l], x, b, 257, T, done); }, 8);
            for (int r = 0; r < 16; ++r) {
                for (float& f : hx) f = nd(rng);
                CK(hipMemcpyAsync(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice, stream));
                CK(hipGraphLaunch(old, stream)); CK(hipGraphLaunch(now, stream));
                equal(a, b, "graph", T, v, r); counter_zero(done);
            }
            CK(hipGraphExecDestroy(old)); CK(hipGraphExecDestroy(now));
        }
        std::printf("PASS %d cases: memcmp of logits, ids, weights, q8 and every grouping byte/sentinel; counter=0\n", cases);
        std::puts("T=1..16; 4/8 waves; both fences and q8 modes; HY holes/empty maps; eager and graph replay.");
        std::puts("Past the original token/wave limit, the frozen reference only extends storage and loops token waves.");
    }
    if (profile) {
        for (float& v : hx) v = nd(rng);
        CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        for (int r = 0; r < 8; ++r) for (Mode mode : {Old, Default, Projection, Split})
            for (int l = 0; l < layers; ++l) project(0, mode == Old ? Fallback : mode, matrices[l], x, b, 257, 4, done);
        CK(hipStreamSynchronize(stream)); counter_zero(done);
        std::puts("PROFILE T=4 old, new, projection-only, split; 8 x 48 calls each");
    } else if (variants) {
        for (float& v : hx) v = nd(rng);
        CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        for (int T = 1; T <= 4; ++T) for (int v = 0; v < 8; ++v) {
            const auto us = time_graphs({
                [&](int l) { project(v, Fallback, matrices[l], x, b, 257, T, done); },
                [&](int l) { project(v, Default, matrices[l], x, b, 257, T, done); }});
            std::printf("VARIANT T=%d LF=%d waves=%d q8=%d old=%.3f new=%.3f us speedup=%.3fx\n",
                T, v & 1, v >= 4 ? 4 : 8, (v / 2) % 2 == 0, us[0], us[1], us[0] / us[1]);
        }
        for (int T = 1; T <= 4; ++T) for (int holes = 0; holes < 3; ++holes) {
            for (int e = 0; e < NE; ++e) hm[e] = holes == 2 || (holes == 1 && e % 2) ? -1 : (e * 73) % NE;
            CK(hipMemcpy(map, hm.data(), hm.size() * 4, hipMemcpyHostToDevice));
            const auto us = time_graphs({
                [&](int) { grouping<true, true><<<1, 256, 0, stream>>>(b, T, 257, map); },
                [&](int) { grouping<true, false><<<1, 256, 0, stream>>>(b, T, 257, map); }});
            std::printf("HY T=%d missing=%s old=%.3f new=%.3f us speedup=%.3fx\n",
                T, holes == 0 ? "none" : holes == 1 ? "half" : "all", us[0], us[1], us[0] / us[1]);
        }
        counter_zero(done);
    } else if (bench || only_bench) {
        for (float& v : hx) v = nd(rng);
        CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        std::printf("BENCH median of 31 interleaved/reversed graph replays; %d rotating matrices, %.1f MiB\n", layers, layers * hw.size() * 4.0 / (1 << 20));
        for (int T = 1; T <= 8; ++T) {
            std::vector<std::function<void(int)>> calls;
            for (Mode mode : {Old, Default, MaskOnly, UnrollOnly, Split, Projection}) {
                // Time the engine's exact off-switch path, not the widened parity reference.
                const Mode timed = mode == Old ? Fallback : mode;
                calls.push_back([&, timed](int l) { project(0, timed, matrices[l], x, b, 257, T, done); });
            }
            const auto us = time_graphs(calls);
            const auto group = time_graphs({
                [&](int) { grouping<false, true><<<1, 256, 0, stream>>>(b, T, 257, nullptr); },
                [&](int) { grouping<false, false><<<1, 256, 0, stream>>>(b, T, 257, nullptr); }});
            std::printf("BENCH T=%d old=%.3f new=%.3f mask_only=%.3f unroll_only=%.3f split=%.3f projection=%.3f group_old=%.3f group_new=%.3f us speedup=%.3fx\n",
                T, us[Old], us[Default], us[MaskOnly], us[UnrollOnly], us[Split], us[Projection], group[0], group[1], us[Old] / us[Default]);
            std::fflush(stdout); counter_zero(done);
        }
    }
    for (auto p : matrices) CK(hipFree(p));
    CK(hipFree(a)); CK(hipFree(b)); CK(hipFree(W)); CK(hipFree(x)); CK(hipFree(map)); CK(hipFree(done));
    CK(hipStreamDestroy(stream));
    return 0;
}
