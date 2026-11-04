// Bitwise router parity and graph microbench, synthetic Ornith geometry; never loads model weights.
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
#include "q36_kernels.cuh"
#include "q36_tp_collective.cuh"
#include "q36_tp_fuse2.cuh"
#include "q36_tp_router_pair.cuh"
#include "router_reference.cuh"

#define CK(x) do { auto e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); std::exit(2); } } while (0)
constexpr int N = 3072, NE = 256, MT = 16, KK = 8;
struct Output {
    float logits[MT * (NE + 1)];
    int32_t ids[MT * KK];
    float weights[MT * KK];
    q36k::QB q[MT * (N / 32)];
    unsigned long long ptr[MT * KK];
    int32_t start[MT * KK + 1], ng, dst[MT * KK], tok[MT * KK];
};
#define blobs reinterpret_cast<const uint8_t*>(0x100000000ull)
constexpr unsigned long long blob = 65536;
static hipStream_t stream;

#define ARGS(o, T, ld, map) (o)->logits, (o)->ids, (o)->weights, T, ld, map, blobs, blob, (o)->ptr, (o)->start, &(o)->ng, (o)->dst, (o)->tok
#define PROJ_ARGS(o) W, x, (o)->logits, N, rows, T, rows, (o)->ids, (o)->weights, blobs, blob, (o)->ptr, (o)->start, &(o)->ng, (o)->dst, (o)->tok, done, (o)->q

template <bool HY, bool Old>
__global__ void grouping(Output* o, int T, int ld, const int32_t* map) {
    if constexpr (Old) router_reference::route_group_body<HY>(ARGS(o, T, ld, map));
    else q36tp_fuse2::route_group_body<HY>(ARGS(o, T, ld, map));
}

template <bool LF, int Warps, bool Q>
static void project(bool old, const float* W, const float* x, Output* o, int rows, int T, unsigned* done) {
    const int grid = ((rows + Warps - 1) / Warps) * T;
    if (old) router_reference::router_group<LF, Warps, Q><<<grid, Warps * 32, 0, stream>>>(PROJ_ARGS(o));
    else q36tp_fuse2::router_group<LF, Warps, Q><<<grid, Warps * 32, 0, stream>>>(PROJ_ARGS(o));
    CK(hipGetLastError());
}
static void project(int variant, bool old, const float* W, const float* x, Output* o, int rows, int T, unsigned* done) {
    switch (variant) {
#define CASE(i, lf, w, q) case i: project<lf, w, q>(old, W, x, o, rows, T, done); break
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

static double time_graph(const std::function<void(int)>& call, int layers = 48) {
    hipGraph_t graph;
    hipGraphExec_t exec;
    CK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (int l = 0; l < layers; ++l) call(l);
    CK(hipStreamEndCapture(stream, &graph));
    CK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int r = 0; r < 8; ++r) CK(hipGraphLaunch(exec, stream));
    CK(hipStreamSynchronize(stream));
    hipEvent_t start, end;
    CK(hipEventCreate(&start)); CK(hipEventCreate(&end));
    std::vector<float> times;
    for (int r = 0; r < 31; ++r) {
        CK(hipEventRecord(start, stream));
        CK(hipGraphLaunch(exec, stream));
        CK(hipEventRecord(end, stream));
        CK(hipEventSynchronize(end));
        float ms;
        CK(hipEventElapsedTime(&ms, start, end));
        times.push_back(ms * 1000.0f / layers);
    }
    std::sort(times.begin(), times.end());
    CK(hipEventDestroy(start)); CK(hipEventDestroy(end));
    CK(hipGraphExecDestroy(exec)); CK(hipGraphDestroy(graph));
    return times[times.size() / 2];
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const bool only_bench = argc > 1 && std::strcmp(argv[1], "--bench-only") == 0;
    const int trials = argc > 2 ? std::atoi(argv[2]) : 24;
    const char* visible = std::getenv("HIP_VISIBLE_DEVICES");
    if (!visible || std::strcmp(visible, "1")) { std::fprintf(stderr, "Use HIP_VISIBLE_DEVICES=1\n"); return 2; }
    CK(hipSetDevice(0));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device 1: %s %s; N=%d experts=%d K=%d\n", prop.name, prop.gcnArchName, N, NE, KK);
    CK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    Output *a, *b;
    float *W, *x;
    int32_t* map;
    unsigned* done;
    CK(hipMalloc(&a, sizeof(Output))); CK(hipMalloc(&b, sizeof(Output)));
    CK(hipMalloc(&W, (NE + 1) * N * sizeof(float)));
    CK(hipMalloc(&x, MT * N * sizeof(float)));
    CK(hipMalloc(&map, NE * sizeof(int32_t)));
    CK(hipMalloc(&done, sizeof(unsigned))); CK(hipMemset(done, 0, sizeof(unsigned)));
    std::mt19937 rng(271828);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hw((NE + 1) * N), hx(MT * N), hl(MT * (NE + 1));
    std::vector<int32_t> hm(NE);
    for (float& v : hw) v = nd(rng) * 0.05f;
    CK(hipMemcpy(W, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
    if (!only_bench) {
        for (int trial = 0; trial < trials; ++trial) {
            for (float& v : hx) v = nd(rng);
            if (trial % 4 == 0) std::fill(hx.begin(), hx.end(), 0.f);
            if (trial % 4 == 1) for (int t = 1; t < MT; ++t) std::copy_n(hx.data(), N, hx.data() + t * N);
            CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
            for (int rows : {256, 257}) for (int T = 1; T <= 8; ++T) for (int v = 0; v < 8; ++v) {
                if (v >= 4 && T > 4) continue;
                fill(a); fill(b);
                project(v, true, W, x, a, rows, T, done);
                project(v, false, W, x, b, rows, T, done);
                equal(a, b, "projection", T, v, trial);
                counter_zero(done);
            }
            for (float& v : hl) v = nd(rng) * (trial % 5 == 0 ? 40.f : 2.f);
            if (trial % 3 == 0) for (float& v : hl) v = std::round(v * 2.f) / 2.f;
            if (trial % 7 == 0) std::fill(hl.begin(), hl.end(), 0.f);
            if (trial % 7 == 1) for (int t = 1; t < MT; ++t) std::copy_n(hl.data(), NE + 1, hl.data() + t * (NE + 1));
            if (trial % 7 == 2) hl[77] = NAN;
            for (int e = 0; e < NE; ++e) hm[e] = trial % 4 == 0 || (trial % 4 == 1 && e % 3) ? -1 : (e * 73) % NE;
            CK(hipMemcpy(map, hm.data(), hm.size() * 4, hipMemcpyHostToDevice));
            for (int T = 1; T <= 8; ++T) for (int threads : {128, 256}) for (int hy = 0; hy < 2; ++hy) {
                if (threads == 128 && T > 4) continue;
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
            }
        }
        std::printf("PASS %d cases: all output bytes, including sentinels and q8; counters reset\n", cases);
    }
    if (bench || only_bench) {
        constexpr int layers = 48;
        std::vector<float*> matrices(layers);
        for (auto& p : matrices) {
            for (float& v : hw) v = nd(rng) * 0.05f;
            CK(hipMalloc(&p, hw.size() * 4));
            CK(hipMemcpy(p, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
        }
        for (float& v : hx) v = nd(rng);
        CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        for (int T = 1; T <= 8; ++T) {
            double us[2];
            for (int old = 1; old >= 0; --old)
                us[old] = time_graph([&](int l) { project(0, old, matrices[l], x, b, 257, T, done); });
            const double gold = time_graph([&](int) { grouping<false, true><<<1, 256, 0, stream>>>(b, T, 257, nullptr); });
            const double gnew = time_graph([&](int) { grouping<false, false><<<1, 256, 0, stream>>>(b, T, 257, nullptr); });
            std::printf("BENCH T=%d old=%.3f new=%.3f speedup=%.3fx group_old=%.3f group_new=%.3f us\n", T, us[1], us[0], us[1] / us[0], gold, gnew);
            std::fflush(stdout);
            counter_zero(done);
        }
        for (auto p : matrices) CK(hipFree(p));
    }
    CK(hipFree(a)); CK(hipFree(b)); CK(hipFree(W)); CK(hipFree(x)); CK(hipFree(map)); CK(hipFree(done));
    CK(hipStreamDestroy(stream));
    return 0;
}
