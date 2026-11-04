// tests/hip/decode_expert_lab.cpp - the decode path's routed experts on this card: what one grouped call costs per
// format pair, split into gate/up (+ SwiGLU + q8_1) and down, against what the card can read (a plain 16-byte-load
// streaming kernel over the same bytes).  Synthetic blobs (random codes are valid for every format), no model.
//
//     hip_decode_expert_lab            the table: us per call and GB/s of weights read, per format pair and window shape
//     hip_decode_expert_lab --hash     one line of output hashes per format pair and plan (fixed seed): two runs with
//                                      different STRATA_* switches must print the same lines when a switch is bitwise
//
// A call is a layer's work in a verify window: T tokens x 10 experts, grouped by expert.  The 24 calls of a card's layers
// run in one graph, every layer with its own experts, so the weights stream from DRAM as they do in the engine.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

// the model's expert geometry: Flash-Next's by default; LAB_EMBD / LAB_FF / LAB_K for another (Qwen3.5-122B-A10B /
// Ornith 1.5: 3072 / 1024 / 8)
const int kE = std::getenv("LAB_EMBD") ? std::atoi(std::getenv("LAB_EMBD")) : 2560;
const int kF = std::getenv("LAB_FF") ? std::atoi(std::getenv("LAB_FF")) : 640;
const int kK = std::getenv("LAB_K") ? std::atoi(std::getenv("LAB_K")) : 10;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "error in %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
template<typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
const char* name_of(int t) {
    switch (t) {
        case 11: return "Q3_K"; case 16: return "IQ2_XXS"; case 18: return "IQ3_XXS"; case 19: return "IQ1_S";
        case 20: return "IQ4_NL"; case 21: return "IQ3_S"; case 22: return "IQ2_S";
        case 23: return "IQ4_XS"; case 42: return "Q2_0"; case 12: return "Q4_K"; case 13: return "Q5_K";
        case 7: return "Q5_1"; case 8: return "Q8_0"; case 2: return "Q4_0"; default: return "?";
    }
}
int block_values(int t) { return t == 20 || t == 7 || t == 8 || t == 2 ? 32 : t == 42 ? 64 : 256; }

std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> ex(2, 8), man(0, 1023), sgn(0, 1);
    uint32_t* w32 = (uint32_t*) w.data();
    for (size_t i = 0; i < w.size() / 4; ++i) w32[i] = rng();
    for (size_t o = 0; o < w.size(); o += bs) {
        const int e = (t == 12 || t == 13) ? 1 + ex(rng) % 4 : ex(rng);
        const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (e << 10) | man(rng));
        std::memcpy(&w[o + (t == 11 ? 108 : 0)], &h, 2);
        if (t == 12 || t == 13 || t == 7) {
            const int em = (t == 7) ? ex(rng) : 1 + ex(rng) % 4;
            const uint16_t m = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (em << 10) | man(rng));
            std::memcpy(&w[o + 2], &m, 2);
        }
    }
    return w;
}

// a plain read of `bytes` from each of *n pointers, 16 bytes a load: what the card streams when nothing else is done
__global__ void stream_kernel(const unsigned long long* __restrict__ ptr, int n, long long bytes, unsigned* __restrict__ out) {
    const long long words = bytes / 16;
    unsigned acc = 0;
    for (int g = blockIdx.y; g < n; g += gridDim.y) {
        const uint4* p = (const uint4*) ptr[g];
        for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (long long) gridDim.x * blockDim.x) {
            const uint4 v = p[i];
            acc += v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (acc == 0x12345678u) out[0] = acc;
}

constexpr int kLayers = 24;

struct Lab {
    k::NativeExpertLayout L;
    int T, K, cap, per;
    size_t slot;
    uint8_t* arena = nullptr;
    uint8_t* xq = nullptr;
    uint8_t* scr = nullptr;
    float* out = nullptr;
    unsigned long long* ptr[kLayers];
    int32_t *start, *n, *dst, *tok;
    int ng = 0;
    size_t scr_bytes = 0;
    size_t launches[3] = {};

    Lab(int gu, int dt, int tokens, int per_layer, std::mt19937& rng, cudaStream_t s) {
        L = k::native_expert_layout(gu, dt, kE, kF);
        T = tokens; K = kK; cap = T * K; per = per_layer;
        slot = (L.bytes + 255) & ~(size_t) 255;
        arena = dalloc<uint8_t>(slot * (size_t) (kLayers * per));
        {   // 4 distinct random blobs, tiled: the reads still stream (every slot is its own memory)
            std::vector<std::vector<uint8_t>> src;
            for (int b = 0; b < 4; ++b) {
                auto blob = random_rows(gu, 2 * kF, kE, rng);
                const auto down = random_rows(dt, kE, kF, rng);
                blob.insert(blob.end(), down.begin(), down.end());
                if (blob.size() != L.bytes) { std::fprintf(stderr, "blob size\n"); std::exit(2); }
                src.push_back(std::move(blob));
            }
            for (int i = 0; i < kLayers * per; ++i)
                ck(cudaMemcpy(arena + slot * (size_t) i, src[(size_t) (i % 4)].data(), L.bytes, cudaMemcpyHostToDevice), "blob");
        }
        for (int l = 0; l < kLayers; ++l) {
            std::vector<unsigned long long> p;
            for (int j = 0; j < per; ++j) p.push_back((unsigned long long) (arena + slot * (size_t) (l * per + j)));
            ptr[l] = dalloc<unsigned long long>(p.size());
            ck(cudaMemcpy(ptr[l], p.data(), p.size() * 8, cudaMemcpyHostToDevice), "ptr");
        }
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) T * kE);
        for (auto& v : x) v = nd(rng);
        float* dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        xq = dalloc<uint8_t>((size_t) T * (kE / 32) * 36);
        k::quantize_q8_1_rows(dx, T, kE, xq, s);
        ck(cudaStreamSynchronize(s), "quantize");
        cudaFree(dx);
        scr_bytes = k::native_expert_scratch_bytes(cap, kF);
        scr = dalloc<uint8_t>(scr_bytes);
        out = dalloc<float>((size_t) cap * kE);
        start = dalloc<int32_t>(cap + 1); n = dalloc<int32_t>(1); dst = dalloc<int32_t>(cap); tok = dalloc<int32_t>(cap);
    }
    ~Lab() {
        cudaFree(arena); cudaFree(xq); cudaFree(scr); cudaFree(out);
        for (auto* p : ptr) cudaFree(p);
        cudaFree(start); cudaFree(n); cudaFree(dst); cudaFree(tok);
        for (auto* p : ptr_off) cudaFree(p);
    }
    // groups of these sizes (entries); a group's entries are different tokens, as the engine's are
    void plan(const std::vector<int>& sizes, std::mt19937& rng) {
        std::vector<int32_t> st(1, 0), tk, ds(cap);
        for (const int m : sizes) {
            if (m < 1 || m > T) { std::fprintf(stderr, "invalid group size\n"); std::exit(2); }
            std::vector<int> t(T);
            std::iota(t.begin(), t.end(), 0);
            std::shuffle(t.begin(), t.end(), rng);
            std::sort(t.begin(), t.begin() + m);
            for (int i = 0; i < m; ++i) tk.push_back(t[(size_t) i]);
            st.push_back(st.back() + m);
        }
        if (st.back() > cap || (int) sizes.size() > per) { std::fprintf(stderr, "plan does not fit\n"); std::exit(2); }
        tk.resize(cap, 0);
        ng = (int) sizes.size();
        std::iota(ds.begin(), ds.end(), 0);
        std::shuffle(ds.begin(), ds.end(), rng);
        ck(cudaMemcpy(start, st.data(), st.size() * 4, cudaMemcpyHostToDevice), "start");
        ck(cudaMemcpy(n, &ng, 4, cudaMemcpyHostToDevice), "n");
        ck(cudaMemcpy(dst, ds.data(), ds.size() * 4, cudaMemcpyHostToDevice), "dst");
        ck(cudaMemcpy(tok, tk.data(), tk.size() * 4, cudaMemcpyHostToDevice), "tok");
    }
    cudaGraphExec_t graph(cudaStream_t s, int phase, int layers, int mode = -1) {
        k::native_expert_set_mode(mode, phase);
        cudaGraph_t g;
        cudaGraphExec_t ge;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
        for (int l = 0; l < layers; ++l)
            k::native_expert_grouped(L, ptr[l], start, n, dst, tok, cap, cap, xq, scr, out, s, 0, T == 1);
        ck(cudaStreamEndCapture(s, &g), "capture");
        size_t nodes = 0;
        ck(cudaGraphGetNodes(g, nullptr, &nodes), "nodes");
        launches[phase] = nodes / (size_t) layers;
        ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
        ck(cudaGraphDestroy(g), "graph");
        k::native_expert_set_mode(-1, 0);
        return ge;
    }
    cudaGraphExec_t stream_graph(cudaStream_t s, long long bytes, size_t off, unsigned* sink) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
        for (int l = 0; l < kLayers; ++l) {
            // pointers offset by `off` cannot be made on the host for a device array without a copy: use a second array
            stream_kernel<<<dim3(64, (unsigned) std::max(ng, 1)), 256, 0, s>>>(ptr_off[l], ng, bytes, sink);
        }
        (void) off;
        ck(cudaStreamEndCapture(s, &g), "capture");
        ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
        ck(cudaGraphDestroy(g), "graph");
        return ge;
    }
    unsigned long long* ptr_off[kLayers] = {};
    void make_ptr_off(size_t off) {
        for (int l = 0; l < kLayers; ++l) {
            std::vector<unsigned long long> p;
            for (int j = 0; j < per; ++j) p.push_back((unsigned long long) (arena + slot * (size_t) (l * per + j) + off));
            if (!ptr_off[l]) ptr_off[l] = dalloc<unsigned long long>(p.size());
            ck(cudaMemcpy(ptr_off[l], p.data(), p.size() * 8, cudaMemcpyHostToDevice), "ptr");
        }
    }
};

float launch_us(cudaStream_t s, cudaGraphExec_t ge, cudaEvent_t e0, cudaEvent_t e1, int layers) {
    ck(cudaEventRecord(e0, s), "record");
    ck(cudaGraphLaunch(ge, s), "launch");
    ck(cudaEventRecord(e1, s), "record");
    ck(cudaEventSynchronize(e1), "bench");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    return 1e3f * ms / (float) layers;
}
float median_us(cudaStream_t s, cudaGraphExec_t ge, int layers = kLayers) {
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    std::vector<float> v;
    for (int i = 0; i < 70; ++i) {
        const float t = launch_us(s, ge, e0, e1, layers);
        if (i >= 20) v.push_back(t);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

uint64_t fnv(const void* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// the window shapes: group sizes of a T-token window (10 experts a token; tokens of one window share some experts)
std::vector<int> shape(int T) {
    if (kK != 10) {   // another model: K a token, each extra token sharing half its experts with the window's
        std::vector<int> v((size_t) kK, 1);
        for (int t = 1; t < T; ++t) {
            for (int j = 0; j < kK / 2; ++j) ++v[(size_t) ((kK / 2 * (t - 1) + j) % v.size())];
            v.insert(v.end(), (size_t) (kK - kK / 2), 1);
        }
        return v;
    }
    switch (T) {
        case 1: return std::vector<int>(10, 1);
        case 2: { std::vector<int> v(14, 1); v.insert(v.end(), 3, 2); return v; }                          // 17 groups, 20
        case 3: { std::vector<int> v(15, 1); v.insert(v.end(), 6, 2); v.insert(v.end(), 1, 3); return v; }  // 22 groups, 30
        case 4: { std::vector<int> v(16, 1); v.insert(v.end(), 9, 2); v.insert(v.end(), 2, 3); return v; } // 27 groups, 40
        default: {
            auto v = shape(4);
            // Synthetic extension: each extra token shares five experts and adds five.
            for (int t = 4; t < T; ++t) {
                for (int j = 0; j < 5; ++j) ++v[(size_t) ((5 * (t - 4) + j) % 27)];
                v.insert(v.end(), 5, 1);
            }
            return v;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool hash = argc > 1 && std::string(argv[1]) == "--hash";
    const bool tpx_fused_t1 = argc > 1 && std::string(argv[1]) == "--tpx-fused-t1";
    const bool tpx_compare = argc > 1 && (std::string(argv[1]) == "--tpx-compare" || tpx_fused_t1);
    int only_gu = -1, only_d = -1, only_T = 0;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--pair") { std::sscanf(argv[i + 1], "%d/%d", &only_gu, &only_d); }
        if (std::string(argv[i]) == "--T") only_T = std::atoi(argv[i + 1]);
    }
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    const int pairs[][2] = {{18, 20}, {22, 20}, {21, 20}, {22, 42}, {18, 42}, {21, 42}, {23, 20}, {12, 7}, {12, 8}, {13, 8},
                            {18, 23}, {23, 23}, {18, 18}, {12, 12}, {12, 13}, {13, 13}, {23, 12}, {2, 2}, {3, 3}, {2, 3},
                            {16, 23}, {19, 23}, {22, 23}, {11, 23}, {8, 8}};   // (+ Ornith 1.5's)
    if (tpx_compare) {
        bool pass = true;
        for (const auto& pr : pairs) {
            if (only_gu >= 0 && (pr[0] != only_gu || pr[1] != only_d)) continue;
            if (!k::native_expert_supported(pr[0], pr[1], kE, kF)) continue;
            std::mt19937 rng(1234u + (unsigned) pr[0] * 100u + (unsigned) pr[1]);
            for (int T = 1; T <= 4; ++T) {
                if (only_T && T != only_T) continue;
                if (tpx_fused_t1 && (T != 1 || (pr[0] != 18 && pr[0] != 23))) continue;
                Lab lab(pr[0], pr[1], T, T * kK, rng, s);
                lab.plan(shape(T), rng);
                k::native_expert_tpx_tune(0, 0);
                k::native_expert_tpx_fused_t1_tune(0);
                const int em = tpx_fused_t1 ? 8 : -1;
                auto old_gu = lab.graph(s, 1, kLayers, em), old_down = lab.graph(s, 2, kLayers, em), old_all = lab.graph(s, 0, kLayers, em);
                if (tpx_fused_t1) k::native_expert_tpx_fused_t1_tune(1);
                else k::native_expert_tpx_tune(1, 0);
                auto new_gu = lab.graph(s, 1, kLayers, em);
                if (!tpx_fused_t1) k::native_expert_tpx_tune(0, 1);
                auto new_down = lab.graph(s, 2, kLayers, em);
                if (!tpx_fused_t1) k::native_expert_tpx_tune(1, 1);
                auto new_all = lab.graph(s, 0, kLayers, em);
                const float gu0 = median_us(s, old_gu), gu1 = median_us(s, new_gu);
                // Both down graphs read hq from the last complete expert call.
                const float d0 = median_us(s, old_down), d1 = median_us(s, new_down);
                ck(cudaGraphLaunch(old_all, s), "old graph");
                ck(cudaStreamSynchronize(s), "old sync");
                std::vector<float> old_out((size_t) lab.cap * kE);
                ck(cudaMemcpy(old_out.data(), lab.out, old_out.size() * 4, cudaMemcpyDeviceToHost), "old out");
                const size_t fa = ((size_t) lab.cap * kF * 4 + 255) & ~(size_t) 255;
                std::vector<uint8_t> old_hq((size_t) lab.cap * (kF / 32) * 36);
                ck(cudaMemcpy(old_hq.data(), lab.scr + 3 * fa, old_hq.size(), cudaMemcpyDeviceToHost), "old hq");
                // The two graphs share buffers. Clear them before the new path:
                // a skipped SwiGLU/q8_1 launch must not inherit the old hq.
                ck(cudaMemset(lab.scr, 0, lab.scr_bytes), "clear scratch before new");
                ck(cudaMemset(lab.out, 0, old_out.size() * 4), "clear output before new");
                ck(cudaGraphLaunch(new_all, s), "new graph");
                ck(cudaStreamSynchronize(s), "new sync");
                std::vector<float> new_out(old_out.size());
                std::vector<uint8_t> new_hq(old_hq.size());
                ck(cudaMemcpy(new_out.data(), lab.out, new_out.size() * 4, cudaMemcpyDeviceToHost), "new out");
                ck(cudaMemcpy(new_hq.data(), lab.scr + 3 * fa, new_hq.size(), cudaMemcpyDeviceToHost), "new hq");
                const bool same_out = std::memcmp(old_out.data(), new_out.data(), old_out.size() * 4) == 0;
                const bool same_hq = std::memcmp(old_hq.data(), new_hq.data(), old_hq.size()) == 0;
                pass &= same_out && same_hq;
                const double gu_mb = shape(T).size() * (double) lab.L.down_off / 1e6;
                const double d_mb = shape(T).size() * (double) (lab.L.bytes - lab.L.down_off) / 1e6;
                std::printf("%-8s/%-7s T=%d groups=%zu GU %.1f -> %.1f us (%.0f -> %.0f GB/s) "
                            "down %.1f -> %.1f us (%.0f -> %.0f GB/s) bits %s/%s\n",
                            name_of(pr[0]), name_of(pr[1]), T, shape(T).size(), gu0, gu1,
                            gu_mb / gu0 * 1e3, gu_mb / gu1 * 1e3, d0, d1,
                            d_mb / d0 * 1e3, d_mb / d1 * 1e3, same_hq ? "hq=OK" : "hq=FAIL",
                            same_out ? "out=OK" : "out=FAIL");
                for (auto g : {old_gu, old_down, old_all, new_gu, new_down, new_all}) cudaGraphExecDestroy(g);
            }
        }
        k::native_expert_tpx_tune(-1, -1);
        k::native_expert_tpx_fused_t1_tune(-1);
        return pass ? 0 : 1;
    }
    if (hash) {
        for (const auto& pr : pairs) {
            if (only_gu >= 0 && (pr[0] != only_gu || pr[1] != only_d)) continue;
            if (!k::native_expert_supported(pr[0], pr[1], kE, kF)) continue;
            std::mt19937 rng(1234u + (unsigned) pr[0] * 100u + (unsigned) pr[1]);
            for (int T = 1; T <= 8; ++T) {
                if (only_T && T != only_T) continue;
                Lab lab(pr[0], pr[1], T, T * kK, rng, s);
                for (int v = 0; v < 3; ++v) {
                    std::vector<int> sizes;
                    if (v == 0) sizes = shape(T);
                    else if (v == 1) sizes = std::vector<int>((size_t) T * kK, 1);
                    else { int left = T * kK; while (left > 0) { const int m = std::min(left, T); sizes.push_back(m); left -= m; } }
                    lab.plan(sizes, rng);
                    ck(cudaMemset(lab.out, 0xFF, (size_t) lab.cap * kE * 4), "memset");
                    ck(cudaMemset(lab.scr, 0x5A, lab.scr_bytes), "memset");
                    k::native_expert_grouped(lab.L, lab.ptr[0], lab.start, lab.n, lab.dst, lab.tok, lab.cap, lab.cap, lab.xq,
                                             lab.scr, lab.out, s, 0);
                    ck(cudaStreamSynchronize(s), "sync");
                    std::vector<float> o((size_t) lab.cap * kE);
                    ck(cudaMemcpy(o.data(), lab.out, o.size() * 4, cudaMemcpyDeviceToHost), "out");
                    // Also compare the fused gate/up output before the down projection.
                    const size_t fa = ((size_t) lab.cap * kF * 4 + 255) & ~(size_t) 255;
                    std::vector<uint8_t> hq((size_t) lab.cap * (kF / 32) * 36);
                    ck(cudaMemcpy(hq.data(), lab.scr + 3 * fa, hq.size(), cudaMemcpyDeviceToHost), "hq");
                    size_t non_finite = 0;
                    for (const float f : o) non_finite += !std::isfinite(f);
                    std::printf("%-8s/%-7s T %d plan %d (%2zu groups): out %016llx hq %016llx (%zu non-finite)\n", name_of(pr[0]),
                                name_of(pr[1]), T, v, sizes.size(), (unsigned long long) fnv(o.data(), o.size() * 4), (unsigned long long) fnv(hq.data(), hq.size()), non_finite);
                }
            }
        }
        return 0;
    }
    std::printf("decode expert lab: us per layer call (median of 50 launches of a %d-layer graph, every layer its own experts)\n"
                "and GB/s of expert weights read; 'stream' = a plain 16-byte-load read of the same bytes\n", kLayers);
    unsigned* sink = dalloc<unsigned>(4);
    {   // what a launch costs by itself: 24 launches of 1 block that reads nothing
        cudaGraph_t g;
        cudaGraphExec_t ge;
        unsigned long long* none = dalloc<unsigned long long>(1);
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
        for (int l = 0; l < kLayers; ++l) stream_kernel<<<dim3(1, 1), 256, 0, s>>>(none, 0, 0, sink);
        ck(cudaStreamEndCapture(s, &g), "capture");
        ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
        std::printf("an empty launch in a graph: %.2f us\n", median_us(s, ge));
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
        for (int l = 0; l < kLayers; ++l) stream_kernel<<<dim3(80, 40), 256, 0, s>>>(none, 0, 0, sink);
        ck(cudaStreamEndCapture(s, &g), "capture");
        ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
        std::printf("an empty launch of 80 x 40 blocks in a graph: %.2f us\n", median_us(s, ge));
    }
    for (const auto& pr : pairs) {
        if (only_gu >= 0 && (pr[0] != only_gu || pr[1] != only_d)) continue;
        if (!k::native_expert_supported(pr[0], pr[1], kE, kF)) { std::printf("%s/%s: not supported\n", name_of(pr[0]), name_of(pr[1])); continue; }
        std::mt19937 rng(99);
        for (int T = 1; T <= 8; ++T) {
            if (only_T && T != only_T) continue;
            Lab lab(pr[0], pr[1], T, T * kK, rng, s);
            const auto sizes = shape(T);
            lab.plan(sizes, rng);
            const double gu_mb = (double) sizes.size() * (double) lab.L.down_off / 1e6;
            const double d_mb = (double) sizes.size() * (double) (lab.L.bytes - lab.L.down_off) / 1e6;
            cudaGraphExec_t g1 = lab.graph(s, 1, kLayers), g2 = lab.graph(s, 2, kLayers), g0 = lab.graph(s, 0, kLayers);
            const float t1 = median_us(s, g1), t2 = median_us(s, g2), t0 = median_us(s, g0);
            lab.make_ptr_off(0);
            cudaGraphExec_t gs1 = lab.stream_graph(s, (long long) lab.L.down_off, 0, sink);
            const float ts1 = median_us(s, gs1);
            lab.make_ptr_off(lab.L.down_off);
            cudaGraphExec_t gs2 = lab.stream_graph(s, (long long) (lab.L.bytes - lab.L.down_off), 0, sink);
            const float ts2 = median_us(s, gs2);
            std::printf("%-8s/%-7s T %d %2zu groups %2d entries | gate/up %7.1f us %5.0f GB/s (stream %6.1f us %4.0f GB/s) | "
                        "down %7.1f us %5.0f GB/s (stream %6.1f us %4.0f GB/s) | all %7.1f us | launches %zu/%zu/%zu\n",
                        name_of(pr[0]), name_of(pr[1]), T, sizes.size(), T * kK, t1, gu_mb / t1 * 1e3, ts1, gu_mb / ts1 * 1e3,
                        t2, d_mb / t2 * 1e3, ts2, d_mb / ts2 * 1e3, t0, lab.launches[1], lab.launches[2], lab.launches[0]);
            std::fflush(stdout);
            for (auto ge : {g1, g2, g0, gs1, gs2}) cudaGraphExecDestroy(ge);
        }
    }
    return 0;
}
