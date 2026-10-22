// Real TP expert-call replay. Use the same GGUF half-FF blob assembly, device
// router/group kernel, capacity arguments, and native_expert_grouped launcher
// as q36_engine.cu. Run both captured paths and compare every intermediate.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "q36_kernels.cuh"
#include "q36_tp_collective.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace K = strata::kernels;

static void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T* alloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset");
    return p;
}
static size_t lcm(size_t a, size_t b) { return a / std::gcd(a, b) * b; }
static size_t block_bytes(int type) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) throw std::runtime_error("unknown block type");
    return (size_t) bb;
}

struct Weight {
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;
    size_t stride = 0;
    int n_expert = 0;
    int layer = -1, rank = 0;
    Weight(const strata::GgufModel& m, const strata::core::ModelGeometry& g, int li, int tp_rank) :
        n_expert((int) g.n_expert), layer(li), rank(tp_rank) {
        const auto name = [&](const char* suffix) { return "blk." + std::to_string(li) + "." + suffix; };
        size_t sg = 0, su = 0, sd = 0;
        const auto* tg = m.find(name("ffn_gate_exps.weight"), &sg);
        const auto* tu = m.find(name("ffn_up_exps.weight"), &su);
        const auto* td = m.find(name("ffn_down_exps.weight"), &sd);
        if (!tg || !tu || !td || tg->type != tu->type || g.n_ff % 2)
            throw std::runtime_error("invalid expert tensors or odd FF");
        if (tg->shape.size() != 3 || tu->shape.size() != 3 || td->shape.size() != 3 ||
            tg->shape[0] != (size_t) g.n_embd || tu->shape[0] != (size_t) g.n_embd ||
            tg->shape[1] != (size_t) g.n_ff || tu->shape[1] != (size_t) g.n_ff ||
            td->shape[0] != (size_t) g.n_ff || td->shape[1] != (size_t) g.n_embd ||
            tg->shape[2] != (size_t) n_expert || tu->shape[2] != (size_t) n_expert || td->shape[2] != (size_t) n_expert)
            throw std::runtime_error("unexpected expert tensor shape");
        int down_block = 0, down_bytes = 0;
        if (!strata::block_geometry(td->type, down_block, down_bytes) || (g.n_ff / 2) % down_block)
            throw std::runtime_error("TP down split crosses a quant block");
        L = K::native_expert_layout((int) tg->type, (int) td->type, g.n_embd, g.n_ff / 2);
        if (!K::native_expert_supported(L.gu_type, L.d_type, L.n_embd, L.n_ff) ||
            !m.in_bounds(*tg, sg) || !m.in_bounds(*tu, su) || !m.in_bounds(*td, sd))
            throw std::runtime_error("unsupported or truncated expert tensor");
        const size_t al = lcm(lcm(block_bytes(L.gu_type), block_bytes(L.d_type)), 16);
        stride = (L.bytes + al - 1) / al * al;
        ck(cudaMalloc(&blobs, (size_t) n_expert * stride + 4096), "blobs");
        ck(cudaMemset(blobs, 0, (size_t) n_expert * stride + 4096), "zero blobs");
        const uint8_t* gate = m.shard(sg).tensor_data(*tg);
        const uint8_t* up = m.shard(su).tensor_data(*tu);
        const uint8_t* down = m.shard(sd).tensor_data(*td);
        const size_t gu = (size_t) L.n_ff * L.gu_row, dsz = (size_t) L.n_embd * L.d_row;
        std::vector<uint8_t> blob(stride, 0);
        for (int e = 0; e < n_expert; ++e) {
            // q36_engine.cu expert_blob(), including the interleaved down rows.
            std::memcpy(blob.data(), gate + (size_t) e * 2 * gu + rank * gu, gu);
            std::memcpy(blob.data() + L.up_off, up + (size_t) e * 2 * gu + rank * gu, gu);
            for (int r = 0; r < L.n_embd; ++r)
                std::memcpy(blob.data() + L.down_off + (size_t) r * L.d_row,
                            down + (size_t) e * 2 * dsz + ((size_t) r * 2 + rank) * L.d_row, L.d_row);
            ck(cudaMemcpy(blobs + (size_t) e * stride, blob.data(), stride, cudaMemcpyHostToDevice), "upload blob");
        }
    }
    ~Weight() { (void) cudaFree(blobs); }
};

struct Call {
    int T, topk, n_expert, cap, groups;
    const Weight& w;
    cudaStream_t stream;
    float *logits, *x, *ewts, *shared, *sgate, *out[2], *combined[2];
    int32_t *ids, *start, *ng, *dst, *tok;
    unsigned long long* ptr;
    uint8_t *xq, *scratch[2];
    cudaGraphExec_t graph[2] = {};
    size_t graph_nodes[2] = {};
    Call(const Weight& weight, int tokens, int k, int ne, cudaStream_t s, bool repeated) :
        T(tokens), topk(k), n_expert(ne), cap(T * k), groups(std::min(cap, ne)), w(weight), stream(s) {
        const int N = (int) w.L.n_embd;
        logits = alloc<float>((size_t) T * ne);
        x = alloc<float>((size_t) T * N);
        ewts = alloc<float>(cap);
        shared = alloc<float>((size_t) T * N);
        sgate = alloc<float>(T);
        ids = alloc<int32_t>(cap + 128); // native MMQ's padded ids convention
        start = alloc<int32_t>(groups + 1);
        ng = alloc<int32_t>(1);
        dst = alloc<int32_t>(cap);
        tok = alloc<int32_t>(cap);
        ptr = alloc<unsigned long long>(groups);
        xq = alloc<uint8_t>((size_t) T * (N / 32) * 36);
        std::mt19937 rng(20261021 + w.layer * 17 + w.rank * 3 + T);
        std::normal_distribution<float> normal(0.f, 1.f);
        std::vector<float> hx((size_t) T * N), hl((size_t) T * ne), hs((size_t) T * N), hg(T);
        for (auto& v : hx) v = normal(rng);
        for (auto& v : hs) v = normal(rng);
        for (auto& v : hg) v = normal(rng);
        for (int t = 0; t < T; ++t)
            for (int e = 0; e < ne; ++e)
                hl[(size_t) t * ne + e] = (repeated ? (e < topk ? 8.f : -8.f) : 0.f) + normal(rng) * (repeated ? .01f : 1.f);
        ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "activation");
        ck(cudaMemcpy(logits, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice), "router logits");
        ck(cudaMemcpy(shared, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice), "shared expert");
        ck(cudaMemcpy(sgate, hg.data(), hg.size() * 4, cudaMemcpyHostToDevice), "shared gate");
        const size_t scratch_bytes = K::native_expert_scratch_bytes(cap, w.L.n_ff);
        K::native_expert_set_mode(0, 0); // gfx1030 engine reference layout
        for (int i = 0; i < 2; ++i) {
            scratch[i] = alloc<uint8_t>(scratch_bytes + 256);
            out[i] = alloc<float>((size_t) cap * N + 64);
            combined[i] = alloc<float>((size_t) T * N);
            K::native_expert_tpx_tune(i, 0);
            cudaGraph_t g;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
            q36k::k_route_group<false><<<1, dim3(32, 8), 0, stream>>>(logits, ids, ewts, T, ne, nullptr,
                w.blobs, w.stride, ptr, start, ng, dst, tok);
            K::native_quantize_q8_1(x, xq, N * T, 1, stream);
            K::native_expert_grouped(w.L, ptr, start, ng, dst, tok, groups, cap, xq, scratch[i], out[i], stream);
            // The TP engine uses this exact kernel after down: put routed
            // outputs back in pick order, weight them, add the shared expert.
            q36tp::partial_moe<<<T, 256, 0, stream>>>(combined[i], out[i], nullptr, ewts, topk,
                                                       shared, sgate, 1, N);
            ck(cudaStreamEndCapture(stream, &g), "end capture");
            ck(cudaGraphGetNodes(g, nullptr, &graph_nodes[i]), "graph nodes");
            ck(cudaGraphInstantiate(&graph[i], g, 0), "instantiate");
            ck(cudaGraphDestroy(g), "destroy graph");
        }
        K::native_expert_tpx_tune(-1, -1);
        K::native_expert_set_mode(-1, 0);
        const char* gu_check = std::getenv("STRATA_Q36_TPX_GU_CHECK");
        if (!(gu_check && std::strcmp(gu_check, "1") == 0) && graph_nodes[0] != graph_nodes[1])
            throw std::runtime_error("GU16 graph has a different number of launches from reference");
    }
    ~Call() {
        for (auto g : graph) (void) cudaGraphExecDestroy(g);
        for (auto p : scratch) (void) cudaFree(p);
        for (auto p : out) (void) cudaFree(p);
        for (auto p : combined) (void) cudaFree(p);
        for (auto p : {static_cast<void*>(logits), static_cast<void*>(x), static_cast<void*>(ewts),
                       static_cast<void*>(shared), static_cast<void*>(sgate),
                       static_cast<void*>(ids), static_cast<void*>(start), static_cast<void*>(ng),
                       static_cast<void*>(dst), static_cast<void*>(tok), static_cast<void*>(ptr),
                       static_cast<void*>(xq)}) (void) cudaFree(p);
    }
    bool compare(const char* label, const void* a, const void* b, size_t bytes) {
        std::vector<uint8_t> va(bytes), vb(bytes);
        ck(cudaMemcpy(va.data(), a, bytes, cudaMemcpyDeviceToHost), "read old");
        ck(cudaMemcpy(vb.data(), b, bytes, cudaMemcpyDeviceToHost), "read new");
        const auto it = std::mismatch(va.begin(), va.end(), vb.begin());
        if (it.first == va.end()) return true;
        std::fprintf(stderr, "FAIL layer=%d rank=%d T=%d %s byte=%zu old=%02x new=%02x\n",
                     w.layer, w.rank, T, label, (size_t) (it.first - va.begin()), *it.first, *it.second);
        return false;
    }
    bool run() {
        ck(cudaGraphLaunch(graph[0], stream), "old graph");
        ck(cudaGraphLaunch(graph[1], stream), "new graph");
        ck(cudaStreamSynchronize(stream), "sync graphs");
        int hng = 0;
        ck(cudaMemcpy(&hng, ng, 4, cudaMemcpyDeviceToHost), "group count");
        if (hng < 1 || hng > groups) throw std::runtime_error("invalid device group count");
        const size_t fa = (((size_t) cap * w.L.n_ff * 4 + 255) & ~(size_t) 255);
        const size_t f = (size_t) cap * w.L.n_ff * 4;
        bool ok = true;
        ok &= compare("gate", scratch[0], scratch[1], f);
        ok &= compare("up", scratch[0] + fa, scratch[1] + fa, f);
        ok &= compare("q8_1", scratch[0] + 3 * fa, scratch[1] + 3 * fa,
                      (size_t) cap * (w.L.n_ff / 32) * 36);
        ok &= compare("down", out[0], out[1], (size_t) cap * w.L.n_embd * 4);
        ok &= compare("combined MoE", combined[0], combined[1], (size_t) T * w.L.n_embd * 4);
        std::printf("layer=%d type=%d/%d rank=%d T=%d groups=%d bits=%s\n", w.layer, w.L.gu_type, w.L.d_type,
                    w.rank, T, hng, ok ? "OK" : "FAIL");
        return ok;
    }
    float us(int which) {
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a), "event"); ck(cudaEventCreate(&b), "event");
        for (int i = 0; i < 5; ++i) ck(cudaGraphLaunch(graph[which], stream), "warm graph");
        ck(cudaStreamSynchronize(stream), "warm sync");
        ck(cudaEventRecord(a, stream), "record a");
        for (int i = 0; i < 40; ++i) ck(cudaGraphLaunch(graph[which], stream), "timed graph");
        ck(cudaEventRecord(b, stream), "record b");
        ck(cudaEventSynchronize(b), "event sync");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
        (void) cudaEventDestroy(a); (void) cudaEventDestroy(b);
        return ms * 1000.f / 40.f;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: hip_tpx_engine_call MODEL.gguf [--layer N] [--rank 0|1] [--device N] [--T 1..4] [--all]\n"); return 2; }
    int layer = -1, rank = -1, device = 0, only_T = 0;
    bool all = false;
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--all") { all = true; continue; }
        if (i + 1 == argc) return 2;
        if (flag == "--layer") layer = std::atoi(argv[++i]);
        else if (flag == "--rank") rank = std::atoi(argv[++i]);
        else if (flag == "--device") device = std::atoi(argv[++i]);
        else if (flag == "--T") only_T = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown flag: %s\n", argv[i]); return 2; }
    }
    if (rank > 1 || only_T > 4 || device < 0) return 2;
    try {
        const auto model = strata::GgufModel::open(argv[1]);
        strata::core::ModelGeometry g;
        int64_t topk = 0;
        std::string err;
        if (!strata::core::geometry_from_gguf(model.meta(), g, topk, err)) throw std::runtime_error(err);
        if (g.n_expert != 256 || topk != 8) throw std::runtime_error("engine route_group supports 256 experts, top-8 only");
        ck(cudaSetDevice(device), "set device");
        cudaStream_t s;
        ck(cudaStreamCreate(&s), "stream");
        bool ok = true;
        bool seen18 = false, seen23 = false;
        for (int li = 0; li < g.n_layers + g.n_mtp_layers; ++li) {
            if (layer >= 0 && li != layer) continue;
            const auto* t = model.find("blk." + std::to_string(li) + ".ffn_gate_exps.weight");
            if (!t || (layer < 0 && t->type != 18 && t->type != 23)) continue;
            if (layer < 0 && !all && ((t->type == 18 && seen18) || (t->type == 23 && seen23))) continue;
            seen18 |= t->type == 18;
            seen23 |= t->type == 23;
            for (int r = 0; r < 2; ++r) {
                if (rank >= 0 && r != rank) continue;
                Weight w(model, g, li, r);
                for (int T = 1; T <= 4; ++T) {
                    if (only_T && T != only_T) continue;
                    for (bool repeated : {false, true}) {
                        Call call(w, T, (int) topk, (int) g.n_expert, s, repeated);
                        ok &= call.run();
                        const float old_us = call.us(0), new_us = call.us(1);
                        std::printf("  route+quant+expert %s: %.1f -> %.1f us\n", repeated ? "repeat" : "varied",
                                    old_us, new_us);
                    }
                }
            }
        }
        ck(cudaStreamDestroy(s), "destroy stream");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hip_tpx_engine_call: %s\n", e.what());
        return 2;
    }
}
