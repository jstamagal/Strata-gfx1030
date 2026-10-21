// CPU-only shard coverage / byte reconstruction and Qwen3.5 head-association test.
// c++ -std=c++17 -Iinclude -Isrc/qwen35 tests/qwen35/tp_plan.cpp -o /tmp/q36-tp-plan && /tmp/q36-tp-plan
#include "q36_tp_plan.hpp"
#include <cassert>
#include <iostream>

int main() {
    strata::core::ModelGeometry g;
    g.n_embd = 3072; g.n_head = 32; g.n_head_kv = 2; g.head_dim = 256;
    g.n_ff = 1024; g.n_ff_shexp = 1024; g.n_vocab = 248320;
    g.ssm_state_size = 128; g.ssm_k_heads = 16; g.ssm_v_heads = 32; g.ssm_value_dim = 4096;
    q36tp::Plan p(g);
    for (int rank = 0; rank < 2; ++rank) {
        auto v = p.slice("blk.0.attn_gate.weight", rank);
        auto a = p.slice("blk.0.ssm_alpha.weight", rank);
        auto qkv = p.slice("blk.0.attn_qkv.weight", rank);
        size_t hlocal = 0;
        for (auto r : v.ranges) for (size_t h = r.begin / g.ssm_state_size; h < (r.begin + r.size) / g.ssm_state_size; ++h) {
            assert(h % g.ssm_k_heads == rank * p.local.ssm_k_heads + hlocal % p.local.ssm_k_heads);
            const auto ar = a.ranges[hlocal / p.local.ssm_k_heads];
            assert(h == ar.begin + hlocal % p.local.ssm_k_heads);
            ++hlocal;
        }
        assert(qkv.size() == (size_t) p.local.ssm_conv_channels);
        assert(p.slice("blk.4.attn_q.weight", rank).size() == (size_t) (2 * p.local.n_head * g.head_dim));
    }
    // Reconstruct a multi-expert tensor exactly from both shards, for every native expert format in the test file.
    for (int type : {0, 1, 8, 11, 12, 16, 18, 19, 20, 21, 23}) for (int axis : {0, 1}) {
        strata::TensorInfo full;
        full.name = axis ? "blk.0.ffn_gate_exps.weight" : "blk.0.ffn_down_exps.weight";
        full.type = type;
        full.shape = axis ? std::vector<uint64_t>{(uint64_t) g.n_embd, (uint64_t) g.n_ff, 3}
                          : std::vector<uint64_t>{(uint64_t) g.n_ff, (uint64_t) g.n_embd, 3};
        std::vector<uint8_t> original(strata::tensor_payload_bytes(full)), restored(original.size(), 0);
        for (size_t i = 0; i < original.size(); ++i) original[i] = (uint8_t) ((i * 31 + i / 251) % 256);
        int be, bb; assert(strata::block_geometry(type, be, bb));
        for (int rank = 0; rank < 2; ++rank) {
            auto t = full;
            auto s = p.slice(t.name, rank);
            auto data = q36tp::gather(t, original.data(), s);
            assert(data.size() * 2 == original.size());
            // Scatter by global quant-block coordinates, independently of gather's byte-stride computation.
            const size_t old_cols = full.shape[0] / be, new_cols = t.shape[0] / be;
            for (size_t e = 0; e < 3; ++e) for (size_t row = 0; row < t.shape[1]; ++row)
                for (size_t col = 0; col < new_cols; ++col) {
                    const size_t rr = axis ? row + rank * t.shape[1] : row;
                    const size_t cc = axis ? col : col + rank * new_cols;
                    std::memcpy(restored.data() + ((e * full.shape[1] + rr) * old_cols + cc) * bb,
                                data.data() + ((e * t.shape[1] + row) * new_cols + col) * bb, bb);
                }
        }
        assert(original == restored);
    }
    strata::TensorInfo bad; bad.name = "bad"; bad.type = 12; bad.shape = {1024, 2};
    bool refused = false;
    try { q36tp::gather(bad, nullptr, {0, {{128, 512}}}); } catch (const std::runtime_error&) { refused = true; }
    assert(refused);
    std::cout << "TP plan: head association, expert byte reconstruction and quant alignment passed\n";
}
