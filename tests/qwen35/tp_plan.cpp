// CPU-only shard coverage / byte reconstruction and Qwen3.5 head-association test.
// c++ -std=c++17 -Iinclude -Isrc/qwen35 tests/qwen35/tp_plan.cpp -o /tmp/q36-tp-plan && /tmp/q36-tp-plan
#include "q36_tp_plan.hpp"
#include <cassert>
#include <iostream>

static std::vector<size_t> indices(const q36tp::Slice& s) {
    std::vector<size_t> out;
    for (auto r : s.ranges) for (size_t i = 0; i < r.size; ++i) out.push_back(r.begin + i);
    return out;
}

static void heads(const strata::core::ModelGeometry& g) {
    const q36tp::Plan p(g);
    const size_t D = g.ssm_state_size, HK = g.ssm_k_heads, HV = g.ssm_v_heads;
    for (int rank = 0; rank < 2; ++rank) {
        const auto qkv = indices(p.slice("blk.0.attn_qkv.weight", rank));
        const auto conv = indices(p.slice("blk.0.ssm_conv1d.weight", rank));
        const auto gate = indices(p.slice("blk.0.attn_gate.weight", rank));
        const auto out = indices(p.slice("blk.0.ssm_out.weight", rank));
        const auto alpha = indices(p.slice("blk.0.ssm_alpha.weight", rank));
        assert(qkv == conv && gate == out);
        for (const char* name : {"ssm_beta.weight", "ssm_dt.bias", "ssm_a"})
            assert(indices(p.slice(std::string("blk.0.") + name, rank)) == alpha);
        for (size_t h = 0; h < HV / 2; ++h) {
            // q36's recurrence uses local h % HK_local. The matching full head has the same K owner
            // and the same broadcast repetition; independent of how Plan represents its segments.
            const size_t kh = h % (HK / 2) + rank * (HK / 2);
            const size_t vh = h / (HK / 2) * HK + kh;
            assert(alpha[h] == vh);
            for (size_t d = 0; d < D; ++d) {
                assert(qkv[(h % (HK / 2)) * D + d] == kh * D + d);
                assert(qkv[(HK / 2 + h % (HK / 2)) * D + d] == (HK + kh) * D + d);
                assert(qkv[(HK + h) * D + d] == (2 * HK + vh) * D + d);
                assert(gate[h * D + d] == vh * D + d);
            }
        }
        const auto q = indices(p.slice("blk.3.attn_q.weight", rank));
        const auto kv = indices(p.slice("blk.3.attn_k.weight", rank));
        assert(kv == indices(p.slice("blk.3.attn_v.weight", rank)));
        const auto wo = indices(p.slice("blk.3.attn_output.weight", rank));
        const size_t HD = g.head_dim, G = g.n_head / g.n_head_kv;
        for (size_t h = 0; h < (size_t)p.local.n_head; ++h) {
            const size_t full_h = rank * p.local.n_head + h;
            // Q/gate pairs stay together, along with their KV group and output-projection columns.
            assert(q[2 * h * HD] == 2 * full_h * HD);
            assert(q[(2 * h + 1) * HD] == (2 * full_h + 1) * HD);
            assert(kv[(h / G) * HD] == (full_h / G) * HD);
            assert(wo[h * HD] == full_h * HD);
        }
        for (const char* name : {"attn_norm.weight", "post_attention_norm.weight", "ssm_norm.weight",
                "attn_q_norm.weight", "attn_k_norm.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
                "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight"})
            assert(p.slice(std::string("blk.0.") + name, rank).axis == -1);
        assert(p.slice("token_embd.weight", rank).axis == -1);
        assert(p.slice("output_norm.weight", rank).axis == -1);
    }
}

int main(int argc, char** argv) {
    strata::core::ModelGeometry g;
    g.n_embd = 3072; g.n_head = 32; g.n_head_kv = 2; g.head_dim = 256;
    g.n_ff = 1024; g.n_ff_shexp = 1024; g.n_vocab = 248320;
    g.ssm_state_size = 128; g.ssm_k_heads = 16; g.ssm_v_heads = 32; g.ssm_value_dim = 4096;
    for (int ratio : {1, 2, 3, 4}) {
        auto shape = g;
        shape.ssm_v_heads = ratio * shape.ssm_k_heads;
        shape.ssm_value_dim = shape.ssm_v_heads * shape.ssm_state_size;
        heads(shape);
    }
    if (argc > 1) {
        const auto model = strata::GgufModel::open(argv[1]);
        int64_t topk;
        std::string err;
        assert(strata::core::geometry_from_gguf(model.meta(), g, topk, err));
        assert(strata::core::check_gguf_layout(model, g, err));
        const q36tp::Plan plan(g);
        heads(g);
        size_t checked = 0;
        for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l) {
            for (const auto& want : strata::core::layer_tensors(plan.local, l)) {
                const auto name = "blk." + std::to_string(l) + "." + want.name;
                const auto* t = model.find(name);
                assert(t);
                int be, bb; assert(strata::block_geometry(t->type, be, bb));
                for (int rank = 0; rank < 2; ++rank) {
                    auto shape = t->shape;
                    const auto s = plan.slice(name, rank);
                    if (s.axis >= 0) {
                        shape[s.axis] = s.size();
                        for (auto r : s.ranges) {
                            assert(r.begin + r.size <= t->shape[s.axis]);
                            assert(s.axis != 0 || (r.begin % be == 0 && r.size % be == 0));
                        }
                    }
                    assert(shape.size() == want.shape.size());
                    for (size_t i = 0; i < shape.size(); ++i) assert(shape[i] == (size_t)want.shape[i]);
                }
                ++checked;
            }
        }
        std::cout << "Actual GGUF: " << checked << " layer tensors match the local engine geometry; HK="
                  << g.ssm_k_heads << " HV=" << g.ssm_v_heads << "\n";
    }
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
        const auto fused = p.slice("blk.3.attn_qkv.weight", rank);
        assert(fused.ranges.size() == 3);
        assert(fused.ranges[0].begin == (size_t)(rank * g.n_head * g.head_dim));
        assert(fused.ranges[1].begin == (size_t)(2 * g.n_head * g.head_dim + rank * g.n_head_kv * g.head_dim / 2));
        assert(fused.ranges[2].begin == fused.ranges[1].begin + (size_t)(g.n_head_kv * g.head_dim));
        assert(p.slice("blk.4.attn_q.weight", rank).size() == (size_t) (2 * p.local.n_head * g.head_dim));
    }
    // Reconstruct a multi-expert tensor exactly from both shards, for every native expert format in the test file.
    for (int type : {0, 1, 8, 11, 12, 16, 18, 19, 20, 21, 22, 23}) for (int axis : {0, 1}) {
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
