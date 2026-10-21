// Two equal qwen35moe tensor shards. Host-only: usable by the loader and CPU tests.
#pragma once
#include "strata/core/family_layout.hpp"
#include <stdexcept>

namespace q36tp {
inline bool enabled(const char* name = "STRATA_Q36_TP") {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, "1") == 0;
}
struct Range { size_t begin, size; };
struct Slice {
    int axis = -1;                       // -1 mirrored; GGUF axes 0 inputs, 1 outputs
    std::vector<Range> ranges;
    size_t size() const { size_t n = 0; for (auto r : ranges) n += r.size; return n; }
};
struct Plan {
    strata::core::ModelGeometry full, local;
    explicit Plan(const strata::core::ModelGeometry& g) : full(g), local(g) {
        auto half = [](int64_t& n) {
            if (n <= 0 || n % 2) throw std::runtime_error("TP needs positive even head and FF dimensions");
            n /= 2;
        };
        half(local.n_head); half(local.n_head_kv); half(local.n_ff); half(local.n_ff_shexp);
        half(local.ssm_k_heads); half(local.ssm_v_heads); half(local.ssm_value_dim);
        local.ssm_conv_channels = 2 * local.ssm_state_size * local.ssm_k_heads + local.ssm_value_dim;
        if (g.n_vocab % 2) throw std::runtime_error("TP needs an even vocabulary size");
    }
    // Repeated segments are independently divided, then concatenated in original order. In Qwen3.5, V head h
    // uses K head h % HK: splitting a contiguous half of V would attach it to the wrong K heads.
    Slice slice(const std::string& name, int rank) const {
        if (rank < 0 || rank > 1) throw std::runtime_error("TP rank must be zero or one");
        auto seg = [rank](int axis, size_t width, size_t repeat = 1) {
            if (width % 2) throw std::runtime_error("TP segment cannot be divided equally");
            Slice s; s.axis = axis;
            for (size_t j = 0; j < repeat; ++j) s.ranges.push_back({j * width + rank * (width / 2), width / 2});
            return s;
        };
        const auto p = name.find('.', name.find('.') + 1);
        const std::string key = name.rfind("blk.", 0) == 0 ? name.substr(p + 1) : name;
        const auto& g = full;
        const size_t kd = g.ssm_state_size * g.ssm_k_heads, ratio = g.ssm_v_heads / g.ssm_k_heads;
        if (key == "attn_qkv.weight") {
            const int64_t layer = std::strtoll(name.c_str() + 4, nullptr, 10);
            if (layer == g.n_layers || strata::core::is_qsa_layer(g, layer)) {
                const size_t qg = 2 * g.n_head * g.head_dim, kv = g.n_head_kv * g.head_dim;
                Slice s = seg(1, qg);
                for (auto r : seg(1, kv, 2).ranges) s.ranges.push_back({qg + r.begin, r.size});
                return s;
            }
            return seg(1, kd, 2 + ratio);
        }
        if (key == "ssm_conv1d.weight") return seg(1, kd, 2 + ratio);
        if (key == "attn_gate.weight") return seg(1, kd, ratio);
        if (key == "ssm_out.weight") return seg(0, kd, ratio);
        if (key == "ssm_alpha.weight" || key == "ssm_beta.weight") return seg(1, g.ssm_k_heads, ratio);
        if (key == "ssm_dt.bias" || key == "ssm_a") return seg(0, g.ssm_k_heads, ratio);
        // Separate full-attention Q stores interleaved [q_head | gate_head] pairs; each shard owns whole KV groups.
        if (key == "attn_q.weight") return seg(1, 2 * g.n_head * g.head_dim);
        if (key == "attn_k.weight" || key == "attn_v.weight") return seg(1, g.n_head_kv * g.head_dim);
        if (key == "attn_output.weight") return seg(0, g.n_head * g.head_dim);
        if (key == "ffn_gate_exps.weight" || key == "ffn_up_exps.weight") return seg(1, g.n_ff);
        if (key == "ffn_down_exps.weight") return seg(0, g.n_ff);
        if (key == "ffn_gate_shexp.weight" || key == "ffn_up_shexp.weight") return seg(1, g.n_ff_shexp);
        if (key == "ffn_down_shexp.weight") return seg(0, g.n_ff_shexp);
        if (key == "output.weight") return seg(1, g.n_vocab);
        if (key == "nextn.shared_head_head.weight" &&
            (enabled("STRATA_Q36_TP_HYBRID") || enabled("STRATA_Q36_TP_MTP_HEAD"))) return seg(1, g.n_vocab);
        return {};                      // norms, embedding, router and MTP input projection
    }
};
// Copy native blocks without dequantization. Each input-column interval must contain whole quant blocks.
inline std::vector<uint8_t> gather(strata::TensorInfo& t, const uint8_t* src, const Slice& s) {
    if (s.axis < 0) return {};
    int be = 0, bb = 0;
    if (!strata::block_geometry(t.type, be, bb) || t.shape.empty() || t.shape[0] % be)
        throw std::runtime_error("TP unsupported quant layout: " + t.name);
    if ((size_t) s.axis >= t.shape.size()) throw std::runtime_error("TP missing tensor axis: " + t.name);
    for (auto r : s.ranges) {
        if (r.begin + r.size > t.shape[s.axis] || (s.axis == 0 && (r.begin % be || r.size % be)))
            throw std::runtime_error("TP shard crosses a quant block or tensor boundary: " + t.name);
    }
    const size_t old_axis = t.shape[s.axis];
    size_t inner = s.axis == 0 ? (size_t) bb : t.shape[0] / be * bb;
    const size_t unit = s.axis == 0 ? (size_t) be : 1;
    size_t outer = 1;
    for (size_t i = s.axis + 1; i < t.shape.size(); ++i) outer *= t.shape[i];
    std::vector<uint8_t> dst(outer * s.size() / unit * inner);
    uint8_t* to = dst.data();
    for (size_t o = 0; o < outer; ++o)
        for (auto r : s.ranges) {
            const size_t nb = r.size / unit * inner;
            std::memcpy(to, src + (o * old_axis + r.begin) / unit * inner, nb);
            to += nb;
        }
    t.shape[s.axis] = s.size();
    return dst;
}
} // namespace q36tp
