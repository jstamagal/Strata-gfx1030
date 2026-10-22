// q36's process-wide RoPE choice. The numerical arithmetic is Flash-Next's
// rope_scaling.hpp helper, transcribed there from llama.cpp's rope_yarn.
#include "strata/kernels/rope_scaling.hpp"

namespace q36rope {
struct Options {
    std::string type; // empty: GGUF, then none
    double factor = 0.0;
    double freq_scale = 0.0;
};

inline bool resolve(const strata::GgufFile& meta, float base, const Options& opt,
                    strata::kernels::RopeScaling& sc, std::string& err) {
    using Type = strata::kernels::RopeScalingType;
    const std::string p = "qwen35moe.";
    sc.freq_base = base;
    std::string type;
    if (const auto* v = meta.get(p + "rope.scaling.type")) type = v->s;
    if (!opt.type.empty()) type = opt.type;
    if (type == "" || type == "none") sc.type = Type::None;
    else if (type == "linear") sc.type = Type::Linear;
    else if (type == "yarn") sc.type = Type::YaRN;
    else { err = "rope.scaling.type must be none, linear or yarn"; return false; }
    if (const auto* v = meta.get(p + "rope.scaling.factor"); v && v->is_num() && v->num() > 0)
        sc.factor = v->num();
    if (opt.factor != 0.0) sc.factor = opt.factor;
    if (const auto* v = meta.get(p + "context_length"); v && v->is_num() && v->num() > 0)
        sc.orig_ctx = v->num();
    if (const auto* v = meta.get(p + "rope.scaling.original_context_length"); v && v->is_num() && v->num() > 0)
        sc.orig_ctx = v->num();
    sc.freq_scale_in = opt.freq_scale;
    sc.ext_factor = sc.type == Type::YaRN ? 1.0 : 0.0;
    if (sc.type == Type::None) {
        const double freq_base = sc.freq_base;
        sc = strata::kernels::RopeScaling{};
        sc.freq_base = freq_base;
    }
    if (const char* why = strata::kernels::rope_scaling_invalid(sc)) {
        err = std::string("invalid RoPE scaling: ") + why;
        return false;
    }
    return true;
}
} // namespace q36rope
