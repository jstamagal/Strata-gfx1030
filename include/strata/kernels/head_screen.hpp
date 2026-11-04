#pragma once
#include <cstddef>
#include <cstdint>
#include "strata/kernels/sampler.hpp"

namespace strata::kernels {
inline constexpr int kHeadScreenGrid = 1024;

// HIP wave32, GGUF Q8_0/Q6_K rows, 2560 weights, 1..8 columns.
// The copy holds Q8_0's upper six bits, Q6_K's upper five, or a certified fitted Q6_K proxy.
// Original weights remain untouched for every exact consumer.
bool head_screen_supported(int type, int n_in, int n_out, int ncols);
bool head_screen_enabled(int type);
int head_screen_max_cols();
// Anything able to promote a tail row or read the full distribution requests exact logits.
inline int head_screen_keep(const SamplerParams& p, bool sampling, bool full_read = false) {
    if (full_read || p.logit_bias || (p.penalty_last_n > 0 &&
        (p.penalty_repeat != 1.0f || p.penalty_freq != 0.0f || p.penalty_present != 0.0f))) return 0;
    if (!sampling || p.greedy || p.temperature <= 0.0f) return 1;
    return p.top_k >= 1 && p.top_k <= 64 ? p.top_k : 0;
}
std::size_t head_screen_copy_bytes(int type, int n_out);
std::size_t head_screen_scratch_bytes();
void head_screen_pack(int type, const void* weights, int n_out, void* copy, void* stream);

// *keep: 1..64 for exact top-K, 0 for every logit exact; device word, read on each replay.
// Scratch is private to this stream/window; no initialization is needed. Capturable.
// Every row whose upper bound reaches its column's threshold is rescored, including ties.
// The remaining logits are upper bounds STRICTLY below the exact K-th value, not probabilities.
// Bit planes use an inflated residual L1 bound; the fitted proxy uses residual/magnitude L2 norms.
// Nonfinite bounds rescore.
void head_screen(int type, const void* copy, const void* weights, const void* x_q8_1,
                 float* logits, int n_out, int ncols, const int32_t* keep, void* scratch, void* stream);

// Fitted two-bit proposal head. Only draft ids/probabilities may change; never use for target logits.
// Unset: gfx1030 Q8_0/2560 only. 0: exact. 1: all explicitly supported formats.
bool head_proxy_enabled(int type, int n_in, int device = -1);
std::size_t head_proxy_reserve_bytes(int type, int n_in, int n_rows, int device = -1, bool owns_proxy = true);
bool head_proxy_supported(int type, int n_in);
std::size_t head_proxy_bytes(int n_rows);
std::size_t head_proxy_scratch_bytes(int n_rows);
void head_proxy_pack(int type, const void* weights, int n_rows, void* proxy, void* stream);
// Replaces the approximate top rows by exact scores from the unchanged main-head subset.
// Writes a subset index (caller maps it), and a probability with an approximate softmax tail.
void head_proxy_pick(int type, const void* proxy, const void* weights, const void* x_q8_1,
                     int n_rows, float* logits, void* scratch, int32_t* pick, float* probability, void* stream);

// Certified Q6_K alternative (STRATA_HEAD_Q6_BITS=2): the fitted copy followed by two row norms.
// ||w-w_proxy||*||x|| bounds truncation; a separate magnitude norm bounds both FP32 evaluations.
std::size_t head_proxy_bound_bytes(int n_rows);
void head_proxy_bound_pack(const void* weights, int n_rows, void* proxy, void* stream);
void head_proxy_screen(const void* proxy, const void* x_q8_1, float* logits, float* block_lower,
                       int n_rows, int ncols, const int32_t* keep, void* stream);

} // namespace strata::kernels
