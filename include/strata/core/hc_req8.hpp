// include/strata/core/hc_req8.hpp - STRATA_HC_REQ8's quantizer: int8 codes and one fp32 scale per 32 contiguous values,
// Q8_0's rule (d = amax / 127, q = round-to-nearest-even(x / d), as x * (1 / d)).  One definition for the loader
// (WeightTable::load_hc_q8 and the layer split's decode copies) and hc_i8_bench, so the bench reads the engine's codes.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace strata::core {

/// `n` (a multiple of 32) values -> n codes in `q`, n / 32 scales in `sc`.
inline void hc_req8_quantize(const float* src, size_t n, int8_t* q, float* sc) {
    for (size_t b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(src[b * 32 + j]));
        const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
        sc[b] = d;
        for (int j = 0; j < 32; ++j) q[b * 32 + j] = (int8_t) std::nearbyint(src[b * 32 + j] * id);
    }
}

}  // namespace strata::core
