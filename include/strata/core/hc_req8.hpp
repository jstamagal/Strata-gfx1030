// include/strata/core/hc_req8.hpp - STRATA_HC_REQ8's quantizer: int8 codes and one fp32 scale per 32 contiguous values,
// Q8_0's rule (d = amax / 127, q = round-to-nearest-even(x / d), as x * (1 / d)); and STRATA_HC_REQ8=2's (the layer split
// only): one fp16 scale per 16 values, the same 1.125 bytes a value (d = fp16(amax / 127), q from the rounded d, clamped
// to +-127).  One definition for the loader (WeightTable::load_hc_q8 and the layer split's decode copies) and hc_i8_bench,
// so the bench reads the engine's codes.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <cstring>

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

/// float -> IEEE binary16 bits, round to nearest even (subnormals, overflow to inf)
inline uint16_t hc_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return (uint16_t) (sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (ax >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);   // 65520 and up round to inf
    if (ax >= 0x38800000u) {                                     // a normal half (2^-14 and up)
        const uint32_t m = ax - 0x38000000u;                     // the exponent rebased from 127 to 15
        return (uint16_t) (sign | ((m + 0x0fffu + ((m >> 13) & 1u)) >> 13));
    }
    float a;
    std::memcpy(&a, &ax, 4);
    return (uint16_t) (sign | (uint32_t) std::nearbyint(a * 16777216.0f));   // subnormal: units of 2^-24 (1024 = 2^-14)
}
/// IEEE binary16 bits -> float (exact)
inline float hc_f16_value(uint16_t h) {
    const uint32_t e = (h >> 10) & 31u, m = h & 1023u;
    float v;
    if (e == 0) {
        v = std::ldexp((float) m, -24);
    } else if (e == 31) {
        v = m != 0 ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    } else {
        const uint32_t x = ((e + 112u) << 23) | (m << 13);
        std::memcpy(&v, &x, 4);
    }
    return (h & 0x8000u) != 0 ? -v : v;
}
/// STRATA_HC_REQ8=2: `n` (a multiple of 16) values -> n codes in `q`, n / 16 fp16 scales (their bits) in `sc`
inline void hc_req8_quantize16(const float* src, size_t n, int8_t* q, uint16_t* sc) {
    for (size_t b = 0; b < n / 16; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(src[b * 16 + j]));
        sc[b] = hc_f16_bits(amax / 127.f);
        const float d = hc_f16_value(sc[b]), id = d > 0.f ? 1.f / d : 0.f;
        for (int j = 0; j < 16; ++j)
            q[b * 16 + j] = (int8_t) std::clamp(std::nearbyint(src[b * 16 + j] * id), -127.f, 127.f);
    }
}

}  // namespace strata::core
