#pragma once
#include "strata/kernels/iq_kernels.hpp"
namespace strata::kernels {
// q36-only entry point; shared engine dispatch and defaults are unchanged.
bool q36_expert_decode(const NativeExpertLayout& L, const unsigned long long* ptr, const int32_t* start,
    const int32_t* groups, const int32_t* dst, const int32_t* tok, int64_t cap_groups, int64_t cap_entries,
    const void* x, void* scratch, float* out, void* stream, bool rows2, bool fuse);
}
