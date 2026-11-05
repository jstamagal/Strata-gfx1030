#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace strata::platform {

// Experimental Linux HIP policy, verified against ROCm 10.3.0a20261023.
// "Immune" means no SVM/USERPTR invalidation for these host buffers. KFD GTT
// BOs still have a TTM eviction fence; this is not a guarantee against TTM,
// GPU resets, or another library explicitly registering ordinary host memory.
inline bool host_immune() {
#if defined(STRATA_USE_HIP) && defined(__linux__)
    static const bool on = [] {
        const char* e = std::getenv("STRATA_HOST_IMMUNE");
        return e && std::strcmp(e, "1") == 0;
    }();
    return on;
#else
    return false;
#endif
}

// Call before the first HIP/HSA call: libhsakmt and ROCclr cache these values.
// Both allocator switches are necessary: disabling USERPTR alone selects SVM
// in this runtime. These are process settings, not machine settings.
inline bool host_immune_bootstrap() {
#if defined(STRATA_USE_HIP) && defined(__linux__)
    if (!host_immune()) return true;
    char threshold[32];
    // ROCclr expresses the threshold in MiB and multiplies in size_t. No
    // addressable Strata buffer can exceed this threshold. Setting XFER_SIZE
    // to zero does NOT disable pinning: rocsettings.cpp clamps it to the min.
    std::snprintf(threshold, sizeof threshold, "%llu",
                  (unsigned long long) (std::numeric_limits<size_t>::max() >> 20));
    if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 1) != 0 ||
        setenv("HSA_SVM_FOR_PAGED_MEM", "0", 1) != 0 ||
        setenv("GPU_PINNED_MIN_XFER_SIZE", threshold, 1) != 0) {
        std::fprintf(stderr, "strata: STRATA_HOST_IMMUNE: cannot configure host allocation\n");
        return false;
    }
    std::fprintf(stderr, "strata: STRATA_HOST_IMMUNE=1: host allocations use GTT; pageable buffer copies use "
                         "persistent ROCclr staging; explicit host registration disabled; TTM eviction remains possible\n");
#endif
    return true;
}
} // namespace strata::platform
