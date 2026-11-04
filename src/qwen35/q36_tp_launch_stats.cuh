// Counts actual HIP kernel submissions during an eager TP forward. The linker
// wraps the runtime entry point, including launches from the kernel libraries.
// A null pointer is the ordinary path; only TP_LAUNCH_STATS enables device queries.
#pragma once
#if defined(STRATA_Q36_HIP_LAUNCH_WRAP)
namespace q36tp_stats {
inline thread_local long long* eager = nullptr;
inline thread_local int device = 0;
}
extern "C" hipError_t __real_hipLaunchKernel(const void*, dim3, dim3, void**, size_t, hipStream_t);
extern "C" hipError_t __wrap_hipLaunchKernel(const void* f, dim3 grid, dim3 block,
                                              void** args, size_t shared, hipStream_t stream) {
    if (q36tp_stats::eager) {
        const int dev = q36tp_stats::device;
        if (dev >= 0 && dev < 2) ++q36tp_stats::eager[dev];
    }
    return __real_hipLaunchKernel(f, grid, block, args, shared, stream);
}
#endif
