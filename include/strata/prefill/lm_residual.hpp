// Layer-major split: upload the incoming residuals once, then keep them on the receiving GPU.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace strata::prefill::detail {
inline bool lm_split_r_vram_from(const char* value) {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

// The caller owns the pinned hand-off until the receiving stage finishes the super-chunk.
// Both streams belong to that stage's device; only the first consumer waits for this upload.
inline cudaError_t lm_stage_upload(void* dst, const void* host, size_t bytes,
                                   cudaStream_t copy, cudaStream_t compute, cudaEvent_t ready) {
    if (bytes == 0) return cudaSuccess;
    cudaError_t e = cudaMemcpyAsync(dst, host, bytes, cudaMemcpyHostToDevice, copy);
    if (e == cudaSuccess) e = cudaEventRecord(ready, copy);
    if (e == cudaSuccess) e = cudaStreamWaitEvent(compute, ready, 0);
    return e;
}
}  // namespace strata::prefill::detail
