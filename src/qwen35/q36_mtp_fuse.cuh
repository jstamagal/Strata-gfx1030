#pragma once

// The two MTP input norms have no dependency on each other. Use one launch with
// the same per-row reduction order as k_rms_ld, so the following eh_proj sees
// the same FP32 values as the separate launches.
__global__ void k_mtp_pair_rms(const float* embedding, const float* seed,
                               const float* embedding_weight, const float* seed_weight,
                               float* cat, int n, float eps) {
    __shared__ float sh[kThreads];
    const int row = blockIdx.x, side = blockIdx.y;
    const float* src = (side ? seed : embedding) + (size_t) row * n;
    const float* weight = side ? seed_weight : embedding_weight;
    float* dst = cat + (size_t) row * 2 * n + (size_t) side * n;
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += src[i] * src[i];
    sum = block_sum(sum, sh);
    const float scale = rsqrtf(sum / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i] * scale * weight[i];
}
