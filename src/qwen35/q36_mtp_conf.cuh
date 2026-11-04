#pragma once

// The Gumbel draft buffer holds logit / temperature + G for the sampler's top-k
// tokens and -infinity elsewhere. Recover their unperturbed scores to measure
// the probability of the actual proposed token, not the top logit's probability.
// This is diagnostic/adaptive control only; the target sampler still decides text.
__global__ void q36_mtp_picked_prob(const float* scores, int n, const int32_t* picked,
                                    uint64_t seed, int64_t pos, float* out) {
    __shared__ float red[1024];
    const int choice = *picked;
    float chosen = -INFINITY;
    if (choice >= 0 && choice < n && isfinite(scores[choice]))
        chosen = scores[choice] + logf(-logf(q36_unif(seed, pos, choice)));
    float part = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (!isfinite(scores[i])) continue;
        const float original = scores[i] + logf(-logf(q36_unif(seed, pos, i)));
        part += expf(original - chosen);
    }
    red[threadIdx.x] = part;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = chosen == -INFINITY ? 0.f : 1.f / red[0];
}
