// Project two verify rows together, reusing each F32 router weight load.
// Each row keeps the original warp's accumulation and shuffle order.
#pragma once
namespace q36tp_router_pair {
__global__ void __launch_bounds__(256) router_group(const float* W, const float* x, float* logits,
        int n_in, int n_out, int n_tok, int ld, int32_t* ids, float* weights,
        const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr,
        int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok,
        unsigned* done) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int cols = (n_out + 7) / 8;
    const int tk = 2 * (blockIdx.x / cols);
    const int o = (blockIdx.x % cols) * 8 + warp;
    if (o < n_out) {
        const float4* wr = (const float4*) (W + (size_t) o * n_in);
        const float4* xr0 = (const float4*) (x + (size_t) tk * n_in);
        const float4* xr1 = tk + 1 < n_tok ? (const float4*) (x + (size_t) (tk + 1) * n_in) : nullptr;
        float a0 = 0.f, a1 = 0.f;
        for (int i = lane; i < n_in / 4; i += 32) {
            const float4 w = wr[i], b0 = xr0[i];
            a0 += w.x * b0.x + w.y * b0.y + w.z * b0.z + w.w * b0.w;
            if (xr1) {
                const float4 b1 = xr1[i];
                a1 += w.x * b1.x + w.y * b1.y + w.z * b1.z + w.w * b1.w;
            }
        }
        for (int m = 16; m > 0; m >>= 1) {
            a0 += __shfl_xor_sync(0xffffffffu, a0, m);
            if (xr1) a1 += __shfl_xor_sync(0xffffffffu, a1, m);
        }
        if (lane == 0) {
            logits[(size_t) tk * n_out + o] = a0;
            if (xr1) logits[(size_t) (tk + 1) * n_out + o] = a1;
        }
    }
    __syncthreads();
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (threadIdx.x == 0) {
        const unsigned old = atomicAdd(done, 1u);
        last = old + 1u == (unsigned) gridDim.x;
        if (last) atomicExch(done, 0u);
    }
    __syncthreads();
    if (last) q36tp_fuse2::route_group_body<false>(logits, ids, weights, n_tok, ld, nullptr,
                                                   blobs, blob, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
}
} // namespace q36tp_router_pair
