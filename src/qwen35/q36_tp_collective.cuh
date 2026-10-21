// Mapped-host transport for two devices. No peer-VRAM polling (not coherent on gfx1030).
// Based on the owner's tp-lab/ar_bench.hip. The consumed handshake also prevents reuse while the peer is reading.
#pragma once
namespace q36tp {
constexpr int join_blocks = 8;
constexpr int flag_stride = 16; // independent cache lines, not a model dimension
#if defined(__HIPCC__)
__device__ inline unsigned acquire(unsigned* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline void release(unsigned* p, unsigned v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline void pause() { __builtin_amdgcn_s_sleep(4); }
#else
__device__ inline unsigned acquire(unsigned* p) { unsigned v = *(volatile unsigned*)p; __threadfence_system(); return v; }
__device__ inline void release(unsigned* p, unsigned v) { __threadfence_system(); *(volatile unsigned*)p = v; }
__device__ inline void pause() {}
#endif
__global__ void join(float* dst, float* mine, const float* peer, size_t n, unsigned* my_flags,
                     unsigned* peer_flags, unsigned* token, int rank) {
    __shared__ unsigned call;
    if (!threadIdx.x) call = token[blockIdx.x] + 1;
    __syncthreads();
    const size_t i0 = (size_t) blockIdx.x * blockDim.x + threadIdx.x, step = (size_t) gridDim.x * blockDim.x;
    const size_t nv = n / 4;
    for (size_t i = i0; i < nv; i += step) ((float4*)mine)[i] = ((const float4*)dst)[i];
    for (size_t i = nv * 4 + i0; i < n; i += step) mine[i] = dst[i];
    __threadfence_system();
    __syncthreads();
    const int f = blockIdx.x * flag_stride;
    if (!threadIdx.x) {
        release(my_flags + f, call);
        while (acquire(peer_flags + f) != call) pause();
    }
    __syncthreads();
    __threadfence_system();
    for (size_t i = i0; i < nv; i += step) {
        float4 a = ((const float4*)dst)[i];
#if defined(__HIPCC__)
        typedef float v4 __attribute__((ext_vector_type(4)));
        const v4 v = __builtin_nontemporal_load((const v4*)peer + i);
        const float4 b = make_float4(v.x, v.y, v.z, v.w);
#else
        const volatile float* p = peer + i * 4;
        const float4 b = make_float4(p[0], p[1], p[2], p[3]);
#endif
        if (rank == 0) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
        else { a.x = b.x + a.x; a.y = b.y + a.y; a.z = b.z + a.z; a.w = b.w + a.w; }
        ((float4*)dst)[i] = a;
    }
    for (size_t i = nv * 4 + i0; i < n; i += step) {
        const float other = ((const volatile float*)peer)[i];
        dst[i] = rank == 0 ? dst[i] + other : other + dst[i];
    }
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) {
        release(my_flags + f + 1, call);
        while (acquire(peer_flags + f + 1) != call) pause();
        token[blockIdx.x] = call;
    }
}
__global__ void partial_moe(float* out, const float* parts, const int32_t* prow, const float* wts, int picks,
                            const float* shared, const float* sgate, int sg_stride, int n) {
    const int t = blockIdx.x;
    const float* wr = wts + (size_t) t * picks;
    const float gs = 1.0f / (1.0f + expf(-sgate[(size_t) t * sg_stride]));
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const size_t first = prow ? prow[t * picks] : t * picks;
        float acc = wr[0] * parts[first * n + i];
        for (int k = 1; k < picks; ++k) {
            const size_t row = prow ? prow[t * picks + k] : t * picks + k;
            acc = fmaf(wr[k], parts[row * n + i], acc);
        }
        out[(size_t) t * n + i] = acc + gs * shared[(size_t) t * n + i];
    }
}
__global__ void ordered_sum(float* mine, const float* peer, size_t n, int rank) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) mine[i] = rank == 0 ? mine[i] + peer[i] : peer[i] + mine[i];
}
} // namespace q36tp
