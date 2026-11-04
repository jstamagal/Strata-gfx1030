// TP decode MoE partial plus mapped-host join. Each join block produces and
// publishes exactly the elements that it will later combine with its peer.
#pragma once
namespace q36tp {
__device__ inline float moe_value(size_t i, const float* parts, const int32_t* prow, const float* wts,
                                  int picks, const float* shared, const float* sgate, int sg_stride, int n) {
    const int t = (int) (i / n), col = (int) (i % n);
    const float* wr = wts + (size_t) t * picks;
    const float gs = 1.0f / (1.0f + expf(-sgate[(size_t) t * sg_stride]));
    const size_t row0 = prow ? prow[t * picks] : t * picks;
    float acc = wr[0] * parts[row0 * n + col];
    for (int k = 1; k < picks; ++k) {
        const size_t row = prow ? prow[t * picks + k] : t * picks + k;
        acc = fmaf(wr[k], parts[row * n + col], acc);
    }
    return acc + gs * shared[i];
}
__global__ void moe_join(float* out, const float* parts, const int32_t* prow, const float* wts, int picks,
                         const float* shared, const float* sgate, int sg_stride, int n, int T, float* mine,
                         const float* peer, unsigned* my_flags, unsigned* peer_flags, unsigned* token, int rank) {
    const size_t count = (size_t) T * n;
    const size_t first = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t step = (size_t) gridDim.x * blockDim.x;
    const size_t nv = count / 4;
    for (size_t i4 = first; i4 < nv; i4 += step) {
        for (int e = 0; e < 4; ++e) {
            const size_t i = i4 * 4 + e;
            const float v = moe_value(i, parts, prow, wts, picks, shared, sgate, sg_stride, n);
            out[i] = v;
            mine[i] = v;
        }
    }
    for (size_t i = nv * 4 + first; i < count; i += step) {
        const float v = moe_value(i, parts, prow, wts, picks, shared, sgate, sg_stride, n);
        out[i] = v;
        mine[i] = v;
    }
    __shared__ unsigned call;
    if (!threadIdx.x) call = token[blockIdx.x] + 1;
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    const int f = blockIdx.x * flag_stride;
    if (!threadIdx.x) {
        release(my_flags + f, call);
        while (acquire(peer_flags + f) != call) pause();
    }
    __syncthreads();
    __threadfence_system();
    for (size_t i = first; i < nv; i += step) {
        float4 a = ((const float4*) out)[i];
#if defined(__HIPCC__)
        typedef float v4 __attribute__((ext_vector_type(4)));
        const v4 v = __builtin_nontemporal_load((const v4*) peer + i);
        const float4 b = make_float4(v.x, v.y, v.z, v.w);
#else
        const volatile float* p = peer + i * 4;
        const float4 b = make_float4(p[0], p[1], p[2], p[3]);
#endif
        if (rank == 0) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
        else { a.x = b.x + a.x; a.y = b.y + a.y; a.z = b.z + a.z; a.w = b.w + a.w; }
        ((float4*) out)[i] = a;
    }
    for (size_t i = nv * 4 + first; i < count; i += step) {
        const float other = ((const volatile float*) peer)[i];
        out[i] = rank == 0 ? out[i] + other : other + out[i];
    }
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) {
        release(my_flags + f + 1, call);
        while (acquire(peer_flags + f + 1) != call) pause();
        token[blockIdx.x] = call;
    }
}
} // namespace q36tp
