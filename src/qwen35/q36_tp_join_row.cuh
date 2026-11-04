// Optional row-owned mapped-host join. A row stays in one CTA, so its RMS reduction
// needs no cross-CTA token barrier. The producer variant publishes MoE output in place.
#pragma once
namespace q36tp_row {

__global__ void partial_moe_publish(float* out, float* mapped, const float* parts,
        const int32_t* prow, const float* wts, int picks, const float* shared,
        const float* sgate, int sg_stride, int n, unsigned* token) {
    const int t = blockIdx.x;
    const size_t stride = (size_t) n + 16;
    float* mine = mapped + (size_t) t * stride + 4;
    unsigned* my_flags = (unsigned*)(mapped + (size_t) t * stride);
    const float* wr = wts + (size_t) t * picks;
    const float gs = 1.0f / (1.0f + expf(-sgate[(size_t) t * sg_stride]));
    const size_t first = prow ? prow[t * picks] : t * picks;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * parts[first * n + i];
        for (int k = 1; k < picks; ++k) {
            const size_t row = prow ? prow[t * picks + k] : t * picks + k;
            acc = fmaf(wr[k], parts[row * n + i], acc);
        }
        const float v = acc + gs * shared[(size_t) t * n + i];
        out[(size_t) t * n + i] = v;
        mine[i] = v;
    }
    // Every writing lane must drain its own mapped-host writes before the CTA's flag is published.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
        q36tp::release(my_flags, token[t] + 1);
}

template <bool Q, bool Published, bool DeferAck = false, bool Skew = false>
__global__ void __launch_bounds__(256) join_norm(float* partial, float* mapped, const float* peer_mapped,
        unsigned* token, int rank,
        float* x, const float* w, float* h, q36k::QB* hq, int n, float eps,
        q36tp_skew::Sample* samples, unsigned* counts, int skew_slot) {
    const int row = blockIdx.x, j = threadIdx.x;
    const size_t base = (size_t) row * n;
    const size_t packed = (size_t) row * ((size_t) n + 16);
    float* mine = mapped + packed + 4;
    const float* peer = peer_mapped + packed + 4;
    unsigned* my_flags = (unsigned*)(mapped + packed);
    unsigned* peer_flags = (unsigned*)(peer_mapped + packed);
    __shared__ unsigned call;
    __shared__ float sh[256];
    if (j == 0) call = token[row] + 1;
    __syncthreads();
    __shared__ unsigned sample_index;
    __shared__ unsigned long long arrived, published_at, peer_at;
    if constexpr (Skew) {
        if (j == 0) {
            sample_index = atomicAdd(counts + skew_slot, 1u) % q36tp_skew::samples_per_join;
            arrived = q36tp_skew::stamp();
        }
        __syncthreads();
    }
    if constexpr (!Published) {
        if constexpr (DeferAck) {
            if (j == 0) while (q36tp::acquire(peer_flags + 1) < call - 1) q36tp::pause();
            __syncthreads();
        }
        const int n4 = n / 4;
        for (int i = j; i < n4; i += 256)
            ((float4*) mine)[i] = ((const float4*) (partial + base))[i];
        for (int i = n4 * 4 + j; i < n; i += 256) mine[i] = partial[base + i];
        __threadfence_system();
        __syncthreads();
    }
    if (j == 0) {
        if constexpr (!Published) q36tp::release(my_flags, call);
        if constexpr (Skew) published_at = q36tp_skew::stamp();
        while (q36tp::acquire(peer_flags) != call) q36tp::pause();
        if constexpr (Skew) peer_at = q36tp_skew::stamp();
    }
    __syncthreads();
    // The flag was acquired by lane zero; make the peer's mapped writes visible to the CTA.
    __threadfence_system();
    const int n4 = n / 4;
    for (int i = j; i < n4; i += 256) {
        float4 a = ((const float4*) (partial + base))[i];
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
        ((float4*) (partial + base))[i] = a;
    }
    for (int i = n4 * 4 + j; i < n; i += 256) {
        const float b = ((const volatile float*) peer)[i];
        partial[base + i] = rank == 0 ? partial[base + i] + b : b + partial[base + i];
    }
    __syncthreads();
    // Once every lane has read the peer row, the peer may reuse that mapped row.
    // The residual/norm below touches only device-local memory.
    if constexpr (DeferAck) {
        if (j == 0) q36tp::release(my_flags + 1, call);
    }
    float* xr = x + base;
    const float* yr = partial + base;
    float* hr = h + base;
    float s = 0.f;
    for (int i = j; i < n; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    sh[j] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (j < o) sh[j] += sh[j + o];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    if constexpr (Q) {
        q36k::QB* qr = hq + (size_t) row * (n >> 5);
        for (int i = j; i < n; i += 256) {
            const float hv = __fmul_rn(__fmul_rn(xr[i], r), w[i]);
            hr[i] = hv;
            q36k::q8_store(qr, i, ((volatile float*) hr)[i], j & 31);
        }
    } else {
        for (int i = j; i < n; i += 256) hr[i] = xr[i] * r * w[i];
    }
    __syncthreads();
    if (j == 0) {
        if constexpr (!DeferAck) {
            q36tp::release(my_flags + 1, call);
            while (q36tp::acquire(peer_flags + 1) != call) q36tp::pause();
        }
#if defined(__HIPCC__)
        __hip_atomic_store(token + row, call, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
        atomicExch(token + row, call);
#endif
        if constexpr (Skew) {
            q36tp_skew::Sample* s = samples +
                (size_t) skew_slot * q36tp_skew::samples_per_join + sample_index;
            *s = {arrived, published_at, peer_at, q36tp_skew::stamp()};
        }
    }
}

template <bool Q, bool Published, bool DeferAck>
inline void launch_join_norm(float* partial, float* mine, const float* peer, unsigned* token, int rank,
        float* x, const float* w, float* h, q36k::QB* hq, int n, float eps, int T, cudaStream_t stream,
        q36tp_skew::Sample* samples, unsigned* counts, int skew_slot) {
    if (samples && T == 1)
        join_norm<Q, Published, DeferAck, true><<<T, 256, 0, stream>>>(
            partial, mine, peer, token, rank, x, w, h, hq, n, eps, samples, counts, skew_slot);
    else
        join_norm<Q, Published, DeferAck, false><<<T, 256, 0, stream>>>(
            partial, mine, peer, token, rank, x, w, h, hq, n, eps, nullptr, nullptr, 0);
}
} // namespace q36tp_row
