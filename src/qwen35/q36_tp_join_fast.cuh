// Row-owned joins with more parallel PCIe reads. The 256 logical RMS lanes and
// reduction tree remain the same as q36tp_row::join_norm.
#pragma once
namespace q36tp_fast {

__device__ __forceinline__ float4 peer4(const float4* p) {
#if defined(__HIPCC__)
    typedef float v4 __attribute__((ext_vector_type(4)));
    const v4 v = __builtin_nontemporal_load((const v4*)p);
    return make_float4(v.x, v.y, v.z, v.w);
#else
    const volatile float* v = (const volatile float*)p;
    return make_float4(v[0], v[1], v[2], v[3]);
#endif
}

__device__ __forceinline__ float4 ordered_add(float4 a, float4 b, int rank) {
    if (rank == 0) return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    return make_float4(b.x + a.x, b.y + a.y, b.z + a.z, b.w + a.w);
}

template <bool Q, bool Published, bool DeferAck, bool Skew>
__global__ void __launch_bounds__(1024) row_wide(float* partial, float* mapped, const float* peer_mapped,
        unsigned* token, int rank, float* x, const float* w, float* h, q36k::QB* hq,
        int n, float eps, bool emit_q8, q36tp_skew::Sample* samples, unsigned* counts, int skew_slot) {
    const int row = blockIdx.x, j = threadIdx.x;
    const size_t base = (size_t)row * n;
    const size_t packed = (size_t)row * ((size_t)n + 16);
    float* mine = mapped + packed + 4;
    const float* peer = peer_mapped + packed + 4;
    unsigned* my_flags = (unsigned*)(mapped + packed);
    unsigned* peer_flags = (unsigned*)(peer_mapped + packed);
    float* xr = x + base;
    float* yr = partial + base;
    float* hr = h + base;
    extern __shared__ float vv[];
    __shared__ float sh[256];
    __shared__ unsigned call, sample_index;
    __shared__ unsigned long long arrived, published_at, peer_at;
    if (j == 0) {
        call = token[row] + 1;
        if constexpr (Skew) {
            sample_index = atomicAdd(counts + skew_slot, 1u) % q36tp_skew::samples_per_join;
            arrived = q36tp_skew::stamp();
        }
    }
    __syncthreads();
    if constexpr (!Published) {
        if constexpr (DeferAck) {
            if (j == 0) while (q36tp::acquire(peer_flags + 1) < call - 1) q36tp::pause();
            __syncthreads();
        }
        for (int i = j; i < n / 4; i += 1024)
            ((float4*)mine)[i] = ((const float4*)yr)[i];
        if (j < n / 4) __threadfence_system();
        __syncthreads();
    }
    if (j == 0) {
        if constexpr (!Published) q36tp::release(my_flags, call);
        if constexpr (Skew) published_at = q36tp_skew::stamp();
        while (q36tp::acquire(peer_flags) != call) q36tp::pause();
        if constexpr (Skew) peer_at = q36tp_skew::stamp();
    }
    __syncthreads();
    if (j < n / 4) __threadfence_system();
    for (int i = j; i < n / 4; i += 1024) {
        const float4 a = ((const float4*)yr)[i];
        const float4 b = peer4(((const float4*)peer) + i);
        const float4 yv = ordered_add(a, b, rank);
        ((float4*)yr)[i] = yv;
        const float4 xv = ((const float4*)xr)[i];
        const float4 sum = make_float4(xv.x + yv.x, xv.y + yv.y, xv.z + yv.z, xv.w + yv.w);
        ((float4*)xr)[i] = sum;
        ((float4*)vv)[i] = sum;
    }
    __syncthreads();
    // The peer row is no longer read. Its producer can reuse it while this CTA normalizes.
    if (j == 0) q36tp::release(my_flags + 1, call);
    if (j < 256) {
        float s = 0.f;
        for (int i = j; i < n; i += 256) s += vv[i] * vv[i];
        sh[j] = s;
    }
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (j < o) sh[j] += sh[j + o];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float)n + eps);
    if constexpr (Q) {
        q36k::QB* qr = hq + (size_t)row * (n >> 5);
        for (int i = j; i < n; i += 1024) {
            const float hv = __fmul_rn(__fmul_rn(vv[i], r), w[i]);
            hr[i] = hv;
            if (emit_q8) q36k::q8_store(qr, i, ((volatile float*)hr)[i], j & 31);
        }
    } else {
        for (int i = j; i < n; i += 1024) hr[i] = vv[i] * r * w[i];
    }
    __syncthreads();
    if (j == 0) {
        if constexpr (!DeferAck)
            while (q36tp::acquire(peer_flags + 1) < call) q36tp::pause();
#if defined(__HIPCC__)
        __hip_atomic_store(token + row, call, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
        atomicExch(token + row, call);
#endif
        if constexpr (Skew) {
            q36tp_skew::Sample* s = samples + (size_t)skew_slot * q36tp_skew::samples_per_join + sample_index;
            *s = {arrived, published_at, peer_at, q36tp_skew::stamp()};
        }
    }
}

// Eight CTAs each own 32 of the original 256 logical RMS lanes. Their 256
// threads fetch those lanes' row elements in parallel from mapped memory, then
// the 32 owners accumulate the same sequence of squares as the old kernel.
// The final CTA runs the original 256-lane tree and the norm/q8 epilogue.
template <bool Q, bool Published, bool DeferAck, bool Skew>
__global__ void __launch_bounds__(512) row_split(float* partial, float* mapped, const float* peer_mapped,
        unsigned* token, unsigned* publish_done, unsigned* sum_done, float* lane_sums, int rank,
        float* x, const float* w, float* h, q36k::QB* hq, int n, float eps,
        bool emit_q8, q36tp_skew::Sample* samples, unsigned* counts, int skew_slot) {
    const int part = blockIdx.x, row = blockIdx.y, j = threadIdx.x;
    const int parts = gridDim.x;
    const size_t base = (size_t)row * n;
    const size_t packed = (size_t)row * ((size_t)n + 16);
    float* mine = mapped + packed + 4;
    const float* peer = peer_mapped + packed + 4;
    unsigned* my_flags = (unsigned*)(mapped + packed);
    unsigned* peer_flags = (unsigned*)(peer_mapped + packed);
    float* xr = x + base;
    float* yr = partial + base;
    float* hr = h + base;
    __shared__ unsigned call;
    __shared__ int last;
    __shared__ float sh[256];
    __shared__ unsigned long long arrived, published_at, peer_at;
    if (j == 0) {
        call = token[row] + 1;
        if constexpr (Skew) arrived = q36tp_skew::stamp();
    }
    __syncthreads();
    const int logical = part * 32 + (j & 31);
    const int first_k = j >> 5;
    if constexpr (!Published) {
        if constexpr (DeferAck) {
            if (j == 0) while (q36tp::acquire(peer_flags + 1) < call - 1) q36tp::pause();
            __syncthreads();
        }
        if (j < 256) for (int k = first_k; logical + 256 * k < n; k += 8) {
            const int i = logical + 256 * k;
            mine[i] = yr[i];
        }
        if (j < 256 && logical + 256 * first_k < n) __threadfence_system();
        __syncthreads();
        if (j == 0) {
#if defined(__HIPCC__)
            const unsigned old = __hip_atomic_fetch_add(publish_done + row, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
#else
            const unsigned old = atomicAdd(publish_done + row, 1u);
#endif
            if (old + 1u == (unsigned)parts) {
#if defined(__HIPCC__)
                __hip_atomic_store(publish_done + row, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
                atomicExch(publish_done + row, 0u);
#endif
                q36tp::release(my_flags, call);
            }
        }
    }
    if (j == 0) {
        if constexpr (Skew) published_at = q36tp_skew::stamp();
        while (q36tp::acquire(peer_flags) != call) q36tp::pause();
        if constexpr (Skew) peer_at = q36tp_skew::stamp();
    }
    __syncthreads();
    if (j < 256 && logical + 256 * first_k < n) __threadfence_system();
    if (j < 256) for (int k = first_k; logical + 256 * k < n; k += 8) {
        const int i = logical + 256 * k;
        const float a = yr[i];
        const float b = ((const volatile float*)peer)[i];
        const float joined = rank == 0 ? a + b : b + a;
        yr[i] = joined;
        xr[i] = xr[i] + joined;
    }
    __syncthreads();
    if (j < 32) {
        float s = 0.f;
        for (int i = logical; i < n; i += 256) {
            const float v = xr[i];
            s += v * v;
        }
        lane_sums[(size_t)row * 256 + logical] = s;
    }
    __threadfence();
    __syncthreads();
    if (j == 0) {
#if defined(__HIPCC__)
        const unsigned old = __hip_atomic_fetch_add(sum_done + row, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
#else
        const unsigned old = atomicAdd(sum_done + row, 1u);
#endif
        last = old + 1u == (unsigned)parts;
        if (last) {
#if defined(__HIPCC__)
            __hip_atomic_store(sum_done + row, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
            atomicExch(sum_done + row, 0u);
#endif
            q36tp::release(my_flags + 1, call);
        }
    }
    __syncthreads();
    if (!last) return;
    if (j < 256) sh[j] = lane_sums[(size_t)row * 256 + j];
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (j < o) sh[j] += sh[j + o];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float)n + eps);
    if constexpr (Q) {
        q36k::QB* qr = hq + (size_t)row * (n >> 5);
        for (int i = j; i < n; i += 512) {
            const float hv = __fmul_rn(__fmul_rn(xr[i], r), w[i]);
            hr[i] = hv;
            if (emit_q8) q36k::q8_store(qr, i, ((volatile float*)hr)[i], j & 31);
        }
    } else {
        for (int i = j; i < n; i += 512) hr[i] = xr[i] * r * w[i];
    }
    __syncthreads();
    if (j == 0) {
        if constexpr (!DeferAck)
            while (q36tp::acquire(peer_flags + 1) < call) q36tp::pause();
#if defined(__HIPCC__)
        __hip_atomic_store(token + row, call, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
        atomicExch(token + row, call);
#endif
        if constexpr (Skew) {
            const unsigned index = atomicAdd(counts + skew_slot, 1u) % q36tp_skew::samples_per_join;
            q36tp_skew::Sample* sample = samples + (size_t)skew_slot * q36tp_skew::samples_per_join + index;
            *sample = {arrived, published_at, peer_at, q36tp_skew::stamp()};
        }
    }
}

template <bool Q, bool Published, bool DeferAck>
inline void launch(float* partial, float* mine, const float* peer, unsigned* token,
        unsigned* publish_done, unsigned* sum_done, float* lane_sums, int rank, float* x, const float* w,
        float* h, q36k::QB* hq, int n, float eps, int T, cudaStream_t stream, int mode, bool emit_q8,
        q36tp_skew::Sample* samples, unsigned* counts, int skew_slot) {
    if (mode == 2) {
        constexpr int parts = 8;
        if (samples && T == 1)
            row_split<Q, Published, DeferAck, true><<<dim3(parts, T), 512, 0, stream>>>(
                partial, mine, peer, token, publish_done, sum_done, lane_sums, rank, x, w, h, hq,
                n, eps, emit_q8, samples, counts, skew_slot);
        else
            row_split<Q, Published, DeferAck, false><<<dim3(parts, T), 512, 0, stream>>>(
                partial, mine, peer, token, publish_done, sum_done, lane_sums, rank, x, w, h, hq,
                n, eps, emit_q8, nullptr, nullptr, 0);
    } else if (mode == 1) {
        const size_t shared_bytes = (size_t)n * sizeof(float);
        if (samples && T == 1)
            row_wide<Q, Published, DeferAck, true><<<T, 1024, shared_bytes, stream>>>(
                partial, mine, peer, token, rank, x, w, h, hq, n, eps, emit_q8, samples, counts, skew_slot);
        else
            row_wide<Q, Published, DeferAck, false><<<T, 1024, shared_bytes, stream>>>(
                partial, mine, peer, token, rank, x, w, h, hq, n, eps, emit_q8, nullptr, nullptr, 0);
    } else {
        q36tp_row::launch_join_norm<Q, Published, DeferAck>(
            partial, mine, peer, token, rank, x, w, h, hq, n, eps, T, stream,
            samples, counts, skew_slot);
    }
}

} // namespace q36tp_fast
