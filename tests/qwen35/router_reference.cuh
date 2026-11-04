// Frozen projection and routing expressions from 137b2d4c, with storage/warp loops extended to 16 tokens.
#pragma once
namespace router_reference {
template <bool HY>
__device__ __forceinline__ void route_group_body(const float* __restrict__ logits, int32_t* __restrict__ ids,
        float* __restrict__ weights, int n_tok, int ld, const int32_t* slot_of,
        const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr,
        int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    constexpr int NE = 256, K = 8, V = NE / 32;
    __shared__ int sids[128], sid[128];
    const int lane = threadIdx.x % 32;
    for (int tk = threadIdx.x / 32; tk < n_tok; tk += blockDim.x / 32) {
        const float* lg = logits + (size_t) tk * ld;
        float values[V];
#pragma unroll
        for (int i = 0; i < V; ++i) values[i] = lg[lane + i * 32];
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < V; ++i) maximum = max(maximum, values[i]);
        maximum = q36k::wmax(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] = q36k::fexp_fm(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = q36k::frcp_fm(q36k::wsum(sum));
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] *= reciprocal;
            if (__isnanf(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < K; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < V; ++i) {
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
                const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
                if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
            }
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                ids[(size_t) tk * K + rank] = expert;
                sids[tk * K + rank] = expert;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        selected_sum = max(q36k::wsum(selected_sum), 6.103515625e-5f);
        const float inverse_selected_sum = q36k::frcp_fm(selected_sum);
        if (lane < K) weights[(size_t) tk * K + lane] = selected * inverse_selected_sum;
    }
    __syncthreads();
    const int n_ent = n_tok * K;
    const int e = threadIdx.x;
    const int id = e < n_ent ? sids[e] : 0;
    const bool valid = e < n_ent && (!HY || slot_of[id] >= 0);
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = sids[e2];
            if (HY && slot_of[id2] < 0) continue;
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K;
        sid[rank] = id;
    }
    __syncthreads();
    int nv = 0;
    for (int e2 = 0; e2 < n_ent; ++e2) nv += (!HY || slot_of[sids[e2]] >= 0) ? 1 : 0;
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int g = 0;
        for (int q = 1; q < rank; ++q) g += sid[q - 1] != sid[q] ? 1 : 0;
        g += rank > 0 ? 1 : 0;
        grp_ptr[g] = (unsigned long long) (blobs + (size_t) (HY ? slot_of[id] : id) * blob);
        grp_start[g] = rank;
    }
    if (e == 0) {
        int G = nv > 0 ? 1 : 0;
        for (int q = 1; q < nv; ++q) G += sid[q - 1] != sid[q] ? 1 : 0;
        grp_start[G] = nv;
        *n_groups = G;
    }
}

template <bool LightFence = false, int Warps = 8, bool QuantizeInput = false>
__global__ void __launch_bounds__(256) router_group(const float* W, const float* x, float* logits,
        int n_in, int n_out, int n_tok, int ld, int32_t* ids, float* weights,
        const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr,
        int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok,
        unsigned* done, q36k::QB* xq) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int cols = (n_out + Warps - 1) / Warps;
    const int tk = blockIdx.x / cols;
    const int o = (blockIdx.x % cols) * Warps + warp;
    if (o < n_out) {
        const float4* wr = (const float4*) (W + (size_t) o * n_in);
        const float4* xr = (const float4*) (x + (size_t) tk * n_in);
        float acc = 0.f;
        for (int i = lane; i < n_in / 4; i += 32) {
            const float4 a = wr[i], b = xr[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        if (lane == 0) logits[(size_t) tk * n_out + o] = acc;
    }
    if constexpr (QuantizeInput) {
        const int qb = (blockIdx.x % cols) * Warps + warp;
        if (qb < n_in / 32) {
            const int i = qb * 32 + lane;
            q36k::q8_store(xq + (size_t)tk * (n_in >> 5), i,
                            ((const volatile float*)x)[(size_t)tk * n_in + i], lane);
        }
    }
    if constexpr (LightFence) {
        if (lane == 0 && o < n_out) __threadfence();
        __syncthreads();
    } else {
        __syncthreads();
        __threadfence();
        __syncthreads();
    }
    __shared__ int last;
    if (threadIdx.x == 0) {
        unsigned old;
#if defined(__HIPCC__)
        if constexpr (LightFence)
            old = __hip_atomic_fetch_add(done, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        else old = atomicAdd(done, 1u);
#else
        old = atomicAdd(done, 1u);
#endif
        last = old + 1u == (unsigned) gridDim.x;
        if (last) atomicExch(done, 0u);
    }
    __syncthreads();
    if (last) route_group_body<false>(logits, ids, weights, n_tok, ld, nullptr,
                                      blobs, blob, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
}
} // namespace router_reference
