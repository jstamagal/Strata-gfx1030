// Included inside q36k after the warp arithmetic helpers.
#pragma once

// Each expert has at most one pick per token. Scan experts, then popcount earlier tokens:
// groups stay in expert-id order and entries stay in their original token/rank order.
template <bool HY>
__device__ __forceinline__ void route_group_mask_body(const float* __restrict__ logits, int32_t* __restrict__ ids,
        float* __restrict__ weights, int n_tok, int ld, const int32_t* slot_of,
        const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr,
        int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    constexpr int NE = 256, K = 8, V = NE / 32;
    __shared__ unsigned masks[NE], prefix[NE], totals[8];
    __shared__ int sids[128];
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int threads = blockDim.x * blockDim.y;
    const int lane = tid & 31;
    for (int id = tid; id < NE; id += threads) masks[id] = 0;
    __syncthreads();
    for (int tk = tid / 32; tk < n_tok; tk += threads / 32) {
        const float* lg = logits + (size_t) tk * ld;
        float values[V];
#pragma unroll
        for (int i = 0; i < V; ++i) values[i] = lg[lane + i * 32];
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < V; ++i) maximum = max(maximum, values[i]);
        maximum = wmax(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] = fexp_fm(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = frcp_fm(wsum(sum));
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
                if (!HY || slot_of[expert] >= 0) atomicOr(masks + expert, 1u << tk);
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        selected_sum = max(wsum(selected_sum), 6.103515625e-5f);
        const float inverse_selected_sum = frcp_fm(selected_sum);
        if (lane < K) weights[(size_t) tk * K + lane] = selected * inverse_selected_sum;
    }
    __syncthreads();
    for (int id = tid; id < NE; id += threads) {
        const unsigned bits = masks[id];
        const unsigned pack = (__popc(bits) << 16) | (bits != 0);
        unsigned inc = pack;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const unsigned up = __shfl_up_sync(0xffffffffu, inc, d, 32);
            if (lane >= d) inc += up;
        }
        prefix[id] = inc - pack;
        if (lane == 31) totals[id / 32] = inc;
    }
    __syncthreads();
    for (int id = tid; id < NE; id += threads) {
        if (!masks[id]) continue;
        unsigned p = prefix[id];
#pragma unroll
        for (int w = 0; w < 8; ++w) if (w < id / 32) p += totals[w];
        const int g = p & 0xffff;
        grp_ptr[g] = (unsigned long long) (blobs + (size_t) (HY ? slot_of[id] : id) * blob);
        grp_start[g] = p >> 16;
    }
    if (tid < n_tok * K) {
        const int id = sids[tid], tk = tid / K;
        const unsigned bits = masks[id];
        if (bits) {
            unsigned p = prefix[id];
#pragma unroll
            for (int w = 0; w < 8; ++w) if (w < id / 32) p += totals[w];
            const int rank = (p >> 16) + __popc(bits & ((1u << tk) - 1u));
            ent_dst[rank] = tid;
            ent_tok[rank] = tk;
        }
    }
    if (tid == 0) {
        unsigned total = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) total += totals[w];
        const int groups = total & 0xffff;
        grp_start[groups] = total >> 16;
        *n_groups = groups;
    }
}
