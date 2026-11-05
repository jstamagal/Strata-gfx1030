// src/kernels/cuda/route_plan.cu - STRATA_ROUTE_PLAN=1: the routing top-10 and the all-resident plan in one launch.
//
// One workgroup of 8 waves.  Wave t < n_tok routes token t with route_multi's code (native_router.cu: the softmax
// over 512 logits held 16 per lane, ten warp-argmax rounds, the renormalised weights) and also leaves its ten ids in
// shared memory.  After one barrier, thread i < n_tok * 10 is plan entry i with resident_plan_kernel's code
// (verify_kernels.cu: groups are the distinct experts in order of first occurrence, a prefix scan over the waves
// places them).  Both bodies are copies, so the ids, the weights and every word of the plan are the two kernels'.
// The plan is the all-resident one (no skip word): a routed expert that is not resident leaves an empty plan and
// raises *plan_err, as there.
#include "strata/kernels/route_plan.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/node_fuse.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kEntries = 128;   // resident_plan's kResidentPlanMax: one thread per entry, at most 4 waves

__device__ __forceinline__ float rp_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float rp_warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}

// route_multi for wave tk < n_tok, one barrier, resident_plan_kernel for thread tid < n_tok * 10.  `load(tk, i)` is
// logit i of token tk.  The caller's kernel declares the shared arrays (kEntries, kEntries, 4, 1).
template <typename Load>
__device__ __forceinline__ void route_plan_body(Load&& load, int32_t* __restrict__ ids_, float* __restrict__ weights_,
                                                int n_tok, const int32_t* __restrict__ res, int n_expert,
                                                const uint8_t* cache_base, const unsigned long long* slot_off,
                                                long long blob, int32_t* __restrict__ pl, long long capx,
                                                volatile uint32_t* plan_err, int32_t* s_ids, int32_t* s_excl,
                                                int32_t* s_wsum, int* s_bad_p) {
    int& s_bad = *s_bad_p;
    const int lane = threadIdx.x;
    const int tk = (int) threadIdx.y;
    const int tid = tk * 32 + lane;
    const int n = n_tok * 10, k = 10;
    if (tid == 0) s_bad = 0;
    if (tk < n_tok) {   // ---- route_multi, wave tk = token tk
        int32_t* ids = ids_ + (size_t) tk * 10;
        float* weights = weights_ + (size_t) tk * 10;
        float values[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = load(tk, lane + i * 32);
        __syncwarp();
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
        maximum = rp_warp_max(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] = expf(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = 1.0f / rp_warp_sum(sum);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] *= reciprocal;
            if (__isnanf(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < 10; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < 16; ++i) {
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
                ids[rank] = expert;
                s_ids[tk * 10 + rank] = expert;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        selected_sum = max(rp_warp_sum(selected_sum), 6.103515625e-5f);
        const float inverse_selected_sum = 1.0f / selected_sum;
        if (lane < 10) weights[lane] = selected * inverse_selected_sum;
    }
    __syncthreads();

    // ---- resident_plan_kernel, thread tid = entry tid (waves 0..3)
    int32_t eid = -1;
    int32_t slot = -1;
    if (tid < n) {
        eid = s_ids[tid];
        slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0) atomicOr(&s_bad, 1);
    }
    __syncthreads();
    if (s_bad) {
        if (tid == 0) {
            pl[0] = 0; pl[1] = 0; pl[2] = 0;
            if (plan_err != nullptr) { *plan_err = 1; __threadfence_system(); }
        }
        return;
    }
    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
    }
    const bool is_first = (tid < n && first_j == tid && slot >= 0);
    const int my_cnt = is_first ? count_same : 0;
    const int my_pack = (my_cnt << 16) | (is_first ? 1 : 0);
    int pref = my_pack;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int up = __shfl_up_sync(0xffffffffu, pref, d);
        if (lane >= d) pref += up;
    }
    if (tid < kEntries) {
        s_excl[tid] = pref - my_pack;
        if (lane == 31) s_wsum[tk] = pref;
    }
    __syncthreads();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (is_first) {
        int tot = s_excl[tid];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < tk) tot += s_wsum[w];
        }
        const int grp_idx = tot & 0xffff;
        const int ent_start = tot >> 16;
        ptr[grp_idx] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        start[grp_idx] = ent_start;
    }
    if (tid < n) {
        const int fj_warp = first_j >> 5;
        int fj_tot = s_excl[first_j];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < fj_warp) fj_tot += s_wsum[w];
        }
        const int out_idx = (fj_tot >> 16) + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        const int groups = (s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3]) & 0xffff;
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
    }
}

// ---- STRATA_T1_DIET (8), HIP gfx10+: route_plan_body with its wave reductions on DPP instead of ds_bpermute (each
// level a lane-crossing VALU move, not an LDS round trip: the ten argmax rounds were ~60 of them in a row), and the
// layer's residency (res[e] and its slot's pointer, for all 512 experts) loaded into LDS by the waves that do not route
// (n_tok <= 4: waves 4-7) while the routing runs, so the plan reads LDS instead of two dependent global loads.  Every
// reduction pairs the same lanes in the same order (rp_xor<M>(v) is __shfl_xor_sync(.., v, M, 32)), every other line
// is route_plan_body's, and the residency values are the same words: ids, weights and the plan are its bits.
#if defined(__HIPCC__) && (defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__))
#define STRATA_RP_DPP 1
template <int M>
__device__ __forceinline__ int rp_xor(int v) {
    static_assert(M == 1 || M == 2 || M == 4 || M == 8 || M == 16, "rp_xor: a wave32 xor butterfly level");
    if constexpr (M == 16)   // the other row of 16, the same lane in it
        return (int) __builtin_amdgcn_permlanex16((unsigned) v, (unsigned) v, 0x76543210u, 0xfedcba98u, false, false);
    else   // row_xmask:M - lane i reads lane i ^ M of its row
        return __builtin_amdgcn_update_dpp(v, v, 0x160 | M, 0xF, 0xF, false);
}
template <int M>
__device__ __forceinline__ float rp_xorf(float v) { return __int_as_float(rp_xor<M>(__float_as_int(v))); }
__device__ __forceinline__ float rp_dpp_sum(float value) {
    value += rp_xorf<16>(value);
    value += rp_xorf<8>(value);
    value += rp_xorf<4>(value);
    value += rp_xorf<2>(value);
    value += rp_xorf<1>(value);
    return value;
}
__device__ __forceinline__ float rp_dpp_max(float value) {
    value = fmaxf(value, rp_xorf<16>(value));
    value = fmaxf(value, rp_xorf<8>(value));
    value = fmaxf(value, rp_xorf<4>(value));
    value = fmaxf(value, rp_xorf<2>(value));
    value = fmaxf(value, rp_xorf<1>(value));
    return value;
}
template <int M>
__device__ __forceinline__ void rp_dpp_arg(float& best, int& expert) {
    const float other = rp_xorf<M>(best);
    const int other_id = rp_xor<M>(expert);
    if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
}
constexpr int kRpExperts = 512;
// waves 4-7 (n_tok <= 4: they route nothing): res[e] and the plan's pointer for e's slot, every expert of the layer
__device__ __forceinline__ void rp_prefetch(int n_tok, const int32_t* __restrict__ res, int n_expert,
                                            const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                                            int32_t* s_res, unsigned long long* s_ptr) {
    const int tk = (int) threadIdx.y;
    if (n_tok > 4 || n_expert != kRpExperts || tk < 4) return;
    const int i0 = (tk - 4) * 32 + (int) threadIdx.x;
    int32_t r[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) r[q] = res[i0 + 128 * q];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int slot = r[q];
        s_res[i0 + 128 * q] = slot;
        s_ptr[i0 + 128 * q] = slot >= 0 ? (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot]
                                                                                        : (size_t) slot * (size_t) blob))
                                        : 0ull;
    }
}
template <typename Load>
__device__ __forceinline__ void route_plan_body_dpp(Load&& load, int32_t* __restrict__ ids_, float* __restrict__ weights_,
                                                    int n_tok, const int32_t* __restrict__ res, int n_expert,
                                                    const uint8_t* cache_base, const unsigned long long* slot_off,
                                                    long long blob, int32_t* __restrict__ pl, long long capx,
                                                    volatile uint32_t* plan_err, int32_t* s_ids, int32_t* s_excl,
                                                    int32_t* s_wsum, int* s_bad_p, const int32_t* s_res,
                                                    const unsigned long long* s_ptr) {
    int& s_bad = *s_bad_p;
    const int lane = threadIdx.x;
    const int tk = (int) threadIdx.y;
    const int tid = tk * 32 + lane;
    const int n = n_tok * 10, k = 10;
    const bool pre = n_tok <= 4 && n_expert == kRpExperts;   // rp_prefetch filled s_res / s_ptr
    if (tid == 0) s_bad = 0;
    if (tk < n_tok) {   // ---- route_multi, wave tk = token tk
        int32_t* ids = ids_ + (size_t) tk * 10;
        float* weights = weights_ + (size_t) tk * 10;
        float values[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = load(tk, lane + i * 32);
        __syncwarp();
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
        maximum = rp_dpp_max(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] = expf(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = 1.0f / rp_dpp_sum(sum);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] *= reciprocal;
            if (__isnanf(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
#pragma unroll
        for (int rank = 0; rank < 10; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < 16; ++i) {
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
            rp_dpp_arg<16>(best, expert);
            rp_dpp_arg<8>(best, expert);
            rp_dpp_arg<4>(best, expert);
            rp_dpp_arg<2>(best, expert);
            rp_dpp_arg<1>(best, expert);
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                ids[rank] = expert;
                s_ids[tk * 10 + rank] = expert;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        selected_sum = max(rp_dpp_sum(selected_sum), 6.103515625e-5f);
        const float inverse_selected_sum = 1.0f / selected_sum;
        if (lane < 10) weights[lane] = selected * inverse_selected_sum;
    }
    __syncthreads();

    // ---- resident_plan_kernel, thread tid = entry tid (waves 0..3)
    int32_t eid = -1;
    int32_t slot = -1;
    if (tid < n) {
        eid = s_ids[tid];
        slot = (eid >= 0 && eid < n_expert) ? (pre ? s_res[eid] : res[eid]) : -1;
        if (slot < 0) atomicOr(&s_bad, 1);
    }
    __syncthreads();
    if (s_bad) {
        if (tid == 0) {
            pl[0] = 0; pl[1] = 0; pl[2] = 0;
            if (plan_err != nullptr) { *plan_err = 1; __threadfence_system(); }
        }
        return;
    }
    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
    }
    const bool is_first = (tid < n && first_j == tid && slot >= 0);
    const int my_cnt = is_first ? count_same : 0;
    const int my_pack = (my_cnt << 16) | (is_first ? 1 : 0);
    int pref = my_pack;
    {   // the inclusive scan as __shfl_up_sync(.., d) levels 1, 2, 4, 8, 16 (row_shr within 16, then row 0's last into row 1)
        int up = __builtin_amdgcn_update_dpp(0, pref, 0x111, 0xF, 0xF, false);   // row_shr:1
        if ((lane & 15) >= 1) pref += up;
        up = __builtin_amdgcn_update_dpp(0, pref, 0x112, 0xF, 0xF, false);       // row_shr:2
        if ((lane & 15) >= 2) pref += up;
        up = __builtin_amdgcn_update_dpp(0, pref, 0x114, 0xF, 0xF, false);       // row_shr:4
        if ((lane & 15) >= 4) pref += up;
        up = __builtin_amdgcn_update_dpp(0, pref, 0x118, 0xF, 0xF, false);       // row_shr:8
        if ((lane & 15) >= 8) pref += up;
        up = __builtin_amdgcn_readlane(pref, 15);                                // lane 15's sum (rows: 0-15, 16-31)
        if (lane >= 16) pref += up;
    }
    if (tid < kEntries) {
        s_excl[tid] = pref - my_pack;
        if (lane == 31) s_wsum[tk] = pref;
    }
    __syncthreads();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (is_first) {
        int tot = s_excl[tid];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < tk) tot += s_wsum[w];
        }
        const int grp_idx = tot & 0xffff;
        const int ent_start = tot >> 16;
        ptr[grp_idx] = pre ? s_ptr[eid]
                           : (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        start[grp_idx] = ent_start;
    }
    if (tid < n) {
        const int fj_warp = first_j >> 5;
        int fj_tot = s_excl[first_j];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < fj_warp) fj_tot += s_wsum[w];
        }
        const int out_idx = (fj_tot >> 16) + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        const int groups = (s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3]) & 0xffff;
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
    }
}
#endif   // STRATA_RP_DPP

__launch_bounds__(256, 1)
__global__ void route_plan_kernel(const float* __restrict__ logits_, int32_t* __restrict__ ids_,
                                  float* __restrict__ weights_, int n_tok, const int32_t* __restrict__ res,
                                  int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                  long long blob, int32_t* __restrict__ pl, long long capx,
                                  volatile uint32_t* plan_err) {
    __shared__ int32_t s_ids[kEntries];
    __shared__ int32_t s_excl[kEntries];
    __shared__ int32_t s_wsum[4];
    __shared__ int s_bad;
    route_plan_body([&](int tk, int i) { return logits_[(size_t) tk * 512 + i]; }, ids_, weights_, n_tok, res, n_expert,
                    cache_base, slot_off, blob, pl, capx, plan_err, s_ids, s_excl, s_wsum, &s_bad);
}
#if defined(__HIPCC__)
// STRATA_T1_DIET (8): route_plan_kernel with route_plan_body_dpp (a device pass without DPP builds the plain body)
__launch_bounds__(256, 1)
__global__ void route_plan_dpp_kernel(const float* __restrict__ logits_, int32_t* __restrict__ ids_,
                                      float* __restrict__ weights_, int n_tok, const int32_t* __restrict__ res,
                                      int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                      long long blob, int32_t* __restrict__ pl, long long capx,
                                      volatile uint32_t* plan_err) {
    __shared__ int32_t s_ids[kEntries];
    __shared__ int32_t s_excl[kEntries];
    __shared__ int32_t s_wsum[4];
    __shared__ int s_bad;
#if defined(STRATA_RP_DPP)
    __shared__ int32_t s_res[kRpExperts];
    __shared__ unsigned long long s_ptr[kRpExperts];
    rp_prefetch(n_tok, res, n_expert, cache_base, slot_off, blob, s_res, s_ptr);
    route_plan_body_dpp([&](int tk, int i) { return logits_[(size_t) tk * 512 + i]; }, ids_, weights_, n_tok, res,
                        n_expert, cache_base, slot_off, blob, pl, capx, plan_err, s_ids, s_excl, s_wsum, &s_bad, s_res,
                        s_ptr);
#else
    route_plan_body([&](int tk, int i) { return logits_[(size_t) tk * 512 + i]; }, ids_, weights_, n_tok, res, n_expert,
                    cache_base, slot_off, blob, pl, capx, plan_err, s_ids, s_excl, s_wsum, &s_bad);
#endif
}
#endif

// ---- STRATA_ROUTE_PLAN=2 (HIP): the router's BF16 projection in the same launch.  Workgroups 0..511 are
// bf16_f32_mmvf_multi_kernel<256, NT>'s (native_bf16.cu: thread t walks pairs t, t + 256, ... with two ordered FMAs
// per pair and column, the wave sums, then the sum of the eight wave sums), one router row each.  Workgroup 512 is
// the routing and the plan above; its first thread waits until all 512 rows are stored.
//
// What crosses workgroups goes the way LLVM's AMDGPU memory model and rocPRIM's look-back scan do it: a row's
// logits are stored, the stores are waited for (an agent-scope fence), and only then the row is counted with an
// atomic add; the reader polls the count with an atomic operation and reads the logits with device-scope atomic
// loads, which bypass the per-CU caches.  The count is reset by the reader, so every launch starts from zero.
__device__ __forceinline__ float rf_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

#if defined(__HIPCC__)
template <int NT>
__launch_bounds__(256)
__global__ void route_fused_kernel(const float* __restrict__ x, long long ldx, const uint16_t* __restrict__ w,
                                   uint32_t* y, int n_in, uint32_t* done, int32_t* __restrict__ ids_,
                                   float* __restrict__ weights_, const int32_t* __restrict__ res, int n_expert,
                                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                                   int32_t* __restrict__ pl, long long capx, volatile uint32_t* plan_err) {
    __shared__ int32_t s_ids[kEntries];
    __shared__ int32_t s_excl[kEntries];
    __shared__ int32_t s_wsum[4];
    __shared__ int s_bad;
    __shared__ float partials[NT][32];
    const int t = int(threadIdx.y) * 32 + int(threadIdx.x);
    if (blockIdx.x < 512u) {
        const uint16_t* row = w + (size_t) blockIdx.x * n_in;
        const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
        if (t < 32) {
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
        }
        __syncthreads();
        float acc[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
        for (int pair = t; pair < n_in / 2; pair += 256) {
            const uint32_t weight = weights2[pair];
            const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const float2 input = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[pair];
                acc[k] = __fmaf_rn(w0, input.x, acc[k]);
                acc[k] = __fmaf_rn(w1, input.y, acc[k]);
            }
        }
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[k] = rf_warp_sum(acc[k]);
        if ((t & 31) == 0)
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t / 32] = acc[k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k) acc[k] = rf_warp_sum(partials[k][t]);
        if (t == 0) {
#pragma unroll
            for (int k = 0; k < NT; ++k)
                __scoped_atomic_store_n(y + (size_t) k * 512 + blockIdx.x, __float_as_uint(acc[k]), __ATOMIC_RELAXED,
                                        __MEMORY_SCOPE_DEVICE);
            __threadfence();          // the stores above have landed before the row is counted
            atomicAdd(done, 1u);
        }
        return;
    }
    if (t == 0) {
        while (atomicAdd(done, 0u) < 512u) __builtin_amdgcn_s_sleep(1);
        __threadfence();
        atomicExch(done, 0u);         // the next launch counts from zero
    }
    __syncthreads();
    route_plan_body([&](int tk, int i) {
                        return __uint_as_float(__scoped_atomic_load_n(y + (size_t) tk * 512 + i, __ATOMIC_RELAXED,
                                                                      __MEMORY_SCOPE_DEVICE));
                    },
                    ids_, weights_, NT, res, n_expert, cache_base, slot_off, blob, pl, capx, plan_err, s_ids, s_excl,
                    s_wsum, &s_bad);
}


// STRATA_T1_DIET (8): route_fused_kernel with route_plan_body_dpp in its last workgroup, the residency loaded while the
// rows are computed (a device pass without DPP builds the plain tail)
template <int NT>
__launch_bounds__(256)
__global__ void route_fused_dpp_kernel(const float* __restrict__ x, long long ldx, const uint16_t* __restrict__ w,
                                   uint32_t* y, int n_in, uint32_t* done, int32_t* __restrict__ ids_,
                                   float* __restrict__ weights_, const int32_t* __restrict__ res, int n_expert,
                                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                                   int32_t* __restrict__ pl, long long capx, volatile uint32_t* plan_err) {
    __shared__ int32_t s_ids[kEntries];
    __shared__ int32_t s_excl[kEntries];
    __shared__ int32_t s_wsum[4];
    __shared__ int s_bad;
    __shared__ float partials[NT][32];
    const int t = int(threadIdx.y) * 32 + int(threadIdx.x);
    if (blockIdx.x < 512u) {
        const uint16_t* row = w + (size_t) blockIdx.x * n_in;
        const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
        if (t < 32) {
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
        }
        __syncthreads();
        float acc[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
        for (int pair = t; pair < n_in / 2; pair += 256) {
            const uint32_t weight = weights2[pair];
            const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const float2 input = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[pair];
                acc[k] = __fmaf_rn(w0, input.x, acc[k]);
                acc[k] = __fmaf_rn(w1, input.y, acc[k]);
            }
        }
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[k] = rf_warp_sum(acc[k]);
        if ((t & 31) == 0)
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t / 32] = acc[k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k) acc[k] = rf_warp_sum(partials[k][t]);
        if (t == 0) {
#pragma unroll
            for (int k = 0; k < NT; ++k)
                __scoped_atomic_store_n(y + (size_t) k * 512 + blockIdx.x, __float_as_uint(acc[k]), __ATOMIC_RELAXED,
                                        __MEMORY_SCOPE_DEVICE);
            __threadfence();          // the stores above have landed before the row is counted
            atomicAdd(done, 1u);
        }
        return;
    }
#if defined(STRATA_RP_DPP)
    __shared__ int32_t s_res[kRpExperts];
    __shared__ unsigned long long s_ptr[kRpExperts];
    rp_prefetch(NT, res, n_expert, cache_base, slot_off, blob, s_res, s_ptr);
#endif
    if (t == 0) {
        while (atomicAdd(done, 0u) < 512u) __builtin_amdgcn_s_sleep(1);
        __threadfence();
        atomicExch(done, 0u);         // the next launch counts from zero
    }
    __syncthreads();
#if defined(STRATA_RP_DPP)
    route_plan_body_dpp([&](int tk, int i) {
                            return __uint_as_float(__scoped_atomic_load_n(y + (size_t) tk * 512 + i, __ATOMIC_RELAXED,
                                                                          __MEMORY_SCOPE_DEVICE));
                        },
                        ids_, weights_, NT, res, n_expert, cache_base, slot_off, blob, pl, capx, plan_err, s_ids, s_excl,
                        s_wsum, &s_bad, s_res, s_ptr);
#else
    route_plan_body([&](int tk, int i) {
                        return __uint_as_float(__scoped_atomic_load_n(y + (size_t) tk * 512 + i, __ATOMIC_RELAXED,
                                                                      __MEMORY_SCOPE_DEVICE));
                    },
                    ids_, weights_, NT, res, n_expert, cache_base, slot_off, blob, pl, capx, plan_err, s_ids, s_excl,
                    s_wsum, &s_bad);
#endif
}

template <int NT>
void launch_fused(const float* x, long long ldx, const uint16_t* w, float* logits, uint32_t* done, int32_t* ids,
                  float* weights, const int32_t* res, int n_expert, const uint8_t* cache_base,
                  const unsigned long long* slot_off, long long blob, int32_t* plan, long long capx, uint32_t* plan_err,
                  cudaStream_t s);
#endif

}  // namespace

// STRATA_T1_DIET (8): the DPP routing body (where it is built: HIP gfx10+); route_plan_fast_tune (tests) overrides
static int g_rp_fast = -1;
void route_plan_fast_tune(int on) { g_rp_fast = on; }
bool route_plan_fast() {
#if defined(__HIPCC__)
    if (g_rp_fast >= 0) return g_rp_fast != 0;
    return t1_diet(kT1Route);
#else
    return false;
#endif
}

namespace {
#if defined(__HIPCC__)
template <int NT>
void launch_fused(const float* x, long long ldx, const uint16_t* w, float* logits, uint32_t* done, int32_t* ids,
                  float* weights, const int32_t* res, int n_expert, const uint8_t* cache_base,
                  const unsigned long long* slot_off, long long blob, int32_t* plan, long long capx, uint32_t* plan_err,
                  cudaStream_t s) {
    if (route_plan_fast())
        route_fused_dpp_kernel<NT><<<513, dim3(32, 8), 0, s>>>(x, ldx, w, reinterpret_cast<uint32_t*>(logits), 2560, done,
                                                                ids, weights, res, n_expert, cache_base, slot_off, blob,
                                                                plan, capx, plan_err);
    else
    route_fused_kernel<NT><<<513, dim3(32, 8), 0, s>>>(x, ldx, w, reinterpret_cast<uint32_t*>(logits), 2560, done, ids,
                                                      weights, res, n_expert, cache_base, slot_off, blob, plan, capx,
                                                      plan_err);
}
#endif
}  // namespace

int route_plan_mode() {
    static const int mode = [] {
        const char* v = std::getenv("STRATA_ROUTE_PLAN");
        if (v == nullptr || v[1] != '\0') return 0;
#if defined(__HIPCC__)
        if (v[0] == '2') return 2;
#else
        if (v[0] == '2') return 1;   // (the fused projection is HIP's; CUDA takes the two launches)
#endif
        return v[0] == '1' ? 1 : 0;
    }();
    return mode;
}
bool route_plan_enabled() { return route_plan_mode() != 0; }

void native_route_plan(const float* logits, int32_t* ids, float* weights, int n_tok, const int32_t* res_layer,
                       int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                       int32_t* plan, long long capx, uint32_t* plan_err, void* stream) {
    if (!stream || !logits || !ids || !weights || !res_layer || !plan || n_tok < 1 || n_tok > 8 || capx < (long long) n_tok * 10)
        throw std::invalid_argument("native_route_plan: needs a stream, device buffers and 1..8 tokens");
    if (route_plan_fast())   // STRATA_T1_DIET (8)
        route_plan_dpp_kernel<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
            logits, ids, weights, n_tok, res_layer, n_expert, cache_base, slot_off, blob, plan, capx, plan_err);
    else
    route_plan_kernel<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, n_tok, res_layer, n_expert, cache_base, slot_off, blob, plan, capx, plan_err);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("native_route_plan: ") + cudaGetErrorString(e));
}

}  // namespace strata::kernels

namespace strata::kernels {
bool native_route_fused(const float* x, long long ldx, const uint16_t* w_router, float* logits, uint32_t* done,
                        int32_t* ids, float* weights, int n_tok, const int32_t* res_layer, int n_expert,
                        const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                        long long capx, uint32_t* plan_err, void* stream) {
#if defined(__HIPCC__)
    if (!stream || !x || !w_router || !logits || !done || !ids || !weights || !res_layer || !plan || n_tok < 1 ||
        n_tok > 8 || (ldx & 1) != 0 || capx < (long long) n_tok * 10 || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("native_route_fused: needs a stream, aligned device buffers and 1..8 tokens");
    const auto s = static_cast<cudaStream_t>(stream);
#define STRATA_RF(N) case N: launch_fused<N>(x, ldx, w_router, logits, done, ids, weights, res_layer, n_expert, cache_base, \
                                             slot_off, blob, plan, capx, plan_err, s); break
    switch (n_tok) { STRATA_RF(1); STRATA_RF(2); STRATA_RF(3); STRATA_RF(4); STRATA_RF(5); STRATA_RF(6); STRATA_RF(7); default: launch_fused<8>(x, ldx, w_router, logits, done, ids, weights, res_layer, n_expert, cache_base, slot_off, blob, plan, capx, plan_err, s); break; }
#undef STRATA_RF
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("native_route_fused: ") + cudaGetErrorString(e));
    return true;
#else
    (void) x; (void) ldx; (void) w_router; (void) logits; (void) done; (void) ids; (void) weights; (void) n_tok;
    (void) res_layer; (void) n_expert; (void) cache_base; (void) slot_off; (void) blob; (void) plan; (void) capx;
    (void) plan_err; (void) stream;
    return false;
#endif
}
}  // namespace strata::kernels
