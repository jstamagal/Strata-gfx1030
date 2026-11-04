#pragma once

#include "q36_attn.cuh"

#if defined(__HIPCC__)
namespace q36attn {

// High levels fold in registers; LDS and the final wave keep the original
// descending-stride block_sum/block_max tree. A head owns Tile consecutive lanes.
template <int G, bool MAX>
__device__ __forceinline__ float attn_tree(float (&v)[G], float* sh) {
    constexpr int Tile = kThreads / G;
#pragma unroll
    for (int s = G / 2; s; s >>= 1)
#pragma unroll
        for (int i = 0; i < s; ++i)
            if constexpr (MAX) v[i] = fmaxf(v[i], v[i + s]);
            else v[i] += v[i + s];
    const int j = threadIdx.x, lane = j % Tile;
    sh[j] = v[0];
    __syncthreads();
#pragma unroll
    for (int s = Tile / 2; s >= 32; s >>= 1) {
        if (lane < s) {
            if constexpr (MAX) sh[j] = fmaxf(sh[j], sh[j + s]);
            else sh[j] += sh[j + s];
        }
        __syncthreads();
    }
    if (lane < 32) {
        float r = sh[j];
#pragma unroll
        for (int s = 16; s; s >>= 1) {
            const float other = __shfl_down_sync(0xffffffffu, r, s, 32);
            if constexpr (MAX) r = fmaxf(r, other);
            else r += other;
        }
        if (lane == 0) sh[j] = r;
    }
    __syncthreads();
    const float r = sh[j - lane];
    __syncthreads();
    return r;
}

// One thread scores a cell for G heads, then owns an output dimension. Packed K
// and batched V loads precede independent heads' ordered FP32 FMA chains.
template <int G, int KLoads, int Phase = 0>
__device__ __forceinline__ void attn_rows(const float* __restrict__ q, const float* __restrict__ kc,
                                         const float* __restrict__ vc, const int32_t* pos0,
                                         int t0, int H, int HKV, int ctx, float scale, int n_chunk,
                                         float* pacc, float* pm, float* pl, int nt,
                                         float sc[8][kChunkCells], float* sh) {
    constexpr int HD = 256, Tile = kThreads / G, VLoads = 8;
    static_assert(G == 4 || G == 8);
    static_assert(64 % KLoads == 0);
    const int groups = H / G;
    if ((int) blockIdx.y >= nt * groups) return;
    const int lt = blockIdx.y / groups, hg = blockIdx.y % groups;
    const int t = t0 + lt, kvh = hg * G / (H / HKV);
    const int npos = *pos0 + t + 1;
    if ((int) blockIdx.x * kChunkCells >= npos) return;
    const int j = threadIdx.x, g = j / Tile, cell = j % Tile;
    for (int c = blockIdx.x; c < n_chunk && c * kChunkCells < npos; c += gridDim.x) {
        const int p0 = c * kChunkCells, n = min(kChunkCells, npos - p0);
        if (j < n) {
            float a[G];
#pragma unroll
            for (int h = 0; h < G; ++h) a[h] = 0.f;
            const uint32_t* src = reinterpret_cast<const uint32_t*>(kc) +
                ((size_t) kvh * ctx + p0 + j) * 65;
            const float ks = __uint_as_float(src[0]);
#pragma unroll 1
            for (int d = 0; d < 64; d += KLoads) {
                uint32_t packed[KLoads];
                float val[4 * KLoads];
#pragma unroll
                for (int i = 0; i < KLoads; ++i) packed[i] = src[d + i + 1];
#pragma unroll
                for (int i = 0; i < 4 * KLoads; ++i)
                    val[i] = ks * (float) (int8_t) (packed[i / 4] >> (8 * (i % 4)));
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const size_t qi = ((size_t) t * H + hg * G + h) * HD;
#pragma unroll
                    for (int i = 0; i < 4 * KLoads; ++i) a[h] += q[qi + 4 * d + i] * val[i];
                }
            }
#pragma unroll
            for (int h = 0; h < G; ++h) sc[h][j] = a[h] * scale;
        }
        __syncthreads();
        const size_t o0 = (size_t) (lt * H + hg * G) * n_chunk + c;
        if constexpr (Phase == 1) {
#pragma unroll
            for (int h = 0; h < G; ++h) pacc[(o0 + (size_t) h * n_chunk) * HD + j] = j < n ? sc[h][j] : -INFINITY;
            __syncthreads();
            continue;
        }
        float s[G];
#pragma unroll
        for (int i = 0; i < G; ++i) s[i] = cell + i * Tile < n ? sc[g][cell + i * Tile] : -INFINITY;
        const float m = attn_tree<G, true>(s, sh);
#pragma unroll
        for (int i = 0; i < G; ++i) {
            const int k = cell + i * Tile;
            s[i] = k < n ? expf(sc[g][k] - m) : 0.f;
            sc[g][k] = s[i];
        }
        const float l = attn_tree<G, false>(s, sh);
        if (cell == 0) { pm[o0 + (size_t) g * n_chunk] = m; pl[o0 + (size_t) g * n_chunk] = l; }
        if constexpr (Phase == 2) continue;
        float acc[G];
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = 0.f;
        int k = 0;
        for (; k + VLoads - 1 < n; k += VLoads) {
            float v[VLoads];
#pragma unroll
            for (int i = 0; i < VLoads; ++i)
                v[i] = kv_load(vc, (size_t) kvh * ctx + p0 + k + i, j, HD, KvType::Int8);
#pragma unroll
            for (int i = 0; i < VLoads; ++i)
#pragma unroll
                for (int h = 0; h < G; ++h) acc[h] += sc[h][k + i] * v[i];
        }
        for (; k < n; ++k) {
            const float v = kv_load(vc, (size_t) kvh * ctx + p0 + k, j, HD, KvType::Int8);
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] += sc[h][k] * v;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) pacc[(o0 + (size_t) h * n_chunk) * HD + j] = acc[h];
        __syncthreads();
    }
}

inline bool attn_rows_eligible(int HD, int H, int HKV, const RowSel& rs, bool stream, bool rows_mode) {
    return !stream && !rows_mode && !rs.cell && !rs.kc && rs.cache_type == KvType::Int8 &&
           HD == 256 && H == 16 * HKV;
}

// Fixed graph grid: group size follows the live device position, not capture's
// host position. Extra head groups and chunks exit before touching LDS.
__global__ void k_attn_part_rows(const float* q, const float* kc, const float* vc, const int32_t* pos0,
                                int t0, int H, int HKV, int ctx, float scale, int n_chunk,
                                float* pacc, float* pm, float* pl, int nt) {
    __shared__ float sc[8][kChunkCells], sh[kThreads];
    if (*pos0 + t0 + nt <= 1024)
        attn_rows<4, 8>(q, kc, vc, pos0, t0, H, HKV, ctx, scale, n_chunk, pacc, pm, pl, nt, sc, sh);
    else
        attn_rows<8, 16>(q, kc, vc, pos0, t0, H, HKV, ctx, scale, n_chunk, pacc, pm, pl, nt, sc, sh);
}

}  // namespace q36attn
#endif
