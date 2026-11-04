#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace q36attn {

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;   // attention split-K chunk

__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}


/// Batch rows (several sessions decoded in one window, one token each): row t reads and writes the cache of session
/// slot[t] at cell[t]; its rotary position is cell[t] + delta[t].  Every field null: one session, the rows are its
/// consecutive cells *pos0 + t (the prompt, a verify window, a decode step).
// Cache rows are head-major [head][ctx][row] (--kv-cache-type): F32, F16, Int8 (a float scale, max |x| / 127 of the
// row, then HD signed bytes), or Int8G64 (HD / 64 half scales, one per 64 values, then HD signed bytes).
// Keep the base pointers opaque to session selection; all addressing uses the selected row size.
enum class KvType { F32, F16, Int8, Int8G64 };
constexpr int kKvGroup = 64;   // Int8G64: the values of one scale
__host__ __device__ size_t kv_row_bytes(KvType type, int HD) {
    return type == KvType::Int8 ? sizeof(float) + (size_t) HD :
           type == KvType::Int8G64 ? (size_t) (HD / kKvGroup) * sizeof(__half) + (size_t) HD :
           (size_t) HD * (type == KvType::F16 ? sizeof(__half) : sizeof(float));
}
__device__ float kv_load(const float* cache, size_t row, int d, int HD, KvType type) {
    if (type == KvType::F32) return cache[row * HD + d];
    if (type == KvType::F16) return __half2float(reinterpret_cast<const __half*>(cache)[row * HD + d]);
    const char* p = reinterpret_cast<const char*>(cache) + row * kv_row_bytes(type, HD);
    if (type == KvType::Int8)
        return *reinterpret_cast<const float*>(p) * (float) reinterpret_cast<const int8_t*>(p + sizeof(float))[d];
    return __half2float(reinterpret_cast<const __half*>(p)[d / kKvGroup]) *
           (float) reinterpret_cast<const int8_t*>(p + (size_t) (HD / kKvGroup) * sizeof(__half))[d];
}

struct RowSel {
    KvType cache_type = KvType::F16;
    const int32_t* cell = nullptr;   // device [T]
    const int32_t* slot = nullptr;   // device [T]
    const int32_t* delta = nullptr;  // device [T]
    float* const* kc = nullptr;      // device [n_sessions]: this layer's K / V base of each session
    float* const* vc = nullptr;
    float* const* kc_hot = nullptr;  // --kv-resident: this layer's ring of each session
    float* const* vc_hot = nullptr;
};

#include "q36_kv_stream.cuh"


/// Split-K attention, one block per (chunk of 256 cells, query row): query row qi = t * H + h (t from t0) sees the
/// cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum; chunks past the query's cells write
/// nothing (the merge stops before them).  STREAM (--kv-resident): each cell from its staging window, ring or host
/// row (q36kvstream::at).  grid (n_chunk, n_rows), block 256, HD == 256.
template <bool STREAM>
__global__ void k_attn_part(const float* q, const float* kc, const float* vc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs,
                            KvStream stream) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
        if constexpr (STREAM) {
            if (rs.kc_hot) {
                stream.khot = rs.kc_hot[rs.slot[t]];
                stream.vhot = rs.vc_hot[rs.slot[t]];
            }
        }
    }
    [[maybe_unused]] const int end = rs.cell ? npos + stream.append_tokens : *pos0 + stream.append_tokens;   // STREAM only
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        float a = 0.f;
        if constexpr (!STREAM) {
            const size_t row = (size_t) kvh * ctx + p;
            for (int i = 0; i < HD; ++i) a += qs[i] * kv_load(kc, row, i, HD, rs.cache_type);
        } else {
            size_t kr = 0;
            const float* kb = q36kvstream::at(kc, stream.khot, stream.kpref, kvh, p, ctx, stream, end, kr);
            for (int i = 0; i < HD; ++i) a += qs[i] * kv_load(kb, kr, i, HD, rs.cache_type);
        }
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = min(kChunkCells, npos - p0);
    float acc = 0.f;
    if constexpr (!STREAM) {
        for (int k = 0; k < n; ++k)
            acc += sc[k] * kv_load(vc, (size_t) kvh * ctx + p0 + k, j, HD, rs.cache_type);
    } else {
        for (int k = 0; k < n; ++k) {
            size_t vr = 0;
            const float* vb = q36kvstream::at(vc, stream.vhot, stream.vpref, kvh, p0 + k, ctx, stream, end, vr);
            acc += sc[k] * kv_load(vb, vr, j, HD, rs.cache_type);
        }
    }
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// k_attn_part for the G query heads of one KV head in one block: each K / V cell of the chunk is read once for the
/// group, not once per head.  The same sums in the same order, so the same numbers.  K goes through LDS in tiles of
/// 16 cells; thread (g, cell) = (j / 16, j % 16) takes one score.  Block x takes chunks x, x + gridDim.x, ... (a
/// graph's grid covers the whole context; a block per chunk would be mostly empty blocks holding 49 KB of LDS each).
/// STREAM: as k_attn_part.  grid (n_slots, n_tok * HKV), block 256, HD == 256.
template <int G, bool STREAM>
__global__ void k_attn_part_gqa(const float* q, const float* kc, const float* vc, const int32_t* pos0, int t0, int H,
                                int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs,
                                KvStream stream) {
    constexpr int HD = 256, kTile = 16, kPad = HD + 4;
    static_assert(G * kTile <= kThreads, "one score per thread");
    __shared__ float qs[G][HD], sc[G][kChunkCells], sh[kThreads];
    __shared__ __align__(16) float kt[kTile][kPad];
    const int lt = blockIdx.y / HKV, kvh = blockIdx.y % HKV, t = t0 + lt;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
        if constexpr (STREAM) {
            if (rs.kc_hot) {
                stream.khot = rs.kc_hot[rs.slot[t]];
                stream.vhot = rs.vc_hot[rs.slot[t]];
            }
        }
    }
    [[maybe_unused]] const int end = rs.cell ? npos + stream.append_tokens : *pos0 + stream.append_tokens;   // STREAM only
    if ((int) blockIdx.x * kChunkCells >= npos) return;
    const int j = threadIdx.x;
#pragma unroll
    for (int g = 0; g < G; ++g) qs[g][j] = q[((size_t) t * H + kvh * G + g) * HD + j];
    const int tg = j / kTile, tc = j % kTile;
    for (int c = blockIdx.x; c < n_chunk && c * kChunkCells < npos; c += gridDim.x) {
        const int p0 = c * kChunkCells, n = min(kChunkCells, npos - p0);
        const size_t kb = (size_t) kvh * ctx + p0;
        for (int c0 = 0; c0 < n; c0 += kTile) {
            __syncthreads();
            for (int x = j; x < kTile * HD / 4; x += kThreads) {
                const int cell = x / (HD / 4), d4 = x % (HD / 4);
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (c0 + cell < n) {
                    if constexpr (!STREAM) {
                        const size_t row = kb + c0 + cell;
                        if (rs.cache_type == KvType::F32)
                            v = reinterpret_cast<const float4*>(kc + row * HD)[d4];
                        else
                            v = make_float4(kv_load(kc, row, d4 * 4, HD, rs.cache_type),
                                            kv_load(kc, row, d4 * 4 + 1, HD, rs.cache_type),
                                            kv_load(kc, row, d4 * 4 + 2, HD, rs.cache_type),
                                            kv_load(kc, row, d4 * 4 + 3, HD, rs.cache_type));
                    } else {
                        size_t row = 0;
                        const float* src = q36kvstream::at(kc, stream.khot, stream.kpref, kvh, p0 + c0 + cell, ctx, stream,
                                                           end, row);
                        v = make_float4(kv_load(src, row, d4 * 4, HD, rs.cache_type),
                                        kv_load(src, row, d4 * 4 + 1, HD, rs.cache_type),
                                        kv_load(src, row, d4 * 4 + 2, HD, rs.cache_type),
                                        kv_load(src, row, d4 * 4 + 3, HD, rs.cache_type));
                    }
                }
                *reinterpret_cast<float4*>(&kt[cell][d4 * 4]) = v;
            }
            __syncthreads();
            if (tg < G && c0 + tc < n) {
                float a = 0.f;
                for (int i = 0; i < HD; ++i) a += qs[tg][i] * kt[tc][i];
                sc[tg][c0 + tc] = a * scale;
            }
        }
        __syncthreads();
        const size_t o0 = (size_t) (lt * H + kvh * G) * n_chunk + c;
        for (int g = 0; g < G; ++g) {
            const float s = j < n ? sc[g][j] : -INFINITY;
            const float m = block_max(s, sh);
            const float e = j < n ? expf(s - m) : 0.f;
            sc[g][j] = e;
            const float l = block_sum(e, sh);
            if (j == 0) {
                pm[o0 + (size_t) g * n_chunk] = m;
                pl[o0 + (size_t) g * n_chunk] = l;
            }
        }
        float acc[G];
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g] = 0.f;
        for (int k = 0; k < n; ++k) {
            float v;
            if constexpr (!STREAM) {
                v = kv_load(vc, (size_t) kvh * ctx + p0 + k, j, HD, rs.cache_type);
            } else {
                size_t vr = 0;
                const float* vb = q36kvstream::at(vc, stream.vhot, stream.vpref, kvh, p0 + k, ctx, stream, end, vr);
                v = kv_load(vb, vr, j, HD, rs.cache_type);
            }
#pragma unroll
            for (int g = 0; g < G; ++g) acc[g] += sc[g][k] * v;
        }
#pragma unroll
        for (int g = 0; g < G; ++g) pacc[(o0 + (size_t) g * n_chunk) * HD + j] = acc[g];
    }
}
/// The chunks merged, times sigmoid(gate): grid (n_rows), block HD.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out, RowSel rs) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

}  // namespace q36attn
