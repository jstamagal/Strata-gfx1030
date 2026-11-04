// The unchanged GGUF rows, in native_mmvq's virtual-thread and FP32 reduction order.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"

namespace strata::kernels::hdetail {
struct X81 { half2 ds; int8_t qs[32]; };
struct Q80 { half d; int8_t qs[32]; };
struct Q6 { uint8_t ql[128], qh[64]; int8_t scales[16]; half d; };
struct Q5 { half2 dm; uint8_t scales[12], qh[32], qs[128]; };
static_assert(sizeof(X81) == 36 && sizeof(Q80) == 34 && sizeof(Q6) == 210 && sizeof(Q5) == 176);

__device__ __forceinline__ int word(const void* p) {
    int x; __builtin_memcpy(&x, p, 4); return x;
}
__device__ __forceinline__ float sum32(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    return v;
}
__device__ __forceinline__ bool finite(float v) { return (__float_as_uint(v) & 0x7f800000u) != 0x7f800000u; }
__device__ __forceinline__ float ninf() { return __int_as_float(0xff800000u); }
__device__ __forceinline__ float pinf() { return __int_as_float(0x7f800000u); }

// Integer codes/scales for consecutive weights of a Q6_K group, without any float rounding.
__device__ __forceinline__ int q6code(const Q6& b, int k) {
    const int h = k / 128, p = k % 128 / 32, l = k % 32;
    const int lo = (b.ql[64 * h + l + 32 * (p & 1)] >> (4 * (p >> 1))) & 15;
    const int hi = (b.qh[32 * h + l] >> (2 * p)) & 3;
    return lo | hi << 4;
}
__device__ __forceinline__ int q6scale(const Q6& b, int k) {
    return b.scales[8 * (k / 128) + k % 32 / 16 + 2 * (k % 128 / 32)];
}
__device__ __forceinline__ void q5scale(const Q5& b, int g, int& s, int& m) {
    if (g < 4) { s = b.scales[g] & 63; m = b.scales[g + 4] & 63; }
    else { s = (b.scales[g + 4] & 15) | ((b.scales[g - 4] >> 6) << 4);
           m = (b.scales[g + 4] >> 4) | ((b.scales[g] >> 6) << 4); }
}

// Each lane emulates four native_mmvq threads; columns never change a row's adds.
template<int TYPE, int NC>
__device__ __forceinline__ void exact_row(const void* weights, const X81* x, int row, int lane, float (&out)[NC]) {
    float p[4][NC] = {};
    if constexpr (TYPE == 14) {
        constexpr int blocks[10] = {0,4,8,1,5,9,2,6,3,7};
        constexpr int waves[10] = {0,0,0,1,1,1,2,2,3,3};
        const Q6* w = static_cast<const Q6*>(weights) + size_t(row) * 10;
#pragma unroll
        for (int c = 0; c < 10; ++c) {
            const Q6& b = w[blocks[c]];
            const int vl = word(b.ql + 4 * lane);
            const int vh = word(b.qh + 4 * (8 * (lane / 16) + lane % 8)) >> (2 * ((lane % 16) / 8));
            const int so = 8 * (lane / 16) + (lane % 16) / 4;
            const int bo = blocks[c] * 8 + 4 * (lane / 16) + (lane % 16) / 8;
            const int v0 = int((uint32_t((vl & 0x0f0f0f0f) | ((vh << 4) & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
            const int v1 = int((uint32_t(((vl >> 4) & 0x0f0f0f0f) | (vh & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
            const float d = __half2float(b.d);
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const X81* a = x + j * 80 + bo;
                float sf = 0.0f;
                sf += __low2float(a[0].ds) * strata_smul(STRATA_DP4A(v0, word(a[0].qs + 4 * (lane % 8)), 0), b.scales[so]);
                sf += __low2float(a[2].ds) * strata_smul(STRATA_DP4A(v1, word(a[2].qs + 4 * (lane % 8)), 0), b.scales[so + 4]);
                p[waves[c]][j] += d * sf;
            }
        }
    } else if constexpr (TYPE == 8) {
        constexpr int off[10] = {0,32,64,8,40,72,16,48,24,56};
        constexpr int waves[10] = {0,0,0,1,1,1,2,2,3,3};
        const Q80* w = static_cast<const Q80*>(weights) + size_t(row) * 80;
#pragma unroll
        for (int c = 0; c < 10; ++c) {
            const int b = lane / 4 + off[c], q = 8 * (lane % 4);
            const int v0 = word(w[b].qs + q), v1 = word(w[b].qs + q + 4);
            const float dw = __half2float(w[b].d);
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const X81& a = x[j * 80 + b];
                int v = STRATA_DP4A(v0, word(a.qs + q), 0);
                v = STRATA_DP4A(v1, word(a.qs + q + 4), v);
                p[waves[c]][j] += dw * __low2float(a.ds) * float(v);
            }
        }
    } else {
        const Q5* w = static_cast<const Q5*>(weights) + size_t(row) * 10;
        const int iqs = 2 * (lane % 16), bo = 2 * ((iqs / 2) / 4), pos = (iqs / 2) % 4;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
#pragma unroll
            for (int slot = 0; slot < 2; ++slot) {
                const int b = (lane + 32 * v) / 16 + 8 * slot;
                if (b >= 10) continue;
                const Q5& wb = w[b];
                const int vl0 = word(wb.qs + 16 * bo + 4 * pos), vl1 = word(wb.qs + 16 * bo + 4 * pos + 16);
                const int vh0 = word(wb.qh + 4 * pos) >> bo, vh1 = word(wb.qh + 4 * pos + 16) >> bo;
                const float2 dm = __half22float2(wb.dm);
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    float sd = 0.0f, sm = 0.0f;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        int sc, mn; q5scale(wb, bo + h, sc, mn);
                        const int w0 = ((vl0 >> (4 * h)) & 0x0f0f0f0f) | (((vh0 >> h) << 4) & 0x10101010);
                        const int w1 = ((vl1 >> (4 * h)) & 0x0f0f0f0f) | (((vh1 >> h) << 4) & 0x10101010);
                        const X81& a = x[j * 80 + b * 8 + bo + h];
                        const int a0 = word(a.qs + 4 * pos), a1 = word(a.qs + 4 * pos + 16);
                        const int dot = STRATA_DP4A(w0, a0, STRATA_DP4A(w1, a1, 0));
                        const int sx = STRATA_DP4A(0x01010101, a0, STRATA_DP4A(0x01010101, a1, 0));
                        sd += __low2float(a.ds) * strata_smul(dot, sc);
                        sm += __low2float(a.ds) * strata_smul(sx, mn);
                    }
                    p[v][j] += dm.x * sd - dm.y * sm;
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        float v = p[0][j]; v += p[1][j]; v += p[2][j]; v += p[3][j];
        out[j] = sum32(v);
    }
}
} // namespace strata::kernels::hdetail
