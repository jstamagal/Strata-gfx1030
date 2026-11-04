// Opt-in q36 decode kernels. Shapes are runtime model dimensions; constants below
// describe wave32, quant blocks and launch geometry, never a model.
#pragma once
#include "q36_kernels.cuh"
#include <cstdlib>
#include <cstring>

namespace q36decode {
inline bool enabled(const char* name) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, "1") == 0;
}
struct Q8Mat {
    const uint8_t* w;
    const half* scales; // optional existing lossless packed copy
    float* out;
    int nb, rows, tiles;
};
// Q8 dot loop adapted from llama.cpp-rdna2 mmvq.cu mul_mat_vec_q and
// vecdotq.cuh vec_dot_q8_0_q8_1 (MIT, third_party/ggml/LICENSE).
// Keep Strata's four original wave partials, then add them in wave order.
// Unlike q36k::q8_row_dot, don't keep all K's weights live at once.
template<int NC>
__device__ __forceinline__ void q8_dot(Q8Mat a, const q36k::QB* x, int row, int lane, float (&out)[NC]) {
    float p[NC][4] = {};
    const int iq = 2 * (lane & 3);
    for (int j = 0; j < a.nb; j += 32) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int k = j + 8 * w + (lane >> 2);
            if (k >= a.nb) continue;
            const size_t b = (size_t) row * a.nb + k;
            int v0, v1;
            float d;
            if (a.scales) {
                const int* v = (const int*) (a.w + b * 32) + iq;
                v0 = v[0]; v1 = v[1]; d = __half2float(a.scales[b]);
            } else {
                const uint16_t* v = (const uint16_t*) (a.w + b * 34);
                v0 = (int) ((uint32_t) v[1 + 2 * iq] | ((uint32_t) v[2 + 2 * iq] << 16));
                v1 = (int) ((uint32_t) v[3 + 2 * iq] | ((uint32_t) v[4 + 2 * iq] << 16));
                d = __half2float(__ushort_as_half(v[0]));
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const q36k::QB* u = x + (size_t) c * a.nb + k;
                const int* q = (const int*) u->qs;
                int si = STRATA_DP4A(v0, q[iq], 0);
                si = STRATA_DP4A(v1, q[iq + 1], si);
                p[c][w] += d * __low2float(u->ds) * float(si);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float s = p[c][0]; s += p[c][1]; s += p[c][2]; s += p[c][3];
        out[c] = q36k::wsum(s);
    }
}
struct AB {
    const float *a = nullptr, *b = nullptr, *x = nullptr, *dt = nullptr, *ssm = nullptr;
    float *alpha = nullptr, *beta = nullptr;
    int n = 0, heads = 0;
};
// Horizontal fusion: disjoint blocks compute independent projections. No grid
// barrier, changed reduction, or intermediate quantization. Also used for one mat.
template<int NC>
__global__ void __launch_bounds__(128) projections(Q8Mat a, Q8Mat b, Q8Mat c, const q36k::QB* x, AB ab) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    int tile = blockIdx.x;
    Q8Mat m = a;
    const int na = a.tiles, nb = b.tiles, nc = c.tiles;
    if (tile >= na + nb + nc) {
        const int o = (tile - na - nb - nc) * 4 + wave;
        if (o >= 2 * ab.heads) return;
        const bool beta = o >= ab.heads;
        const int r = beta ? o - ab.heads : o;
        const float4* w = (const float4*) ((beta ? ab.b : ab.a) + (size_t) r * ab.n);
#pragma unroll
        for (int t = 0; t < NC; ++t) {
            const float4* h = (const float4*) (ab.x + (size_t) t * ab.n);
            float acc = 0.f;
            for (int i = lane; i < ab.n / 4; i += 32) {
                const float4 v = w[i], u = h[i];
                acc += v.x * u.x + v.y * u.y + v.z * u.z + v.w * u.w;
            }
            acc = q36k::wsum(acc);
            if (lane == 0) {
                if (beta) ab.beta[(size_t) t * ab.heads + r] = q36k::sigm(acc);
                else {
                    const float v = acc + ab.dt[r];
                    ab.alpha[(size_t) t * ab.heads + r] = (v > 20.f ? v : log1pf(expf(v))) * ab.ssm[r];
                }
            }
        }
        return;
    }
    if (tile >= na + nb) { tile -= na + nb; m = c; }
    else if (tile >= na) { tile -= na; m = b; }
    for (int row = tile * 4 + wave; row < m.rows; row += m.tiles * 4) {
        float res[NC];
        q8_dot<NC>(m, x, row, lane, res);
        if (lane == 0)
#pragma unroll
            for (int t = 0; t < NC; ++t) m.out[(size_t) t * m.rows + row] = res[t];
    }
}
inline bool project(Q8Mat a, Q8Mat b, Q8Mat c, const void* x, AB ab, int T, cudaStream_t st) {
    const int grid = a.tiles + b.tiles + c.tiles + (2 * ab.heads + 3) / 4;
#define Q36_PROJECT(N) case N: projections<N><<<grid, 128, 0, st>>>(a, b, c, (const q36k::QB*) x, ab); return true;
    switch (T) { Q36_PROJECT(1) Q36_PROJECT(2) Q36_PROJECT(3) Q36_PROJECT(4) }
#undef Q36_PROJECT
    return false;
}
// Shared expert: eight waves each compute four gate/up pairs; a whole block
// then quantizes the 32 hidden values. Avoid the old 1024-thread workgroup.
template<int NC>
__global__ void __launch_bounds__(256) shared_gu(Q8Mat g, Q8Mat u, const q36k::QB* x, q36k::QB* h) {
    __shared__ float gs[NC][32], us[NC][32];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int r = wave; r < 32; r += 8) {
        float a[NC], b[NC];
        q8_dot<NC>(g, x, blockIdx.x * 32 + r, lane, a);
        q8_dot<NC>(u, x, blockIdx.x * 32 + r, lane, b);
        if (lane == 0)
#pragma unroll
            for (int t = 0; t < NC; ++t) { gs[t][r] = a[t]; us[t][r] = b[t]; }
    }
    __syncthreads();
    if (wave < NC) {
        const float gi = gs[wave][lane];
        const float v = __fmul_rn(__fdividef(gi, __fadd_rn(1.f, __expf(-gi))), us[wave][lane]);
        // A store prevents HIP contracting the product into the quantizer sum.
        gs[wave][lane] = v;
        q36k::q8_store(h + (size_t) wave * (g.rows / 32), blockIdx.x * 32 + lane,
                      ((volatile float*) gs[wave])[lane], lane);
    }
}
inline bool shared(Q8Mat g, Q8Mat u, const void* x, void* h, int T, cudaStream_t st) {
#define Q36_SHARED(N) case N: shared_gu<N><<<g.rows / 32, 256, 0, st>>>(g, u, (const q36k::QB*) x, (q36k::QB*) h); return true;
    switch (T) { Q36_SHARED(1) Q36_SHARED(2) Q36_SHARED(3) Q36_SHARED(4) }
#undef Q36_SHARED
    return false;
}
// Initial/device-boundary norm and its quantization, same tree as k_rms.
__global__ void __launch_bounds__(256) rms_q(const float* x, const float* w, float* h, q36k::QB* q, int n, float eps) {
    __shared__ float sh[256];
    const size_t base = (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) s += x[base + i] * x[base + i];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 256) {
        h[base + i] = __fmul_rn(__fmul_rn(x[base + i], r), w[i]);
        q36k::q8_store(q + base / 32, i, ((volatile float*) h)[base + i], threadIdx.x & 31);
    }
}
inline size_t moe_lds(int n, int k) { return (size_t) k * sizeof(const float*) + (size_t) (n + 256) * sizeof(float); }
// Same arithmetic and 256-lane norm tree as k_moe_out_small, runtime LDS.
__global__ void __launch_bounds__(1024) moe_wide(float* x, const float* parts, const int32_t* prow, const float* wts,
        int K, const float* shared, const float* sgate, int sgs, const float* wn, float* h, int n, float eps, q36k::QB* hq) {
    extern __shared__ double storage[];
    const float** pk = (const float**) storage;
    float* vv = (float*) (pk + K);
    float* sh = vv + n;
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        pk[k] = parts + (size_t) (prow ? prow[t * K + k] : t * K + k) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K;
    const float gs = q36k::sigm(sgate[(size_t) t * sgs]);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v; vv[i] = v;
    }
    if (!wn) return;
    __syncthreads();
    if (threadIdx.x < 256) {
        float s = 0.f;
        for (int i = threadIdx.x; i < n; i += 256) s += vv[i] * vv[i];
        sh[threadIdx.x] = s;
    }
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        h[(size_t) t * n + i] = __fmul_rn(__fmul_rn(vv[i], r), wn[i]);
        if (hq) q36k::q8_store(hq + (size_t) t * (n / 32), i, ((volatile float*) h)[(size_t) t * n + i], threadIdx.x & 31);
    }
}
} // namespace q36decode
