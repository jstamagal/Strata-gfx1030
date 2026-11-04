// src/prefill/moe_q4.cu - see include/strata/prefill/moe_q4.hpp.  The prompt path's Q4_0 / Q4_1 experts on RDNA2.
//
// The arithmetic.  A Q4_0 block is 32 weights w = d_w * (q - 8), q in 0..15: {f16 d_w; 16 bytes}, weight v < 16 in
// the low nibble of byte v, weight 16 + v in the high nibble of byte v.  A Q4_1 block is w = d_w * q + m_w:
// {f16 d_w; f16 m_w; 16 bytes}.  An activation block of 32 values is x = d_x * a, a = rint(x / d_x) in -127..127.
// Over one block
//     Q4_0:  sum w x = d_w * d_x * (sum q a - 8 * sum a)
//     Q4_1:  sum w x = d_x * (d_w * sum q a + m_w * sum a)
// The codes go into the int8 dot as they are (0..15 is a valid s8), and the zero point comes off as an INTEGER:
// sum a is an exact int (at most 32 * 127), computed once per token row and block in the kernel and shared by every
// weight row of the tile.  Nothing is rounded before the one multiply by the two scales, so a zero weight (q = 8)
// contributes exactly zero.  (llama.cpp's chain takes the offset from the q8_1 block's FP16 sum; on a real Q4_0
// expert that cancellation is where its error comes from - see the note in kernels/cuda/iq_kernels.cu.)
// |sum q a| <= 32 * 15 * 127 < 2^16: exact in int32 and in FP32.
//
// The tile.  One workgroup of 256 threads (8 waves) takes 64 weight rows x 32 token rows of one expert and walks K
// in chunks of 128 values.  Thread t owns weight row (t & 63) and the 8 token rows of group (t >> 6):
//   - the weights never touch LDS: a thread reads its row's 4 blocks (72 or 80 bytes) from the expert buffer and
//     widens the nibbles into 32 registers, once per chunk, used against all 8 token rows;
//   - the activations are staged in LDS (32 rows x 144 bytes = 4.6 KB).  Two waves share a token group, so every
//     LDS read of an activation word has one address for the whole wave;
//   - 8 float accumulators a thread, no shared-memory writes in the hot loop, no atomics, one store per output.
// LDS is 5.1 KB a workgroup, so the tile does not bound how many workgroups a CU keeps resident.
#include "strata/prefill/moe_q4.hpp"
#include "strata/kernels/gfx_arch.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill::q4 {
namespace {

constexpr int kTypeQ4_0 = 2, kTypeQ4_1 = 3;   // ggml_type
constexpr int THREADS = 256;
constexpr int TI = 64;                 // weight rows a tile
constexpr int TJ = 32;                 // token rows a tile
constexpr int JG = 8;                  // token rows a thread
constexpr int CHUNK = 128;             // K values a step
constexpr int CHUNK_BYTES = 16 + CHUNK;   // {float d[4]; s8 q[128]}
constexpr int CHUNK_WORDS = CHUNK_BYTES / 4;   // 36

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill q4 experts: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__device__ __forceinline__ float bits_float(uint32_t u) {
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}
__device__ __forceinline__ uint32_t float_bits(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}
__device__ __forceinline__ uint32_t load32(const uint8_t* p) {   // any alignment (a Q4_0 block is 18 bytes)
    uint32_t u;
    __builtin_memcpy(&u, p, 4);
    return u;
}
__device__ __forceinline__ float load_f16(const uint8_t* p) {
    unsigned short h;
    __builtin_memcpy(&h, p, 2);
    return __half2float(__ushort_as_half(h));
}

// ---- the activation quantizer -----------------------------------------------------------------------------------
// One workgroup: 8 chunks (1,024 values) of one row.  A lane holds 4 values, 8 lanes a 32-value block, a wave a chunk.
__global__ void __launch_bounds__(THREADS) q4_quantize_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                                              uint8_t* __restrict__ xq, int cols, int64_t ld) {
    const int t = threadIdx.x;
    const int64_t row = blockIdx.y;
    const int v0 = ((int) blockIdx.x * THREADS + t) * 4;        // first of this lane's 4 values
    const bool in = v0 < cols;                                   // cols is a multiple of 128: a whole wave is in or out
    const int64_t src = ids != nullptr ? (int64_t) ids[row] : row;
    const float* xr = x + src * ld;
    float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
    if (in) {
        f0 = xr[v0];
        f1 = xr[v0 + 1];
        f2 = xr[v0 + 2];
        f3 = xr[v0 + 3];
    }
    float amax = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
    // the block's 8 lanes: every lane of the wave takes part (no lane has left yet)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    if (!in) return;
    const float d = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q0 = (int) rintf(f0 * inv), q1 = (int) rintf(f1 * inv), q2 = (int) rintf(f2 * inv), q3 = (int) rintf(f3 * inv);
    const uint32_t packed = ((uint32_t) q0 & 0xffu) | (((uint32_t) q1 & 0xffu) << 8) | (((uint32_t) q2 & 0xffu) << 16) |
                            (((uint32_t) q3 & 0xffu) << 24);
    const int chunk = v0 / CHUNK, in_chunk = v0 % CHUNK;
    uint32_t* c = (uint32_t*) (xq + (size_t) row * (size_t) (cols / CHUNK) * CHUNK_BYTES + (size_t) chunk * CHUNK_BYTES);
    c[4 + in_chunk / 4] = packed;
    if ((t & 7) == 0) c[in_chunk / 32] = float_bits(d);
}

// ---- the product ------------------------------------------------------------------------------------------------
struct Args {
    const uint8_t* w;        // expert e's matrix at w + e * expert_bytes, its rows w_cols / 32 blocks each
    const uint8_t* xq;       // rows of (w_cols / 128) chunks
    const int32_t* bounds;   // n + 1
    const int32_t* ids;      // dst row of an activation row
    float* dst;
    int64_t ld_dst;
    uint64_t expert_bytes;
    int w_rows, w_cols;
};

template <bool Q41>
__global__ void __launch_bounds__(THREADS) q4_product_kernel(Args a) {
    __shared__ __align__(16) uint32_t act[TJ * CHUNK_WORDS];   // the tile's token rows, one chunk
    __shared__ int asum[TJ * 4];                               // sum of a row's int8 over each of the chunk's 4 blocks
    constexpr int BLOCK_BYTES = Q41 ? 20 : 18;
    constexpr int QS_OFF = Q41 ? 4 : 2;
    const int t = threadIdx.x;
    const int e = blockIdx.z;
    const int r0 = a.bounds[e], r1 = a.bounds[e + 1];
    const int j0 = r0 + (int) blockIdx.y * TJ;
    if (j0 >= r1) return;                                      // the same for the whole workgroup
    const int nj = min(TJ, r1 - j0);
    const int i = (int) blockIdx.x * TI + (t & (TI - 1));
    const bool irow = i < a.w_rows;
    const int jg = t >> 6;
    const int chunks = a.w_cols / CHUNK;
    const size_t x_row = (size_t) chunks * CHUNK_BYTES;
    const uint8_t* wr = a.w + (size_t) e * a.expert_bytes + (size_t) (irow ? i : 0) * (size_t) (a.w_cols / 32) * BLOCK_BYTES;
    const uint8_t* xt = a.xq + (size_t) j0 * x_row;

    float acc[JG];
#pragma unroll
    for (int jj = 0; jj < JG; ++jj) acc[jj] = 0.0f;

    for (int c = 0; c < chunks; ++c) {
        // stage the 32 token rows' chunk: 1,152 words, 4.5 a thread; rows past the expert's last one read as zero
        for (int n = t; n < TJ * CHUNK_WORDS; n += THREADS) {
            const int j = n / CHUNK_WORDS, k = n - j * CHUNK_WORDS;
            uint32_t v = 0;
            if (j < nj) v = ((const uint32_t*) (xt + (size_t) j * x_row + (size_t) c * CHUNK_BYTES))[k];
            act[n] = v;
        }
        __syncthreads();
        if (t < TJ * 4) {                                      // one thread per (token row, block): its exact integer sum
            const uint32_t* q = act + (t >> 2) * CHUNK_WORDS + 4 + (t & 3) * 8;
            int s = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) s = STRATA_DP4A((int) q[k], 0x01010101, s);
            asum[t] = s;
        }
        __syncthreads();
        if (irow) {
            uint32_t w[32];
            float dw[4], mw[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint8_t* blk = wr + (size_t) (c * 4 + b) * BLOCK_BYTES;
                dw[b] = load_f16(blk);
                mw[b] = Q41 ? load_f16(blk + 2) : 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t r = load32(blk + QS_OFF + 4 * k);
                    w[b * 8 + k] = r & 0x0f0f0f0fu;            // weights 4k .. 4k+3
                    w[b * 8 + 4 + k] = (r >> 4) & 0x0f0f0f0fu; // weights 16+4k .. 16+4k+3
                }
            }
#pragma unroll
            for (int jj = 0; jj < JG; ++jj) {
                const int j = jg * JG + jj;
                const uint32_t* aj = act + j * CHUNK_WORDS;
                float part = 0.0f;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int s = 0;
#pragma unroll
                    for (int k = 0; k < 8; ++k) s = STRATA_DP4A((int) w[b * 8 + k], (int) aj[4 + b * 8 + k], s);
                    const float dx = bits_float(aj[b]);
                    const int sa = asum[j * 4 + b];
                    if constexpr (Q41) part += dx * (dw[b] * (float) s + mw[b] * (float) sa);
                    else part += (dw[b] * dx) * (float) (s - 8 * sa);
                }
                acc[jj] += part;
            }
        }
        __syncthreads();                                       // every read of this chunk is done before the next is staged
    }
    if (!irow) return;
#pragma unroll
    for (int jj = 0; jj < JG; ++jj) {
        const int j = jg * JG + jj;
        if (j < nj) {
            const int64_t row = a.ids != nullptr ? (int64_t) a.ids[j0 + j] : (int64_t) (j0 + j);
            a.dst[row * a.ld_dst + i] = acc[jj];
        }
    }
}

// ---- the wide tiles (STRATA_PF_Q4_WIDE=1, opt-in) ------------------------------------------------------------------
// The same arithmetic, the same output bits, about 3.5x (Q4_0) and 2x (Q4_1) the 64 x 32 tile's speed on a V620
// (hip_prefill_q4_parity prints both).  What the 64 x 32 tile spends its time on is LDS: every 4 dots wait for a
// 16-byte ds_read of activations, and a compute unit serves those far slower than its lanes issue dots (measured: 9
// reads per 32 dots run at 9.6 T MAC/s, 9 per 64 at 18.0, none at 22-23).  So:
//
//   Q4_0 (q4_sload_kernel): no LDS and no barrier at all.  One workgroup is 256 weight rows x 32 token rows; thread t
//   owns weight row t and all 32 token rows.  The tile's token rows are the same for every thread, so their address
//   is uniform and the compiler reads a row's chunk with three scalar loads (s_load_dwordx16 x 2 + x4, 144 bytes into
//   SGPRs); the dots take the activation word straight from a scalar register.  The zero point goes into the
//   weights' bytes, q - 8 as a signed byte (-8..7), once per chunk and thread, so the int8 dot is sum (q - 8) a
//   itself - the same integer as sum q a - 8 sum a, with no per-row sums to keep.  A weight block is read by one
//   thread (the 64 x 32 tile reads it four times) and serves 32 token rows instead of 8.
//
//   Q4_1 (q4_blk_kernel): its m_w * sum a term needs each (token row, block)'s integer sum once per workgroup, which
//   is what LDS is for; so it keeps the staged activations and halves the reads per dot instead: 128 threads, each
//   with two weight rows in registers against every activation word it reads.
//
// A token row past the expert's last one is skipped (a uniform branch), so a tile with 3 rows does 3 rows of work.
constexpr int WI = 256;                // weight rows a workgroup
constexpr int WN = mmq::kGatherGroupMax;   // experts a launch: each has its own weight pointers

// The wide tiles' arguments: expert e's rows [0, split) start at w0[e], rows [split, w_rows) at w1[e].  So a launch
// reads an expert where it lies (its blob: gate at one offset, up at another) as well as from a group buffer.
struct ArgsW {
    const uint8_t* w0[WN];
    const uint8_t* w1[WN];
    const uint8_t* xq;       // rows of (w_cols / 128) chunks
    const int32_t* bounds;   // n + 1
    const int32_t* ids;      // dst row of an activation row
    float* dst;
    int64_t ld_dst;
    int w_rows, w_cols, split;
};
__device__ __forceinline__ const uint8_t* row_ptr(const ArgsW& a, int e, int i, size_t w_row) {
    const int ii = i < a.w_rows ? i : 0;
    return ii < a.split ? a.w0[e] + (size_t) ii * w_row : a.w1[e] + (size_t) (ii - a.split) * w_row;
}

// R weight rows a thread (256 / R threads a workgroup), TJ token rows a workgroup, KS 32-value blocks a step.
// IDENT: the 64 x 32 tile's float order (a chunk's 4 blocks summed first; needs KS == 4).
template <bool Q41, int R, int KS, int TJ, bool IDENT>
__global__ void __launch_bounds__(WI / R) q4_blk_kernel(ArgsW a) {
    static_assert(!IDENT || KS == 4, "the chunk's partial sum needs the whole chunk in one step");
    constexpr int NT = WI / R;
    __shared__ __align__(16) uint32_t act[TJ * CHUNK_WORDS];
    __shared__ int asum[TJ * 4];                               // Q4_1 only
    constexpr int BLOCK_BYTES = Q41 ? 20 : 18;
    constexpr int QS_OFF = Q41 ? 4 : 2;
    constexpr int ROW16 = CHUNK_WORDS / 4;                     // a row's chunk in 16-byte pieces: 9
    const int t = threadIdx.x;
    const int e = blockIdx.z;
    const int r0 = a.bounds[e], r1 = a.bounds[e + 1];
    const int j0 = r0 + (int) blockIdx.y * TJ;
    if (j0 >= r1) return;                                      // the same for the whole workgroup
    const int nj = min(TJ, r1 - j0);
    const int i0 = (int) blockIdx.x * WI + t;                  // this thread's rows: i0 + r * NT
    const int chunks = a.w_cols / CHUNK;
    const size_t x_row = (size_t) chunks * CHUNK_BYTES;
    const size_t w_row = (size_t) (a.w_cols / 32) * BLOCK_BYTES;
    const uint8_t* xt = a.xq + (size_t) j0 * x_row;

    float acc[R * TJ];
#pragma unroll
    for (int q = 0; q < R * TJ; ++q) acc[q] = 0.0f;

    for (int c = 0; c < chunks; ++c) {
        for (int n = t; n < nj * ROW16; n += NT) {
            const int j = n / ROW16, k = n - j * ROW16;
            uint4 v;
            __builtin_memcpy(&v, xt + (size_t) j * x_row + (size_t) c * CHUNK_BYTES + (size_t) k * 16, 16);
            *(uint4*) (act + n * 4) = v;
        }
        __syncthreads();
        if constexpr (Q41) {
            for (int n = t; n < TJ * 4; n += NT) {
                const uint32_t* q = act + (n >> 2) * CHUNK_WORDS + 4 + (n & 3) * 8;
                int s = 0;
                if ((n >> 2) < nj) {
#pragma unroll
                    for (int k = 0; k < 8; ++k) s = STRATA_DP4A((int) q[k], 0x01010101, s);
                }
                asum[n] = s;
            }
            __syncthreads();
        }
#pragma unroll
        for (int ks = 0; ks < 4; ks += KS) {
            uint32_t w[R][KS * 8];
            float dw[R][KS], mw[R][KS];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int i = i0 + r * NT;
                const uint8_t* wr = row_ptr(a, e, i, w_row);
#pragma unroll
                for (int b = 0; b < KS; ++b) {
                    const uint8_t* blk = wr + (size_t) (c * 4 + ks + b) * BLOCK_BYTES;
                    dw[r][b] = load_f16(blk);
                    mw[r][b] = Q41 ? load_f16(blk + 2) : 0.0f;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t x = load32(blk + QS_OFF + 4 * k);
                        uint32_t lo = x & 0x0f0f0f0fu, hi = (x >> 4) & 0x0f0f0f0fu;
                        if constexpr (!Q41) {                  // q - 8 in each byte, as a signed byte (no borrow crosses)
                            lo = ((lo | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
                            hi = ((hi | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
                        }
                        w[r][b * 8 + k] = lo;                  // weights 4k .. 4k+3
                        w[r][b * 8 + 4 + k] = hi;              // weights 16+4k .. 16+4k+3
                    }
                }
            }
#pragma unroll
            for (int jj = 0; jj < TJ; ++jj) {
                if (jj < nj) {
                    const uint32_t* aj = act + jj * CHUNK_WORDS;
                    float part[R];
#pragma unroll
                    for (int r = 0; r < R; ++r) part[r] = 0.0f;
#pragma unroll
                    for (int b = 0; b < KS; ++b) {
                        uint32_t av[8];
#pragma unroll
                        for (int k = 0; k < 8; ++k) av[k] = aj[4 + (ks + b) * 8 + k];
                        const float dx = bits_float(aj[ks + b]);
                        const float sa = Q41 ? (float) asum[jj * 4 + ks + b] : 0.0f;
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            int s = 0;
#pragma unroll
                            for (int k = 0; k < 8; ++k) s = STRATA_DP4A((int) w[r][b * 8 + k], (int) av[k], s);
                            float v;
                            if constexpr (Q41) v = dx * (dw[r][b] * (float) s + mw[r][b] * sa);
                            else v = (dw[r][b] * dx) * (float) s;
                            if constexpr (IDENT) part[r] += v;
                            else acc[r * TJ + jj] += v;
                        }
                    }
                    if constexpr (IDENT) {
#pragma unroll
                        for (int r = 0; r < R; ++r) acc[r * TJ + jj] += part[r];
                    }
                }
            }
        }
        __syncthreads();                                       // every read of this chunk is done before the next is staged
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int i = i0 + r * NT;
        if (i < a.w_rows) {
#pragma unroll
            for (int jj = 0; jj < TJ; ++jj) {
                if (jj < nj) {
                    const int64_t row = a.ids != nullptr ? (int64_t) a.ids[j0 + jj] : (int64_t) (j0 + jj);
                    a.dst[row * a.ld_dst + i] = acc[r * TJ + jj];
                }
            }
        }
    }
}

// R weight rows a thread (256 / R threads a workgroup), TJ token rows a workgroup; the activations by scalar loads
// (`xq` 4-byte aligned)
template <bool Q41, int R, int TJ>
__global__ void __launch_bounds__(WI / R) q4_sload_kernel(ArgsW a) {
    constexpr int NT = WI / R;
    constexpr int BLOCK_BYTES = Q41 ? 20 : 18;
    constexpr int QS_OFF = Q41 ? 4 : 2;
    const int t = threadIdx.x;
    const int e = blockIdx.z;
    const int r0 = a.bounds[e], r1 = a.bounds[e + 1];
    const int j0 = r0 + (int) blockIdx.y * TJ;
    if (j0 >= r1) return;
    const int nj = min(TJ, r1 - j0);
    const int i0 = (int) blockIdx.x * WI + t;                  // this thread's rows: i0 + r * NT
    const int chunks = a.w_cols / CHUNK;
    const int x_row_w = chunks * CHUNK_WORDS;                  // a token row in words
    const size_t w_row = (size_t) (a.w_cols / 32) * BLOCK_BYTES;
    const uint32_t* xc = (const uint32_t*) a.xq + (size_t) j0 * (size_t) x_row_w;   // the tile's first row, chunk c
    float acc[R * TJ];
#pragma unroll
    for (int q = 0; q < R * TJ; ++q) acc[q] = 0.0f;
    for (int c = 0; c < chunks; ++c, xc += CHUNK_WORDS) {
        uint32_t w[R][32];
        float dw[R][4], mw[R][4];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int i = i0 + r * NT;
            const uint8_t* wr = row_ptr(a, e, i, w_row);
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint8_t* blk = wr + (size_t) (c * 4 + b) * BLOCK_BYTES;
                dw[r][b] = load_f16(blk);
                mw[r][b] = Q41 ? load_f16(blk + 2) : 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t x = load32(blk + QS_OFF + 4 * k);
                    uint32_t lo = x & 0x0f0f0f0fu, hi = (x >> 4) & 0x0f0f0f0fu;
                    if constexpr (!Q41) {                      // q - 8 in each byte, as a signed byte (no borrow crosses)
                        lo = ((lo | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
                        hi = ((hi | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
                    }
                    w[r][b * 8 + k] = lo;                      // weights 4k .. 4k+3
                    w[r][b * 8 + 4 + k] = hi;                  // weights 16+4k .. 16+4k+3
                }
            }
        }
        const uint32_t* aj = xc;
#pragma unroll
        for (int jj = 0; jj < TJ; ++jj, aj += x_row_w) {
            if (jj < nj) {
                float part[R];
#pragma unroll
                for (int r = 0; r < R; ++r) part[r] = 0.0f;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const float dx = bits_float(aj[b]);
                    float sa = 0.0f;
                    if constexpr (Q41) {                       // the row's integer sum over the block
                        int si = 0;
#pragma unroll
                        for (int k = 0; k < 8; ++k) si = STRATA_DP4A(0x01010101, (int) aj[4 + b * 8 + k], si);
                        sa = (float) si;
                    }
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        int s = 0;
#pragma unroll
                        for (int k = 0; k < 8; ++k) s = STRATA_DP4A((int) w[r][b * 8 + k], (int) aj[4 + b * 8 + k], s);
                        if constexpr (Q41) part[r] += dx * (dw[r][b] * (float) s + mw[r][b] * sa);
                        else part[r] += (dw[r][b] * dx) * (float) s;
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r * TJ + jj] += part[r];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int i = i0 + r * NT;
        if (i < a.w_rows) {
#pragma unroll
            for (int jj = 0; jj < TJ; ++jj) {
                if (jj < nj) {
                    const int64_t row = a.ids != nullptr ? (int64_t) a.ids[j0 + jj] : (int64_t) (j0 + jj);
                    a.dst[row * a.ld_dst + i] = acc[r * TJ + jj];
                }
            }
        }
    }
}

template <bool Q41, int R, int TJ>
void launch_sload(const ArgsW& a, const mmq::Product& p, cudaStream_t s) {
    const dim3 g((unsigned) ((p.w_rows + WI - 1) / WI), (unsigned) ((p.max_rows + TJ - 1) / TJ), (unsigned) p.n);
    q4_sload_kernel<Q41, R, TJ><<<g, WI / R, 0, s>>>(a);
}

template <bool Q41, int R, int KS, int TJ, bool IDENT>
void launch_blk(const ArgsW& a, const mmq::Product& p, cudaStream_t s) {
    const dim3 g((unsigned) ((p.w_rows + WI - 1) / WI), (unsigned) ((p.max_rows + TJ - 1) / TJ), (unsigned) p.n);
    q4_blk_kernel<Q41, R, KS, TJ, IDENT><<<g, WI / R, 0, s>>>(a);
}

// ---- IQ4_NL on the 256-row tiles (STRATA_PF_NL=1, opt-in) -----------------------------------------------------------
// An IQ4_NL block is a Q4_0 block with another meaning of the code: {f16 d; 16 bytes}, weight v < 16 in the low
// nibble of byte v, weight 16 + v in the high nibble, w = d * kvalues_iq4nl[q].  39 of the IQ3_S pack's 48 layers
// keep their down projection in it.  This is q4_sload_kernel for that format, written to give llama.cpp's MMQ kernel
// output bit for bit, so it takes MMQ's own activations and repeats its arithmetic:
//   - the activations are block_q8_1_mmq in the D4 layout as quantize_mmq_q8_1 (or swiglu_quant) wrote them:
//     {float d[4]; int8 q[128]} for 128 values of one row, block (k / 128, row) at index (k / 128) * ncols_y + row.
//     The rows of a tile are 36 words apart, at an address that is the same for every thread: scalar loads;
//   - per 32 values MMQ adds d_w * d_x * (float) sum q a to the output's sum, in K order
//     (ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a / vec_dot_q8_0_q8_1_impl); so does this.
// What MMQ does and this does not: it walks K in steps of 256, so for a 640-column matrix it also adds four blocks
// that lie past the row (the next row's, times activations that are zero).  Those terms are +-0; they can turn an
// output of exactly -0 into +0 and change nothing else.  And it never reads past a matrix, so a down product can
// read an expert in its cache slot (STRATA_PF_DIRECT=1).
// The codes are decoded two at a time through a 256-entry table (byte -> the int8 of its low and of its high nibble).
struct NlPairs {
    uint16_t v[256];
    constexpr NlPairs() : v() {
        constexpr int kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};   // kvalues_iq4nl
        for (int b = 0; b < 256; ++b) v[b] = (uint16_t) (((unsigned) kv[b & 15] & 0xffu) | (((unsigned) kv[b >> 4] & 0xffu) << 8));
    }
};
static __device__ const NlPairs nl_pairs{};

struct ArgsN {
    const uint8_t* w0[WN];
    const uint8_t* w1[WN];
    const uint32_t* y;       // block_q8_1_mmq, D4 layout
    const int32_t* bounds;   // n + 1
    const int32_t* ids;      // dst row of an activation row
    float* dst;
    int64_t ld_dst;
    int64_t ncols_y;         // activation rows in y (the block index's stride)
    int w_rows, w_cols, split;
};

// STRATA_PF_NL_PERM=1 (opt-in): the 16 IQ4_NL values held in four registers and picked with v_perm_b32 (llama.cpp's
// get_int_from_table_16 on HIP) instead of 4 lookups of nl_pairs per 32-bit word, each a 32-address gather through
// the vector memory path.  The same int8 values, so the same dp4a sums and the same bits.
__device__ __forceinline__ void nl_perm_decode(uint32_t x, uint32_t& lo, uint32_t& hi) {
    constexpr uint32_t v0 = 0xBFAD9881u, v1 = 0xF6EADDCFu, v2 = 0x26190D01u, v3 = 0x71594535u;   // kvalues_iq4nl bytes
    const uint32_t qe = x, qo = x >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v1, v0, qe & 0x07070707u), ol = __builtin_amdgcn_perm(v1, v0, qo & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v3, v2, qe & 0x07070707u), oh = __builtin_amdgcn_perm(v3, v2, qo & 0x07070707u);
    lo = __builtin_amdgcn_perm(eh, el, 0x03020100u | ((qe & 0x08080808u) >> 1));
    hi = __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((qo & 0x08080808u) >> 1));
}
template <int TJ, bool PERM = false>
__global__ void __launch_bounds__(WI) nl_sload_kernel(ArgsN a) {
    constexpr int BLOCK_BYTES = 18;
    const int t = threadIdx.x;
    const int e = blockIdx.z;
    const int r0 = a.bounds[e], r1 = a.bounds[e + 1];
    const int j0 = r0 + (int) blockIdx.y * TJ;
    if (j0 >= r1) return;
    const int nj = min(TJ, r1 - j0);
    const int i = (int) blockIdx.x * WI + t;
    const int chunks = a.w_cols / CHUNK;
    const size_t w_row = (size_t) (a.w_cols / 32) * BLOCK_BYTES;
    const int ii = i < a.w_rows ? i : 0;
    const uint8_t* wr = ii < a.split ? a.w0[e] + (size_t) ii * w_row : a.w1[e] + (size_t) (ii - a.split) * w_row;
    float acc[TJ];
#pragma unroll
    for (int q = 0; q < TJ; ++q) acc[q] = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        uint32_t w[32];
        float dw[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint8_t* blk = wr + (size_t) (c * 4 + b) * BLOCK_BYTES;
            dw[b] = load_f16(blk);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t x = load32(blk + 2 + 4 * k);
                if constexpr (PERM) {
                    nl_perm_decode(x, w[b * 8 + k], w[b * 8 + 4 + k]);
                } else {
                const uint32_t t0 = nl_pairs.v[x & 0xffu], t1 = nl_pairs.v[(x >> 8) & 0xffu],
                               t2 = nl_pairs.v[(x >> 16) & 0xffu], t3 = nl_pairs.v[x >> 24];
                w[b * 8 + k] = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);   // weights 4k .. 4k+3
                w[b * 8 + 4 + k] = (t0 >> 8) | ((t1 >> 8) << 8) | ((t2 >> 8) << 16) | ((t3 >> 8) << 24);   // 16+4k .. 16+4k+3
                }
            }
        }
        const uint32_t* aj = a.y + ((size_t) c * (size_t) a.ncols_y + (size_t) j0) * CHUNK_WORDS;
#pragma unroll
        for (int jj = 0; jj < TJ; ++jj, aj += CHUNK_WORDS) {
            if (jj < nj) {
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const float dx = bits_float(aj[b]);
                    int sm = 0;
#pragma unroll
                    for (int k = 0; k < 8; ++k) sm = STRATA_DP4A((int) w[b * 8 + k], (int) aj[4 + b * 8 + k], sm);
                    acc[jj] += dw[b] * dx * (float) sm;    // MMQ's expression and order: d8_0 * d8_1 * (float) sumi
                }
            }
        }
    }
    if (i >= a.w_rows) return;
#pragma unroll
    for (int jj = 0; jj < TJ; ++jj) {
        if (jj < nj) {
            const int64_t row = a.ids != nullptr ? (int64_t) a.ids[j0 + jj] : (int64_t) (j0 + jj);
            a.dst[row * a.ld_dst + i] = acc[jj];
        }
    }
}

// ---- Q2_0 down products on the same tiles (STRATA_PF_Q2=1, opt-in) ------------------------------------------------
// A Q2_0 block: {f16 d; 16 bytes} for 64 values, value j in bits 2 (j % 4) of byte j / 4, w = d * (q - 1).  9 of the
// IQ3_S pack's 48 layers keep their down projection in it, and llama.cpp's MMQ ran them at about half the IQ4_NL
// tile's speed.  MMQ reads Q2_0 through its q8_0 dot (ggml_cuda_mmq_load_tiles_q2_0 + vec_dot_q8_0_q8_1_dp4a): per 32
// values d_w * d_x * (float) sum q a, in K order, from the D4 activations - which this repeats, as the IQ4_NL tile
// does (and with its one exception, the sign of an exact zero past a 640-column row).
// Four codes of a byte to four int8 (q - 1): spread the crumbs to bytes, then subtract 1 per byte without a borrow.
__device__ __forceinline__ uint32_t q2_byte4(uint32_t x) {
    const uint32_t v = (x & 0x03u) | ((x & 0x0cu) << 6) | ((x & 0x30u) << 12) | ((x & 0xc0u) << 18);
    return ((v | 0x80808080u) - 0x01010101u) ^ 0x80808080u;
}
template <int TJ>
__global__ void __launch_bounds__(WI) q2_sload_kernel(ArgsN a) {
    constexpr int BLOCK_BYTES = 18;   // 64 values
    const int t = threadIdx.x;
    const int e = blockIdx.z;
    const int r0 = a.bounds[e], r1 = a.bounds[e + 1];
    const int j0 = r0 + (int) blockIdx.y * TJ;
    if (j0 >= r1) return;
    const int nj = min(TJ, r1 - j0);
    const int i = (int) blockIdx.x * WI + t;
    const int chunks = a.w_cols / CHUNK;
    const size_t w_row = (size_t) (a.w_cols / 64) * BLOCK_BYTES;
    const int ii = i < a.w_rows ? i : 0;
    const uint8_t* wr = ii < a.split ? a.w0[e] + (size_t) ii * w_row : a.w1[e] + (size_t) (ii - a.split) * w_row;
    float acc[TJ];
#pragma unroll
    for (int q = 0; q < TJ; ++q) acc[q] = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        uint32_t w[32];
        float dw[4];
#pragma unroll
        for (int bb = 0; bb < 2; ++bb) {
            const uint8_t* blk = wr + (size_t) (c * 2 + bb) * BLOCK_BYTES;
            const float d = load_f16(blk);
            dw[2 * bb] = d;
            dw[2 * bb + 1] = d;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t x = load32(blk + 2 + 4 * k);   // values 16k .. 16k+15 of the block
#pragma unroll
                for (int m = 0; m < 4; ++m) w[bb * 16 + 4 * k + m] = q2_byte4((x >> (8 * m)) & 0xffu);
            }
        }
        const uint32_t* aj = a.y + ((size_t) c * (size_t) a.ncols_y + (size_t) j0) * CHUNK_WORDS;
#pragma unroll
        for (int jj = 0; jj < TJ; ++jj, aj += CHUNK_WORDS) {
            if (jj < nj) {
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const float dx = bits_float(aj[b]);
                    int sm = 0;
#pragma unroll
                    for (int k = 0; k < 8; ++k) sm = STRATA_DP4A((int) w[b * 8 + k], (int) aj[4 + b * 8 + k], sm);
                    acc[jj] += dw[b] * dx * (float) sm;
                }
            }
        }
    }
    if (i >= a.w_rows) return;
#pragma unroll
    for (int jj = 0; jj < TJ; ++jj) {
        if (jj < nj) {
            const int64_t row = a.ids != nullptr ? (int64_t) a.ids[j0 + jj] : (int64_t) (j0 + jj);
            a.dst[row * a.ld_dst + i] = acc[jj];
        }
    }
}

int g_nl_tj = 0;                       // set_nl_tile's (tests); 0: the launch's own choice

int g_variant = -1;                    // set_variant's (tests); -1: the environment decides

int variant() {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_Q4_WIDE");
        return v != nullptr && v[0] == '1' ? 1 : 0;
    }();
    return g_variant >= 0 ? g_variant : env;
}

bool device_is_gfx103x() {
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        cudaDeviceProp p;
        const bool ok = cudaGetDeviceProperties(&p, dev) == cudaSuccess && std::strncmp(p.gcnArchName, "gfx103", 6) == 0;
        per_dev[dev] = ok ? 2 : 1;
    }
    return per_dev[dev] == 2;
}

}  // namespace

bool covers(int t) { return t == kTypeQ4_0 || t == kTypeQ4_1; }

void set_nl_tile(int tj) { g_nl_tj = tj; }

bool gfx103x() { return device_is_gfx103x(); }

bool nl_on() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_PF_NL");
        return v != nullptr && v[0] == '1';
    }();
    return env && device_is_gfx103x();
}

// The down shape only (K 640).  Measured on a V620 against llama.cpp's MMQ, 16 experts: down x 40 rows 0.100 ->
// 0.095 ms, x 320 rows 15.2 -> 17.2 T MAC/s, a skewed 5,120 rows 0.610 -> 0.493 ms; gate/up (K 2560) loses, x 40 rows
// 0.166 -> 0.193 ms, x 320 rows 18.3 -> 17.0 T MAC/s, so that shape stays with MMQ (no pack here has it in IQ4_NL).
bool nl_fits(int t, int64_t w_cols) { return t == 20 /* GGML_TYPE_IQ4_NL */ && w_cols == 640 && nl_on(); }

void nl_run(const mmq::Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    if (p.w_cols % CHUNK != 0 || p.n > WN || ((uintptr_t) p.xq & 3) != 0) {
        std::fprintf(stderr, "prefill iq4_nl experts: %lld columns, %d experts: not covered\n", (long long) p.w_cols, p.n);
        std::exit(1);
    }
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "strata: prompt kernels: IQ4_NL experts on the 256-row tiles (STRATA_PF_NL=1: MMQ's output "
                             "bits)\n");
    }
    ArgsN a{};
    for (int e = 0; e < p.n; ++e) {
        a.w0[e] = p.each ? (const uint8_t*) p.w0[e] : (const uint8_t*) p.w + (size_t) e * p.expert_bytes;
        a.w1[e] = p.each ? (const uint8_t*) p.w1[e] : a.w0[e];
    }
    a.split = p.each ? (int) p.w_split : (int) p.w_rows;
    a.y = (const uint32_t*) p.xq;
    a.bounds = p.bounds;
    a.ids = p.ids;
    a.dst = p.dst;
    a.ld_dst = p.ld_dst;
    a.ncols_y = p.total_rows;
    a.w_rows = (int) p.w_rows;
    a.w_cols = (int) p.w_cols;
    const cudaStream_t s = (cudaStream_t) stream;
    // 32 token rows a tile.  48 and 64 (an expert's weights decoded once for more rows) were measured on a V620 and
    // lose: 16 experts x 40 rows of down take 0.095 ms at 32, 0.145 at 48, 0.147 at 64 (the accumulators and the
    // decoded weights no longer fit the registers).  set_nl_tile keeps them for a test.
    // STRATA_PF_NL_TJ=16, 20, 24 or 40 (opt-in): another tile height; each output's sum runs in the same order
    // whatever the height, so the bits are the same.
    static const int env_tj = [] {
        const char* v = std::getenv("STRATA_PF_NL_TJ");
        const int t = v ? std::atoi(v) : 0;
        return t == 16 || t == 20 || t == 24 || t == 40 ? t : 0;
    }();
    const int tj = g_nl_tj > 0 ? g_nl_tj : env_tj > 0 ? env_tj : 32;
    const dim3 g((unsigned) ((p.w_rows + WI - 1) / WI), (unsigned) ((p.max_rows + tj - 1) / tj), (unsigned) p.n);
    static const bool perm = [] { const char* v = std::getenv("STRATA_PF_NL_PERM"); return v != nullptr && v[0] == '1'; }();
    if (perm && tj == 32) nl_sload_kernel<32, true><<<g, WI, 0, s>>>(a);
    else if (tj == 64) nl_sload_kernel<64><<<g, WI, 0, s>>>(a);
    else if (tj == 48) nl_sload_kernel<48><<<g, WI, 0, s>>>(a);
    else if (tj == 40) nl_sload_kernel<40><<<g, WI, 0, s>>>(a);
    else if (tj == 24) nl_sload_kernel<24><<<g, WI, 0, s>>>(a);
    else if (tj == 20) nl_sload_kernel<20><<<g, WI, 0, s>>>(a);
    else if (tj == 16) nl_sload_kernel<16><<<g, WI, 0, s>>>(a);
    else nl_sload_kernel<32><<<g, WI, 0, s>>>(a);
    ck(cudaGetLastError(), "product (iq4_nl)");
}

bool q2_on() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_PF_Q2");
        return v != nullptr && v[0] == '1';
    }();
    return env && device_is_gfx103x();
}
bool q2_fits(int t, int64_t w_cols) { return t == 42 /* GGML_TYPE_Q2_0 */ && w_cols == 640 && q2_on(); }

void q2_run(const mmq::Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    if (p.w_cols % CHUNK != 0 || p.n > WN || ((uintptr_t) p.xq & 3) != 0) {
        std::fprintf(stderr, "prefill q2_0 experts: %lld columns, %d experts: not covered\n", (long long) p.w_cols, p.n);
        std::exit(1);
    }
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "strata: prompt kernels: Q2_0 down experts on the 256-row tiles (STRATA_PF_Q2=1: MMQ's output "
                             "bits)\n");
    }
    ArgsN a{};
    for (int e = 0; e < p.n; ++e) {
        a.w0[e] = p.each ? (const uint8_t*) p.w0[e] : (const uint8_t*) p.w + (size_t) e * p.expert_bytes;
        a.w1[e] = p.each ? (const uint8_t*) p.w1[e] : a.w0[e];
    }
    a.split = p.each ? (int) p.w_split : (int) p.w_rows;
    a.y = (const uint32_t*) p.xq;
    a.bounds = p.bounds;
    a.ids = p.ids;
    a.dst = p.dst;
    a.ld_dst = p.ld_dst;
    a.ncols_y = p.total_rows;
    a.w_rows = (int) p.w_rows;
    a.w_cols = (int) p.w_cols;
    const cudaStream_t s = (cudaStream_t) stream;
    // 24 token rows: hip_prefill_expert_bench, 16 x 40 rows 0.089 ms (16: 0.095, 20: 0.099, 32: 0.108), skewed 5,120
    // rows 0.483 ms (0.564, 0.555, 0.569); set_nl_tile sets both tiles' height (tests)
    const int tj = g_nl_tj > 0 ? g_nl_tj : 24;
    const dim3 g((unsigned) ((p.w_rows + WI - 1) / WI), (unsigned) ((p.max_rows + tj - 1) / tj), (unsigned) p.n);
    if (tj == 16) q2_sload_kernel<16><<<g, WI, 0, s>>>(a);
    else if (tj == 20) q2_sload_kernel<20><<<g, WI, 0, s>>>(a);
    else if (tj == 24) q2_sload_kernel<24><<<g, WI, 0, s>>>(a);
    else q2_sload_kernel<32><<<g, WI, 0, s>>>(a);
    ck(cudaGetLastError(), "product (q2_0)");
}

void set_variant(int v) { g_variant = v; }

bool direct() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_PF_DIRECT");
        return v != nullptr && v[0] == '1';
    }();
    return env && enabled();
}

bool enabled() {
    // STRATA_PF_Q4=0 turns it off (those experts take the FP16 path again); unset or anything else: on, on gfx103x
    static const bool env_on = [] {
        const char* v = std::getenv("STRATA_PF_Q4");
        return !(v != nullptr && v[0] == '0');
    }();
    if (!env_on || !device_is_gfx103x()) return false;
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "strata: prompt kernels: Q4_0 / Q4_1 experts on the RDNA2 int8 kernel (64 x 32 tiles, exact integer "
                             "zero point; STRATA_PF_Q4=0 turns it off)\n");
    }
    return true;
}

bool fits(int t, int64_t w_rows) { return covers(t) && w_rows > 0 && enabled(); }

void quantize(const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    if (cols <= 0 || cols % CHUNK != 0) {
        std::fprintf(stderr, "prefill q4 experts: %lld columns is not a multiple of %d\n", (long long) cols, CHUNK);
        std::exit(1);
    }
    const dim3 grid((unsigned) ((cols + THREADS * 4 - 1) / (THREADS * 4)), (unsigned) rows, 1);
    q4_quantize_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(x, ids, (uint8_t*) xq, (int) cols, ld);
    ck(cudaGetLastError(), "quantize");
}

void run(const mmq::Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    if (!covers(p.type) || p.w_cols % CHUNK != 0) {
        std::fprintf(stderr, "prefill q4 experts: type %d with %lld columns is not covered\n", p.type, (long long) p.w_cols);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    // STRATA_PF_Q4_WIDE=1 (the scalar loads need the activations word-aligned: they are, the quantizer writes words)
    const bool wide = (variant() == 1 || p.each) && ((uintptr_t) p.xq & 3) == 0 && p.n <= WN;
    if (p.each && !wide) {
        std::fprintf(stderr, "prefill q4 experts: a product over the experts' own blobs needs the wide tiles\n");
        std::exit(1);
    }
    if (wide) {
        static bool said = false, said_each = false;
        if (!said && g_variant < 0) {
            said = true;
            std::fprintf(stderr, "strata: prompt kernels: Q4_0 / Q4_1 experts on the 256 x 32 tiles (STRATA_PF_Q4_WIDE=1: "
                                 "the same output bits)\n");
        }
        if (p.each && !said_each) {
            said_each = true;
            std::fprintf(stderr, "strata: prompt kernels: Q4_0 / Q4_1 experts read from their own blobs, no group copy "
                                 "(STRATA_PF_DIRECT=1)\n");
        }
        ArgsW a{};
        for (int e = 0; e < p.n; ++e) {
            a.w0[e] = p.each ? (const uint8_t*) p.w0[e] : (const uint8_t*) p.w + (size_t) e * p.expert_bytes;
            a.w1[e] = p.each ? (const uint8_t*) p.w1[e] : a.w0[e];
        }
        a.split = p.each ? (int) p.w_split : (int) p.w_rows;
        a.xq = (const uint8_t*) p.xq;
        a.bounds = p.bounds;
        a.ids = p.ids;
        a.dst = p.dst;
        a.ld_dst = p.ld_dst;
        a.w_rows = (int) p.w_rows;
        a.w_cols = (int) p.w_cols;
        // (Q4_0 with 64 token rows a tile, so that a 2,048-token chunk's 40 rows an expert decode its weights once:
        // measured slower on a V620, 16 experts x 40 rows 0.163 -> 0.221 ms, x 320 rows 22.5 -> 17.4 T MAC/s)
        if (p.type == kTypeQ4_1) launch_blk<true, 2, 4, 32, true>(a, p, s);
        else launch_sload<false, 1, 32>(a, p, s);
        ck(cudaGetLastError(), "product (wide)");
        return;
    }
    Args a{};
    a.w = (const uint8_t*) p.w;
    a.xq = (const uint8_t*) p.xq;
    a.bounds = p.bounds;
    a.ids = p.ids;
    a.dst = p.dst;
    a.ld_dst = p.ld_dst;
    a.expert_bytes = (uint64_t) p.expert_bytes;
    a.w_rows = (int) p.w_rows;
    a.w_cols = (int) p.w_cols;
    const dim3 grid((unsigned) ((p.w_rows + TI - 1) / TI), (unsigned) ((p.max_rows + TJ - 1) / TJ), (unsigned) p.n);
    if (p.type == kTypeQ4_1) q4_product_kernel<true><<<grid, THREADS, 0, s>>>(a);
    else q4_product_kernel<false><<<grid, THREADS, 0, s>>>(a);
    ck(cudaGetLastError(), "product");
}

}  // namespace strata::prefill::q4
