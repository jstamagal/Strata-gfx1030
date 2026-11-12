// IQ3_S down rows for RDNA2 (Ornith 1.5 AD-IQ3_XXS: IQ3_XXS gate/up, IQ3_S down, FF 1024, or 512 a card with TP).
// Included inside iq_kernels.cu's anonymous namespace, after q36_tpx_expert.cuh.
//
// A down row is n_ff / 32 parts of 32 weights (16 with TP, 32 without) and one lane takes one part, so a row is 16 or 32
// lanes and a 256-thread block takes 16 or 8 rows a pass.  What differs from native_down_multi_kernel / tpx_down_half_kernel:
//   - the lane's part of the activation never changes with the row, so its 8 words and its scale are read once per
//     entry and stay in registers while the block walks its rows;
//   - a part's weight bytes are five loads (8 codes, the high bits, 4 sign bytes, the scale nibble, d), none split in
//     2-byte halves;
//   - the 9-bit codebook is 2 KB of LDS, staged once a block and paid over every pass of the block, not eight global
//     table loads a part that the streaming weights keep pushing out of the cache;
//   - the signs come from arithmetic: nibble * 0x00204081 & 0x01010101 puts sign bit k in byte k, and every codebook
//     byte is odd, so (g ^ m) + (m & 0x01010101) negates the chosen bytes with no carry between them;
//   - the row sum is DPP row_xmask moves inside the row's 16 lanes (permlanex16 first for a 32-lane row).
// The integers are Split<21>'s and the floats are row_dot_multi's: (d * ds) * sumi per part, then the xor butterfly
// 16 (a 16-lane row adds the reference's idle lanes' 0 there), 8, 4, 2, 1.  Every output is bit for bit
// native_down_multi_kernel<21>'s (tests/hip/decode_expert_lab.cpp --tpx-compare --pair 18/21).
#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906)
#define STRATA_Q36_DOWN_IQ3S_BUILT 1

template<int PL>   // lanes a row: 16 or 32
__global__ void __launch_bounds__(256) q36_down_iq3s_kernel(const unsigned long long* __restrict__ grp_ptr,
    const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups,
    const int32_t* __restrict__ ent_dst, const block_q8_1* __restrict__ hq,
    NativeExpertLayout L, float* __restrict__ out) {
    const int ng = *n_groups;
    if ((int) blockIdx.y >= ng) return;
    __shared__ uint32_t s_grid[512];
    for (int i = threadIdx.x; i < 512; i += 256) s_grid[i] = iq3s_grid[i];
    __syncthreads();
    constexpr int RP = 256 / PL;                     // rows a pass
    const int t = threadIdx.x & (PL - 1);            // this lane's part of the row
    const int j = t & 7;                             // ... inside its 256-value block
    const int row_of = threadIdx.x / PL;
    const int n_embd = (int) L.n_embd, hb = (int) (L.n_ff / 32);
    const size_t part_off = L.down_off + (size_t) (t >> 3) * sizeof(block_iq3_s);
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* base = (const uint8_t*) grp_ptr[g] + part_off;
        const int e1 = grp_start[g + 1];
        for (int e = grp_start[g]; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            int x[GRP_NC][8];
            float ds[GRP_NC];
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) {
                const block_q8_1* a = hq + (size_t) (e + min(c, n - 1)) * hb + t;
                ds[c] = __low2float(a->ds);
#pragma unroll
                for (int l = 0; l < 8; ++l) x[c][l] = get_int_b4(a->qs, l);
            }
            for (int r = blockIdx.x * RP + row_of; r < n_embd; r += gridDim.x * RP) {
                const uint8_t* b = base + (size_t) r * L.d_row;
                uint16_t dh;
                uint32_t q[2], sg;
                __builtin_memcpy(&dh, b, 2);
                __builtin_memcpy(q, b + 2 + 8 * j, 8);
                __builtin_memcpy(&sg, b + 74 + 4 * j, 4);
                const uint32_t qh = b[66 + j];
                const int ls = 1 + 2 * ((b[106 + (j >> 1)] >> (4 * (j & 1))) & 15);
                int w[8];
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const uint32_t code = ((q[l >> 2] >> (8 * (l & 3))) & 0xff) | (((qh >> l) & 1) << 8);
                    const uint32_t one = (((sg >> (4 * l)) & 15) * 0x00204081u) & 0x01010101u;
                    w[l] = (int) ((s_grid[code] ^ ((one << 8) - one)) + one);
                }
                const float dw = __half2float(reinterpret_cast<const __half&>(dh));
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) {
                    if (c < n) {
                        int sumi = 0;
#pragma unroll
                        for (int l = 0; l < 8; ++l) sumi = ggml_cuda_dp4a(w[l], x[c][l], sumi);
                        sumi = strata_smul(sumi, ls);
                        const float d = dw * ds[c];
                        float p = 0.0f;
                        p += d * sumi;
                        const float v = PL == 32 ? s26ts::tplain<16>(p) : s26ts::tplain<8>(p);
                        if (t == 0) out[(size_t) ent_dst[e + c] * n_embd + r] = v;
                    }
                }
            }
        }
    }
}

// STRATA_Q36_DOWN_IQ3S=0 keeps the other down kernels (the A/B switch); the lab's "reference" override does too
bool launch_q36_down_iq3s(const NativeExpertLayout& L, unsigned gy, cudaStream_t s, const unsigned long long* grp_ptr,
                          const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                          const block_q8_1* hq, float* out) {
    static const bool off = [] { const char* v = std::getenv("STRATA_Q36_DOWN_IQ3S"); return v && v[0] == '0'; }();
    // passes of a block over its rows, so the staged codebook is paid once per 64 (32) rows.  2x Radeon Pro V620, 8 groups
    // of 1 entry, n_embd 3072: FF 512 14.1 / 13.4 / 13.0 / 14.2 us at 2 / 4 / 8 / 16 passes (27 groups, 4 tokens: 37.9 /
    // 36.1 / 37.1 / 40.5), FF 1024 32.2 / 31.0 / 30.9 / 31.4
    constexpr int passes = 4;
    if (off || g_tpx_down == 0 || L.d_type != 21 || L.tail_off || L.had_block > 0 || L.n_embd <= 0) return false;
    if (L.n_ff == 512) {
        const dim3 grid((unsigned) ((L.n_embd + 16 * passes - 1) / (16 * passes)), gy);
        q36_down_iq3s_kernel<16><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    } else if (L.n_ff == 1024) {
        const dim3 grid((unsigned) ((L.n_embd + 8 * passes - 1) / (8 * passes)), gy);
        q36_down_iq3s_kernel<32><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    } else {
        return false;
    }
    return true;
}
#else
inline bool launch_q36_down_iq3s(const NativeExpertLayout&, unsigned, cudaStream_t, const unsigned long long*, const int32_t*,
                                 const int32_t*, const int32_t*, const block_q8_1*, float*) { return false; }
#endif
