// Included at the end of iq_kernels.cu's strata::kernels namespace: reuses the
// already compiled dot decoders, staged codebooks and reference launchers.
namespace {
template<int TY, bool STAGE>
__device__ __forceinline__ void q36_expert_dot(const uint8_t* w, const block_q8_1* x, const int (&off)[GRP_NC],
        int n, int nb, int lane, float (&v)[GRP_NC], const uint32_t* grid) {
    if constexpr (kSplit<TY>) row_dot_multi<TY, GRP_NC, false, STAGE>(w, x, off, n, nb, lane, v, grid);
    else {
#pragma unroll
        for (int c = 0; c < GRP_NC; ++c)
            if (c < n) v[c] = row_dot<TY>(w, x + off[c], nb, lane);
    }
}
// 32 hidden columns / block, eight waves, four consecutive pairs per wave.
// Preserve each dot's lane ownership and tree, and decode weights once for up
// to four entries. No full activation staging (furnace's LDS warning applies).
template<int TG>
__global__ void __launch_bounds__(256) q36_gu_quant(const unsigned long long* ptr, const int32_t* start,
        const int32_t* groups, const int32_t* tok, const block_q8_1* x, NativeExpertLayout L, block_q8_1* hq) {
    if ((int) blockIdx.y >= *groups) return;
    constexpr bool STAGE = kStageIqGrid<TG>;
    STRATA_SHARED_ALIGN16_U32(grid_buf, IqGridWords<TG, STAGE>::value);
    const uint32_t* grid = stage_iq_grid<TG, STAGE>(grid_buf, threadIdx.x, 256);
    __shared__ float gates[GRP_NC][32], ups[GRP_NC][32];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const uint8_t* blob = (const uint8_t*) ptr[blockIdx.y];
    const int xb = L.n_embd / 32, nb = L.n_embd / Fmt<TG>::qk;
    for (int e = start[blockIdx.y]; e < start[blockIdx.y + 1]; e += GRP_NC) {
        const int n = min(GRP_NC, start[blockIdx.y + 1] - e);
        int off[GRP_NC] = {};
        for (int c = 0; c < n; ++c) off[c] = tok[e + c] * xb;
        for (int r = wave; r < 32; r += 8) {
            const size_t at = (size_t) (blockIdx.x * 32 + r) * L.gu_row;
            float g[GRP_NC], u[GRP_NC];
            q36_expert_dot<TG, STAGE>(blob + at, x, off, n, nb, lane, g, grid);
            q36_expert_dot<TG, STAGE>(blob + L.up_off + at, x, off, n, nb, lane, u, grid);
            if (lane < n) { gates[lane][r] = g[lane]; ups[lane][r] = u[lane]; }
        }
        __syncthreads();
        if (wave < n) {
            // Explicit memory boundary like swiglu_q8_1_entries_kernel: no
            // contraction from the SwiGLU product into the quantizer's sum.
            const float g = gates[wave][lane];
            gates[wave][lane] = (g / (1.f + __expf(-g))) * ups[wave][lane];
            q8_1_store(((volatile float*) gates[wave])[lane], hq,
                       (long long) (e + wave) * L.n_ff + blockIdx.x * 32 + lane);
        }
        __syncthreads();
    }
}
// RDNA2's two-output-row idea, with one original wave reduction per row.
// Simple decoders only: interleave loads and reuse the activation operands,
// without doubling the live decoded IQ codebook vectors.
template<int TD>
__global__ void __launch_bounds__(128) q36_down_rows2(const unsigned long long* ptr, const int32_t* start,
        const int32_t* groups, const int32_t* dst, const block_q8_1* x, NativeExpertLayout L, float* out) {
    if ((int) blockIdx.y >= *groups) return;
    using F = Fmt<TD>;
    const int lane = threadIdx.x & 31, r = blockIdx.x * 8 + 2 * (threadIdx.x >> 5);
    if (r >= L.n_embd) return;
    const uint8_t* w = (const uint8_t*) ptr[blockIdx.y] + L.down_off + (size_t) r * L.d_row;
    const int nb = L.n_ff / F::qk, xb = L.n_ff / 32;
    for (int e = start[blockIdx.y]; e < start[blockIdx.y + 1]; e += GRP_NC) {
        const int n = min(GRP_NC, start[blockIdx.y + 1] - e);
        float a[GRP_NC] = {}, b[GRP_NC] = {};
        for (int k = lane; k < nb * F::ipb; k += 32) {
            const int kb = k / F::ipb, iq = F::step * (k % F::ipb);
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) if (c < n) {
                const block_q8_1* u = x + (size_t) (e + c) * xb + kb * (F::qk / 32);
                if constexpr (TD == 2 || TD == 3) {
                    a[c] += F::dot(w, u, kb, iq);
                    b[c] += F::dot(w + (r + 1 < L.n_embd ? L.d_row : 0), u, kb, iq);
                } else {
                    const auto wa = Split<TD>::template load<false>(w, kb, iq);
                    const auto wb = Split<TD>::template load<false>(w + (r + 1 < L.n_embd ? L.d_row : 0), kb, iq);
                    a[c] += Split<TD>::apply(wa, u, iq);
                    b[c] += Split<TD>::apply(wb, u, iq);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < GRP_NC; ++c) if (c < n) {
            const float va = warp_sum(a[c]), vb = warp_sum(b[c]);
            if (lane == 0) {
                out[(size_t) dst[e + c] * L.n_embd + r] = va;
                if (r + 1 < L.n_embd) out[(size_t) dst[e + c] * L.n_embd + r + 1] = vb;
            }
        }
    }
}
} // namespace
bool q36_expert_decode(const NativeExpertLayout& L, const unsigned long long* ptr, const int32_t* start,
    const int32_t* groups, const int32_t* dst, const int32_t* tok, int64_t cap_groups, int64_t cap_entries,
    const void* x, void* scratch, float* out, void* stream, bool rows2, bool fuse) {
    if (L.tail_off || L.had_block || L.n_ff % 32 || cap_groups <= 0 || cap_entries <= 0) return false;
    // Preserve alternate global experimental layouts, whose arithmetic may differ.
    if (g_old_kernels || g_grouped_v1 || exp_mode() != 0 || g_exp_phase != 0 || expdec_mode()) return false;
    const bool gu_ok = L.gu_type == 16 || L.gu_type == 18 || L.gu_type == 19 || L.gu_type == 22 ||
                       L.gu_type == 23 || L.gu_type == 11 || L.gu_type == 12 || L.gu_type == 8 ||
                       L.gu_type == 2 || L.gu_type == 3;
    const bool dn_ok = d_qk(L.d_type) > 0;
    if (!gu_ok || !dn_ok) return false;
    rows2 = rows2 && (L.d_type == 12 || L.d_type == 8 || L.d_type == 2 || L.d_type == 3) && L.n_ff <= L.n_embd;
    if (!rows2 && !fuse) return false;
    const size_t f = (size_t) cap_entries * L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x;
    const auto st = (cudaStream_t) stream;
    const dim3 gg((unsigned) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS), (unsigned) cap_groups);
    const dim3 gf((unsigned) (L.n_ff / 32), (unsigned) cap_groups);
#define Q36_GU(TY) case TY: if (fuse) q36_gu_quant<TY><<<gf, 256, 0, st>>>(ptr, start, groups, tok, X, L, hq); \
    else launch_gu<TY>(gg, st, ptr, start, groups, tok, X, L, gate, up); break;
    switch (L.gu_type) { Q36_GU(16) Q36_GU(18) Q36_GU(19) Q36_GU(22) Q36_GU(23) Q36_GU(11) Q36_GU(12) Q36_GU(8) Q36_GU(2) Q36_GU(3) }
#undef Q36_GU
    if (!fuse) swiglu_q8_1_entries_kernel<<<(unsigned) ((cap_entries * L.n_ff + 255) / 256), 256, 0, st>>>(gate, up, start, groups, L.n_ff, hq);
    const dim3 gd((unsigned) ((L.n_embd + 7) / 8), (unsigned) cap_groups);
    if (rows2) {
#define Q36_DN(TY) case TY: q36_down_rows2<TY><<<gd, 128, 0, st>>>(ptr, start, groups, dst, hq, L, out); break;
        switch (L.d_type) { Q36_DN(12) Q36_DN(8) Q36_DN(2) Q36_DN(3) }
#undef Q36_DN
    } else {
#define Q36_DN(TY) case TY: launch_down<TY>(gd, st, ptr, start, groups, dst, hq, L, out); break;
        switch (L.d_type) { STRATA_D_FMTS(Q36_DN) }
#undef Q36_DN
    }
    check("q36_expert_decode");
    return true;
}
