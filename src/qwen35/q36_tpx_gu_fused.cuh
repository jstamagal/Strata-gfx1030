// T=1 TP gate/up: mode 8's sums and q8_1 epilogue, with the one activation
// entry of each routed group read from L1 instead of copied into dynamic LDS.
// Included in iq_kernels.cu after its GridOf, DotG and q8_1 helpers.
#pragma once

int g_tpx_gu_fused_t1 = -1;  // -1: exact environment switch, 0/1: lab override

template<int TG>
__global__ void __launch_bounds__(256) tpx_gu_fused_t1_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok,
    const block_q8_1* __restrict__ xq, NativeExpertLayout L, block_q8_1* __restrict__ hq) {
    using GT = typename GridOf<TG>::T;
    using F = Fmt<TG>;
    __shared__ GT sgrid[GridOf<TG>::N];
    __shared__ float res[64];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const GT* gsrc = GridOf<TG>::src();
    for (int i = tid; i < GridOf<TG>::N; i += 256) sgrid[i] = gsrc[i];
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32), hb = (int) (L.n_ff / 32);
    const int r0 = blockIdx.x * 32;
    // The normal mode-8 kernel uses the same p/i0/i1/lane decomposition. A
    // group may have repeated entries even when the host supplied a T=1-sized
    // capacity, so retain the entry loop for exactness in every graph.
    for (int e = grp_start[g]; e < grp_start[g + 1]; ++e) {
        const block_q8_1* x = xq + (size_t) ent_tok[e] * xb;
        for (int p = 0; p < 4; ++p) {
            const int i0 = p * 16 + warp, i1 = i0 + 8;
            const uint8_t* w0 = blob + (i0 < 32 ? (size_t) 0 : L.up_off) + (size_t) (r0 + (i0 & 31)) * L.gu_row;
            const uint8_t* w1 = blob + (i1 < 32 ? (size_t) 0 : L.up_off) + (size_t) (r0 + (i1 & 31)) * L.gu_row;
            float s0 = 0.0f, s1 = 0.0f;
            for (int k = lane; k < nb * F::ipb; k += 32) {
                const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
                const block_q8_1* xk = x + kbx * (F::qk / 32);
                const float a = DotG<TG, 1>::f(w0, xk, kbx, iqs, sgrid);
                const float b = DotG<TG, 1>::f(w1, xk, kbx, iqs, sgrid);
                s0 += a;
                s1 += b;
            }
            const float a = warp_sum(s0), b = warp_sum(s1);
            if (lane == 0) { res[i0] = a; res[i1] = b; }
        }
        __syncthreads();
        if (warp == 0) {
            const float gg = res[lane], uu = res[32 + lane];
            const float xi = (gg / (1.0f + __expf(-gg))) * uu;
            float amax = fabsf(xi), sum = xi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            const float d = q8_1_finite(amax / 127.0f);
            const int8_t q = q8_1_quant(xi, d, amax);
            block_q8_1* y = hq + (size_t) e * hb + blockIdx.x;
            y->qs[lane] = q;
            if (lane == 0) y->ds = q8_1_ds(d, sum);
        }
        __syncthreads();
    }
}
