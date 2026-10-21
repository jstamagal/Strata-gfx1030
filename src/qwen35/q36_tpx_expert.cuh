// TP decode: two independent 16-lane rows per wave for the half-width experts.
// Included inside iq_kernels.cu's anonymous namespace, after Fmt/Split and GRP_NC.
// Each lane performs the work of default lanes
// t and t+16 in separate accumulators. Their final addition is the default
// warp's offset-16 step; offsets 8..1 follow unchanged. At 16 parts per down
// row, default lanes 16..31 contribute zero. Thus the FP additions for each
// output retain their order, including the offset-16 addition of zero.

int g_tpx_gu = -1, g_tpx_down = -1; // -1: exact environment switch; 0/1: lab override
inline bool tpx_switch(int override_value, const char* name) {
    if (override_value >= 0) return override_value == 1;
    const char* v = std::getenv(name);
    return v && v[0] == '1' && v[1] == '\0';
}
inline void tpx_expert_trace(const char* phase, const NativeExpertLayout& L, bool selected) {
    if (!tpx_switch(-1, "STRATA_Q36_TPX_TRACE")) return;
    static std::map<std::tuple<int, int64_t, int64_t, bool>, bool> seen;
    const int type = phase[0] == 'g' ? L.gu_type : L.d_type;
    if (!seen.emplace(std::make_tuple(type + (phase[0] == 'd' ? 1000 : 0), L.n_embd, L.n_ff, selected), true).second) return;
    std::fprintf(stderr, "strata-q36 TPX expert: %s type=%d K=%lld FF=%lld %s\n",
                 phase, type, (long long) L.n_embd, (long long) L.n_ff, selected ? "selected" : "fallback");
}

// Diagnostic graph node: compare a real grouped call with the old gate/up
// kernel, then restore its outputs so the rest of the decode uses reference
// values.  h/hq are idle scratch until SwiGLU/quantize below this call.
__device__ unsigned tpx_gu_mismatch_seen[64] = {};
__global__ void tpx_gu_compare_restore(float* gate, float* up, const float* old_gate,
                                        const float* old_up, const int32_t* grp_start, const int32_t* n_groups,
                                        const int32_t* ent_tok, int cap_groups, int cap_entries, int n_ff, int type) {
    const int groups = *n_groups;
    const int entries = groups >= 0 && groups <= cap_groups ? grp_start[groups] : 0;
    const size_t n = (size_t) max(0, min(entries, cap_entries)) * n_ff;
    if (threadIdx.x == 0) {
        for (size_t i = 0; i < n; ++i) {
            if (__float_as_uint(gate[i]) != __float_as_uint(old_gate[i]) ||
                __float_as_uint(up[i]) != __float_as_uint(old_up[i])) {
                const int entry = (int) (i / n_ff);
                int group = 0;
                while (group + 1 < groups && grp_start[group + 1] <= entry) ++group;
                if (type >= 0 && type < 64 && atomicAdd(&tpx_gu_mismatch_seen[type], 1u) < 8)
                    printf("strata-q36 TPX GU mismatch: type=%d groups=%d entry=%d token=%d group=%d row=%d gate=%08x/%08x up=%08x/%08x\n",
                           type, groups, entry, ent_tok[entry], group, (int) (i % n_ff),
                           __float_as_uint(gate[i]), __float_as_uint(old_gate[i]),
                           __float_as_uint(up[i]), __float_as_uint(old_up[i]));
                break;
            }
        }
    }
    __syncthreads();
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        gate[i] = old_gate[i];
        up[i] = old_up[i];
    }
}

template<int TY, bool STAGE_GRID, bool USE_SPLIT = kSplit<TY> && TY != 12 && TY != 8>
struct TpxPart;
template<int TY, bool STAGE_GRID>
struct TpxPart<TY, STAGE_GRID, true> {
    using W = typename Split<TY>::W;
    __device__ static W load(const uint8_t* row, int b, int i, const uint32_t* grid) {
        return Split<TY>::template load<STAGE_GRID>(row, b, i, grid);
    }
    __device__ static float apply(const W& w, const block_q8_1* x, int i) { return Split<TY>::apply(w, x, i); }
};
template<int TY, bool STAGE_GRID>
struct TpxPart<TY, STAGE_GRID, false> {
    struct W { const uint8_t* row; int b, i; };
    __device__ static W load(const uint8_t* row, int b, int i, const uint32_t*) { return {row, b, i}; }
    __device__ static float apply(const W& w, const block_q8_1* x, int) { return Fmt<TY>::dot(w.row, x, w.b, w.i); }
};

template<int TY, int NC, bool STAGE_GRID>
__device__ __forceinline__ void tpx_dot_parts(const uint8_t* row, const block_q8_1* x,
                                               const int (&off)[NC], int count, int parts, int t, float (&sum)[NC],
                                               const uint32_t* grid = nullptr) {
    using F = Fmt<TY>;
    using S = TpxPart<TY, STAGE_GRID>;
    float lo[NC] = {}, hi[NC] = {};
    for (int base = 0; base < parts; base += 32) {
        const int kl = base + t, kh = kl + 16;
        const int bl = kl / F::ipb, bh = kh / F::ipb;
        const int il = F::step * (kl % F::ipb), ih = F::step * (kh % F::ipb);
        if (kl < parts) {
            const typename S::W wl = S::load(row, bl, il, grid);
#pragma unroll
            for (int c = 0; c < NC; ++c) if (c < count)
                lo[c] += S::apply(wl, x + off[c] + bl * (F::qk / 32), il);
        }
        if (kh < parts) {
            const typename S::W wh = S::load(row, bh, ih, grid);
#pragma unroll
            for (int c = 0; c < NC; ++c) if (c < count)
                hi[c] += S::apply(wh, x + off[c] + bh * (F::qk / 32), ih);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) if (c < count) {
        float s = __fadd_rn(lo[c], hi[c]);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        sum[c] = s;
    }
}

template<int TY, bool STAGE_GRID>
__global__ void __launch_bounds__(256) tpx_gu_half_kernel(const unsigned long long* __restrict__ grp_ptr,
    const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups,
    const int32_t* __restrict__ ent_tok, const block_q8_1* __restrict__ xq,
    NativeExpertLayout L, float* __restrict__ gate, float* __restrict__ up) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    const int t = threadIdx.x & 15;
    STRATA_SHARED_ALIGN16_U32(s_grid_buf, IqGridWords<TY, STAGE_GRID>::value);
    const uint32_t* s_grid = stage_iq_grid<TY, STAGE_GRID>(s_grid_buf, threadIdx.x, 256);
    const int row = blockIdx.x * 16 + (threadIdx.x >> 4);
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int xb = (int) L.n_embd / 32;
    const int parts = (int) (L.n_embd / Fmt<TY>::qk) * Fmt<TY>::ipb;
    float* dst = is_up ? up : gate;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const float sc = L.tail_off ? ((const float*) (grp_ptr[g] + L.tail_off))[is_up ? 1 : 0] : 1.0f;
        const int e1 = grp_start[g + 1];
        for (int e = grp_start[g]; e < e1; e += GRP_NC) {
            const int count = min(GRP_NC, e1 - e);
            int xoff[GRP_NC];
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) xoff[c] = ent_tok[e + min(c, count - 1)] * xb;
            float sum[GRP_NC] = {};
            tpx_dot_parts<TY, GRP_NC, STAGE_GRID>(wr, xq, xoff, count, parts, t, sum, s_grid);
            if (t < count) dst[(size_t) (e + t) * L.n_ff + r] = sum[t] * sc;
        }
    }
}

template<int TY>
__global__ void __launch_bounds__(256) tpx_down_half_kernel(const unsigned long long* __restrict__ grp_ptr,
    const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups,
    const int32_t* __restrict__ ent_dst, const block_q8_1* __restrict__ hq,
    NativeExpertLayout L, float* __restrict__ out) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    const int t = threadIdx.x & 15;
    const int r = blockIdx.x * 16 + (threadIdx.x >> 4);
    if (r >= L.n_embd) return;
    const size_t off = L.down_off + (size_t) r * L.d_row;
    const int hb = (int) L.n_ff / 32;
    const int parts = (int) (L.n_ff / Fmt<TY>::qk) * Fmt<TY>::ipb;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const float sd = L.tail_off ? ((const float*) (grp_ptr[g] + L.tail_off))[2] : 1.0f;
        const int e1 = grp_start[g + 1];
        for (int e = grp_start[g]; e < e1; e += GRP_NC) {
            const int count = min(GRP_NC, e1 - e);
            int xoff[GRP_NC];
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) xoff[c] = (e + min(c, count - 1)) * hb;
            float sum[GRP_NC] = {};
            tpx_dot_parts<TY, GRP_NC, false>(wr, hq, xoff, count, parts, t, sum);
            if (t < count) out[(size_t) ent_dst[e + t] * L.n_embd + r] = sum[t] * sd;
        }
    }
}
