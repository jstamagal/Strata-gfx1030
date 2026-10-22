// TP decode launch fusions. Original kernels remain unchanged for the default path.
#pragma once
namespace q36tp_fuse2 {

// The two products are independent once QKV, Z and the input norm are ready. Convolution blocks
// retain k_conv_seq's per-channel history and reduction order; the remaining blocks retain
// k_gdn_ab's warp-per-row FP32 dot product and epilogue. Only T <= 4, non-batch decode uses this.
__global__ void __launch_bounds__(128) conv_ab(float* hist, const float* qkv, const float* w, float* hc,
        int C, int qk_heads, float eps, int T, bool write, const float* Wa, const float* Wb,
        const float* x, const float* dt, const float* ssm_a, float* alpha, float* beta, int n_in, int HV) {
    const int conv_blocks = C / 128;
    const int ab_blocks = (2 * HV + 3) / 4;
    const int bid = blockIdx.x;
    if (bid < conv_blocks) {
        __shared__ float part[4];
        const int c = bid * 128 + threadIdx.x;
        float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
        const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
        const bool qk = bid < qk_heads;
        for (int t = 0; t < T; ++t) {
            const float xv = qkv[(size_t) t * C + c];
            const float sum = v0 * w0 + v1 * w1 + v2 * w2 + xv * w3;
            v0 = v1;
            v1 = v2;
            v2 = xv;
            float y = sum / (1.0f + __expf(-sum));
            if (qk) {
                float sq = y * y;
                for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
                if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
                __syncthreads();
                const float ss = part[0] + part[1] + part[2] + part[3];
                __syncthreads();
                y *= rsqrtf(ss + eps);
            }
            hc[(size_t) t * C + c] = y;
        }
        if (!write) return;
        hist[c * 3] = v0;
        hist[c * 3 + 1] = v1;
        hist[c * 3 + 2] = v2;
    } else {
        const int ab = bid - conv_blocks;
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        const int o = (ab % ab_blocks) * 4 + warp;
        if (o >= 2 * HV) return;
        const int t = ab / ab_blocks;
        const bool isb = o >= HV;
        const int r = isb ? o - HV : o;
        const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
        const float4* xr = (const float4*) (x + (size_t) t * n_in);
        float acc = 0.f;
        for (int i = lane; i < n_in / 4; i += 32) {
            const float4 a = wr[i], b = xr[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        if (lane != 0) return;
        if (isb) beta[(size_t) t * HV + r] = 1.0f / (1.0f + expf(-acc));
        else {
            const float v = acc + dt[r];
            alpha[(size_t) t * HV + r] = (v > 20.f ? v : log1pf(expf(v))) * ssm_a[r];
        }
    }
}

// The original GDN sequence with its output quantized in the producing warp.
// The output is read back after the FP32 store, as in the standalone quantizer.
__global__ void __launch_bounds__(512) gdn_seq_q(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write, q36k::QB* hq) {
    constexpr int S = 128, RG = 4, RPG = 32;
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t) t * ZV + head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
            const size_t i = (size_t) t * ZV + head * S + col;
            q36k::q8_store(hq + (size_t) t * (ZV >> 5), head * S + col, ((volatile float*) y)[i], col & 31);
        }
        __syncthreads();   // sk / sq / the reductions are rewritten by the next token
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

// The router projection keeps the same warp-per-row summation as k_gemv_f32. Its final block performs
// the exact route/group body after all projection blocks have published their logits. The counter is
// per rank and reset by the final block, so graph replay and consecutive layers use the same address.
// The selection and grouping expressions are copied verbatim from q36k::k_route_group.
template <bool HY>
__device__ __forceinline__ void route_group_body(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                                     float* __restrict__ weights, int n_tok, int ld, const int32_t* slot_of,
                                                     const uint8_t* blobs, unsigned long long blob,
                                                     unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                                                     int32_t* ent_dst, int32_t* ent_tok) {
    constexpr int NE = 256, K = 8, V = NE / 32;
    __shared__ int sids[64], sid[64];
    const int tk = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (tk < n_tok) {
        const float* lg = logits + (size_t) tk * ld;
        float values[V];
#pragma unroll
        for (int i = 0; i < V; ++i) values[i] = lg[lane + i * 32];
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < V; ++i) maximum = max(maximum, values[i]);
        maximum = q36k::wmax(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] = q36k::fexp_fm(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = q36k::frcp_fm(q36k::wsum(sum));
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] *= reciprocal;
            if (__isnanf(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < K; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < V; ++i) {
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
                ids[(size_t) tk * K + rank] = expert;
                sids[tk * K + rank] = expert;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        selected_sum = max(q36k::wsum(selected_sum), 6.103515625e-5f);
        const float inverse_selected_sum = q36k::frcp_fm(selected_sum);
        if (lane < K) weights[(size_t) tk * K + lane] = selected * inverse_selected_sum;
    }
    __syncthreads();
    const int n_ent = n_tok * K;
    const int e = tk * 32 + lane;   // 0 .. 255: the entries are the first n_ent threads
    const int id = e < n_ent ? sids[e] : 0;
    const bool valid = e < n_ent && (!HY || slot_of[id] >= 0);
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = sids[e2];
            if (HY && slot_of[id2] < 0) continue;
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K;
        sid[rank] = id;
    }
    __syncthreads();
    int nv = 0;
    for (int e2 = 0; e2 < n_ent; ++e2) nv += (!HY || slot_of[sids[e2]] >= 0) ? 1 : 0;
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int g = 0;
        for (int q = 1; q < rank; ++q) g += sid[q - 1] != sid[q] ? 1 : 0;
        g += rank > 0 ? 1 : 0;
        grp_ptr[g] = (unsigned long long) (blobs + (size_t) (HY ? slot_of[id] : id) * blob);
        grp_start[g] = rank;
    }
    if (e == 0) {
        int G = nv > 0 ? 1 : 0;
        for (int q = 1; q < nv; ++q) G += sid[q - 1] != sid[q] ? 1 : 0;
        grp_start[G] = nv;
        *n_groups = G;
    }
}

template <bool LightFence = false, int Warps = 8>
__global__ void __launch_bounds__(256) router_group(const float* W, const float* x, float* logits,
        int n_in, int n_out, int n_tok, int ld, int32_t* ids, float* weights,
        const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr,
        int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok,
        unsigned* done) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int cols = (n_out + Warps - 1) / Warps;
    const int tk = blockIdx.x / cols;
    const int o = (blockIdx.x % cols) * Warps + warp;
    if (o < n_out) {
        const float4* wr = (const float4*) (W + (size_t) o * n_in);
        const float4* xr = (const float4*) (x + (size_t) tk * n_in);
        float acc = 0.f;
        for (int i = lane; i < n_in / 4; i += 32) {
            const float4 a = wr[i], b = xr[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        if (lane == 0) logits[(size_t) tk * n_out + o] = acc;
    }
    if constexpr (LightFence) {
        // Only warp leaders stored logits. Drain those stores before the CTA
        // announces completion; the other lanes have nothing to fence.
        if (lane == 0 && o < n_out) __threadfence();
        __syncthreads();
    } else {
        __syncthreads();
        __threadfence();
        __syncthreads();
    }
    __shared__ int last;
    if (threadIdx.x == 0) {
        unsigned old;
#if defined(__HIPCC__)
        if constexpr (LightFence)
            old = __hip_atomic_fetch_add(done, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        else old = atomicAdd(done, 1u);
#else
        old = atomicAdd(done, 1u);
#endif
        last = old + 1u == (unsigned) gridDim.x;
        if (last) atomicExch(done, 0u);
    }
    __syncthreads();
    if (last) route_group_body<false>(logits, ids, weights, n_tok, ld, nullptr,
                                      blobs, blob, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
}

// The join publishes/sums exactly as q36tp::join. The norm consumes the completed joined rows
// in the same launch. A device-local token barrier replaces the inter-kernel ordering point;
// all eight join blocks can reside together on gfx1030. Q selects the existing q8 epilogue.
template <bool Q>
__global__ void __launch_bounds__(256) join_norm(float* partial, float* mine, const float* peer,
        size_t count, unsigned* my_flags, unsigned* peer_flags, unsigned* token, int rank,
        float* x, const float* w, float* h, q36k::QB* hq, int n, float eps, int T) {
    __shared__ unsigned call;
    if (threadIdx.x == 0) call = token[blockIdx.x] + 1;
    __syncthreads();
    const size_t i0 = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t step = (size_t) gridDim.x * blockDim.x;
    const size_t nv = count / 4;
    for (size_t i = i0; i < nv; i += step) ((float4*) mine)[i] = ((const float4*) partial)[i];
    for (size_t i = nv * 4 + i0; i < count; i += step) mine[i] = partial[i];
    __threadfence_system();
    __syncthreads();
    const int f = blockIdx.x * q36tp::flag_stride;
    if (threadIdx.x == 0) {
        q36tp::release(my_flags + f, call);
        while (q36tp::acquire(peer_flags + f) != call) q36tp::pause();
    }
    __syncthreads();
    __threadfence_system();
    for (size_t i = i0; i < nv; i += step) {
        float4 a = ((const float4*) partial)[i];
#if defined(__HIPCC__)
        typedef float v4 __attribute__((ext_vector_type(4)));
        const v4 v = __builtin_nontemporal_load((const v4*) peer + i);
        const float4 b = make_float4(v.x, v.y, v.z, v.w);
#else
        const volatile float* p = peer + i * 4;
        const float4 b = make_float4(p[0], p[1], p[2], p[3]);
#endif
        if (rank == 0) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
        else { a.x = b.x + a.x; a.y = b.y + a.y; a.z = b.z + a.z; a.w = b.w + a.w; }
        ((float4*) partial)[i] = a;
    }
    for (size_t i = nv * 4 + i0; i < count; i += step) {
        const float other = ((const volatile float*) peer)[i];
        partial[i] = rank == 0 ? partial[i] + other : other + partial[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        q36tp::release(my_flags + f + 1, call);
        while (q36tp::acquire(peer_flags + f + 1) != call) q36tp::pause();
#if defined(__HIPCC__)
        __hip_atomic_store(token + blockIdx.x, call, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
        atomicExch(token + blockIdx.x, call);
#endif
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int b = 0; b < q36tp::join_blocks; ++b) {
#if defined(__HIPCC__)
            while (__hip_atomic_load(token + b, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != call) q36tp::pause();
#else
            while (atomicAdd(token + b, 0u) != call) q36tp::pause();
#endif
        }
    }
    __syncthreads();
    if ((int) blockIdx.x >= T) return;

    __shared__ float sh[256];
    const int row = blockIdx.x;
    float* xr = x + (size_t) row * n;
    const float* yr = partial + (size_t) row * n;
    float* hr = h + (size_t) row * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    if constexpr (Q) {
        q36k::QB* qr = hq + (size_t) row * (n >> 5);
        for (int i = threadIdx.x; i < n; i += 256) {
            const float hv = __fmul_rn(__fmul_rn(xr[i], r), w[i]);
            hr[i] = hv;
            q36k::q8_store(qr, i, ((volatile float*) hr)[i], threadIdx.x & 31);
        }
    } else {
        for (int i = threadIdx.x; i < n; i += 256) hr[i] = xr[i] * r * w[i];
    }
}

} // namespace q36tp_fuse2
