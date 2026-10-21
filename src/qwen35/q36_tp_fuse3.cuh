// Short-window TP GDN. Each V-head block owns its state and V convolution; Q/K
// convolution is repeated by heads sharing a K head, with one designated writer.
// This removes the inter-grid conv -> GDN dependency without a global barrier.
#pragma once
namespace q36tp_fuse3 {

__global__ void __launch_bounds__(512) conv_ab_gdn_q(
        float* hist, const float* qkv, const float* cw, float* hc, int C, int HK,
        const float* Wa, const float* Wb, const float* hin, const float* dt,
        const float* ssm_a, float* alpha, float* beta, int N, bool ab_ready,
        float* state, const float* z, const float* gamma, float eps,
        float* y, q36k::QB* yq, int HV, int T, bool write) {
    constexpr int S = 128, RG = 4, RPG = 32;
    const int head = blockIdx.x, qh = head % HK;
    const int col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int ZV = HV * S;
    __shared__ float cv[3][S], conv_sum[2][4], ab[2];
    __shared__ float red_kv[RG][S], red_o[RG][S], wsum[S * RG / 32];
    const int channel = rg == 0 ? qh * S + col : rg == 1 ? (HK + qh) * S + col : (2 * HK + head) * S + col;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f, w0 = 0.f, w1 = 0.f, w2 = 0.f, w3 = 0.f;
    if (rg < 3) {
        v0 = hist[channel * 3]; v1 = hist[channel * 3 + 1]; v2 = hist[channel * 3 + 2];
        w0 = cw[channel * 4]; w1 = cw[channel * 4 + 1];
        w2 = cw[channel * 4 + 2]; w3 = cw[channel * 4 + 3];
    }
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t row_stride = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        if (!ab_ready && tid < 64) {
            const bool isb = tid >= 32;
            const int lane = tid & 31;
            const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) head * N);
            const float4* xr = (const float4*) (hin + (size_t) t * N);
            float acc = 0.f;
            for (int i = lane; i < N / 4; i += 32) {
                const float4 a = wr[i], b = xr[i];
                acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
            if (lane == 0) {
                if (isb) ab[1] = beta[(size_t) t * HV + head] = 1.0f / (1.0f + expf(-acc));
                else {
                    const float v = acc + dt[head];
                    ab[0] = alpha[(size_t) t * HV + head] = (v > 20.f ? v : log1pf(expf(v))) * ssm_a[head];
                }
            }
        }
        if (rg < 3) {
            const float xv = qkv[(size_t) t * C + channel];
            const float sum = v0 * w0 + v1 * w1 + v2 * w2 + xv * w3;
            v0 = v1; v1 = v2; v2 = xv;
            float yc = sum / (1.0f + __expf(-sum));
            if (rg < 2) {
                float sq = yc * yc;
                for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
                if ((col & 31) == 0) conv_sum[rg][col >> 5] = sq;
            }
            cv[rg][col] = yc;
        }
        __syncthreads();
        if (rg < 2) {
            const float ss = conv_sum[rg][0] + conv_sum[rg][1] + conv_sum[rg][2] + conv_sum[rg][3];
            cv[rg][col] *= rsqrtf(ss + eps);
        }
        __syncthreads();
        if (rg < 3 && (rg == 2 || head == qh)) hc[(size_t) t * C + channel] = cv[rg][col];
        const float g = __expf(ab_ready ? alpha[(size_t) t * HV + head] : ab[0]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], cv[1][rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (cv[2][col] - g * kv_col) * (ab_ready ? beta[(size_t) t * HV + head] : ab[1]);
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], cv[1][rg * RPG + r] * delta);
            o = fmaf(s[r], cv[0][rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) { oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S); sq_part = oc * oc; }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            const size_t i = (size_t) t * ZV + head * S + col;
            y[i] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
            q36k::q8_store(yq + (size_t) t * (ZV >> 5), head * S + col, ((volatile float*) y)[i], col & 31);
        }
        __syncthreads();
    }
    if (!write) return;
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    if (rg == 2) {
        hist[channel * 3] = v0;
        hist[channel * 3 + 1] = v1;
        hist[channel * 3 + 2] = v2;
    }
}

// The Q/K history update is deferred until every V-head block has read its old
// history. The existing partial-MoE launch is a later stream dependency, so this
// side work adds no launch and does not depend on the two-rank rendezvous.
__global__ void partial_moe_hist(float* out, const float* parts, const int32_t* prow,
        const float* wts, int picks, const float* shared, const float* sgate,
        int sg_stride, int n, float* hist, const float* qkv, int C, int qk_channels, int T) {
    const int t = blockIdx.x;
    const float* wr = wts + (size_t) t * picks;
    const float gs = 1.0f / (1.0f + expf(-sgate[(size_t) t * sg_stride]));
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const size_t first = prow ? prow[t * picks] : t * picks;
        float acc = wr[0] * parts[first * n + i];
        for (int k = 1; k < picks; ++k) {
            const size_t row = prow ? prow[t * picks + k] : t * picks + k;
            acc = fmaf(wr[k], parts[row * n + i], acc);
        }
        out[(size_t) t * n + i] = acc + gs * shared[(size_t) t * n + i];
    }
    if (t == 0) for (int c = threadIdx.x; c < qk_channels; c += blockDim.x) {
        float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
        for (int row = 0; row < T; ++row) { v0 = v1; v1 = v2; v2 = qkv[(size_t) row * C + c]; }
        hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
    }
}
} // namespace q36tp_fuse3
