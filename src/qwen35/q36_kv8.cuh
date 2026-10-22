// Dense q36 KV: one int8 code per value, one stored FP16 scale per 64 values.
// K and V each use [HKV][context][HD] codes followed by [HKV][context][HD/64] scales.
// This is the same row/group arithmetic as Flash-Next's kv_q8.cu.
namespace q36kv8 {
constexpr int group = 64;
inline size_t bytes(int hkv, int ctx, int hd) {
    const size_t rows = (size_t) hkv * ctx;
    return rows * hd + rows * (hd / group) * sizeof(__half);
}
__device__ __forceinline__ int8_t* codes(float* base) { return reinterpret_cast<int8_t*>(base); }
__device__ __forceinline__ const int8_t* codes(const float* base) { return reinterpret_cast<const int8_t*>(base); }
__device__ __forceinline__ __half* scales(float* base, size_t rows, int hd) {
    return reinterpret_cast<__half*>(codes(base) + rows * hd);
}
__device__ __forceinline__ const __half* scales(const float* base, size_t rows, int hd) {
    return reinterpret_cast<const __half*>(codes(base) + rows * hd);
}
__device__ __forceinline__ float load(const float* base, size_t row, int d, size_t rows, int hd) {
    return (float) codes(base)[row * hd + d] * __half2float(scales(base, rows, hd)[row * (hd / group) + d / group]);
}
} // namespace q36kv8

// The FP32 path's q/k normalization and RoPE, with a 64-value group reduction before writing K/V.
// Each K block has 256 threads, including inactive lanes for heads shorter than 256.
__global__ void k_attn_prep_q8(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                               float* q, float* gate, float* kc, float* vc, int H, int HKV, int HD, int n_rot,
                               const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                               const int32_t* mtab, RowSel rs) {
    __shared__ float sh[kThreads], buf[256], warp_max[8];
    const int head = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    if (rs.kc) { kc = rs.kc[rs.slot[t]]; vc = rs.vc[rs.slot[t]]; }
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    float v = i < HD ? buf[i] : 0.f;
    const int half = n_rot / 2;
    if (i < HD && i < n_rot) {
        const int p = i < half ? i : i - half;
        const int rp = mtab ? __ldg(mtab + (size_t) pos * 3 + p % 3) : pos + dl;
        const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        if (i < HD) {
            q[((size_t) t * H + head) * HD + i] = v;
            gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
        }
        return;
    }
    const int kh = head - H;
    const size_t row = (size_t) kh * ctx + pos, rows = (size_t) HKV * ctx;
    const float vv = i < HD ? vx[((size_t) t * HKV + kh) * HD + i] : 0.f;
    float ka = fabsf(v), va = fabsf(vv);
    for (int o = 16; o > 0; o >>= 1) {
        ka = fmaxf(ka, __shfl_xor_sync(0xffffffffu, ka, o));
        va = fmaxf(va, __shfl_xor_sync(0xffffffffu, va, o));
    }
    if ((i & 31) == 0) { warp_max[i >> 5] = ka; sh[i >> 5] = va; }
    __syncthreads();
    if (i >= HD) return;
    const int g = i / q36kv8::group;
    const float ks = __half2float(__float2half_rn(fmaxf(warp_max[2 * g], warp_max[2 * g + 1]) / 127.f));
    const float vs = __half2float(__float2half_rn(fmaxf(sh[2 * g], sh[2 * g + 1]) / 127.f));
    const int kq = ks > 0.f ? max(-127, min(127, __float2int_rn(v / ks))) : 0;
    const int vq = vs > 0.f ? max(-127, min(127, __float2int_rn(vv / vs))) : 0;
    q36kv8::codes(kc)[row * HD + i] = (int8_t) kq;
    q36kv8::codes(vc)[row * HD + i] = (int8_t) vq;
    if (i % q36kv8::group == 0) {
        q36kv8::scales(kc, rows, HD)[row * (HD / q36kv8::group) + g] = __float2half_rn(ks);
        q36kv8::scales(vc, rows, HD)[row * (HD / q36kv8::group) + g] = __float2half_rn(vs);
    }
}
