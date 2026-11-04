// Q4_0 dense rows: one wave owns the four partial sums of the native four-wave row.
// Each lane keeps the native k-block and q-byte order. The partials are added in native wave order
// before the same XOR reduction, so STRATA_Q36_VERIFY can compare every output bit.
#pragma once
namespace q36q4 {
template<int NC>
__global__ void rows(const uint8_t* __restrict__ w, const q36k::QB* __restrict__ x,
                     float* __restrict__ y, int nb, int nr) {
    const int r = blockIdx.x;
    if (r >= nr) return;
    const int lane = threadIdx.x;
    const int iqs = 2 * (lane & 1);
    const int kb0 = lane >> 1;
    float partial[NC][4] = {};
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        for (int kb = kb0 + 16 * p; kb < nb; kb += 64) {
            const uint8_t* b = w + ((size_t)r * nb + kb) * 18;
            const auto* h = reinterpret_cast<const uint16_t*>(b);
            const float d = __half2float(__ushort_as_half(h[0]));
            const auto* q = reinterpret_cast<const uint16_t*>(b + 2);
            const int v0 = (int)q[2 * iqs] | ((int)q[2 * iqs + 1] << 16);
            const int v1 = (int)q[2 * (iqs + 1)] | ((int)q[2 * (iqs + 1) + 1] << 16);
            const int lo0 = v0 & 0x0f0f0f0f, hi0 = (v0 >> 4) & 0x0f0f0f0f;
            const int lo1 = v1 & 0x0f0f0f0f, hi1 = (v1 >> 4) & 0x0f0f0f0f;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const q36k::QB& a = x[(size_t)c * nb + kb];
                const int* u = reinterpret_cast<const int*>(a.qs);
                int s = 0;
                s = STRATA_DP4A(lo0, u[iqs], s);
                s = STRATA_DP4A(hi0, u[iqs + 4], s);
                s = STRATA_DP4A(lo1, u[iqs + 1], s);
                s = STRATA_DP4A(hi1, u[iqs + 5], s);
                const float2 ds = __half22float2(a.ds);
                partial[c][p] += d * (s * ds.x - 4 * ds.y);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float z = partial[c][0];
        z += partial[c][1]; z += partial[c][2]; z += partial[c][3];
        z = q36k::wsum(z);
        if (lane == 0) y[(size_t)c * nr + r] = z;
    }
}
inline void launch(const void* w, const void* x, float* y, int n_in, int n_out, int nc, cudaStream_t st) {
    const auto* q = static_cast<const q36k::QB*>(x);
    const auto* b = static_cast<const uint8_t*>(w);
#define Q36_Q4_CASE(N) case N: rows<N><<<n_out, 32, 0, st>>>(b, q, y, n_in / 32, n_out); break
    switch (nc) {
        Q36_Q4_CASE(1); Q36_Q4_CASE(2); Q36_Q4_CASE(3); Q36_Q4_CASE(4);
        Q36_Q4_CASE(5); Q36_Q4_CASE(6); Q36_Q4_CASE(7); Q36_Q4_CASE(8);
    }
#undef Q36_Q4_CASE
}
} // namespace q36q4
