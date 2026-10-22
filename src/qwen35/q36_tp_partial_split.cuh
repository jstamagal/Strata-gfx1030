// Split the TP MoE partial across CTAs without changing an element's reduction order.
#pragma once
namespace q36tp_partial {
template <bool Publish>
__global__ void __launch_bounds__(256) combine(float* out, float* mapped, const float* parts,
        const int32_t* prow, const float* wts, int picks, const float* shared,
        const float* sgate, int sg_stride, int n, unsigned* token, unsigned* done) {
    const int t = blockIdx.y;
    const float* wr = wts + (size_t) t * picks;
    const float gs = 1.0f / (1.0f + expf(-sgate[(size_t) t * sg_stride]));
    float* mine = nullptr;
    unsigned* flag = nullptr;
    if constexpr (Publish) {
        const size_t base = (size_t) t * ((size_t) n + 16);
        mine = mapped + base + 4;
        flag = (unsigned*) (mapped + base);
    }
    const size_t first = prow ? prow[t * picks] : t * picks;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float acc = wr[0] * parts[first * n + i];
        for (int k = 1; k < picks; ++k) {
            const size_t row = prow ? prow[t * picks + k] : t * picks + k;
            acc = fmaf(wr[k], parts[row * n + i], acc);
        }
        const float v = acc + gs * shared[(size_t) t * n + i];
        out[(size_t) t * n + i] = v;
        if constexpr (Publish) mine[i] = v;
    }
    if constexpr (Publish) {
        // Each lane drains its mapped stores. The final CTA for this token releases
        // the flag only after every CTA has fenced its own stores.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
#if defined(__HIPCC__)
            const unsigned old = __hip_atomic_fetch_add(done + t, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
#else
            const unsigned old = atomicAdd(done + t, 1u);
#endif
            if (old + 1u == gridDim.x) {
#if defined(__HIPCC__)
                __hip_atomic_store(done + t, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
                atomicExch(done + t, 0u);
#endif
                q36tp::release(flag, token[t] + 1);
            }
        }
    }
}
} // namespace q36tp_partial
