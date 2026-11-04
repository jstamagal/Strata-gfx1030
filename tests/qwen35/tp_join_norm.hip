// Differential two-card test: the old join plus add/norm/q8 sequence versus the joined consumer.
// hipcc -O2 --offload-arch=gfx1030 -Iinclude -Iinclude/strata/hip_compat -Isrc/qwen35 tests/qwen35/tp_join_norm.hip -o /tmp/tp-join-norm
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "strata/hip_compat/cuda_runtime.h"
#include "q36_kernels.cuh"
#include "q36_tp_collective.cuh"
#include "q36_tp_fuse2.cuh"
#define CHECK(x) do { const hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)

__global__ void add_rms_reference(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[256];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = 128; k; k >>= 1) {
        if ((int) threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 256) hr[i] = xr[i] * r * w[i];
}

int main() {
    constexpr int max_n = 3072, max_t = 4;
    float *part[2], *x[2], *h[2], *w[2], *host[2], *mapped[2][2];
    q36k::QB* q[2];
    unsigned *flags_h[2], *flags[2][2], *token[2];
    hipStream_t stream[2];
    for (int d = 0; d < 2; ++d) {
        CHECK(hipSetDevice(d));
        CHECK(hipMalloc((void**) &part[d], max_t * max_n * 4));
        CHECK(hipMalloc((void**) &x[d], max_t * max_n * 4));
        CHECK(hipMalloc((void**) &h[d], max_t * max_n * 4));
        CHECK(hipMalloc((void**) &w[d], max_n * 4));
        CHECK(hipMalloc((void**) &q[d], max_t * (max_n / 32) * sizeof(q36k::QB)));
        CHECK(hipMalloc((void**) &token[d], q36tp::join_blocks * sizeof(unsigned)));
        CHECK(hipMemset(token[d], 0, q36tp::join_blocks * sizeof(unsigned)));
        CHECK(hipHostMalloc((void**) &host[d], max_t * max_n * 4, hipHostMallocMapped | hipHostMallocPortable));
        CHECK(hipHostMalloc((void**) &flags_h[d], q36tp::join_blocks * q36tp::flag_stride * sizeof(unsigned),
                            hipHostMallocMapped | hipHostMallocPortable));
        std::fill(flags_h[d], flags_h[d] + q36tp::join_blocks * q36tp::flag_stride, 0);
        CHECK(hipStreamCreateWithFlags(&stream[d], hipStreamNonBlocking));
    }
    for (int d = 0; d < 2; ++d) {
        CHECK(hipSetDevice(d));
        for (int r = 0; r < 2; ++r) {
            CHECK(hipHostGetDevicePointer((void**) &mapped[d][r], host[r], 0));
            CHECK(hipHostGetDevicePointer((void**) &flags[d][r], flags_h[r], 0));
        }
    }
    for (int n : {512, 1024, 3072}) for (int T = 1; T <= max_t; ++T) for (bool quant : {false, true}) {
        const size_t count = (size_t) n * T;
        std::vector<float> hp[2], hx[2], hw[2];
        for (int d = 0; d < 2; ++d) {
            hp[d].resize(count); hx[d].resize(count); hw[d].resize(n);
            for (size_t i = 0; i < count; ++i) {
                hp[d][i] = (float) ((i * 19 + d * 13) % 103) * 0.013f;
                hx[d][i] = (float) ((i * 7 + d * 5) % 67) * -0.011f;
            }
            for (int i = 0; i < n; ++i) hw[d][i] = 0.9f + (float) (i % 17) * 0.007f;
            CHECK(hipSetDevice(d));
            CHECK(hipMemcpy(part[d], hp[d].data(), count * 4, hipMemcpyHostToDevice));
            CHECK(hipMemcpy(x[d], hx[d].data(), count * 4, hipMemcpyHostToDevice));
            CHECK(hipMemcpy(w[d], hw[d].data(), n * 4, hipMemcpyHostToDevice));
            q36tp::join<<<q36tp::join_blocks, 256, 0, stream[d]>>>(part[d], mapped[d][d], mapped[d][1-d], count,
                flags[d][d], flags[d][1-d], token[d], d);
            if (quant) q36k::k_add_rms_q<<<T, 256, 0, stream[d]>>>(x[d], part[d], w[d], h[d], q[d], n, 1e-6f);
            else add_rms_reference<<<T, 256, 0, stream[d]>>>(x[d], part[d], w[d], h[d], n, 1e-6f);
        }
        std::vector<float> ref_x[2], ref_h[2];
        std::vector<uint8_t> ref_q[2];
        for (int d = 0; d < 2; ++d) {
            CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(stream[d]));
            ref_x[d].resize(count); ref_h[d].resize(count);
            CHECK(hipMemcpy(ref_x[d].data(), x[d], count * 4, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(ref_h[d].data(), h[d], count * 4, hipMemcpyDeviceToHost));
            if (quant) {
                ref_q[d].resize((size_t) T * (n / 32) * sizeof(q36k::QB));
                CHECK(hipMemcpy(ref_q[d].data(), q[d], ref_q[d].size(), hipMemcpyDeviceToHost));
            }
        }
        for (int graph = 0; graph < 2; ++graph) {
            hipGraphExec_t exec[2] = {};
            if (graph) for (int d = 0; d < 2; ++d) {
                CHECK(hipSetDevice(d)); hipGraph_t captured;
                CHECK(hipStreamBeginCapture(stream[d], hipStreamCaptureModeThreadLocal));
                if (quant) q36tp_fuse2::join_norm<true><<<q36tp::join_blocks, 256, 0, stream[d]>>>(
                    part[d], mapped[d][d], mapped[d][1-d], count, flags[d][d], flags[d][1-d], token[d], d,
                    x[d], w[d], h[d], q[d], n, 1e-6f, T);
                else q36tp_fuse2::join_norm<false><<<q36tp::join_blocks, 256, 0, stream[d]>>>(
                    part[d], mapped[d][d], mapped[d][1-d], count, flags[d][d], flags[d][1-d], token[d], d,
                    x[d], w[d], h[d], nullptr, n, 1e-6f, T);
                CHECK(hipStreamEndCapture(stream[d], &captured));
                CHECK(hipGraphInstantiate(&exec[d], captured, nullptr, nullptr, 0));
                CHECK(hipGraphDestroy(captured));
            }
            for (int rep = 0; rep < 3; ++rep) {
                for (int d = 0; d < 2; ++d) {
                    CHECK(hipSetDevice(d));
                    CHECK(hipMemcpyAsync(part[d], hp[d].data(), count * 4, hipMemcpyHostToDevice, stream[d]));
                    CHECK(hipMemcpyAsync(x[d], hx[d].data(), count * 4, hipMemcpyHostToDevice, stream[d]));
                    if (graph) CHECK(hipGraphLaunch(exec[d], stream[d]));
                    else if (quant) q36tp_fuse2::join_norm<true><<<q36tp::join_blocks, 256, 0, stream[d]>>>(
                        part[d], mapped[d][d], mapped[d][1-d], count, flags[d][d], flags[d][1-d], token[d], d,
                        x[d], w[d], h[d], q[d], n, 1e-6f, T);
                    else q36tp_fuse2::join_norm<false><<<q36tp::join_blocks, 256, 0, stream[d]>>>(
                        part[d], mapped[d][d], mapped[d][1-d], count, flags[d][d], flags[d][1-d], token[d], d,
                        x[d], w[d], h[d], nullptr, n, 1e-6f, T);
                }
                for (int d = 0; d < 2; ++d) {
                    CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(stream[d]));
                    std::vector<float> got_x(count), got_h(count);
                    CHECK(hipMemcpy(got_x.data(), x[d], count * 4, hipMemcpyDeviceToHost));
                    CHECK(hipMemcpy(got_h.data(), h[d], count * 4, hipMemcpyDeviceToHost));
                    if (std::memcmp(got_x.data(), ref_x[d].data(), count * 4) ||
                        std::memcmp(got_h.data(), ref_h[d].data(), count * 4)) {
                        std::fprintf(stderr, "join_norm mismatch n=%d T=%d q=%d graph=%d rep=%d rank=%d\n",
                                     n, T, quant, graph, rep, d); return 1;
                    }
                    if (quant) {
                        std::vector<uint8_t> got(ref_q[d].size());
                        CHECK(hipMemcpy(got.data(), q[d], got.size(), hipMemcpyDeviceToHost));
                        if (got != ref_q[d]) {
                            std::fprintf(stderr, "join_norm q8 mismatch n=%d T=%d graph=%d rep=%d rank=%d\n",
                                         n, T, graph, rep, d); return 1;
                        }
                    }
                }
            }
            if (graph) for (int d = 0; d < 2; ++d) { CHECK(hipSetDevice(d)); CHECK(hipGraphExecDestroy(exec[d])); }
        }
    }
    std::puts("TP join_norm passed: bitwise x, h, q8; T=1..4; both ranks; eager and graph replay");
}
