// Two-card bitwise check of the fused MoE combine and mapped-host join.
// hipcc -O2 --offload-arch=gfx1030 -Isrc/qwen35 tests/qwen35/tp_moe_join.hip -o /tmp/tp-moe-join
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "q36_tp_collective.cuh"
#include "q36_tp_moe_join.cuh"

#define CHECK(x) do { const hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)

int main() {
    constexpr int max_n = 1024, max_t = 4, picks = 8;
    float *parts[2], *wts[2], *shared[2], *gate[2], *out[2], *mapped[2][2], *host[2];
    unsigned *flags_h[2], *flags[2][2], *token[2];
    hipStream_t stream[2];
    for (int d = 0; d < 2; ++d) {
        CHECK(hipSetDevice(d));
        CHECK(hipMalloc((void**) &parts[d], (size_t) max_t * picks * max_n * sizeof(float)));
        CHECK(hipMalloc((void**) &wts[d], max_t * picks * sizeof(float)));
        CHECK(hipMalloc((void**) &shared[d], max_t * max_n * sizeof(float)));
        CHECK(hipMalloc((void**) &gate[d], max_t * sizeof(float)));
        CHECK(hipMalloc((void**) &out[d], max_t * max_n * sizeof(float)));
        CHECK(hipMalloc((void**) &token[d], q36tp::join_blocks * sizeof(unsigned)));
        CHECK(hipMemset(token[d], 0, q36tp::join_blocks * sizeof(unsigned)));
        CHECK(hipHostMalloc((void**) &host[d], max_t * max_n * sizeof(float), hipHostMallocMapped | hipHostMallocPortable));
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
    for (const int n : {96, 257, max_n}) for (int T = 1; T <= max_t; ++T) {
        const size_t count = (size_t) T * n;
        for (int d = 0; d < 2; ++d) {
            std::vector<float> hp(count * picks), hw((size_t) T * picks), hs(count), hg(T);
            for (size_t i = 0; i < hp.size(); ++i) hp[i] = 0.001f * (float) ((i * 13 + d * 7) % 101);
            for (size_t i = 0; i < hw.size(); ++i) hw[i] = 0.01f * (float) (i % picks + 1);
            for (size_t i = 0; i < hs.size(); ++i) hs[i] = 0.002f * (float) ((i + d * 11) % 37);
            for (int t = 0; t < T; ++t) hg[t] = 0.1f * (float) (t + d);
            CHECK(hipSetDevice(d));
            CHECK(hipMemcpy(parts[d], hp.data(), hp.size() * sizeof(float), hipMemcpyHostToDevice));
            CHECK(hipMemcpy(wts[d], hw.data(), hw.size() * sizeof(float), hipMemcpyHostToDevice));
            CHECK(hipMemcpy(shared[d], hs.data(), hs.size() * sizeof(float), hipMemcpyHostToDevice));
            CHECK(hipMemcpy(gate[d], hg.data(), hg.size() * sizeof(float), hipMemcpyHostToDevice));
            q36tp::partial_moe<<<T, 256, 0, stream[d]>>>(out[d], parts[d], nullptr, wts[d], picks,
                                                          shared[d], gate[d], 1, n);
            q36tp::join<<<q36tp::join_blocks, 256, 0, stream[d]>>>(out[d], mapped[d][d], mapped[d][1-d], count,
                flags[d][d], flags[d][1-d], token[d], d);
        }
        std::vector<float> reference[2];
        for (int d = 0; d < 2; ++d) {
            CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(stream[d]));
            reference[d].resize(count);
            CHECK(hipMemcpy(reference[d].data(), out[d], count * sizeof(float), hipMemcpyDeviceToHost));
        }
        for (int graph = 0; graph < 2; ++graph) {
            hipGraphExec_t exec[2] = {};
            if (graph) for (int d = 0; d < 2; ++d) {
                CHECK(hipSetDevice(d));
                hipGraph_t captured;
                CHECK(hipStreamBeginCapture(stream[d], hipStreamCaptureModeThreadLocal));
                q36tp::moe_join<<<q36tp::join_blocks, 256, 0, stream[d]>>>(out[d], parts[d], nullptr, wts[d], picks,
                    shared[d], gate[d], 1, n, T, mapped[d][d], mapped[d][1-d], flags[d][d], flags[d][1-d], token[d], d);
                CHECK(hipStreamEndCapture(stream[d], &captured));
                CHECK(hipGraphInstantiate(&exec[d], captured, nullptr, nullptr, 0));
                CHECK(hipGraphDestroy(captured));
            }
            for (int rep = 0; rep < 3; ++rep) {
                for (int d = 0; d < 2; ++d) {
                    CHECK(hipSetDevice(d));
                    if (graph) CHECK(hipGraphLaunch(exec[d], stream[d]));
                    else q36tp::moe_join<<<q36tp::join_blocks, 256, 0, stream[d]>>>(out[d], parts[d], nullptr, wts[d], picks,
                        shared[d], gate[d], 1, n, T, mapped[d][d], mapped[d][1-d], flags[d][d], flags[d][1-d], token[d], d);
                }
                for (int d = 0; d < 2; ++d) {
                    CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(stream[d]));
                    std::vector<float> got(count);
                    CHECK(hipMemcpy(got.data(), out[d], count * sizeof(float), hipMemcpyDeviceToHost));
                    if (std::memcmp(got.data(), reference[d].data(), count * sizeof(float))) {
                        std::fprintf(stderr, "MoE join mismatch: rank=%d n=%d T=%d graph=%d rep=%d\n", d, n, T, graph, rep);
                        return 1;
                    }
                }
            }
            if (graph) for (int d = 0; d < 2; ++d) {
                CHECK(hipSetDevice(d)); CHECK(hipGraphExecDestroy(exec[d]));
            }
        }
    }
    std::puts("TP MoE join passed: bitwise parity, T=1..4, both ranks, eager and graph replay");
}
