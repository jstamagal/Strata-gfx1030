#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "q36_tp_collective.cuh"
#define CHECK(x) do { auto e = (x); if(e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); return 1; } } while(0)
__global__ void fill(float* out, size_t n, int rank) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)(i % 97) + rank;
}
__global__ void check(const float* out, size_t n, unsigned* bad) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && out[i] != (float)(2*(i%97)+1)) atomicAdd(bad, 1u);
}
int main(int argc, char** argv) {
    const int width = argc > 1 ? std::atoi(argv[1]) : 3072;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 100;
    if (width <= 0 || iters <= 0) return 2;
    const size_t cap = (size_t)width * 8;
    float *host[2], *mapped[2][2], *buf[2];
    unsigned *flags_h[2], *flags[2][2], *token[2], *bad[2];
    hipStream_t st[2]; hipGraphExec_t gx[2];
    for (int d = 0; d < 2; ++d) {
        CHECK(hipSetDevice(d));
        CHECK(hipHostMalloc((void**)&host[d], cap * 4, hipHostMallocMapped | hipHostMallocPortable));
        CHECK(hipHostMalloc((void**)&flags_h[d], q36tp::join_blocks * q36tp::flag_stride * sizeof(unsigned), hipHostMallocMapped | hipHostMallocPortable));
        std::fill(flags_h[d], flags_h[d] + q36tp::join_blocks * q36tp::flag_stride, 0);
        CHECK(hipMalloc((void**)&buf[d], cap * 4));
        CHECK(hipMalloc((void**)&token[d], q36tp::join_blocks * sizeof(unsigned)));
        CHECK(hipMemset(token[d], 0, q36tp::join_blocks * sizeof(unsigned)));
        CHECK(hipMalloc((void**)&bad[d], sizeof(unsigned)));
        CHECK(hipMemset(bad[d], 0, sizeof(unsigned)));
        CHECK(hipDeviceSynchronize());
        CHECK(hipStreamCreateWithFlags(&st[d], hipStreamNonBlocking));
    }
    for (int d = 0; d < 2; ++d) {
        CHECK(hipSetDevice(d));
        for(int r = 0; r < 2; ++r) {
            CHECK(hipHostGetDevicePointer((void**)&mapped[d][r], host[r], 0));
            CHECK(hipHostGetDevicePointer((void**)&flags[d][r], flags_h[r], 0));
        }
    }
    // Alternate shapes, then repeat the same captured calls. Tokens must advance on the GPU, including replay.
    for (int graph = 0; graph < 2; ++graph) for (int rows = 1; rows <= 8; ++rows) {
        const size_t n = (size_t)width * rows;
        auto launch = [&](int d) {
            fill<<<(n+255)/256, 256, 0, st[d]>>>(buf[d], n, d);
            q36tp::join<<<q36tp::join_blocks, 256, 0, st[d]>>>(buf[d], mapped[d][d], mapped[d][1-d], n,
                flags[d][d], flags[d][1-d], token[d], d);
            check<<<(n+255)/256, 256, 0, st[d]>>>(buf[d], n, bad[d]);
        };
        if (graph) for (int d = 0; d < 2; ++d) {
            CHECK(hipSetDevice(d)); hipGraph_t g;
            CHECK(hipStreamBeginCapture(st[d], hipStreamCaptureModeThreadLocal));
            launch(d);
            CHECK(hipStreamEndCapture(st[d], &g));
            CHECK(hipGraphInstantiate(&gx[d], g, nullptr, nullptr, 0));
            CHECK(hipGraphDestroy(g));
        }
        for (int it = 0; it < iters; ++it) for (int d = 0; d < 2; ++d) {
            CHECK(hipSetDevice(d));
            if (graph) CHECK(hipGraphLaunch(gx[d], st[d])); else launch(d);
        }
        for (int d = 0; d < 2; ++d) {
            CHECK(hipSetDevice(d)); CHECK(hipStreamSynchronize(st[d]));
            std::vector<float> out(n);
            CHECK(hipMemcpy(out.data(), buf[d], n*4, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i) if(out[i] != (float)(2*(i%97)+1)) {
                std::fprintf(stderr, "join mismatch graph=%d rows=%d rank=%d at %zu\n", graph, rows, d, i); return 1;
            }
            unsigned errors = 0;
            CHECK(hipMemcpy(&errors, bad[d], sizeof(unsigned), hipMemcpyDeviceToHost));
            if (errors) { std::fprintf(stderr, "%u mismatches across earlier joins on rank %d\n", errors, d); return 1; }
            if(graph) CHECK(hipGraphExecDestroy(gx[d]));
        }
    }
    std::puts("TP join passed: T=1..8, both ranks, eager and graph replay");
}
