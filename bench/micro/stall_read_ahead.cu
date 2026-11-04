#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define GPU(x) do { auto e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)
__global__ void scan(const uint32_t* p, size_t n, unsigned long long* sink) {
    unsigned long long sum = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        sum += ((const volatile uint32_t*) p)[i];
    if (sum) atomicAdd(sink, sum);
}
int main() {
    cudaStream_t s, side;
    cudaEvent_t a, b, fork, join;
    GPU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    GPU(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    GPU(cudaEventCreate(&a)); GPU(cudaEventCreate(&b));
    GPU(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    GPU(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
    uint32_t *weights, *hot, *flush;
    unsigned long long *sink, *lines;
    constexpr size_t mb = 1048576, max_bytes = 128 * mb, hot_bytes = 16 * mb, flush_bytes = 384 * mb;
    GPU(cudaMalloc((void**) &weights, max_bytes)); GPU(cudaMalloc((void**) &hot, hot_bytes));
    GPU(cudaMalloc((void**) &flush, flush_bytes)); GPU(cudaMalloc((void**) &sink, 8)); GPU(cudaMalloc((void**) &lines, 8));
    GPU(cudaMemset(weights, 1, max_bytes)); GPU(cudaMemset(hot, 2, hot_bytes)); GPU(cudaMemset(flush, 3, flush_bytes));
    GPU(cudaMemset(sink, 0, 8));
    uint32_t *flag, *mapped;
    GPU(cudaHostAlloc((void**) &flag, 4, cudaHostAllocMapped)); GPU(cudaHostGetDevicePointer((void**) &mapped, flag, 0));
    auto run = [&](const uint32_t* p, size_t bytes) { scan<<<128,256,0,s>>>(p, bytes / 4, sink); };
    auto time = [&](const uint32_t* p, size_t bytes) {
        cudaEventRecord(a, s); run(p, bytes); cudaEventRecord(b, s); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms * 1000.0f;
    };
    strata::kernels::StallReadAhead ranges;
    ranges.p[0] = weights; ranges.bytes[0] = max_bytes; ranges.n = 1;
    // Graph fork/join and the already-completed flag must perform no reads.
    *flag = 1; GPU(cudaMemset(lines, 0, 8));
    GPU(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    GPU(cudaEventRecord(fork, s)); GPU(cudaStreamWaitEvent(side, fork, 0));
    strata::kernels::stall_read_ahead(ranges, mapped, 1, nullptr, max_bytes, 4, side, lines, 8);
    GPU(cudaEventRecord(join, side)); GPU(cudaStreamWaitEvent(s, join, 0));
    cudaGraph_t graph; cudaGraphExec_t exec;
    GPU(cudaStreamEndCapture(s, &graph)); GPU(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    GPU(cudaGraphLaunch(exec, s)); GPU(cudaStreamSynchronize(s));
    unsigned long long count = 1;
    GPU(cudaMemcpy(&count, lines, 8, cudaMemcpyDeviceToHost));
    if (count != 0) { std::fprintf(stderr, "completed flag read %llu lines\n", count); return 1; }
    *flag = 0;
    GPU(cudaGraphLaunch(exec, s)); GPU(cudaStreamSynchronize(s));
    GPU(cudaMemcpy(&count, lines, 8, cudaMemcpyDeviceToHost));
    if (count != max_bytes / 64) { std::fprintf(stderr, "bounded pass read %llu lines\n", count); return 1; }
    // A live host completion cuts a large pass short; the flag is coherent mapped memory.
    GPU(cudaMemset(lines, 0, 8)); *flag = 0;
    run(flush, flush_bytes); GPU(cudaStreamSynchronize(s));
    GPU(cudaGraphLaunch(exec, s));
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    const auto stopped = std::chrono::steady_clock::now();
    *(volatile uint32_t*) flag = 1;
    GPU(cudaStreamSynchronize(s));
    const double stop_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - stopped).count();
    GPU(cudaMemcpy(&count, lines, 8, cudaMemcpyDeviceToHost));
    std::fprintf(stderr, "read-ahead checks: graph fork/join PASS, completed flag PASS, full bounded pass PASS, host stop %llu/%zu lines, completion %.2f us after host stop\n",
                 count, max_bytes / 64, stop_us);
    if (count >= max_bytes / 64) { std::fprintf(stderr, "host stop did not interrupt the pass\n"); return 1; }
    std::puts("read_mb,blocks,chunks,arm,next_us,hot_after_us,prefetch_us");
    for (int read_mb : {8,32,64,128}) for (int blocks : {4,8,16}) for (int chunks : {1,8}) for (int trial = 0; trial < 24; ++trial) {
        const bool on = (trial % 4 == 1 || trial % 4 == 2);
        ranges.bytes[0] = (size_t) read_mb * mb;
        run(flush, flush_bytes); run(hot, hot_bytes); GPU(cudaStreamSynchronize(s));
        float pf_us = 0;
        if (on) {
            *flag = 0;
            GPU(cudaEventRecord(a, s));
            strata::kernels::stall_read_ahead(ranges, mapped, 1, nullptr, ranges.bytes[0], blocks, s, nullptr, chunks);
            GPU(cudaEventRecord(b, s)); GPU(cudaEventSynchronize(b));
            GPU(cudaEventElapsedTime(&pf_us, a, b)); pf_us *= 1000;
        }
        const float hot_us = time(hot, hot_bytes);
        const float next_us = time(weights, ranges.bytes[0]);
        std::printf("%d,%d,%d,%s,%.3f,%.3f,%.3f\n", read_mb, blocks, chunks, on ? "on" : "off", next_us, hot_us, pf_us);
    }
    GPU(cudaGetLastError()); GPU(cudaDeviceSynchronize());
    GPU(cudaGraphExecDestroy(exec)); GPU(cudaGraphDestroy(graph));
    GPU(cudaFreeHost(flag)); GPU(cudaFree(weights)); GPU(cudaFree(hot)); GPU(cudaFree(flush)); GPU(cudaFree(sink)); GPU(cudaFree(lines));
    GPU(cudaEventDestroy(a)); GPU(cudaEventDestroy(b)); GPU(cudaEventDestroy(fork)); GPU(cudaEventDestroy(join));
    GPU(cudaStreamDestroy(side)); GPU(cudaStreamDestroy(s));
    return 0;
}
