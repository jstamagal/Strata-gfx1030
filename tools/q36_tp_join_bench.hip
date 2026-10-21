// Graph-replayed, two-card mapped-host join timing. Build/run commands are in NOTES.md.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HC(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); \
} } while (0)

struct Times { unsigned long long publish, wait, read, ack, total; };
__device__ unsigned load_flag(unsigned* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ void store_flag(unsigned* p, unsigned v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void joined(float* dst, float* mine, const float* peer, unsigned* own_flags,
                       unsigned* peer_flags, unsigned* token, int rank, int n, Times* times, int iter) {
    __shared__ unsigned call;
    __shared__ unsigned long long t0, t1, t2, t3, t4;
    if (threadIdx.x == 0) { call = token[blockIdx.x] + 1; t0 = clock64(); }
    __syncthreads();
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    for (int i = i0; i < n / 4; i += step)
        ((float4*) mine)[i] = ((const float4*) dst)[i];
    __threadfence_system();
    __syncthreads();
    const int f = blockIdx.x * 16;
    if (threadIdx.x == 0) {
        store_flag(own_flags + f, call);
        t1 = clock64();
        while (load_flag(peer_flags + f) != call) __builtin_amdgcn_s_sleep(4);
        t2 = clock64();
    }
    __syncthreads();
    __threadfence_system();
    for (int i = i0; i < n / 4; i += step) {
        typedef float v4 __attribute__((ext_vector_type(4)));
        const v4 v = __builtin_nontemporal_load((const v4*) peer + i);
        const float4 b = make_float4(v.x, v.y, v.z, v.w);
        float4 a = ((const float4*) dst)[i];
        if (rank == 0) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
        else { a.x = b.x + a.x; a.y = b.y + a.y; a.z = b.z + a.z; a.w = b.w + a.w; }
        ((float4*) dst)[i] = a;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        t3 = clock64();
        store_flag(own_flags + f + 1, call);
        while (load_flag(peer_flags + f + 1) != call) __builtin_amdgcn_s_sleep(4);
        token[blockIdx.x] = call;
        t4 = clock64();
        times[iter * gridDim.x + blockIdx.x] = {t1-t0, t2-t1, t3-t2, t4-t3, t4-t0};
    }
}

struct Card {
    hipStream_t st{};
    hipGraph_t graph{};
    hipGraphExec_t exec{};
    hipEvent_t begin{}, end{};
    float *local{}, *mapped[2]{};
    unsigned *flags[2]{}, *token{};
    Times* times{};
    hipDeviceProp_t prop{};
};
int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 1;
    const int N = argc > 2 ? std::atoi(argv[2]) : 3072;
    const int blocks = argc > 3 ? std::atoi(argv[3]) : 8;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 256;
    const int first = argc > 5 ? std::atoi(argv[5]) : 1;
    if (T < 1 || N < 1 || (T * N) % 4 || blocks < 1 || blocks > 8 || reps < 2 || first < 0 || first > 1) return 2;
    int devices = 0;
    HC(hipGetDeviceCount(&devices));
    if (devices < 2) { std::fprintf(stderr, "two HIP devices required\n"); return 2; }
    const int n = T * N;
    float* host[2]{};
    unsigned* host_flags[2]{};
    for (int d = 0; d < 2; ++d) {
        HC(hipHostMalloc(&host[d], (size_t)n * sizeof(float), hipHostMallocPortable | hipHostMallocMapped));
        HC(hipHostMalloc(&host_flags[d], (size_t)8 * 16 * sizeof(unsigned), hipHostMallocPortable | hipHostMallocMapped));
        std::fill(host_flags[d], host_flags[d] + 8 * 16, 0u);
    }
    Card c[2];
    for (int d = 0; d < 2; ++d) {
        HC(hipSetDevice(d));
        HC(hipGetDeviceProperties(&c[d].prop, d));
        HC(hipStreamCreate(&c[d].st));
        HC(hipEventCreate(&c[d].begin)); HC(hipEventCreate(&c[d].end));
        HC(hipMalloc(&c[d].local, (size_t)n * sizeof(float)));
        HC(hipMalloc(&c[d].token, (size_t)blocks * sizeof(unsigned)));
        HC(hipMalloc(&c[d].times, (size_t)reps * blocks * sizeof(Times)));
        HC(hipMemset(c[d].token, 0, (size_t)blocks * sizeof(unsigned)));
        std::vector<float> init(n, (float)(d+1));
        HC(hipMemcpy(c[d].local, init.data(), (size_t)n * sizeof(float), hipMemcpyHostToDevice));
        for (int r = 0; r < 2; ++r) {
            HC(hipHostGetDevicePointer((void**)&c[d].mapped[r], host[r], 0));
            HC(hipHostGetDevicePointer((void**)&c[d].flags[r], host_flags[r], 0));
        }
        HC(hipStreamBeginCapture(c[d].st, hipStreamCaptureModeThreadLocal));
        for (int i = 0; i < reps; ++i)
            hipLaunchKernelGGL(joined, dim3(blocks), dim3(256), 0, c[d].st,
                c[d].local, c[d].mapped[d], c[d].mapped[1-d], c[d].flags[d], c[d].flags[1-d],
                c[d].token, d, n, c[d].times, i);
        HC(hipStreamEndCapture(c[d].st, &c[d].graph));
        HC(hipGraphInstantiate(&c[d].exec, c[d].graph, nullptr, nullptr, 0));
    }
    for (int i = 0; i < 2; ++i) {
        const int d = i == 0 ? first : 1-first;
        HC(hipSetDevice(d));
        HC(hipEventRecord(c[d].begin, c[d].st));
        HC(hipGraphLaunch(c[d].exec, c[d].st));
        HC(hipEventRecord(c[d].end, c[d].st));
    }
    for (int d = 0; d < 2; ++d) {
        HC(hipSetDevice(d));
        HC(hipEventSynchronize(c[d].end));
        float ms = 0;
        HC(hipEventElapsedTime(&ms, c[d].begin, c[d].end));
        std::vector<Times> v((size_t)reps * blocks);
        HC(hipMemcpy(v.data(), c[d].times, v.size() * sizeof(Times), hipMemcpyDeviceToHost));
        double sums[5]{};
        for (int i = blocks; i < (int)v.size(); ++i) {
            sums[0] += v[i].publish; sums[1] += v[i].wait; sums[2] += v[i].read;
            sums[3] += v[i].ack; sums[4] += v[i].total;
        }
        const double divisor = (double)(reps - 1) * blocks * c[d].prop.clockRate / 1000.0;
        const double wait = sums[1] + sums[3], transfer = sums[0] + sums[2];
        std::printf("rank %d first=%d T=%d N=%d blocks=%d graph_us=%.2f ",
                    d, first, T, N, blocks, ms * 1000.0f / reps);
        std::printf("CTA_us publish=%.2f peer_wait=%.2f read_sum=%.2f ack_wait=%.2f ",
                    sums[0]/divisor, sums[1]/divisor, sums[2]/divisor, sums[3]/divisor);
        std::printf("wait_fraction=%.1f%% transfer_fraction=%.1f%%\n",
                    100.0 * wait / (wait + transfer), 100.0 * transfer / (wait + transfer));
    }
}
