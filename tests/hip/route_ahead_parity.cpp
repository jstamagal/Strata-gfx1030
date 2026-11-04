// Synthetic expert byte readers exercise the production route predictor, ring wrap, hint misses and slot events.
// No model. Timings are for this copy/reader fixture, not an inference speed prediction.
#include "strata/prefill/route_ahead.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int E = 128, L = 3, Chunks = 3, Ring = 64, Group = 16, Resident = 16, Parts = 4;
constexpr size_t Blob = 1u << 20, Words = Blob / sizeof(uint32_t);
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
__global__ void read_blob(const uint32_t* input, uint32_t* out) {
    __shared__ uint32_t part[256];
    uint32_t sum = 2166136261u ^ (blockIdx.x * blockDim.x + threadIdx.x + 1);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < Words; i += gridDim.x * blockDim.x)
        sum = (sum ^ input[i]) * 16777619u;
    part[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) part[threadIdx.x] ^= part[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = part[0];
}
struct Result { std::vector<uint32_t> hashes; uint64_t copied = 0, predicted = 0, hit = 0; double ms = 0; };
Result run(bool hints, const uint32_t* host, const uint32_t* cache) {
    uint32_t *stage = nullptr, *output = nullptr;
    ck(cudaMalloc(&stage, (size_t) Ring * Blob), "stage allocation");
    const size_t outputs = (size_t) Chunks * L * E * Parts;
    ck(cudaMalloc(&output, outputs * sizeof(uint32_t)), "hash allocation");
    ck(cudaMemset(output, 0, outputs * sizeof(uint32_t)), "zero hashes");
    cudaStream_t copy{}, compute{};
    ck(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking), "copy stream");
    ck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), "compute stream");
    cudaEvent_t copied[Ring]{}, used[Ring]{};
    int used_of[Ring]{}; bool live[Ring]{};
    for (int i = 0; i < Ring; ++i) {
        used_of[i] = i;
        ck(cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming), "copied event");
        ck(cudaEventCreateWithFlags(&used[i], cudaEventDisableTiming), "used event");
    }
    strata::prefill::route_ahead::History history;
    history.prepare(L, E, false);
    Result result;
    auto upload = [&](int l, int e, int sl) {
        if (live[sl]) ck(cudaStreamWaitEvent(copy, used[used_of[sl]], 0), "slot release wait");
        ck(cudaMemcpyAsync(stage + (size_t) sl * Words, host + ((size_t) l * E + e) * Words,
                           Blob, cudaMemcpyHostToDevice, copy), "expert upload");
        ck(cudaEventRecord(copied[sl], copy), "expert ready");
        live[sl] = true;
        ++result.copied;
    };
    const auto t0 = std::chrono::steady_clock::now();
    for (int chunk = 0; chunk < Chunks; ++chunk) {
        const int bank = hints && history.any() ? strata::prefill::route_ahead::bank_slots(16, Ring, Group) : 0;
        const int regular = Ring - bank;
        size_t issued = 0, consumed = 0;
        for (int l = 0; l < L; ++l) {
            std::vector<int> slot(E, -1);
            auto predicted = history.predict(l, bank, [](int32_t e) { return e >= Resident; });
            for (size_t i = 0; i < predicted.size(); ++i) {
                slot[(size_t) predicted[i]] = regular + (int) i;
                upload(l, predicted[i], regular + (int) i);
                ++result.predicted;
            }
            std::vector<int32_t> cnt(E), order, seq;
            for (int e = 0; e < E; ++e) {
                if ((e + 2 * chunk + l) % 9 != 0) {
                    cnt[(size_t) e] = 1 + (e * 17 + l * 3 + chunk) % 37;
                    order.push_back(e);
                    if (e >= Resident && slot[(size_t) e] < 0) seq.push_back(e);
                }
                if (slot[(size_t) e] >= 0 && cnt[(size_t) e] > 0) ++result.hit;
            }
            history.remember(l, cnt.data());
            const size_t start = issued, end = start + seq.size();
            auto issue = [&](size_t limit) {
                while (issued < std::min(end, limit)) {
                    upload(l, seq[issued - start], (int) (issued % regular));
                    ++issued;
                }
            };
            issue(consumed + regular);
            for (const int e : order) {
                const uint32_t* src = nullptr;
                int normal_slot = -1;
                if (e < Resident) src = cache + ((size_t) l * Resident + e) * Words;
                else if (slot[(size_t) e] >= 0) {
                    const int sl = slot[(size_t) e];
                    ck(cudaStreamWaitEvent(compute, copied[sl], 0), "hint ready wait");
                    src = stage + (size_t) sl * Words;
                } else {
                    normal_slot = (int) (consumed % regular);
                    ck(cudaStreamWaitEvent(compute, copied[normal_slot], 0), "exact expert ready wait");
                    src = stage + (size_t) normal_slot * Words;
                }
                read_blob<<<Parts, 256, 0, compute>>>(src, output + (((size_t) chunk * L + l) * E + e) * Parts);
                ck(cudaGetLastError(), "byte reader");
                if (normal_slot >= 0) {
                    ck(cudaEventRecord(used[normal_slot], compute), "normal slot release");
                    used_of[normal_slot] = normal_slot;
                    ++consumed;
                    issue(consumed + regular);
                }
            }
            if (!predicted.empty()) {
                ck(cudaEventRecord(used[regular], compute), "bank release");
                for (int sl = regular; sl < Ring; ++sl) used_of[sl] = regular;
            }
        }
    }
    ck(cudaStreamSynchronize(compute), "reader drain");
    ck(cudaStreamSynchronize(copy), "copy drain");
    result.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    result.hashes.resize(outputs);
    ck(cudaMemcpy(result.hashes.data(), output, outputs * sizeof(uint32_t), cudaMemcpyDeviceToHost), "hash readback");
    for (int i = 0; i < Ring; ++i) { ck(cudaEventDestroy(copied[i]), "destroy copied"); ck(cudaEventDestroy(used[i]), "destroy used"); }
    ck(cudaStreamDestroy(copy), "destroy copy"); ck(cudaStreamDestroy(compute), "destroy compute");
    ck(cudaFree(stage), "free stage"); ck(cudaFree(output), "free hashes");
    return result;
}
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    uint32_t* host = nullptr;
    ck(cudaHostAlloc((void**) &host, (size_t) L * E * Blob, cudaHostAllocPortable), "source arena");
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e)
            for (size_t i = 0; i < Words; ++i)
                host[((size_t) l * E + e) * Words + i] = (uint32_t) (i * 13 + e * 997 + l * 1000003);
    bool ok = true;
    for (int d = 0; d < devices; ++d) {
        ck(cudaSetDevice(d), "device");
        uint32_t* cache = nullptr;
        ck(cudaMalloc(&cache, (size_t) L * Resident * Blob), "resident cache");
        for (int l = 0; l < L; ++l)
            ck(cudaMemcpy(cache + (size_t) l * Resident * Words, host + (size_t) l * E * Words,
                          (size_t) Resident * Blob, cudaMemcpyHostToDevice), "resident initial fill");
        const auto reference = run(false, host, cache), hinted = run(true, host, cache);
        const bool equal = reference.hashes == hinted.hashes;
        const bool exercised = hinted.predicted > hinted.hit && hinted.hit > 0;
        std::printf("GPU%d 1-MiB expert byte fixture: reference %.3f ms/%llu copies; hints %.3f ms/%llu copies, "
                    "predicted %llu hit %llu waste %llu; all hashes %s, hit+miss exercised %s\n", d,
                    reference.ms, (unsigned long long) reference.copied, hinted.ms, (unsigned long long) hinted.copied,
                    (unsigned long long) hinted.predicted, (unsigned long long) hinted.hit,
                    (unsigned long long) (hinted.predicted - hinted.hit), equal ? "identical" : "DIFFERENT",
                    exercised ? "yes" : "NO");
        ok = ok && equal && exercised;
        ck(cudaFree(cache), "free resident cache");
    }
    ck(cudaFreeHost(host), "free source arena");
    std::puts(ok ? "PASS: exact expert bytes despite predictions, misses and wrapping" : "FAIL");
    return ok ? 0 : 1;
}
