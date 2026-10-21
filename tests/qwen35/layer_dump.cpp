// CPU exercise of the production trace writer. CUDA calls below only emulate transfers;
// GPU stream ordering still requires the paired engine runs documented in NOTES.md.
// c++ -std=c++17 -Isrc/qwen35 tests/qwen35/layer_dump.cpp -o /tmp/q36-layer-dump
// /tmp/q36-layer-dump /tmp/q36-dump-test.bin
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
using cudaStream_t = void*;
constexpr int cudaHostAllocPortable = 1, cudaMemcpyDeviceToHost = 2;
int cudaHostAlloc(void** p, size_t n, int) { *p = std::malloc(n); return *p ? 0 : 1; }
int cudaFreeHost(void* p) { std::free(p); return 0; }
int cudaMemcpyAsync(void* d, const void* s, size_t n, int, cudaStream_t) { std::memcpy(d, s, n); return 0; }
int cudaStreamSynchronize(cudaStream_t) { return 0; }
#define CK(op) do { if ((op) != 0) std::abort(); } while (0)
#include "q36_layer_dump.cuh"

int main(int argc, char** argv) {
    assert(argc == 2);
    setenv("STRATA_Q36_DUMP_LAYERS", argv[1], 1);
    constexpr int width = 7, layers = 2;
    q36dump::Layers dump;
    int64_t pos = 0;
    for (int rows : {1, 8, 29, 2}) {
        std::vector<int32_t> tokens(rows), cells(rows), slots(rows);
        for (int i = 0; i < rows; ++i) { tokens[i] = 100 + pos + i; cells[i] = pos + i; slots[i] = 0; }
        std::vector<float> x(rows * width, -1.0f);
        // A real warmup must leave no records or call-number gaps.
        dump.begin(width, layers, rows, pos, false, false, 0, true, tokens.data(), nullptr, nullptr, nullptr);
        dump.mixer(x.data(), nullptr); dump.finish(0, x.data(), nullptr);
        dump.begin(width, layers, rows, pos, rows == 8, false, 0, false, tokens.data(),
                   rows == 2 ? cells.data() : nullptr, rows == 2 ? slots.data() : nullptr, nullptr);
        for (int layer = 0; layer < layers; ++layer) {
            for (int i = 0; i < rows * width; ++i) x[i] = (float)(1000 * layer + pos * width + i);
            dump.mixer(x.data(), nullptr);
            for (float& f : x) f += 0.5f;
            dump.finish(layer, x.data(), nullptr);
        }
        pos += rows;
    }
}
