#include "strata/core/expert_warm.hpp"
#include <cstdio>
#include <thread>
#include <vector>
using strata::core::NextExpertWarm;
static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
int main() {
    NextExpertWarm warm(2, 8, 64);
    std::vector<uint16_t> router(8 * 64);
    for (int e = 0; e < 8; ++e) for (int i = 0; i < 64; ++i) {
        float f = (float) e; uint32_t bits; std::memcpy(&bits, &f, 4); router[e * 64 + i] = (uint16_t) (bits >> 16);
    }
    CHECK(warm.set_router(1, router.data(), 256, 128));
    float x[128]; std::fill(x, x + 128, 1);
    int32_t res[8]; std::fill(res, res + 8, -1); res[7] = 0;
    std::vector<std::vector<uint8_t>> memory(8, std::vector<uint8_t>(256));
    std::vector<const uint8_t*> blobs; for (auto& v : memory) blobs.push_back(v.data());
    for (int workers : {1,2,4,8}) {
        CHECK(warm.prepare(1, x, 2, 2, res, blobs));
        std::atomic<bool> stop{false}; std::vector<std::thread> threads;
        for (int w = 0; w < workers; ++w) threads.emplace_back([&,w] { NextExpertWarm::job(&warm, w, workers, stop); });
        for (auto& t : threads) t.join();
        CHECK(warm.best == 6);
    }
    CHECK(warm.prepare(1, x, 2, 2, res, blobs));
    std::atomic<bool> stop{true};
    NextExpertWarm::job(&warm, 0, 1, stop); CHECK(warm.best == -1);
    res[6] = 1;
    CHECK(warm.prepare(1, x, 2, 2, res, blobs)); stop = false;
    NextExpertWarm::job(&warm, 0, 1, stop); CHECK(warm.best == -1);
    CHECK(!warm.prepare(0, x, 2, 2, res, blobs));
    CHECK(!warm.prepare(1, x, 17, 2, res, blobs));
    std::printf("expert_warm_test: %s (parallel predictor, CPU-only selection, cancellation)\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
