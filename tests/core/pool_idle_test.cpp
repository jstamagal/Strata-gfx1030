#include "strata/kernels/cpu/pool.hpp"
#include <atomic>
#include <cstdio>
#include <thread>
using strata::kernels::cpu::ExpertPool;
struct Work {
    std::atomic<int> active{0}, entered{0}, exited{0};
    static void job(void* ctx, int, int, const std::atomic<bool>& stop) {
        auto& w = *(Work*) ctx;
        ++w.entered; ++w.active;
        while (!stop.load(std::memory_order_acquire)) std::this_thread::yield();
        --w.active; ++w.exited;
    }
};
int main() {
    ExpertPool pool(4, false);
    Work w;
    for (int i = 0; i < 3000; ++i) {
        pool.idle_start(&Work::job, &w);
        if (i % 2) while (w.active == 0) std::this_thread::yield();
        if (i % 3) pool.idle_stop();
        else pool.run(nullptr, 0);
        if (w.active != 0 || w.entered != w.exited) { std::fputs("pool_idle_test: active work escaped cancellation\n", stderr); return 1; }
    }
    pool.idle_start(&Work::job, &w);
    pool.idle_start(&Work::job, &w);
    pool.idle_stop();
    std::printf("pool_idle_test: PASS (%d entered and joined)\n", w.entered.load());
    return 0;
}
