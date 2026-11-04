#pragma once
#include <array>
#include <atomic>
#include <thread>

namespace strata::program {

// A copy owns an idle stage until it lands; the host keeps serving the other stages.
class AdaptGap {
public:
    bool enabled = false;
    std::atomic<bool> stopping{false};
    bool launch(int st) {
        if (!enabled) return true;
        int idle = 0;
        return state_[(size_t) st].compare_exchange_strong(idle, 1, std::memory_order_acq_rel);
    }
    void idle(int st) {
        int running = 1;
        state_[(size_t) st].compare_exchange_strong(running, 0, std::memory_order_acq_rel);
    }
    void copy_begin(int st) {
        if (!enabled) return;
        while (!stopping.load(std::memory_order_acquire)) {
            int idle = 0;
            if (state_[(size_t) st].compare_exchange_weak(idle, 2, std::memory_order_acq_rel)) return;
            std::this_thread::yield();
        }
    }
    void copy_end(int st) {
        if (enabled) state_[(size_t) st].store(0, std::memory_order_release);
    }
private:
    std::array<std::atomic<int>, 4> state_{};
};

} // namespace strata::program
