#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace strata::core {

// Calibrated from this request's exposed drafts, never from censored suffixes.
class DraftCumAdapt {
public:
    explicit DraftCumAdapt(float min_p = 0.7f, float threshold = -1.0f)
        : base_(threshold >= 0.0f && std::isfinite(threshold) ? std::clamp(threshold, 0.01f, 0.99f)
                : min_p > 0.0f ? std::clamp(min_p * min_p, 0.01f, 0.99f) : 0.5f), cur_(base_) {}
    static int bin(float p) {
        if (!std::isfinite(p)) return 0;
        return std::clamp((int) (std::clamp(p, 0.0f, 1.0f) * 10.0f), 0, 9);
    }
    float rate(float p) const {
        if (!std::isfinite(p) || p <= 0.0f) return 0.0f;
        const int b = bin(p);
        const double prior = std::clamp((double) p, 0.0, 1.0);
        return (float) ((ok_[b] + 2.0 * prior) / (seen_[b] + 2.0));
    }
    float threshold() const { return std::clamp(cur_ * (1.0f + cpu_pressure_), 0.01f, 0.99f); }
    float cpu_pressure() const { return cpu_pressure_; }
    void observe_cpu(uint64_t cpu_bytes, uint64_t routed_bytes) {
        if (routed_bytes == 0) return;
        cpu_bytes_ = cpu_bytes_ * decay_ + (double) std::min(cpu_bytes, routed_bytes);
        routed_bytes_ = routed_bytes_ * decay_ + (double) routed_bytes;
        cpu_pressure_ = (float) (cpu_bytes_ / routed_bytes_);
    }
    bool keep(const float* probs, int j) const {
        float joint = 1.0f;
        for (int i = 0; i <= j; ++i) joint *= rate(probs[i]);
        return joint >= threshold();
    }
    int rows(const float* probs, int n, int cap) const {
        int t = 1;
        while (t < cap && t <= n && keep(probs, t - 1)) ++t;
        return t;
    }
    void observe(const float* probs, int n, int accepted) {
        if (n <= 0) return;
        accepted = std::clamp(accepted, 0, n);
        double joint = 1.0, expected = 0;
        for (int j = 0; j < n; ++j) { joint *= rate(probs[j]); expected += joint; }
        observed_ = observed_ * decay_ + accepted;
        expected_ = expected_ * decay_ + expected;
        for (int b = 0; b < 10; ++b) { seen_[b] *= decay_; ok_[b] *= decay_; }
        for (int j = 0; j < n && j <= accepted; ++j) {
            const int b = bin(probs[j]);
            ++seen_[b];
            if (j < accepted) ++ok_[b];
        }
        if (expected_ >= 4.0) {
            const double ratio = (observed_ + 2.0) / (expected_ + 2.0);
            cur_ = std::clamp(base_ / (float) std::clamp(ratio, 0.67, 2.0), 0.01f, 0.99f);
        }
    }
    double seen(int b) const { return seen_[(size_t) b]; }
    double accepted(int b) const { return ok_[(size_t) b]; }
private:
    static constexpr double decay_ = 0.97;
    std::array<double, 10> seen_{}, ok_{};
    float base_, cur_, cpu_pressure_ = 0.0f;
    double cpu_bytes_ = 0.0, routed_bytes_ = 0.0;
    double observed_ = 0.0, expected_ = 0.0;
};

} // namespace strata::core
