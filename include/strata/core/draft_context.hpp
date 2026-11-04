#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {

class DraftContextRows {
public:
    static constexpr int capacity = 512, pick_reserve = 64, prompt_share = 256;
    void init(int vocab, const std::vector<int32_t>& subset) {
        base_.assign((size_t) vocab, 0);
        for (int32_t t : subset) if (t >= 0 && t < vocab) base_[(size_t) t] = 1;
        seen_ = base_;
        ids_.reserve(capacity);
    }
    void begin(const int64_t* prompt, int64_t n) {
        seen_ = base_; ids_.clear(); flushed_ = 0;
        std::vector<int32_t> recent;
        recent.reserve(prompt_share);
        for (int64_t i = n; i-- > 0 && recent.size() < prompt_share;) {
            const int64_t t = prompt[i];
            if (t < 0 || t >= (int64_t) seen_.size() || seen_[(size_t) t]) continue;
            seen_[(size_t) t] = 1; recent.push_back((int32_t) t);
        }
        ids_.assign(recent.rbegin(), recent.rend());
    }
    void note(const int32_t* ids, int n, bool candidates) {
        if (!ids) return;
        const size_t cap = candidates ? capacity - pick_reserve : capacity;
        for (int i = 0; i < n && ids_.size() < cap; ++i) {
            const int32_t t = ((const volatile int32_t*) ids)[i];
            if (t < 0 || t >= (int32_t) seen_.size() || seen_[(size_t) t]) continue;
            seen_[(size_t) t] = 1; ids_.push_back(t);
        }
    }
    const std::vector<int32_t>& ids() const { return ids_; }
    size_t flushed() const { return flushed_; }
    void commit() { flushed_ = ids_.size(); }
private:
    std::vector<uint8_t> base_, seen_;
    std::vector<int32_t> ids_;
    size_t flushed_ = 0;
};

} // namespace strata::core
