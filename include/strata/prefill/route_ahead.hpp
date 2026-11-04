// Previous-chunk routing is a bounded copy hint, never a substitute for the current router.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::prefill::route_ahead {
// A scalar applies to every device; a list names devices individually ("0,16" targets the slow second link).
// Invalid, negative, or absent values disable the hint. No allocation or state change when disabled.
inline int requested_from(const char* text, int device) {
    if (text == nullptr || *text == '\0') return 0;
    device = std::max(0, device);
    int first = 0, selected = 0, index = 0;
    const char* p = text;
    for (;;) {
        if (*p < '0' || *p > '9') return 0;
        char* end = nullptr;
        const long value = std::strtol(p, &end, 10);
        if (value < 0 || value > 64 || (*end != '\0' && *end != ',')) return 0;
        if (index == 0) first = (int) value;
        if (index == device) selected = (int) value;
        ++index;
        if (*end == '\0') return index == 1 ? first : selected;
        p = end + 1;
    }
}
inline int bank_slots(int requested, int ring, int group) {
    // The normal copy ring must still hold two complete MMQ groups.
    return std::max(0, std::min({requested, 64, ring - 2 * group}));
}

class History {
public:
    void prepare(int layers, int experts, bool reset) {
        if (layers != layers_ || experts != experts_ || reset) {
            layers_ = layers; experts_ = experts;
            counts_.assign((size_t) layers * experts, 0);
            known_.assign((size_t) layers, 0);
        }
    }
    bool any() const { return std::find(known_.begin(), known_.end(), (uint8_t) 1) != known_.end(); }
    void remember(int layer, const int32_t* counts) {
        std::copy(counts, counts + experts_, counts_.begin() + (size_t) layer * experts_);
        known_[(size_t) layer] = 1;
    }
    template<class Eligible> std::vector<int32_t> predict(int layer, int limit, Eligible eligible) const {
        std::vector<int32_t> ids;
        if (limit <= 0 || layer < 0 || layer >= layers_ || !known_[(size_t) layer]) return ids;
        const int32_t* c = counts_.data() + (size_t) layer * experts_;
        for (int32_t e = 0; e < experts_; ++e) if (c[e] > 0 && eligible(e)) ids.push_back(e);
        std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) {
            return c[a] != c[b] ? c[a] > c[b] : a < b;
        });
        if (ids.size() > (size_t) limit) ids.resize((size_t) limit);
        return ids;
    }
private:
    int layers_ = 0, experts_ = 0;
    std::vector<int32_t> counts_;
    std::vector<uint8_t> known_;
};
}  // namespace strata::prefill::route_ahead
