#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace strata::qwen35::mtp {

// Counts a draft at position i only when that position was actually proposed.
struct Stats {
    bool enabled = false;
    std::vector<long long> offered, reached, accepted;

    void record(int n, int kept) {
        if (!enabled) return;
        if ((int) offered.size() < n) { offered.resize(n); reached.resize(n); accepted.resize(n); }
        for (int i = 0; i < n; ++i) {
            ++offered[i];
            if (i <= kept) ++reached[i];
            if (i < kept) ++accepted[i];
        }
    }

    void print(const char* label) const {
        if (!enabled) return;
        for (size_t i = 0; i < offered.size(); ++i)
            std::fprintf(stderr, "strata-q36: MTP %s draft %zu: %lld/%lld offered (%.1f%%), %lld/%lld reached (%.1f%%)\n",
                         label, i + 1, accepted[i], offered[i], offered[i] ? 100.0 * accepted[i] / offered[i] : 0.0,
                         accepted[i], reached[i], reached[i] ? 100.0 * accepted[i] / reached[i] : 0.0);
    }

    void clear() { offered.clear(); reached.clear(); accepted.clear(); }
};

// Try the full window periodically so a change of topic can restore its length.
// Conditional acceptance is the chance that an additional drafted token pays for its MTP pass.
struct Adaptive {
    bool enabled = false;
    int steps = 0;
    static constexpr int window = 4;
    static constexpr int probe_every = 12;
    std::vector<std::vector<uint8_t>> recent;

    int choose(int maximum) const {
        if (!enabled || maximum < 2 || steps < window || steps % probe_every == 0) return maximum;
        int count = 1;
        for (int i = 1; i < maximum; ++i) {
            if (i >= (int) recent.size() || recent[i].size() < 4) break;
            int good = 0;
            for (uint8_t v : recent[i]) good += v;
            if (good * 2 < (int) recent[i].size()) break;
            ++count;
        }
        return count;
    }

    void record(int n, int kept) {
        if (!enabled) return;
        ++steps;
        if ((int) recent.size() < n) recent.resize(n);
        for (int i = 0; i < n; ++i) {
            auto& r = recent[i];
            if (r.size() == window) r.erase(r.begin());
            r.push_back(i < kept);
        }
    }
};

// The latest continuation of the last 4 or 3 tokens in already emitted text.
inline void lookup(const std::vector<int32_t>& history, std::vector<int32_t>& out, int max_drafts) {
    out.clear();
    const int end = (int) history.size();
    for (int n = 4; n >= 3; --n) {
        if (end < n + 1) continue;
        for (int j = end - n - 1; j >= 0; --j) {
            bool same = true;
            for (int k = 0; k < n && same; ++k) same = history[j + k] == history[end - n + k];
            if (!same) continue;
            for (int k = j + n; k < end && (int) out.size() < max_drafts; ++k) out.push_back(history[k]);
            if (!out.empty()) return;
        }
    }
}

}  // namespace strata::qwen35::mtp
