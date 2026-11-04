#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

namespace strata::program {

struct TierSwap {
    float gain;
    int32_t layer, in, out;
    int32_t lout = -1;
    int32_t out_layer() const { return lout < 0 ? layer : lout; }
};

struct TierRankOptions {
    bool across = false;
    bool once = false;
    float min_gain = 1.5f;
    float once_bar = 0.9f;
    size_t cap = 0;
};

// Groups name a device and an exact blob size, not a peer expert tier.
inline std::vector<TierSwap> rank_tier(const std::vector<float>& usage, const std::vector<int32_t>& res,
                                      int ne, const std::vector<int>& groups, const std::vector<int32_t>& prior,
                                      const std::vector<uint8_t>& excluded, const TierRankOptions& opt) {
    std::vector<TierSwap> swaps;
    if (ne <= 0 || usage.size() != res.size() || usage.size() != groups.size() * (size_t) ne ||
        prior.size() != usage.size() || excluded.size() != usage.size() || opt.cap == 0) return swaps;
    struct Ent { float u; int rank, l, e; };
    auto key = [](const Ent& x) { return std::make_tuple(x.rank, x.l, x.e); };
    auto hot = [&](const Ent& a, const Ent& b) { return a.u != b.u ? a.u > b.u : key(a) < key(b); };
    auto cold = [&](const Ent& a, const Ent& b) { return a.u != b.u ? a.u < b.u : key(a) > key(b); };
    const int nl = (int) groups.size();
    std::vector<int> buckets = groups;
    if (!opt.across) for (int l = 0; l < nl; ++l) buckets[l] = l;
    std::vector<int> unique = buckets;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    for (int bucket : unique) {
        std::vector<Ent> cand, vict;
        for (int l = 0; l < nl; ++l) {
            if (buckets[l] != bucket) continue;
            for (int e = 0; e < ne; ++e) {
                const size_t i = (size_t) l * ne + e;
                if (res[i] >= 0) vict.push_back({usage[i], prior[i], l, e});
                else if (!excluded[i] && usage[i] >= (opt.across ? 1.5f : 2.0f))
                    cand.push_back({usage[i], prior[i], l, e});
            }
        }
        const size_t nc = std::min(cand.size(), vict.size());
        if (opt.across) {
            std::sort(cand.begin(), cand.end(), hot);
            std::partial_sort(vict.begin(), vict.begin() + nc, vict.end(), cold);
        } else {
            // Keep the original per-layer tie order for the timing-only switch.
            std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.u > b.u; });
            std::partial_sort(vict.begin(), vict.begin() + nc, vict.end(), [](auto& a, auto& b) { return a.u < b.u; });
        }
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].u < vict[i].u + opt.min_gain) break;
            swaps.push_back({cand[i].u - vict[i].u, cand[i].l, cand[i].e, vict[i].e, vict[i].l});
        }
    }
    if (opt.across) std::stable_sort(swaps.begin(), swaps.end(), [](auto& a, auto& b) { return a.gain > b.gain; });
    else std::sort(swaps.begin(), swaps.end(), [](auto& a, auto& b) { return a.gain > b.gain; });
    if (swaps.size() > opt.cap) swaps.resize(opt.cap);
    if (!opt.once || swaps.size() >= opt.cap) return swaps;
    std::vector<uint8_t> taken(res.size(), 0);
    for (const auto& s : swaps) {
        taken[(size_t) s.layer * ne + s.in] = 1;
        taken[(size_t) s.out_layer() * ne + s.out] = 1;
    }
    std::vector<TierSwap> extra;
    const float floor = opt.across ? 1.5f : 2.0f;
    for (int bucket : unique) {
        std::vector<Ent> cand, vict;
        for (int l = 0; l < nl; ++l) {
            if (buckets[l] != bucket) continue;
            for (int e = 0; e < ne; ++e) {
                const size_t i = (size_t) l * ne + e;
                if (taken[i]) continue;
                if (res[i] >= 0 && usage[i] < floor - opt.once_bar) vict.push_back({usage[i], prior[i], l, e});
                else if (res[i] < 0 && !excluded[i] && usage[i] >= opt.once_bar && usage[i] < floor)
                    cand.push_back({usage[i], prior[i], l, e});
            }
        }
        const size_t nc = std::min(cand.size(), vict.size());
        std::sort(cand.begin(), cand.end(), hot);
        std::partial_sort(vict.begin(), vict.begin() + nc, vict.end(), cold);
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].u < vict[i].u + opt.once_bar) break;
            extra.push_back({cand[i].u - vict[i].u, cand[i].l, cand[i].e, vict[i].e, vict[i].l});
        }
    }
    std::stable_sort(extra.begin(), extra.end(), [](auto& a, auto& b) { return a.gain > b.gain; });
    const size_t n = std::min({extra.size(), opt.cap - swaps.size(), size_t(16)});
    swaps.insert(swaps.end(), extra.begin(), extra.begin() + n);
    return swaps;
}

} // namespace strata::program
