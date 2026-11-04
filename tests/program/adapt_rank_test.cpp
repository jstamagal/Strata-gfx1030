#include "strata/program/adapt_rank.hpp"
#include "strata/program/adapt_gap.hpp"
#include <cstdio>
#include <random>
#include <set>
#include <thread>

using namespace strata::program;
static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    TierRankOptions o;
    o.across = true; o.cap = 160;
    std::vector<float> u{4, 0, 0, 0, 1, 5, 0, 0};
    std::vector<int32_t> r{-1, -1, -1, -1, 0, -1, -1, -1}, p{0,1,2,3,4,5,6,7};
    std::vector<uint8_t> excluded(8, 0);
    auto s = rank_tier(u, r, 4, {0,0}, p, excluded, o);
    CHECK(s.size() == 1 && s[0].layer == 1 && s[0].in == 1 && s[0].out_layer() == 1);
    u[0] = 6;
    s = rank_tier(u, r, 4, {0,0}, p, excluded, o);
    CHECK(s.size() == 1 && s[0].layer == 0 && s[0].out_layer() == 1);
    s = rank_tier(u, r, 4, {0,1}, p, excluded, o);
    CHECK(s.size() == 1 && s[0].layer == s[0].out_layer());
    excluded[0] = 1;
    s = rank_tier(u, r, 4, {0,0}, p, excluded, o);
    CHECK(s[0].layer == 1);
    o.once = true;
    u.assign(8, 0); u[0] = 0.95f;
    excluded[0] = 0;
    s = rank_tier(u, r, 4, {0,0}, p, excluded, o);
    CHECK(s.size() == 1 && s[0].layer == 0 && s[0].gain == 0.95f);
    o.once = false;
    CHECK(rank_tier(u, r, 4, {0,0}, p, excluded, o).empty());
    o.once = true; u[0] = 0.89f;
    CHECK(rank_tier(u, r, 4, {0,0}, p, excluded, o).empty());
    u.assign(8, 3); r.assign(8, 0); r[0] = r[1] = -1; u[4] = u[5] = 0;
    s = rank_tier(u, r, 4, {0,0}, p, excluded, o);
    CHECK(s.size() == 2 && s[0].in == 0 && s[0].out == 1 && s[0].out_layer() == 1);
    const int ne = 512, nl = 48, n = ne * nl;
    u.assign(n, 0); r.assign(n, -1); p.resize(n); excluded.assign(n, 0);
    std::vector<int> groups(nl);
    for (int l = 0; l < nl; ++l) groups[l] = (l / 24) * 2 + l % 2;
    for (int i = 0; i < n; ++i) { p[i] = i; if (i % 2) r[i] = i / 2; else u[i] = 1; }
    s = rank_tier(u, r, ne, groups, p, excluded, o);
    CHECK(s.size() == 16);
    std::mt19937 gen(932);
    for (int round = 0; round < 300; ++round) {
        for (int i = 0; i < n; ++i) u[i] = (gen() % 130) * 0.1f;
        s = rank_tier(u, r, ne, groups, p, excluded, o);
        CHECK(s.size() <= o.cap);
        std::set<size_t> ins, outs;
        for (const auto& x : s) {
            const size_t in = (size_t) x.layer * ne + x.in, out = (size_t) x.out_layer() * ne + x.out;
            CHECK(r[in] < 0 && r[out] >= 0);
            CHECK(groups[x.layer] == groups[x.out_layer()]);
            CHECK(ins.insert(in).second && outs.insert(out).second);
            CHECK(x.gain >= o.once_bar);
        }
    }
    AdaptGap gap;
    gap.enabled = true;
    CHECK(gap.launch(0));
    CHECK(!gap.launch(0));
    std::atomic<bool> copying{false}, landed{false};
    std::thread copier([&] { gap.copy_begin(0); copying = true; while (!landed) std::this_thread::yield(); gap.copy_end(0); });
    CHECK(!copying);
    gap.idle(0);
    while (!copying) std::this_thread::yield();
    CHECK(!gap.launch(0));
    CHECK(gap.launch(1));
    landed = true; copier.join();
    CHECK(gap.launch(0));
    gap.stopping = true;
    std::thread cancel([&] { gap.copy_begin(0); gap.copy_end(0); });
    cancel.join();
    std::printf("adapt_rank_test: %s (ranking, promotions, device/size groups, copy leases)\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
