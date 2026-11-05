// src/spec/draft_policy_test.cpp - DraftPolicy: when does a lookup window beat the MTP's?
//
// Simulated rounds with the costs measured on the RTX 5070 (window-cost: ~10 ms more per token) check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap.
#include "strata/spec/draft_policy.hpp"

#include <algorithm>
#include <cstdio>

using strata::spec::DraftPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
double cost(int t) { return 19.0 + 10.5 * (t - 1); }   // ms per round, measured shape
}  // namespace

int main() {
    std::printf("draft_policy_test\n");
    {
        DraftPolicy p(3);
        check(p.set_fixed_costs(" 8.2, 9.3, 10.4 "), "fixed costs: a complete positive profile is accepted");
        bool invalid = true;
        for (const char* s : {"", "8,9", "8,9,10,11", "8,9,0", "8,9,-1", "8,9,nan", "8,9,inf",
                              "8,9,1e999", "8,9,1e-999", "8,9,10,", "8,9,10junk", "8,,10"})
            invalid &= !p.set_fixed_costs(s) && p.cost_ms(3) == 10.4;
        check(invalid, "fixed costs: invalid/partial profiles leave the policy unchanged");
        for (int i = 0; i < 100; ++i) p.observe(false, 3, 1, 0, 1000.0 + i);
        check(p.cost_ms(3) == 10.4 && p.cost_ms(1) == 8.2, "fixed costs: neither observed nor unseen widths learn time");
        p.set_robust(true);
        p.set_context(30000);
        p.observe_chain(2, 1, 2, 12, 9999.0);
        p.defer_cost(5, 3, 123456.0);
        p.apply_costs(6);
        check(p.cost_ms(3) == 10.4, "fixed costs: context buckets, chains and deferred costs stay fixed");
        DraftPolicy copy = p;
        copy.observe(false, 3, 0, 0, 1e10);
        check(copy.cost_ms(3) == p.cost_ms(3), "fixed costs: generation policy copies retain the profile");
    }
    {
        // The live agent trace's first split: the same outcomes at different T3 wall costs take different lookups.
        DraftPolicy fast(8), slow(8);
        for (int i = 0; i < 300; ++i) {
            fast.observe(false, 3, 1 + (i & 1), 0, 17.94);
            slow.observe(false, 3, 1 + (i & 1), 0, 20.23);
        }
        for (int a : {3, 2, 2}) {
            fast.observe(true, 8, a, 3, 24.92);
            slow.observe(true, 8, a, 3, 24.92);
        }
        check(!fast.choose(3, 7, 5).lookup && slow.choose(3, 7, 5).lookup,
              "bimodal: timing alone flips MTP T3 versus lookup T8");
        const char* profile = "14.3505,15.5044,17.9436,24.1993,28.9211,33.6429,38.3647,24.9153";
        check(fast.set_fixed_costs(profile) && slow.set_fixed_costs(profile), "bimodal: opt-in profile applied to both histories");
        bool same = true;
        for (int i = 0; i < 1000; ++i) {
            const auto f = fast.choose(3, 7, 5), s = slow.choose(3, 7, 5);
            same &= f.lookup == s.lookup && f.t == s.t;
            const int a = std::min(f.t - 1, i % 4);
            fast.observe(f.lookup, f.t, a, 5, 0.01 + (i % 5));
            slow.observe(s.lookup, s.t, a, 5, 40.0 + (i % 17) * 31.25);
        }
        check(same && fast.lookup_rate(5) == slow.lookup_rate(5),
              "bimodal: 1000 decisions agree despite adversarial timing jitter");
        const double q_before = fast.lookup_rate(5);
        fast.observe(true, 8, 0, 5, 1.0);
        slow.observe(true, 8, 0, 5, 99999.0);
        check(fast.lookup_rate(5) < q_before && fast.lookup_rate(5) == slow.lookup_rate(5),
              "fixed costs: acceptance is still learned, not frozen");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));   // MTP windows of 4: 3 tokens each
        const DraftPolicy::Pick k = p.choose(4, 0, 0);
        check(!k.lookup && k.t == 4, "no proposal: the MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));
        for (int t = 2; t <= 6; ++t) p.observe(false, t, 0, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 0, 4, cost(6));      // short matches, all rejected
        check(p.lookup_rate(4) < 0.15, "rejected short-match drafts: their rate falls below 0.15");
        check(!p.choose(4, 5, 4).lookup, "rejected short-match drafts: no longer taken");
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 5, 30, cost(6));     // long matches, all accepted
        check(p.lookup_rate(30) > 0.9, "accepted long-match drafts: their rate rises above 0.9");
        const DraftPolicy::Pick k = p.choose(4, 5, 30);
        check(k.lookup && k.t == 6, "accepted long matches: the full lookup window is taken");
        check(!p.choose(4, 5, 4).lookup, "buckets are separate: short matches still not taken");
        check(p.choose(4, 20, 30).t <= 6, "never beyond the window cap");
    }
    {
        DraftPolicy p(8);
        for (int i = 0; i < 50; ++i) p.observe(false, 3, 2, 0, cost(3));      // a very good MTP: 3 of 3 tokens
        for (int t = 2; t <= 8; ++t) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 4, 2, 8, cost(4));       // lookup at q ~ 0.67
        check(!p.choose(3, 7, 8).lookup, "a mediocre lookup does not replace a strong MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a near-perfect MTP, only size 4 seen
        const DraftPolicy::Pick k = p.choose(4, 5, 40);
        check(k.lookup && k.t == 6, "an unmeasured size is probed for a confident lookup");
        for (int i = 0; i < 3; ++i) p.observe(true, 6, 5, 40, 3.0 * cost(6));   // it turns out very expensive
        check(!p.choose(4, 5, 40).lookup, "after the probes, the measured cost decides");
    }
    {
        // #1252: a wide size priced high in one slow stretch is measured again after a while
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a strong MTP
        for (int t = 2; t <= 6; ++t) for (int i = 0; i < 4; ++i) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 6; ++i) p.observe(true, 6, 5, 40, 4.0 * cost(6));  // the wide size, in a slow stretch
        for (int i = 0; i < 30; ++i) p.observe(true, 4, 3, 40, cost(4));       // and the short window's rate is fine
        bool saw6 = false;
        for (int i = 0; i < 400 && !saw6; ++i) {
            const DraftPolicy::Pick k = p.choose(4, 5, 40);
            if (k.lookup && k.t == 6) { saw6 = true; p.observe(true, 6, 5, 40, cost(6)); }
            else p.observe(k.lookup, k.t, k.t - 1, 40, cost(k.t));
        }
        check(saw6, "a stale wide size is tried again within a few hundred rounds");
    }
    {
        // --lookup-chain: rows cheap (UMA-like) and the chain always accepted -> chained; rows dear or the chain
        // always rejected -> not
        auto cheap = [](int t) { return 40.0 + 2.0 * (t - 1); };
        DraftPolicy p(8);
        for (int t = 2; t <= 8; ++t)
            for (int i = 0; i < 5; ++i) p.observe(false, t, 1, 0, cheap(t));
        for (int i = 0; i < 30; ++i) p.observe(false, 4, 2, 0, cheap(4));
        for (int i = 0; i < 30; ++i) p.observe_chain(4, 3, 6, 12, cheap(7));    // every chained token accepted
        check(p.chain(4, 0.9, 3, 12) == 3, "chain: cheap rows, always accepted -> all 3 chained");
        DraftPolicy q(8);
        for (int t = 2; t <= 8; ++t)
            for (int i = 0; i < 5; ++i) q.observe(false, t, 1, 0, cost(t));
        for (int i = 0; i < 30; ++i) q.observe_chain(4, 3, 3, 4, cost(7));       // reached, never accepted
        check(q.chain(4, 0.9, 3, 4) == 0, "chain: dear rows, never accepted -> none");
        check(q.chain(4, 0.9, 0, 4) == 0 && q.chain(8, 1.0, 3, 40) == 0, "chain: nothing proposed / no room -> none");
    }
    {
        // The stale-size check must also run when a shorter chain already pays. Synthetic costs isolate the
        // selection policy: even a size that stays expensive is retried, but only once per stale interval.
        DraftPolicy p(6);
        for (int i = 0; i < 3; ++i) p.observe(false, 4, 3, 0, 40.0);
        for (int i = 0; i < 3; ++i) p.observe_chain(4, 1, 4, 40, 42.0);
        for (int i = 0; i < 3; ++i) p.observe_chain(4, 2, 5, 40, 1000.0);
        check(p.chain(4, 1.0, 2, 40) == 1, "chain: the profitable shorter chain initially wins");
        int probes = 0, last_probe = -1;
        bool spaced = true, confident = true;
        for (int i = 0; i < 1000; ++i) {
            const int k = p.chain(4, 1.0, 2, 40);
            if (k == 2) {
                ++probes;
                spaced &= last_probe < 0 ? i >= 300 : i - last_probe > 300;
                last_probe = i;
                confident &= p.chain(4, 0.5, 2, 40) < 2;
            }
            p.observe_chain(4, k, 3 + k, 40, k == 2 ? 1000.0 : 42.0);
        }
        check(probes > 0 && probes <= 3 && spaced,
              "chain: stale full size is re-probed despite a profitable shorter chain");
        check(probes > 0 && confident, "chain: stale re-probes retain the MTP confidence gate");
    }
    {
        DraftPolicy p(12);
        for (int i = 0; i < 20; ++i) {
            p.observe(false, 3, 1, 0, 15.0);
            p.observe(true, 12, 11, 32, 25.0);
        }
        const auto k = p.choose(3, 23, 32);
        check(k.lookup && k.t == 12, "extended lookup respects twelve-row cap");
        check(p.cost_ms(12) == 25.0, "extended size records its own measured cost");
    }
    {
        // Gated proposals: within one generation the policy copy learns from committed outcomes only. A size
        // whose cost is still guessed is probed until kProbes rounds of it have run, timed or not, not every round.
        DraftPolicy live(6);
        for (int i = 0; i < 50; ++i) live.observe(false, 4, 3, 0, cost(4));      // a strong MTP
        live.observe(true, 6, 5, 8, 3.0 * cost(6));                             // size 6 timed once, dear
        DraftPolicy frozen = live, copy = live;
        int frozen_probes = 0, copy_probes = 0;
        for (int i = 0; i < 20; ++i) {
            const DraftPolicy::Pick f = frozen.choose(4, 5, 8);
            frozen_probes += f.lookup && f.t == 6;
            const DraftPolicy::Pick c = copy.choose(4, 5, 8);
            copy_probes += c.lookup && c.t == 6;
            copy.observe_committed(c.lookup, c.lookup ? c.t : 4, c.lookup ? 5 : 3, 8);
        }
        check(frozen_probes == 20, "gated: a frozen copy probes the guessed size every round");
        check(copy_probes == 2, "gated: committed outcomes end the probes after kProbes rounds of the size");
        check(copy.cost_ms(6) == live.cost_ms(6), "gated: committed outcomes leave the measured costs alone");
    }
    {
        // Gated proposals: a round's time reaches the generation's copy only once the committed position has passed
        // every counter it depends on. Cheap costs from earlier text stop steering long lookups once the dear
        // repetitive-text times are applied; a held time is not a new probe.
        DraftPolicy live(6);
        for (int t = 2; t <= 6; ++t) for (int i = 0; i < 5; ++i) live.observe(false, t, 1, 0, cost(t) * 0.2);   // cheap stretch
        DraftPolicy copy = live;
        check(copy.choose(4, 5, 40).lookup && copy.choose(4, 5, 40).t == 6, "delayed: cheap inherited costs take the full lookup");
        const double before = copy.cost_ms(6);
        copy.observe_committed(true, 6, 5, 40);
        copy.defer_cost(110, 6, 40.0 * cost(6));
        copy.apply_costs(110);
        check(copy.cost_ms(6) == before, "delayed: a time is held while its counters can still be drawn");
        copy.apply_costs(111);
        check(copy.cost_ms(6) > before, "delayed: applied once the committed position passes its last counter");
        for (int i = 0; i < 40; ++i) {
            copy.observe_committed(true, 6, 5, 40);
            copy.defer_cost(120 + i, 6, 40.0 * cost(6));
            copy.apply_costs(121 + i);
        }
        check(copy.choose(4, 5, 40).t < 6, "delayed: dear measured times shorten the lookup window");
        DraftPolicy fresh(6);
        fresh.observe_committed(true, 6, 5, 40);
        fresh.defer_cost(5, 6, 30.0);
        fresh.apply_costs(6);
        fresh.observe_committed(true, 6, 5, 40);
        check(fresh.choose(4, 5, 40).lookup && fresh.choose(4, 5, 40).t == 6,
              "delayed: a held time does not count as another probe");
    }
    {
        // --lookup-max-t: only the allowed sizes are taken; a proposal between two sizes takes the smaller; a size
        // disabled (its graph found no VRAM) is never proposed again; sizes past 8 work like the others
        DraftPolicy p(16);
        p.set_sizes((1u << 1) | (1u << 2) | (1u << 3) | (1u << 5) | (1u << 8) | (1u << 16));
        for (int i = 0; i < 50; ++i) p.observe(false, 3, 2, 0, cost(3));
        for (int i = 0; i < 40; ++i) p.observe(true, 16, 15, 40, 19.0 + 3.0 * 15);   // long windows: cheap, all accepted
        for (int i = 0; i < 40; ++i) p.observe(true, 8, 7, 40, 19.0 + 3.0 * 7);
        for (int i = 0; i < 40; ++i) p.observe(true, 5, 4, 40, 19.0 + 3.0 * 4);
        const DraftPolicy::Pick a = p.choose(3, 15, 40);
        check(a.lookup && a.t == 16, "sizes: a full 16-token lookup window when it pays");
        const DraftPolicy::Pick b = p.choose(3, 10, 40);
        check(b.lookup && b.t == 8, "sizes: a 10-token proposal takes the 8-token size");
        bool ok = true;
        for (int k = 1; k <= 15; ++k) {
            const DraftPolicy::Pick c = p.choose(3, k, 40);
            ok &= !c.lookup || p.allowed(c.t);
        }
        check(ok, "sizes: never a size outside the mask");
        p.disable(16);
        const DraftPolicy::Pick d = p.choose(3, 15, 40);
        check(d.lookup && d.t == 8, "sizes: a disabled size falls back to the next one");
        check(p.fit(7) == 5 && p.fit(16) == 8 && p.fit(1) == 1, "sizes: fit() finds the largest allowed size");
        DraftPolicy q(16);
        q.set_sizes((1u << 1) | (1u << 2) | (1u << 3) | (1u << 8));
        bool cok = true;
        for (int k = 0; k <= 13; ++k) {
            const int n = q.chain(3, 1.0, k, 40);
            cok &= n == 0 || q.allowed(3 + n);
        }
        check(cok, "sizes: a chained window's total is an allowed size");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
