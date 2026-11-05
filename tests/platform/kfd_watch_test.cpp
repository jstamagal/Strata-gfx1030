// kfd_watch_test (#267 on AMD): the KFD SMI event parser, the eviction episodes (nesting, rescheduled restores, a lost
// restore), the evicted time inside a span, and the deadline rule that counts only time the GPU could run.  No GPU.
#include "strata/platform/kfd_watch.hpp"

#include <cstdio>

using strata::platform::KfdEpisodes;
using strata::platform::KfdEvent;
using strata::platform::kfd_past_limit;
using strata::platform::parse_kfd_event;

static int bad = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++bad; } \
    } while (0)

static KfdEvent ev(const char* s) {
    KfdEvent e;
    CHECK(parse_kfd_event(s, e));
    return e;
}

int main() {
    // 1. the kernel's lines (KFD_EVENT_FMT_QUEUE_EVICTION / _RESTORE, behind "%x ")
    {
        KfdEvent e;
        CHECK(parse_kfd_event("9 442385122507158 -1444524 d90b 0\n", e));
        CHECK(e.kind == KfdEvent::Evict && e.ns == 442385122507158ll && e.pid == 1444524 && e.node == 0xd90b && e.trigger == 0);
        CHECK(parse_kfd_event("9 5 -7 1a09 1", e) && e.trigger == 1 && e.node == 0x1a09);
        CHECK(parse_kfd_event("a 442384692793980 -1444524 d90b R\n", e));
        CHECK(e.kind == KfdEvent::Restore && e.rescheduled && e.node == 0xd90b);
        CHECK(parse_kfd_event("a 442384811485234 -1444524 300f 0", e) && e.kind == KfdEvent::Restore && !e.rescheduled);
        // other events and junk
        CHECK(!parse_kfd_event("b 442402605907094 -1444524 @7efd29901(2001) 1a09 2", e));   // UNMAP_FROM_GPU
        CHECK(!parse_kfd_event("1 -1444524 strata", e));                                    // VMFAULT
        CHECK(!parse_kfd_event("", e));
        CHECK(!parse_kfd_event("9 12 1444524 d90b 0", e));   // no '-' before the pid
        CHECK(!parse_kfd_event("a 12 -5 d90b X", e));
        CHECK(!parse_kfd_event(nullptr, e));
        CHECK(std::string(strata::platform::kfd_trigger_name(0)) == "SVM");
        CHECK(std::string(strata::platform::kfd_trigger_name(1)) == "USERPTR");
    }
    // 2. one episode over three nodes, with failed restores in between
    {
        KfdEpisodes p;
        CHECK(!p.on_event(ev("9 1000 -1 d90b 0")));
        CHECK(!p.on_event(ev("9 1010 -1 1a09 0")));
        CHECK(!p.on_event(ev("9 1020 -1 300f 0")));
        CHECK(p.open() && p.open_since() == 1000);
        for (int i = 0; i < 5; ++i) CHECK(!p.on_event(ev("a 2000 -1 d90b R")));
        CHECK(p.current().retries == 5);
        CHECK(!p.on_event(ev("a 3000 -1 d90b 0")));
        CHECK(!p.on_event(ev("a 3010 -1 1a09 0")));
        CHECK(p.open());
        CHECK(p.on_event(ev("a 3020 -1 300f 0")));
        CHECK(!p.open());
        CHECK(p.last().begin == 1000 && p.last().end == 3020 && p.last().retries == 5 && p.last().trigger == 0);
        CHECK(p.episodes() == 1 && p.retries() == 5 && p.total_ns(99999) == 2020 && p.by_trigger(0) == 1);
        // a restore with nothing evicted is ignored
        CHECK(!p.on_event(ev("a 4000 -1 d90b 0")));
        CHECK(!p.open() && p.episodes() == 1);
    }
    // 3. nested evictions (SVM, then USERPTR on the same nodes): the queues run only after both restores
    {
        KfdEpisodes p;
        p.on_event(ev("9 100 -1 d90b 0"));
        p.on_event(ev("9 100 -1 1a09 0"));
        p.on_event(ev("9 150 -1 d90b 1"));
        p.on_event(ev("9 150 -1 1a09 1"));
        CHECK(!p.on_event(ev("a 200 -1 d90b 0")));
        CHECK(!p.on_event(ev("a 200 -1 1a09 0")));
        CHECK(p.open());   // one eviction still holds each node
        CHECK(!p.on_event(ev("a 400 -1 d90b 0")));
        CHECK(p.on_event(ev("a 400 -1 1a09 0")));
        CHECK(!p.open() && p.last().end - p.last().begin == 300 && p.last().trigger == 0);
    }
    // 4. evicted time inside a span: closed episodes clipped to it, the open one up to its end
    {
        KfdEpisodes p;
        p.on_event(ev("9 1000 -1 1 0"));
        p.on_event(ev("a 2000 -1 1 0"));     // [1000, 2000]
        p.on_event(ev("9 5000 -1 1 2"));
        p.on_event(ev("a 5500 -1 1 0"));     // [5000, 5500]
        p.on_event(ev("9 8000 -1 1 0"));     // open from 8000
        CHECK(p.evicted_between(0, 10000) == 1000 + 500 + 2000);
        CHECK(p.evicted_between(1500, 5200) == 500 + 200);
        CHECK(p.evicted_between(2000, 5000) == 0);
        CHECK(p.evicted_between(9000, 9500) == 500);
        CHECK(p.evicted_between(5, 5) == 0);
        CHECK(p.by_trigger(2) == 1 && p.by_trigger(0) == 2);   // the open one counts
        CHECK(p.longest_ns(12000) == 4000);
        CHECK(p.total_ns(10000) == 1000 + 500 + 2000);
        // 5. a restore event that never came: closed from outside
        CHECK(p.force_close(9000));
        CHECK(!p.open() && p.last().end == 9000 && p.episodes() == 3);
        CHECK(!p.force_close(9500));
        // the node is free again: a new eviction opens a new episode
        p.on_event(ev("9 9600 -1 1 0"));
        CHECK(p.open() && p.open_since() == 9600);
    }
    // 6. more episodes than the ring keeps: the totals stay exact, the span sum sees the kept ones
    {
        KfdEpisodes p;
        char b[64];
        const int n = KfdEpisodes::kKeep + 10;
        for (int i = 0; i < n; ++i) {
            std::snprintf(b, sizeof b, "9 %d -1 1 0", i * 100);
            p.on_event(ev(b));
            std::snprintf(b, sizeof b, "a %d -1 1 0", i * 100 + 10);
            p.on_event(ev(b));
        }
        CHECK(p.episodes() == (uint64_t) n && p.total_ns(1ll << 40) == 10ll * n);
        CHECK(p.evicted_between((n - 5) * 100, n * 100) == 50);
    }
    // 7. the deadline rule
    {
        CHECK(!kfd_past_limit(19000, 0, 20000, 600000));           // within the limit
        CHECK(kfd_past_limit(20001, 0, 20000, 600000));            // past it, nothing evicted: as before
        CHECK(!kfd_past_limit(30000, 15000, 20000, 600000));       // 15 s of 30 evicted: 15 s of GPU time, wait on
        CHECK(kfd_past_limit(36000, 15000, 20000, 600000));        // 21 s of GPU time: past
        CHECK(kfd_past_limit(600001, 600000, 20000, 600000));      // the cap ends an eviction that never ends
        CHECK(!kfd_past_limit(600000, 599000, 20000, 0));          // no cap
    }
    std::printf(bad == 0 ? "kfd_watch_test: OK\n" : "kfd_watch_test: %d FAILED\n", bad);
    return bad == 0 ? 0 : 1;
}
