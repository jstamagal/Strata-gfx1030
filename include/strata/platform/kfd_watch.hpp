// include/strata/platform/kfd_watch.hpp - #267 on AMD: when the kernel has this process's GPU queues evicted.
//
// gfx10 has no XNACK, so KFD keeps GPU-mapped host memory valid by stopping the process instead: any change to a page
// of a registered range (compaction moving it, reclaim or swap taking it, a THP split or collapse, a free that returns
// it) evicts EVERY queue of the process on every GPU until a restore worker has re-validated the ranges.  The restore
// retries every few ms and keeps failing while pages in the ranges are still moving.  ROCm's libhsakmt registers
// hipHostRegister memory - and the pageable buffers ROCclr pins in place for copies over 64 KiB - as SVM ranges
// (HSA_USE_SVM, default on) whose deregistration is a no-op (HSA_SVM_HOST_UNREGISTER_DEBUG=1 makes it real), so those
// ranges stay GPU-mapped until the memory is unmapped.  A GPU that is evicted for 25-70 s looks exactly like a verify
// window whose flags nobody raised.
//
// STRATA_KFD_WATCH (Linux, default 1; 0 = off) reads KFD's own event stream for this process (AMDKFD_IOC_SMI_EVENTS:
// QUEUE_EVICTION / QUEUE_RESTORE, no root needed for the own process) on one thread, keeps the episodes, and
//   * prints one line per eviction episode of STRATA_KFD_WATCH_LOG_MS (default 250) or more, and while one lasts,
//   * answers how much of a stretch of time the queues spent evicted, so the #267 deadlines (the verify window's 20 s,
//     the serve watchdog's limit) count only time the GPU could run (STRATA_KFD_EVICT_WAIT, default 1; 0 = wall time,
//     as before), up to STRATA_KFD_EVICT_WAIT_MAX_S (default 600) of wall time in all.
// The episode bookkeeping and the event parser are header-only (tests/platform/kfd_watch_test.cpp); the thread is
// src/platform/kfd_watch.cpp.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::platform {

/// One line of a KFD SMI event fd: "<id hex> <payload>".  Only the queue events are parsed.
struct KfdEvent {
    enum Kind { Other = 0, Evict, Restore } kind = Other;
    int64_t ns = 0;            ///< the kernel's CLOCK_BOOTTIME stamp
    int pid = 0;
    uint32_t node = 0;         ///< KFD gpu_id
    int trigger = -1;          ///< Evict: KFD_QUEUE_EVICTION_TRIGGER_* (0 SVM, 1 USERPTR, 2 TTM, 3 SUSPEND, 4/5 CRIU)
    bool rescheduled = false;  ///< Restore: 'R' = the restore failed and was queued again (still evicted)
};

inline const char* kfd_trigger_name(int t) {
    switch (t) {
        case 0: return "SVM";
        case 1: return "USERPTR";
        case 2: return "TTM";
        case 3: return "SUSPEND";
        case 4: return "CRIU-checkpoint";
        case 5: return "CRIU-restore";
        default: return "?";
    }
}

/// "9 <ns> -<pid> <node hex> <trigger>" (eviction), "a <ns> -<pid> <node hex> <R|0>" (restore).  False for any other
/// event or a line that does not parse.
inline bool parse_kfd_event(const char* s, KfdEvent& ev) {
    ev = KfdEvent{};
    if (s == nullptr) return false;
    char* end = nullptr;
    const long id = std::strtol(s, &end, 16);
    if (end == s || (id != 9 && id != 10)) return false;
    s = end;
    const long long ns = std::strtoll(s, &end, 10);
    if (end == s) return false;
    s = end;
    while (*s == ' ') ++s;
    if (*s != '-') return false;
    ++s;
    const long pid = std::strtol(s, &end, 10);
    if (end == s) return false;
    s = end;
    const unsigned long node = std::strtoul(s, &end, 16);
    if (end == s) return false;
    s = end;
    while (*s == ' ') ++s;
    ev.ns = (int64_t) ns;
    ev.pid = (int) pid;
    ev.node = (uint32_t) node;
    if (id == 9) {
        const long t = std::strtol(s, &end, 10);
        if (end == s) return false;
        ev.kind = KfdEvent::Evict;
        ev.trigger = (int) t;
    } else {
        if (*s != 'R' && *s != '0') return false;
        ev.kind = KfdEvent::Restore;
        ev.rescheduled = *s == 'R';
    }
    return true;
}

/// The episodes of one process.  KFD evicts and restores all of a process's GPUs together; an episode runs from the
/// first node's eviction to the restore that leaves no node evicted.  Evictions nest (an SVM and a userptr eviction at
/// once: KFD counts them per node, qpd->evicted, and the queues run again only when every one is restored), so each
/// node keeps a count; a rescheduled restore ('R') restores nothing.  Times are ns on one clock (the caller converts).
class KfdEpisodes {
public:
    static constexpr int kNodes = 16;
    static constexpr int kKeep = 256;   ///< closed episodes kept for evicted_between (a ring; older ones are summed)

    struct Episode {
        int64_t begin = 0, end = 0;
        int trigger = -1;
        uint32_t retries = 0;
    };

    /// Feed one event; returns true when it closed an episode (last() is that one).
    bool on_event(const KfdEvent& e) {
        if (e.kind == KfdEvent::Evict) {
            const int i = slot(e.node);
            if (i < 0) return false;
            if (evicted_[i]++ == 0) ++n_evicted_;
            if (!open_) {
                open_ = true;
                cur_ = Episode{};
                cur_.begin = e.ns;
                cur_.trigger = e.trigger;
            }
            return false;
        }
        if (e.kind != KfdEvent::Restore || !open_) return false;
        if (e.rescheduled) { ++cur_.retries; return false; }
        const int i = slot(e.node);
        if (i < 0 || evicted_[i] == 0) return false;
        if (--evicted_[i] > 0) return false;
        --n_evicted_;
        if (n_evicted_ > 0) return false;
        close(e.ns);
        return true;
    }

    /// End an open episode from outside (a restore whose event was lost: the sysfs evicted_ms moved).
    bool force_close(int64_t ns) {
        if (!open_) return false;
        for (int& c : evicted_) c = 0;
        n_evicted_ = 0;
        close(ns > cur_.begin ? ns : cur_.begin);
        return true;
    }

    bool open() const { return open_; }
    int64_t open_since() const { return open_ ? cur_.begin : 0; }
    const Episode& current() const { return cur_; }
    const Episode& last() const { return last_; }
    uint64_t episodes() const { return episodes_ + (open_ ? 1 : 0); }
    uint64_t retries() const { return retries_ + (open_ ? cur_.retries : 0); }
    int64_t longest_ns(int64_t now) const {
        const int64_t o = open_ && now > cur_.begin ? now - cur_.begin : 0;
        return o > longest_ ? o : longest_;
    }
    uint64_t by_trigger(int t) const { return t >= 0 && t < 8 ? trig_[t] + (open_ && cur_.trigger == t ? 1 : 0) : 0; }
    /// all evicted time up to `now` (the open episode up to now included)
    int64_t total_ns(int64_t now) const { return total_ + (open_ && now > cur_.begin ? now - cur_.begin : 0); }

    /// Evicted time inside [a, b]: the open episode (up to b) and the kept closed ones.  Closed episodes older than the
    /// ring are not seen; at 256 kept that is far beyond any deadline's span.
    int64_t evicted_between(int64_t a, int64_t b) const {
        if (b <= a) return 0;
        int64_t s = 0;
        auto add = [&](int64_t x, int64_t y) {
            if (x < a) x = a;
            if (y > b) y = b;
            if (y > x) s += y - x;
        };
        const int n = kept_ < kKeep ? kept_ : kKeep;
        for (int k = 0; k < n; ++k) add(ring_[k].begin, ring_[k].end);
        if (open_) add(cur_.begin, b);
        return s;
    }

private:
    int slot(uint32_t node) {
        for (int i = 0; i < n_nodes_; ++i) if (nodes_[i] == node) return i;
        if (n_nodes_ == kNodes) return -1;
        nodes_[n_nodes_] = node;
        return n_nodes_++;
    }
    void close(int64_t ns) {
        cur_.end = ns;
        open_ = false;
        const int64_t d = cur_.end - cur_.begin;
        total_ += d;
        if (d > longest_) longest_ = d;
        retries_ += cur_.retries;
        if (cur_.trigger >= 0 && cur_.trigger < 8) ++trig_[cur_.trigger];
        ++episodes_;
        ring_[kept_ % kKeep] = cur_;
        ++kept_;
        last_ = cur_;
    }
    uint32_t nodes_[kNodes] = {};
    int evicted_[kNodes] = {};
    int n_nodes_ = 0, n_evicted_ = 0;
    bool open_ = false;
    Episode cur_, last_;
    Episode ring_[kKeep];
    int kept_ = 0;
    uint64_t episodes_ = 0, retries_ = 0, trig_[8] = {};
    int64_t total_ = 0, longest_ = 0;
};

/// The #267 deadline rule: a wait of `wall_ms` with `evicted_ms` of it evicted is past `limit_ms` when the time the GPU
/// could run is, or when the wall time passes `cap_ms` (an eviction that never ends still ends the engine).
inline bool kfd_past_limit(double wall_ms, double evicted_ms, double limit_ms, double cap_ms) {
    if (wall_ms <= limit_ms) return false;
    if (cap_ms > 0.0 && wall_ms > cap_ms) return true;
    return wall_ms - evicted_ms > limit_ms;
}

/// Start the watcher thread once (Linux, STRATA_KFD_WATCH not 0, /dev/kfd present); `log` gets its lines.  Also prints
/// the KFD sysfs evicted_ms this process has so far (the load's share).  No-op elsewhere.
void kfd_watch_start(std::FILE* log, const char* when);
/// The watcher is running.
bool kfd_watch_on();
/// The queues are evicted right now; `for_ms` (optional) gets how long.
bool kfd_evicted_now(double* for_ms = nullptr);
/// ms of [since, now] (std::chrono::steady_clock ms) the queues spent evicted (0 when the watcher is off).
double kfd_evicted_ms_between(double since_steady_ms, double now_steady_ms);
/// The #267 deadline with STRATA_KFD_EVICT_WAIT: true when a wait that began at `since` (steady ms) is past `limit_ms`
/// of time the GPU could run (or past the cap).  Without the watcher, or with STRATA_KFD_EVICT_WAIT=0: wall time.
bool kfd_deadline_passed(double since_steady_ms, double now_steady_ms, double limit_ms);
/// One line about the episodes so far (empty when the watcher is off): "KFD: 12 queue evictions, 3.4 s evicted ...".
std::string kfd_summary();

}  // namespace strata::platform
