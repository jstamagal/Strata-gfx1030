// src/platform/kfd_watch.cpp - see include/strata/platform/kfd_watch.hpp.
#include "strata/platform/kfd_watch.hpp"

#include "strata/core/progress.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/platform/thread_name.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#endif

namespace strata::platform {

namespace {

std::atomic<bool> g_on{false};
std::mutex g_mu;
KfdEpisodes g_ep;   // guarded by g_mu; times are steady-clock ns

int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' ? std::atoi(v) : dflt;
}

[[maybe_unused]] int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

[[maybe_unused]] std::string stage_now() {
    const strata::core::Progress& p = strata::core::progress();
    std::string s = p.where.load(std::memory_order_relaxed);
    const int64_t d = p.detail.load(std::memory_order_relaxed);
    if (d >= 0) s += " " + std::to_string((long long) d);
    if (!p.busy.load(std::memory_order_relaxed)) s += " (no request)";
    return s;
}

#if defined(__linux__)
// _IOWR('K', 0x1F, struct kfd_ioctl_smi_events_args { u32 gpuid; u32 anon_fd; }), as linux/kfd_ioctl.h has it
struct SmiArgs {
    uint32_t gpuid;
    uint32_t anon_fd;
};
constexpr unsigned long kIocSmiEvents = (3ul << 30) | ((unsigned long) sizeof(SmiArgs) << 16) | ((unsigned long) 'K' << 8) | 0x1Ful;
constexpr uint64_t kMaskQueue = (1ull << (9 - 1)) | (1ull << (10 - 1));   // QUEUE_EVICTION, QUEUE_RESTORE

int64_t boot_minus_steady_ns() {
    timespec b{}, m{};
    clock_gettime(CLOCK_BOOTTIME, &b);
    clock_gettime(CLOCK_MONOTONIC, &m);
    return ((int64_t) b.tv_sec - (int64_t) m.tv_sec) * 1000000000ll + ((int64_t) b.tv_nsec - (int64_t) m.tv_nsec);
}

std::vector<uint32_t> gpu_ids() {
    std::vector<uint32_t> out;
    const char* base = "/sys/class/kfd/kfd/topology/nodes";
    DIR* d = opendir(base);
    if (d == nullptr) return out;
    while (dirent* e = readdir(d)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        const std::string p = std::string(base) + "/" + e->d_name + "/gpu_id";
        if (std::FILE* f = std::fopen(p.c_str(), "r")) {
            unsigned long id = 0;
            if (std::fscanf(f, "%lu", &id) == 1 && id != 0) out.push_back((uint32_t) id);
            std::fclose(f);
        }
    }
    closedir(d);
    return out;
}

/// KFD's own count for this process and GPU (ms; updated when a restore completes), -1 when unreadable
long long sysfs_evicted_ms(uint32_t gpu) {
    char p[128];
    std::snprintf(p, sizeof p, "/sys/class/kfd/kfd/proc/%d/stats_%u/evicted_ms", (int) getpid(), (unsigned) gpu);
    long long v = -1;
    if (std::FILE* f = std::fopen(p, "r")) {
        if (std::fscanf(f, "%lld", &v) != 1) v = -1;
        std::fclose(f);
    }
    return v;
}

void watch_loop(std::FILE* log, std::vector<int> fds, uint32_t gpu0) {
    strata::platform::name_this_thread("strata-kfd");
    strata::aux_cpus::pin_current_thread();
    const int log_ms = env_int("STRATA_KFD_WATCH_LOG_MS", 250);
    const int64_t pid = (int64_t) getpid();
    std::vector<pollfd> pf;
    for (int fd : fds) pf.push_back(pollfd{fd, POLLIN, 0});
    int64_t next_note_ns = 0;       // the open episode's next "still evicted" line
    long long sys_at_open = -1;     // sysfs evicted_ms when the open episode began (a lost restore event moves it)
    int64_t sys_checked_ns = 0;
    static char buf[16384];   // the KFD event fifo is 8 KiB: one read takes all of it
    for (;;) {
        const int r = poll(pf.data(), (nfds_t) pf.size(), 250);
        const int64_t off = boot_minus_steady_ns();
        if (r > 0) {
            for (pollfd& p : pf) {
                if ((p.revents & POLLIN) == 0) continue;
                const ssize_t n = read(p.fd, buf, sizeof buf - 1);
                if (n <= 0) continue;
                buf[n] = '\0';
                for (char* line = buf; line != nullptr && *line != '\0';) {
                    char* nl = std::strchr(line, '\n');
                    if (nl != nullptr) *nl = '\0';
                    KfdEvent ev;
                    if (parse_kfd_event(line, ev) && ev.pid == pid) {
                        ev.ns -= off;   // CLOCK_BOOTTIME -> steady
                        std::lock_guard<std::mutex> g(g_mu);
                        const bool was_open = g_ep.open();
                        const bool closed = g_ep.on_event(ev);
                        if (!was_open && g_ep.open()) {
                            next_note_ns = g_ep.open_since() + 2000000000ll;
                            sys_at_open = sysfs_evicted_ms(gpu0);
                            sys_checked_ns = steady_ns();
                        }
                        if (closed) {
                            const KfdEpisodes::Episode& e = g_ep.last();
                            const double ms = (double) (e.end - e.begin) / 1e6;
                            if (ms >= (double) log_ms && log != nullptr) {
                                std::fprintf(log, "strata: KFD evicted this process's GPU queues for %.0f ms (trigger %s, %u "
                                                  "failed restores) during %s; %llu evictions, %.1f s evicted since the watch "
                                                  "began\n",
                                             ms, kfd_trigger_name(e.trigger), (unsigned) e.retries, stage_now().c_str(),
                                             (unsigned long long) g_ep.episodes(), (double) g_ep.total_ns(e.end) / 1e9);
                                std::fflush(log);
                            }
                        }
                    }
                    line = nl != nullptr ? nl + 1 : nullptr;
                }
            }
        }
        const int64_t now = steady_ns();
        std::lock_guard<std::mutex> g(g_mu);
        if (!g_ep.open()) continue;
        // a restore whose event was lost (the KFD event fifo is 8 KiB): KFD's own counter moved, the queues are back
        if (now - sys_checked_ns >= 1000000000ll) {
            sys_checked_ns = now;
            const long long s = sysfs_evicted_ms(gpu0);
            if (sys_at_open >= 0 && s > sys_at_open && now - g_ep.open_since() > 1000000000ll) {
                g_ep.force_close(now);
                if (log != nullptr) {
                    std::fprintf(log, "strata: KFD: the restore event of an eviction was lost; KFD's evicted_ms moved "
                                      "(%lld -> %lld ms), so the queues run again\n", sys_at_open, s);
                    std::fflush(log);
                }
                continue;
            }
        }
        if (now >= next_note_ns && log != nullptr) {
            const KfdEpisodes::Episode& c = g_ep.current();
            const double s = (double) (now - c.begin) / 1e9;
            std::fprintf(log, "strata: KFD: this process's GPU queues have been evicted for %.0f s (trigger %s, %u failed "
                              "restores) during %s - the kernel is moving, reclaiming or freeing pages of GPU-registered host "
                              "memory; nothing runs on the GPU until it restores them (#267)\n",
                         s, kfd_trigger_name(c.trigger), (unsigned) c.retries, stage_now().c_str());
            std::fflush(log);
            const int64_t step = now - c.begin;   // 2, 4, 8, 16 ... s
            next_note_ns = c.begin + 2 * (step > 2000000000ll ? step : 2000000000ll);
        }
    }
}
#endif

}  // namespace

void kfd_watch_start(std::FILE* log, const char* when) {
#if defined(__linux__)
    static std::once_flag once;
    std::call_once(once, [log, when] {
        if (env_int("STRATA_KFD_WATCH", 1) == 0) return;
        const std::vector<uint32_t> ids = gpu_ids();
        if (ids.empty()) return;   // no AMD GPU (or no KFD): nothing to watch
        const int kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
        if (kfd < 0) return;
        std::vector<int> fds;
        std::string sys;
        for (uint32_t id : ids) {
            SmiArgs a{id, 0};
            if (ioctl(kfd, kIocSmiEvents, &a) != 0) continue;
            const int fd = (int) a.anon_fd;
            if (write(fd, &kMaskQueue, sizeof kMaskQueue) != (ssize_t) sizeof kMaskQueue) { close(fd); continue; }
            fds.push_back(fd);
            const long long ms = sysfs_evicted_ms(id);
            if (ms >= 0) sys += (sys.empty() ? "" : " / ") + std::to_string(ms);
        }
        if (fds.empty()) { close(kfd); return; }   // the /dev/kfd fd stays open with the watcher: the event fds need it
        if (log != nullptr) {
            std::fprintf(log, "strata: KFD %s: this process's GPU queues were evicted %s ms so far (KFD stats per GPU); "
                              "watching for evictions on %zu GPU node(s) (STRATA_KFD_WATCH=0: off; #267 deadlines count "
                              "only time the GPU could run: STRATA_KFD_EVICT_WAIT=%d)\n",
                         when != nullptr ? when : "", sys.empty() ? "?" : sys.c_str(), fds.size(),
                         env_int("STRATA_KFD_EVICT_WAIT", 1) != 0 ? 1 : 0);
            std::fflush(log);
        }
        g_on.store(true);
        std::thread(watch_loop, log, fds, ids.front()).detach();
    });
#else
    (void) log;
    (void) when;
#endif
}

bool kfd_watch_on() { return g_on.load(); }

bool kfd_evicted_now(double* for_ms) {
    if (!g_on.load()) return false;
    const int64_t now = steady_ns();
    std::lock_guard<std::mutex> g(g_mu);
    if (!g_ep.open()) return false;
    if (for_ms != nullptr) *for_ms = (double) (now - g_ep.open_since()) / 1e6;
    return true;
}

double kfd_evicted_ms_between(double since_steady_ms, double now_steady_ms) {
    if (!g_on.load()) return 0.0;
    std::lock_guard<std::mutex> g(g_mu);
    return (double) g_ep.evicted_between((int64_t) (since_steady_ms * 1e6), (int64_t) (now_steady_ms * 1e6)) / 1e6;
}

bool kfd_deadline_passed(double since_steady_ms, double now_steady_ms, double limit_ms) {
    const double wall = now_steady_ms - since_steady_ms;
    if (wall <= limit_ms) return false;
    static const bool wait = env_int("STRATA_KFD_EVICT_WAIT", 1) != 0;
    if (!wait || !g_on.load()) return true;
    static const double cap_ms = 1000.0 * (double) env_int("STRATA_KFD_EVICT_WAIT_MAX_S", 600);
    // the waits poll this in tight loops once past the wall-time limit: one answer per ms per wait, so the watcher
    // thread is not kept off its lock
    thread_local double c_since = -1.0, c_limit = -1.0, c_at = -1e300;
    thread_local bool c_past = false;
    if (since_steady_ms == c_since && limit_ms == c_limit && now_steady_ms - c_at < 1.0) return c_past;
    const double ev = kfd_evicted_ms_between(since_steady_ms, now_steady_ms);
    const bool past = kfd_past_limit(wall, ev, limit_ms, cap_ms);
    c_since = since_steady_ms;
    c_limit = limit_ms;
    c_at = now_steady_ms;
    c_past = past;
    if (!past && ev > 0.0) {   // the wall-time deadline alone would have ended the engine here: say so, now and then
        static std::atomic<int64_t> said_ns{0};
        const int64_t now = steady_ns();
        int64_t was = said_ns.load();
        if (now - was > 5000000000ll && said_ns.compare_exchange_strong(was, now)) {
            std::fprintf(stderr, "strata: KFD: a %.0f s #267 deadline has passed in wall time (%.1f s), but the kernel kept "
                                 "the GPU queues evicted for %.1f s of it: waiting on (STRATA_KFD_EVICT_WAIT; at most %.0f s)\n",
                         limit_ms / 1000.0, wall / 1000.0, ev / 1000.0, cap_ms / 1000.0);
            std::fflush(stderr);
        }
    }
    return past;
}

std::string kfd_summary() {
    if (!g_on.load()) return std::string();
    const int64_t now = steady_ns();
    std::lock_guard<std::mutex> g(g_mu);
    char b[320];
    int n = std::snprintf(b, sizeof b, "KFD: %llu queue evictions (SVM %llu, USERPTR %llu, TTM %llu), %.1f s evicted in all, "
                                       "longest %.0f ms, %llu failed restores",
                          (unsigned long long) g_ep.episodes(), (unsigned long long) g_ep.by_trigger(0),
                          (unsigned long long) g_ep.by_trigger(1), (unsigned long long) g_ep.by_trigger(2),
                          (double) g_ep.total_ns(now) / 1e9, (double) g_ep.longest_ns(now) / 1e6,
                          (unsigned long long) g_ep.retries());
    if (g_ep.open() && n > 0 && n < (int) sizeof b)
        std::snprintf(b + n, sizeof b - (size_t) n, "; EVICTED NOW for %.1f s (trigger %s)",
                      (double) (now - g_ep.open_since()) / 1e9, kfd_trigger_name(g_ep.current().trigger));
    return b;
}

}  // namespace strata::platform
