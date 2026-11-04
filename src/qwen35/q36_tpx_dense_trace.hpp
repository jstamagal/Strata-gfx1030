#pragma once
#include "strata/kernels/native_mmvq.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

namespace q36tpx {
inline bool dense_trace_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_Q36_TPX_DENSE_TRACE"); return v && std::strcmp(v, "1") == 0; }();
    return on;
}
// Called while the T=1 graph is captured. One line per TP rank, named call
// site, quant format, shard shape and actual engine route. The byte count is
// the GGUF weight payload touched by one invocation, independent of T.
inline void dense_trace(const std::string& name, int type, int n_in, int n_out,
                        const char* route, int T, const void* weights = nullptr) {
    if (!dense_trace_on() || T != 1) return;
    int dev = -1;
    (void) cudaGetDevice(&dev);
    static std::mutex lock;
    static std::set<std::tuple<int, std::string, int, int, int, std::string>> seen;
    // Keep the layer number: a distinct matrix pointer can use a different
    // route if a packed copy was dropped to make room for experts.
    std::lock_guard<std::mutex> guard(lock);
    if (!seen.emplace(dev, name, type, n_in, n_out, route).second) return;
    const size_t bytes = strata::kernels::native_mmvq_weight_bytes(type, n_in, n_out);
    std::fprintf(stderr, "strata-q36 TPX_DENSE rank=%d name=%s type=%d K=%d M=%d route=%s bytes=%zu MB=%.3f weights=%p\n",
                 dev, name.c_str(), type, n_in, n_out, route, bytes, (double) bytes / 1e6, weights);
}
}
