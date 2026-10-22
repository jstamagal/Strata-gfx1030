// Diagnostic timestamps for the row-owned TP join. Cross-device GPU clocks do
// not share a documented epoch, so compare the two local wait durations.
#pragma once
namespace q36tp_skew {
constexpr unsigned samples_per_join = 128;
struct Sample {
    unsigned long long arrival, publish, peer, done;
};
__device__ inline unsigned long long stamp() {
#if defined(__HIPCC__)
    return (unsigned long long)wall_clock64();
#else
    return (unsigned long long)clock64();
#endif
}
} // namespace q36tp_skew
