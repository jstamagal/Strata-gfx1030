// q36 int8 KV with host backing, a recent GPU ring, and a graph-safe cold-window prefetch.
// The host allocation contains every cell. The GPU ring contains the last `resident` cells.
// The prefetch window is per device and is reused from one attention layer to the next.
namespace q36kvstream {
__device__ __forceinline__ float load(const float* cold, const float* hot, const float* pref,
                                       int kvh, int pos, int d, int HD, int HKV, int ctx,
                                       int resident, int pref_cap, int end) {
    if (resident > 0) {
        const int cold_end = max(0, end - resident);
        if (pref && pref_cap > 0) {
            const int len = min(pref_cap, cold_end);
            if (pos >= cold_end - len && pos < cold_end) {
                const size_t row = (size_t) kvh * pref_cap + pos - (cold_end - len);
                return q36kv8::load(pref, row, d, (size_t) HKV * pref_cap, HD);
            }
        }
        if (hot && pos >= cold_end) {
            const size_t row = (size_t) kvh * resident + pos % resident;
            return q36kv8::load(hot, row, d, (size_t) HKV * resident, HD);
        }
    }
    return q36kv8::load(cold, (size_t) kvh * ctx + pos, d, (size_t) HKV * ctx, HD);
}

// One launch pulls the cold cells adjacent to the recent ring into the device's
// shared staging buffer. The position is read on device, so HIP graph replay sees
// the current window; all data stays on the calling card.
__global__ void prefetch(const float* kc, const float* vc, float* pk, float* pv,
                         const int32_t* pos0, int append_tokens, int ctx, int resident,
                         int cap, int HKV, int HD) {
    const int end = *pos0 + append_tokens;
    const int cold_end = max(0, end - resident);
    const int len = min(cap, cold_end);
    if (!len) return;
    const int start = cold_end - len;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t code_n = (size_t) HKV * cap * HD;
    if (i < code_n) {
        const int h = (int) (i / ((size_t) cap * HD));
        const int cell = (int) ((i / HD) % cap), d = (int) (i % HD);
        if (cell < len) {
            const size_t src = ((size_t) h * ctx + start + cell) * HD + d;
            q36kv8::codes(pk)[i] = q36kv8::codes(kc)[src];
            q36kv8::codes(pv)[i] = q36kv8::codes(vc)[src];
        }
    }
    const int groups = HD / q36kv8::group;
    const size_t scale_n = (size_t) HKV * cap * groups;
    if (i < scale_n) {
        const int h = (int) (i / ((size_t) cap * groups));
        const int cell = (int) ((i / groups) % cap), g = (int) (i % groups);
        if (cell < len) {
            const size_t src = ((size_t) h * ctx + start + cell) * groups + g;
            q36kv8::scales(pk, (size_t) HKV * cap, HD)[i] = q36kv8::scales(kc, (size_t) HKV * ctx, HD)[src];
            q36kv8::scales(pv, (size_t) HKV * cap, HD)[i] = q36kv8::scales(vc, (size_t) HKV * ctx, HD)[src];
        }
    }
}
} // namespace q36kvstream
