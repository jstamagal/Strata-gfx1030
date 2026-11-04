// --kv-resident N (q36): every cell's K / V in mapped pinned host memory (the cold copy, written by every step), the
// last `resident` cells in a device ring (cell pos at ring row pos % resident), and per device a staging window of the
// `prefetch` cold cells just below the ring, refilled on device before each attention layer (the position is read on
// device, so graph replay sees the current window).  Rows are the cache type's rows (kv_row_bytes): every
// --kv-cache-type streams.  Included after KvType / RowSel.
struct KvStream {
    float* khot = nullptr;   // the current session's ring (rows mode: RowSel's tables per row)
    float* vhot = nullptr;
    float* kpref = nullptr;  // the staging window (the current session only; null in rows mode)
    float* vpref = nullptr;
    int resident = 0;        // ring cells (0: no streaming)
    int prefetch = 0;        // staging cells (0: none)
    int append_tokens = 0;   // the rows this step writes (T)
};

namespace q36kvstream {
// A rejected draft's rows wrap over the ring's oldest cells; those cells stay right in the host copy until reused.
inline constexpr int rollback_guard = 8;   // q36's largest verify window (Engine::kVerify)

/// The buffer holding cell pos of KV head kvh, and its row there.  `end`: one past the last cell this step writes.
__device__ __forceinline__ const float* at(const float* cold, const float* hot, const float* pref, int kvh, int pos, int ctx,
                                           const KvStream& s, int end, size_t& row) {
    const int cold_end = max(0, end - s.resident);
    if (pref && s.prefetch > 0) {
        const int len = min(s.prefetch, cold_end);
        if (pos >= cold_end - len && pos < cold_end) {
            row = (size_t) kvh * s.prefetch + (size_t) (pos - (cold_end - len));
            return pref;
        }
    }
    if (hot && pos >= cold_end + rollback_guard) {
        row = (size_t) kvh * s.resident + (size_t) (pos % s.resident);
        return hot;
    }
    row = (size_t) kvh * ctx + pos;
    return cold;
}

/// The cold cells adjacent to the ring ([max(0, end - resident) - len, max(0, end - resident)), len <= cap) into the
/// device's staging window [HKV][cap][row], in 4-byte words (every row size is a multiple of 4).
/// grid1(HKV * cap * row_words).
__global__ void prefetch(const float* kc, const float* vc, float* pk, float* pv, const int32_t* pos0, int append_tokens,
                         int ctx, int resident, int cap, int HKV, int row_words) {
    const int end = *pos0 + append_tokens;
    const int cold_end = max(0, end - resident);
    const int len = min(cap, cold_end);
    if (!len) return;
    const int start = cold_end - len;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t per_head = (size_t) cap * row_words;
    if (i >= (size_t) HKV * per_head) return;
    const int h = (int) (i / per_head);
    const int cell = (int) ((i % per_head) / row_words), w = (int) (i % row_words);
    if (cell >= len) return;
    const size_t src = ((size_t) h * ctx + start + cell) * row_words + w;
    reinterpret_cast<uint32_t*>(pk)[i] = reinterpret_cast<const uint32_t*>(kc)[src];
    reinterpret_cast<uint32_t*>(pv)[i] = reinterpret_cast<const uint32_t*>(vc)[src];
}
}  // namespace q36kvstream
