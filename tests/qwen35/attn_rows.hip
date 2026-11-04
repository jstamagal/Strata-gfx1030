// Ordered split-K attention parity and timing; synthetic inputs, no model.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "q36_attn.cuh"
#include "q36_attn_rows.cuh"

using namespace q36attn;
#define CHECK(x) do { const auto e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); std::exit(2); } } while (0)
struct Buffer {
    void* p = nullptr;
    size_t bytes;
    explicit Buffer(size_t n) : bytes(n) { CHECK(hipMalloc(&p, n)); }
    ~Buffer() { CHECK(hipFree(p)); }
    float* f() const { return (float*) p; }
    int32_t* i() const { return (int32_t*) p; }
    void put(const void* src, size_t n) { CHECK(hipMemcpy(p, src, n, hipMemcpyHostToDevice)); }
    void clear() { CHECK(hipMemset(p, 0xa5, bytes)); }
};
struct Result {
    Buffer acc, m, l, out;
    Result(int nt, int H, int nc) : acc((size_t) nt * H * nc * 256 * 4),
        m((size_t) nt * H * nc * 4), l(m.bytes), out((size_t) (nt + 4) * H * 256 * 4) {}
    void clear() { acc.clear(); m.clear(); l.clear(); out.clear(); }
};
bool equal(const Buffer& a, const Buffer& b, const char* what) {
    std::vector<uint32_t> x(a.bytes / 4), y(x.size());
    CHECK(hipMemcpy(x.data(), a.p, a.bytes, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(y.data(), b.p, b.bytes, hipMemcpyDeviceToHost));
    if (!std::memcmp(x.data(), y.data(), a.bytes)) return true;
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) {
        std::fprintf(stderr, "%s mismatch at %zu: %08x != %08x\n", what, i, x[i], y[i]); break;
    }
    return false;
}
void fill_cache(Buffer& dst, KvType type, std::mt19937& rng) {
    const size_t rows = dst.bytes / kv_row_bytes(type, 256);
    std::vector<uint8_t> data(dst.bytes);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t r = 0; r < rows; ++r) {
        auto* row = data.data() + r * kv_row_bytes(type, 256);
        if (type == KvType::Int8) {
            const float scale = .001f + .02f * std::abs(dist(rng));
            std::memcpy(row, &scale, 4);
            for (int d = 0; d < 256; ++d) ((int8_t*) (row + 4))[d] = (int) (rng() % 255) - 127;
        } else if (type == KvType::Int8G64) {
            for (int g = 0; g < 4; ++g) ((__half*) row)[g] = __float2half(.001f + .02f * std::abs(dist(rng)));
            for (int d = 0; d < 256; ++d) ((int8_t*) (row + 8))[d] = (int) (rng() % 255) - 127;
        } else for (int d = 0; d < 256; ++d) {
            const float v = dist(rng);
            if (type == KvType::F16) ((__half*) row)[d] = __float2half(v);
            else ((float*) row)[d] = v;
        }
    }
    dst.put(data.data(), data.size());
}
__global__ void evict(uint32_t* buf, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = buf[i] * 1664525u + 1013904223u;
}
template <int Phase>
__global__ void profile_rows(const float* q, const float* k, const float* v, const int32_t* pos,
                              int t0, int H, int HKV, int cap, int nc, float* acc, float* m, float* l, int nt) {
    __shared__ float sc[8][kChunkCells], sh[kThreads];
    if (*pos + t0 + nt <= 1024)
        attn_rows<4, 8, Phase>(q, k, v, pos, t0, H, HKV, cap, .0625f, nc, acc, m, l, nt, sc, sh);
    else
        attn_rows<8, 16, Phase>(q, k, v, pos, t0, H, HKV, cap, .0625f, nc, acc, m, l, nt, sc, sh);
}
struct Fixture {
    static constexpr int cap = 65792, nc = cap / 256, max_nt = 8;
    int H, HKV;
    KvType type;
    Buffer k, v, q, gate, pos;
    Result ref, old, got;
    hipStream_t st;
    RowSel selection;
    KvStream stream;
    bool stream_on = false, rows_mode = false;
    int head_grid = nc, cells_now = 1;
    Fixture(KvType ty, int heads = 16, int kvheads = 1) : H(heads), HKV(kvheads), type(ty),
        k((size_t) cap * HKV * kv_row_bytes(ty, 256)), v(k.bytes),
        q((size_t) (max_nt + 4) * H * 256 * 4), gate(q.bytes), pos(4),
        ref(max_nt, H, nc), old(max_nt, H, nc), got(max_nt, H, nc) {
        CHECK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
        selection.cache_type = type;
        std::mt19937 rng(1271 + H + (int) type);
        fill_cache(k, type, rng); fill_cache(v, type, rng);
        std::vector<float> h(q.bytes / 4);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto& x : h) x = dist(rng);
        q.put(h.data(), q.bytes);
        for (auto& x : h) x = dist(rng);
        gate.put(h.data(), gate.bytes);
    }
    ~Fixture() { CHECK(hipStreamDestroy(st)); }
    void position(int cells) { cells_now = cells; int p = cells - 1; pos.put(&p, 4); }
    void launch(int mode, Result& r, int nt, int t0 = 0, bool merge = true, int slots = 64) {
        const RowSel rs = selection;
        if (mode >= 2 && !attn_rows_eligible(256, H, HKV, rs, stream_on, rows_mode)) {
            launch(1, r, nt, t0, merge, slots); return;
        }
        if (mode <= 1) {
            auto original = [&]<bool S>() {
                if (mode == 0)
                    k_attn_part<S><<<dim3(head_grid, nt * H), 256, 0, st>>>(q.f(), k.f(), v.f(), pos.i(), t0,
                        H, HKV, cap, .0625f, nc, r.acc.f(), r.m.f(), r.l.f(), rs, stream);
                else if (H / HKV == 16)
                    k_attn_part_gqa<16, S><<<dim3(std::min(slots, 32), nt * HKV), 256, 0, st>>>(q.f(), k.f(), v.f(), pos.i(), t0,
                        H, HKV, cap, .0625f, nc, r.acc.f(), r.m.f(), r.l.f(), rs, stream);
                else
                    k_attn_part_gqa<8, S><<<dim3(std::min(slots, 32), nt * HKV), 256, 0, st>>>(q.f(), k.f(), v.f(), pos.i(), t0,
                        H, HKV, cap, .0625f, nc, r.acc.f(), r.m.f(), r.l.f(), rs, stream);
            };
            if (stream_on) original.template operator()<true>();
            else original.template operator()<false>();
        } else if (mode == 2) {
            k_attn_part_rows<<<dim3(slots, nt * H / 4), 256, 0, st>>>(q.f(), k.f(), v.f(), pos.i(), t0,
                H, HKV, cap, .0625f, nc, r.acc.f(), r.m.f(), r.l.f(), nt);
        } else {
            auto call = [&]<int Phase>() {
                profile_rows<Phase><<<dim3(slots, nt * H / 4), 256, 0, st>>>(
                    q.f(), k.f(), v.f(), pos.i(), t0, H, HKV, cap, nc, r.acc.f(), r.m.f(), r.l.f(), nt);
            };
            if (mode == 3) call.template operator()<1>();
            else call.template operator()<2>();
        }
        if (merge) k_attn_merge<<<nt * H, 256, 0, st>>>(r.acc.f(), r.m.f(), r.l.f(), gate.f(), pos.i(), t0, H, nc, r.out.f(), rs);
        CHECK(hipGetLastError());
    }
    bool same() {
        bool ok = true;
        for (Result* r : {&old, &got})
            ok = equal(ref.acc, r->acc, "pacc") && equal(ref.m, r->m, "pm") &&
                 equal(ref.l, r->l, "pl") && equal(ref.out, r->out, "merged") && ok;
        return ok;
    }
    bool compare(int nt, int cells, int t0 = 0, bool graph = false, int slots = 64, bool cold_ref = false) {
        ref.clear(); old.clear(); got.clear(); CHECK(hipDeviceSynchronize()); position(cells);
        auto run = [&] {
            const bool saved = stream_on;
            if (cold_ref) stream_on = false;
            launch(0, ref, nt, t0);
            stream_on = saved;
            launch(1, old, nt, t0); launch(2, got, nt, t0, true, slots);
        };
        bool ok = true;
        if (graph) {
            CHECK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal)); run();
            hipGraph_t g; hipGraphExec_t gx;
            CHECK(hipStreamEndCapture(st, &g)); CHECK(hipGraphInstantiate(&gx, g, nullptr, nullptr, 0));
            // Position is data, not a capture-time constant. Replay across a chunk boundary.
            for (int p : {cells, 255, 257, 1023, 1024, 1025, 4096, cells}) {
                position(p); CHECK(hipGraphLaunch(gx, st)); CHECK(hipStreamSynchronize(st));
                if (!same()) { std::fprintf(stderr, "graph replay position=%d\n", p); ok = false; break; }
            }
            CHECK(hipStreamSynchronize(st)); CHECK(hipGraphExecDestroy(gx)); CHECK(hipGraphDestroy(g));
        } else { run(); CHECK(hipStreamSynchronize(st)); }
        ok = same() && ok;
        if (!ok) std::fprintf(stderr, "FAIL type=%d H=%d HKV=%d nt=%d cells=%d t0=%d graph=%d\n", (int) type, H, HKV, nt, cells, t0, graph);
        return ok;
    }
    int fallbacks() {
        int cases = 0;
        Buffer k2(k.bytes), v2(v.bytes), kt(2 * sizeof(float*)), vt(2 * sizeof(float*));
        Buffer cells((max_nt + 4) * 4), slots(cells.bytes);
        std::mt19937 rng(9127);
        fill_cache(k2, type, rng); fill_cache(v2, type, rng);
        float* keys[] = {k.f(), k2.f()}; float* vals[] = {v.f(), v2.f()};
        kt.put(keys, sizeof(keys)); vt.put(vals, sizeof(vals));
        int32_t cp[max_nt + 4], sp[max_nt + 4];
        const int positions[] = {0, 254, 255, 256, 999, 4095, 29999, 65535};
        for (int i = 0; i < max_nt + 4; ++i) { cp[i] = positions[i % 8]; sp[i] = i % 2; }
        cells.put(cp, sizeof(cp)); slots.put(sp, sizeof(sp));
        selection.cell = cells.i(); selection.slot = slots.i();
        selection.kc = (float* const*) kt.p; selection.vc = (float* const*) vt.p;
        rows_mode = true;
        if (attn_rows_eligible(256, H, HKV, selection, false, true)) return -1;
        for (int nt : {1, 4, 8}) {
            if (!compare(nt, 1, 2, true)) return -1; ++cases;
        }
        stream_on = true; stream.resident = cap; stream.append_tokens = 8;
        selection.kc_hot = selection.kc; selection.vc_hot = selection.vc;
        for (int nt : {1, 4, 8}) {
            if (!compare(nt, 1, 2, true, 64, true)) return -1; ++cases;
        }
        selection = {}; selection.cache_type = type; rows_mode = false;
        constexpr int resident = 512, prefetch = 256;
        const size_t rb = kv_row_bytes(type, 256);
        Buffer kh((size_t) HKV * resident * rb), vh(kh.bytes);
        Buffer kp((size_t) HKV * prefetch * rb), vp(kp.bytes);
        std::vector<uint8_t> ck(k.bytes), cv(v.bytes), hk(kh.bytes), hv(vh.bytes), pk(kp.bytes), pv(vp.bytes);
        CHECK(hipMemcpy(ck.data(), k.p, k.bytes, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(cv.data(), v.p, v.bytes, hipMemcpyDeviceToHost));
        stream = {kh.f(), vh.f(), kp.f(), vp.f(), resident, prefetch, 0};
        if (attn_rows_eligible(256, H, HKV, selection, true, false)) return -1;
        for (int nt : {1, 4, 8}) for (int count : {64, 1000, 4096, 30000, 65536}) {
            const int end = count + nt - 1, cold_end = std::max(0, end - resident);
            const int len = std::min(prefetch, cold_end);
            for (int h = 0; h < HKV; ++h) {
                for (int p = cold_end; p < end; ++p) {
                    const size_t src = ((size_t) h * cap + p) * rb;
                    const size_t dst = ((size_t) h * resident + p % resident) * rb;
                    std::memcpy(hk.data() + dst, ck.data() + src, rb);
                    std::memcpy(hv.data() + dst, cv.data() + src, rb);
                }
                for (int p = 0; p < len; ++p) {
                    const size_t src = ((size_t) h * cap + cold_end - len + p) * rb;
                    const size_t dst = ((size_t) h * prefetch + p) * rb;
                    std::memcpy(pk.data() + dst, ck.data() + src, rb);
                    std::memcpy(pv.data() + dst, cv.data() + src, rb);
                }
            }
            kh.put(hk.data(), hk.size()); vh.put(hv.data(), hv.size());
            kp.put(pk.data(), pk.size()); vp.put(pv.data(), pv.size());
            stream.append_tokens = nt;
            if (!compare(nt, count, 0, false, 64, true)) return -1; ++cases;
        }
        stream_on = false; stream = {};
        return cases;
    }
    float timing(int mode, int nt, bool cold, Buffer& trash, bool whole = false) {
        hipGraph_t graph; hipGraphExec_t gx;
        const int batch = cold ? 1 : 16, reps = cold ? 12 : 8;
        CHECK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
        constexpr int slots = 64;
        head_grid = cells_now + nt - 1 <= 1024 ? 4 : nc;
        for (int i = 0; i < batch; ++i) launch(mode, got, nt, 0, whole, slots);
        CHECK(hipStreamEndCapture(st, &graph)); CHECK(hipGraphInstantiate(&gx, graph, nullptr, nullptr, 0));
        head_grid = nc;
        hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
        CHECK(hipGraphLaunch(gx, st)); CHECK(hipStreamSynchronize(st));
        std::vector<float> times;
        for (int rep = 0; rep < reps; ++rep) {
            if (cold) evict<<<(trash.bytes / 4 + 255) / 256, 256, 0, st>>>((uint32_t*) trash.p, trash.bytes / 4);
            CHECK(hipEventRecord(a, st)); CHECK(hipGraphLaunch(gx, st)); CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
            float ms; CHECK(hipEventElapsedTime(&ms, a, b)); times.push_back(ms * 1000 / batch);
        }
        std::sort(times.begin(), times.end());
        CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(b)); CHECK(hipGraphExecDestroy(gx)); CHECK(hipGraphDestroy(graph));
        return times[times.size() / 2];
    }
};
int main(int argc, char** argv) {
    const bool bench = argc > 1 && !std::strcmp(argv[1], "--bench");
    CHECK(hipSetDevice(0)); hipDeviceProp_t props; CHECK(hipGetDeviceProperties(&props, 0));
    std::fprintf(stderr, "device 0: %s (%s), %d HIP multiprocessors, wave%d\n",
                 props.name, props.gcnArchName, props.multiProcessorCount, props.warpSize);
    if (props.warpSize != 32 || std::strncmp(props.gcnArchName, "gfx1030", 7) != 0) {
        std::fputs("SKIP: gfx1030 wave32 test\n", stderr); return 77;
    }
    if (argc > 2 || (argc == 2 && !bench && std::strcmp(argv[1], "--profile"))) {
        std::fputs("usage: q36_attn_rows [--bench|--profile]\n", stderr); return 2;
    }
    if (argc > 1 && !std::strcmp(argv[1], "--profile")) {
        Buffer trash(256ul << 20); Fixture f(KvType::Int8);
        for (int nt : {1, 4}) for (int cells : {64, 256, 1000, 4096, 30000, 65536}) {
            f.position(cells);
            // Ablations have different register allocation; their timings are not additive.
            std::printf("nt=%d ctx=%d full_us=%.3f scores_only_us=%.3f scores_softmax_us=%.3f\n", nt, cells,
                f.timing(2, nt, false, trash), f.timing(3, nt, false, trash), f.timing(4, nt, false, trash));
        }
    } else if (bench) {
        Buffer trash(256ul << 20); CHECK(hipMemset(trash.p, 1, trash.bytes));
        Fixture f(KvType::Int8);
        std::puts("nt,cells,old_warm_us,new_warm_us,head_short_warm_us,old_cold_us,new_cold_us,head_short_cold_us,old_total_warm_us,new_total_warm_us");
        for (int nt = 1; nt <= 8; ++nt) for (int cells : {1, 64, 255, 256, 257, 1000, 1024, 1025, 4096, 5000, 16384, 30000, 65536}) {
            f.position(cells);
            const float ow = f.timing(1, nt, false, trash), nw = f.timing(2, nt, false, trash);
            const float hw = cells + nt - 1 <= 1024 ? f.timing(0, nt, false, trash) : NAN;
            const float oc = f.timing(1, nt, true, trash), nc = f.timing(2, nt, true, trash);
            const float hc = cells + nt - 1 <= 1024 ? f.timing(0, nt, true, trash) : NAN;
            const float ot = f.timing(1, nt, false, trash, true), nwt = f.timing(2, nt, false, trash, true);
            std::printf("%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", nt, cells, ow, nw, hw, oc, nc, hc, ot, nwt); std::fflush(stdout);
        }
    } else {
        int cases = 0;
        for (KvType type : {KvType::Int8, KvType::F16, KvType::F32, KvType::Int8G64}) {
            Fixture f(type);
            for (int nt = 1; nt <= 8; ++nt) for (int cells : {1, 64, 255, 256, 257, 1000, 1024, 1025, 4096, 30000, 65536}) {
                if (!f.compare(nt, cells)) return 1; ++cases;
            }
            for (int cells : {252, 253, 254, 255, 256, 257, 1022, 1023, 1024, 4094}) {
                if (!f.compare(8, cells, 3, true)) return 1; ++cases;
            }
            const int fallback_cases = f.fallbacks();
            if (fallback_cases < 0) return 1;
            cases += fallback_cases;
            if (type == KvType::Int8) for (int slots : {1, 3, 7, 32, 64, 257}) {
                if (!f.compare(4, 30000, 0, true, slots)) return 1; ++cases;
            }
            std::vector<float> stress(f.q.bytes / 4);
            for (size_t i = 0; i < stress.size(); ++i) stress[i] = (int(i % 29) - 14) * 32.f;
            f.q.put(stress.data(), f.q.bytes);
            for (int nt : {1, 4, 8}) for (int cells : {1, 255, 257, 1000, 30000}) {
                if (!f.compare(nt, cells)) return 1; ++cases;
            }
            CHECK(hipMemset(f.k.p, 0, f.k.bytes)); CHECK(hipMemset(f.v.p, 0, f.v.bytes));
            std::fill(stress.begin(), stress.end(), -0.f); f.q.put(stress.data(), f.q.bytes);
            for (int cells : {1, 255, 257, 30000}) {
                if (!f.compare(8, cells, 0, true)) return 1; ++cases;
            }
            std::printf("type=%d: random, saturated softmax, signed zero, STREAM, RowSel passed\n", (int) type); std::fflush(stdout);
        }
        for (int heads : {8, 16, 32}) {
            Fixture f(KvType::Int8, heads, heads == 32 ? 2 : 1);
            for (int nt : {1, 4, 8}) for (int cells : {1, 255, 1000, 30000}) {
                if (!f.compare(nt, cells, 2, true)) return 1; ++cases;
            }
        }
        std::printf("PASS: %d cases, pacc/pm/pl/merged and untouched guards bitwise equal to both original kernels\n", cases);
    }
    CHECK(hipDeviceSynchronize());
}
