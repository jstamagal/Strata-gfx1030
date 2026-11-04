// tests/hip/mmvq_row_parity.cpp - STRATA_MMVQ_ROW's dense-projection kernel (one wavefront per row, no shared
// memory) against the default native_mmvq kernels, bit for bit: every GGUF type they share, the decode window's
// shapes and two awkward ones, 1 to 8 activation columns, every (rows a wavefront, wavefronts a block) layout, with
// and without STRATA_TSUM's reduction (the environment decides which; run it both ways).  Then STRATA_MMVQ_SET's
// launch of two or three matrices on one input against each matrix's own default call.  `--bench` then times the
// decode shapes: another of 1.2 GB of weight copies every call (so no cache holds them), microseconds a call and GB/s of weights read.
// GPU, synthetic, no model.
#include <hip/hip_runtime.h>
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

namespace {
struct Type { int id; const char* name; int block_elems, block_bytes, d_off, d_count; };
// d_off / d_count: where the block's FP16 scales sit (kept finite and small; every other byte is random)
const Type kTypes[] = {{14, "Q6_K", 256, 210, 208, 1}, {13, "Q5_K", 256, 176, 0, 2}, {12, "Q4_K", 256, 144, 0, 2},
                       {23, "IQ4_XS", 256, 136, 0, 1}, {8, "Q8_0", 32, 34, 0, 1},    {11, "Q3_K", 256, 110, 108, 1},
                       {2, "Q4_0", 32, 18, 0, 1},      {6, "Q5_0", 32, 22, 0, 1},    {20, "IQ4_NL", 32, 18, 0, 1},
                       {42, "Q2_0", 64, 18, 0, 1}};
const Type& type_of(int id) {
    for (const Type& t : kTypes) if (t.id == id) return t;
    std::exit(3);
}

uint16_t f16_bits(float f) {   // a finite value well inside FP16
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = u >> 31, exp = ((u >> 23) & 0xFF) - 127 + 15, man = (u >> 13) & 0x3FF;
    return (uint16_t) ((sign << 15) | (exp << 10) | man);
}

std::mt19937 rng(20261013);

std::vector<uint8_t> make_weights(const Type& t, int n_in, int n_out) {
    std::vector<uint8_t> b((size_t) n_out * (n_in / t.block_elems) * t.block_bytes);
    std::uniform_real_distribution<float> scale(0.002f, 0.2f);
    uint32_t* p32 = (uint32_t*) b.data();
    for (size_t i = 0; i < b.size() / 4; ++i) p32[i] = rng();
    for (size_t i = b.size() & ~(size_t) 3; i < b.size(); ++i) b[i] = (uint8_t) rng();
    for (size_t o = 0; o + t.block_bytes <= b.size(); o += t.block_bytes)
        for (int k = 0; k < t.d_count; ++k) {
            const uint16_t h = f16_bits((rng() & 1 ? -1.0f : 1.0f) * scale(rng));
            std::memcpy(&b[o + t.d_off + 2 * k], &h, 2);
        }
    return b;
}

void* to_device(const void* h, size_t bytes) {
    void* d = nullptr;
    CHECK(hipMalloc(&d, bytes));
    CHECK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice));
    return d;
}

struct Shape { int type, n_in, n_out; const char* what; };
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    using namespace strata::kernels;
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool tpx_trace = argc > 2 && std::string(argv[1]) == "--tpx-trace";
    const bool tpx_t1 = argc > 1 && (std::string(argv[1]) == "--tpx-t1" || tpx_trace);
    const bool tpx = argc > 1 && (std::string(argv[1]) == "--tpx" || tpx_t1);
    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    const int kR[] = {1, 2, 4}, kW[] = {1, 2, 4, 8};

    if (tpx) {
        native_mmvq_row_tune(0, 0);
        std::vector<Shape> cases = {{23, 3072, 1536, "TP IQ4 small"}, {23, 3072, 4096, "TP IQ4 dense"},
                               {12, 3072, 1536, "TP Q4 small"}, {12, 3072, 4096, "TP Q4 dense"},
                               {13, 3072, 4096, "TP Q5 dense"}, {8, 3072, 256, "TP Q8 256"},
                               {8, 3072, 512, "TP Q8 512"}, {8, 3072, 1024, "TP Q8 1024"},
                               {8, 3072, 1536, "TP Q8 1536"}, {8, 3072, 3072, "TP Q8 3072"},
                               {8, 3072, 4096, "TP Q8 dense"}, {8, 8192, 3072, "TP Q8 long K"},
                               {8, 3072, 62080, "TP head subset"}, {8, 3072, 124160, "TP full head"}};
        if (tpx_trace) {
            cases.clear();
            std::ifstream input(argv[2]);
            if (!input) { std::fprintf(stderr, "cannot read TPX dense trace: %s\n", argv[2]); return 2; }
            std::set<std::tuple<int, int, int>> seen;
            std::string line;
            auto field = [](const std::string& s, const char* key) -> int {
                const size_t at = s.find(key);
                return at == std::string::npos ? -1 : std::atoi(s.c_str() + at + std::strlen(key));
            };
            while (std::getline(input, line)) {
                if (line.find("TPX_DENSE rank=") == std::string::npos) continue;
                const int type = field(line, " type="), K = field(line, " K="), M = field(line, " M=");
                const bool supported = type == 8 || type == 23 || type == 12 || type == 13 || type == 14 || type == 11 ||
                                       type == 2 || type == 6 || type == 20 || type == 42;
                if (supported && K > 0 && M > 0 && seen.emplace(type, K, M).second)
                    cases.push_back({type, K, M, "TP trace"});
            }
            if (cases.empty()) { std::fprintf(stderr, "TPX dense trace contained no supported matrices\n"); return 2; }
        }
        bool pass = true;
        for (const auto& sh : cases) {
            const auto tune = [&](int rows) {
                if (sh.type == 8) native_mmvq_tpx_q8_tune(rows);
                else native_mmvq_tpx_k_tune(rows);
            };
            const auto hw = make_weights(type_of(sh.type), sh.n_in, sh.n_out);
            const int copies = std::max(2, std::min(16, (int) (((size_t) 256 << 20) / hw.size())));
            std::vector<void*> dw;
            for (int i = 0; i < copies; ++i) dw.push_back(to_device(hw.data(), hw.size()));
            std::vector<void*> packed_dev;
            if (sh.type == 8) {
                std::vector<uint8_t> packed(hw.size());
                native_q8_0_pack_host(hw.data(), packed.data(), sh.n_in, sh.n_out);
                for (void* w : dw) {
                    void* p = to_device(packed.data(), packed.size());
                    native_q8_0_packed_register(w, p, sh.n_in, sh.n_out);
                    packed_dev.push_back(p);
                }
            }
            std::vector<float> hx((size_t) 4 * sh.n_in);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (float& v : hx) v = nd(rng);
            void* dx = to_device(hx.data(), hx.size() * 4);
            void* dq = nullptr;
            CHECK(hipMalloc(&dq, native_q8_1_bytes(sh.n_in, 4)));
            native_quantize_q8_1((const float*) dx, dq, sh.n_in, 4, stream);
            float *y0 = nullptr, *y1 = nullptr;
            CHECK(hipMalloc((void**) &y0, (size_t) 4 * sh.n_out * 4));
            CHECK(hipMalloc((void**) &y1, (size_t) 4 * sh.n_out * 4));
            for (int T = 1; T <= (tpx_t1 ? 1 : 4); ++T) {
                tune(0);
                native_mmvq(sh.type, dw[0], dq, y0, sh.n_in, sh.n_out, T, stream);
                CHECK(hipStreamSynchronize(stream));
                const size_t bytes = (size_t) T * sh.n_out * 4;
                std::vector<uint8_t> ref(bytes), got(bytes);
                CHECK(hipMemcpy(ref.data(), y0, bytes, hipMemcpyDeviceToHost));
                auto time_us = [&](int rows) {
                    tune(rows);
                    hipGraph_t graph;
                    hipGraphExec_t exec;
                    CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
                    for (int i = 0; i < 32; ++i)
                        native_mmvq(sh.type, dw[(size_t) i % copies], dq, y1, sh.n_in, sh.n_out, T, stream);
                    CHECK(hipStreamEndCapture(stream, &graph));
                    CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                    hipEvent_t a, b;
                    CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
                    double best = 1e30;
                    for (int rep = 0; rep < 8; ++rep) {
                        CHECK(hipEventRecord(a, stream));
                        CHECK(hipGraphLaunch(exec, stream));
                        CHECK(hipEventRecord(b, stream));
                        CHECK(hipEventSynchronize(b));
                        float ms = 0;
                        CHECK(hipEventElapsedTime(&ms, a, b));
                        if (rep >= 2) best = std::min(best, (double) ms * 1000.0 / 32.0);
                    }
                    hipEventDestroy(a); hipEventDestroy(b);
                    hipGraphExecDestroy(exec); hipGraphDestroy(graph);
                    return best;
                };
                const double old_us = time_us(0);
                std::printf("%-15s T=%d %.1f MB old %.1f us %.0f GB/s", sh.what, T, hw.size() / 1e6,
                            old_us, hw.size() / old_us / 1e3);
                for (int rows : {1, 2, 4}) {
                    tune(rows);
                    native_mmvq(sh.type, dw[0], dq, y1, sh.n_in, sh.n_out, T, stream);
                    CHECK(hipStreamSynchronize(stream));
                    CHECK(hipMemcpy(got.data(), y1, bytes, hipMemcpyDeviceToHost));
                    const bool same = got == ref;
                    pass &= same;
                    const double us = time_us(rows);
                    std::printf(" | r%d %.1f us %.0f GB/s %s", rows, us, hw.size() / us / 1e3,
                                same ? "bits=OK" : "bits=FAIL");
                }
                std::printf("\n");
            }
            for (void* w : dw) if (sh.type == 8) native_q8_0_packed_unregister(w);
            for (void* w : dw) hipFree(w);
            for (void* p : packed_dev) hipFree(p);
            hipFree(dx); hipFree(dq); hipFree(y0); hipFree(y1);
        }
        native_mmvq_tpx_q8_tune(-1);
        native_mmvq_tpx_k_tune(-1);
        native_mmvq_row_tune(-1, -1);
        return pass ? 0 : 1;
    }

    if (!bench) {
        // every type on a decode shape and on two awkward ones (rows that do not fill the last block, a short row)
        std::vector<Shape> shapes;
        for (const Type& t : kTypes) {
            shapes.push_back({t.id, 2560, 1030, "2560 -> 1030"});
            shapes.push_back({t.id, 6144, 517, "6144 -> 517"});
            shapes.push_back({t.id, 768, 259, "768 -> 259"});
            shapes.push_back({t.id, 256, 67, "256 -> 67"});
        }
        int bad = 0, cases = 0;
        for (const Shape& sh : shapes) {
            const Type& t = type_of(sh.type);
            if (sh.n_in % t.block_elems != 0) continue;
            const std::vector<uint8_t> hw = make_weights(t, sh.n_in, sh.n_out);
            void* dw = to_device(hw.data(), hw.size());
            std::vector<float> hx((size_t) 8 * sh.n_in);
            std::normal_distribution<float> nd(0.0f, 1.5f);
            for (auto& v : hx) v = nd(rng);
            void* dx = to_device(hx.data(), hx.size() * 4);
            void* dq = nullptr;
            CHECK(hipMalloc(&dq, native_q8_1_bytes(sh.n_in, 8)));
            native_quantize_q8_1((const float*) dx, dq, sh.n_in, 8, stream);
            float *dref = nullptr, *dy = nullptr;
            const size_t ybytes = (size_t) 8 * sh.n_out * 4;
            CHECK(hipMalloc((void**) &dref, ybytes));
            CHECK(hipMalloc((void**) &dy, ybytes));
            std::vector<float> href((size_t) 8 * sh.n_out), hy((size_t) 8 * sh.n_out);
            int shape_bad = 0;
            for (int nc = 1; nc <= 8; ++nc) {
                const size_t nb = (size_t) nc * sh.n_out * 4;
                native_mmvq_row_tune(0, 0);   // the default kernels
                CHECK(hipMemsetAsync(dref, 0xff, ybytes, stream));
                native_mmvq(sh.type, dw, dq, dref, sh.n_in, sh.n_out, nc, stream);
                CHECK(hipStreamSynchronize(stream));
                CHECK(hipMemcpy(href.data(), dref, nb, hipMemcpyDeviceToHost));
                size_t nonfinite = 0, zeros = 0;
                for (size_t i = 0; i < (size_t) nc * sh.n_out; ++i) {
                    if (!(href[i] - href[i] == 0.0f)) ++nonfinite;
                    if (href[i] == 0.0f) ++zeros;
                }
                if (nonfinite != 0 || zeros > (size_t) nc * sh.n_out / 2) {
                    std::printf("  %s %s x%d: the reference itself is not a usable ruler (%zu non-finite, %zu zeros)\n",
                                t.name, sh.what, nc, nonfinite, zeros);
                    ++shape_bad;
                }
                for (int r : kR)
                    for (int w : kW) {
                        native_mmvq_row_tune(r, w);
                        CHECK(hipMemsetAsync(dy, 0xee, ybytes, stream));
                        native_mmvq(sh.type, dw, dq, dy, sh.n_in, sh.n_out, nc, stream);
                        CHECK(hipStreamSynchronize(stream));
                        CHECK(hipMemcpy(hy.data(), dy, ybytes, hipMemcpyDeviceToHost));
                        ++cases;
                        bool ok = std::memcmp(hy.data(), href.data(), nb) == 0;
                        // nothing past the nc columns may be written
                        const uint8_t* tail = (const uint8_t*) hy.data() + nb;
                        for (size_t i = 0; ok && i < ybytes - nb; ++i) ok = tail[i] == 0xee;
                        if (!ok) {
                            size_t diff = 0;
                            for (size_t i = 0; i < (size_t) nc * sh.n_out; ++i)
                                diff += std::memcmp(&hy[i], &href[i], 4) != 0;
                            if (shape_bad++ < 4)
                                std::printf("  MISMATCH %s %s x%d rows %d wavefronts %d: %zu of %zu values differ\n", t.name,
                                            sh.what, nc, r, w, diff, (size_t) nc * sh.n_out);
                        }
                    }
            }
            std::printf("%-7s %-13s 1-8 columns x 12 layouts: %s\n", t.name, sh.what, shape_bad ? "DIFFERENT" : "bitwise equal");
            bad += shape_bad;
            hipFree(dw); hipFree(dx); hipFree(dq); hipFree(dref); hipFree(dy);
        }
        native_mmvq_row_tune(-1, -1);
        // ---- STRATA_MMVQ_SET: two and three matrices on one input in one launch, each against its own default call
        {
            const int set_types[] = {14, 13, 12, 23, 2, 8};
            const int n_in = 2560, outs[3] = {1030, 517, 259};
            std::vector<float> hx((size_t) 8 * n_in);
            std::normal_distribution<float> nd(0.0f, 1.5f);
            for (auto& v : hx) v = nd(rng);
            void* dx = to_device(hx.data(), hx.size() * 4);
            void* dq = nullptr;
            CHECK(hipMalloc(&dq, native_q8_1_bytes(n_in, 8)));
            native_quantize_q8_1((const float*) dx, dq, n_in, 8, stream);
            int set_bad = 0, sets = 0, refused = 0;
            for (int i = 0; i < 6; ++i)
                for (int n_items = 2; n_items <= 3; ++n_items) {
                    void* dw[3];
                    float *dref[3], *dy[3];
                    NativeMmvqItem items[3];
                    for (int m = 0; m < n_items; ++m) {
                        const Type& t = type_of(set_types[(i + (n_items == 2 ? 3 * m : m)) % 6]);
                        const std::vector<uint8_t> hw = make_weights(t, n_in, outs[m]);
                        dw[m] = to_device(hw.data(), hw.size());
                        CHECK(hipMalloc((void**) &dref[m], (size_t) 8 * outs[m] * 4));
                        CHECK(hipMalloc((void**) &dy[m], (size_t) 8 * outs[m] * 4));
                        items[m] = {t.id, dw[m], dy[m], outs[m]};
                    }
                    for (int nc = 1; nc <= 8; ++nc) {
                        native_mmvq_row_tune(0, 0);
                        for (int m = 0; m < n_items; ++m) {
                            CHECK(hipMemsetAsync(dref[m], 0xee, (size_t) 8 * outs[m] * 4, stream));
                            CHECK(hipMemsetAsync(dy[m], 0xee, (size_t) 8 * outs[m] * 4, stream));
                            native_mmvq(items[m].type, dw[m], dq, dref[m], n_in, outs[m], nc, stream);
                        }
                        native_mmvq_set_tune(1);
                        const bool took = native_mmvq_set(items, n_items, dq, n_in, nc, stream);
                        native_mmvq_set_tune(-1);
                        CHECK(hipStreamSynchronize(stream));
                        ++sets;
                        if (!took) { ++refused; continue; }
                        for (int m = 0; m < n_items; ++m) {
                            std::vector<uint8_t> a((size_t) 8 * outs[m] * 4), b(a.size());
                            CHECK(hipMemcpy(a.data(), dref[m], a.size(), hipMemcpyDeviceToHost));
                            CHECK(hipMemcpy(b.data(), dy[m], b.size(), hipMemcpyDeviceToHost));
                            if (a != b && set_bad++ < 6)
                                std::printf("  SET MISMATCH %s (matrix %d of %d) x%d\n", type_of(items[m].type).name, m, n_items, nc);
                        }
                    }
                    for (int m = 0; m < n_items; ++m) { hipFree(dw[m]); hipFree(dref[m]); hipFree(dy[m]); }
                }
            native_mmvq_row_tune(-1, -1);
            std::printf("sets of 2 and 3 matrices (Q6_K Q5_K Q4_K IQ4_XS Q4_0 Q8_0 in every place), 1-8 columns: %d launches, "
                        "%d refused, %s\n", sets, refused, set_bad || refused ? "DIFFERENT" : "bitwise equal");
            bad += set_bad + refused;
            hipFree(dx); hipFree(dq);
        }
        std::printf("%s (%d layout runs, STRATA_TSUM %s)\n", bad ? "FAILED" : "PASSED", cases,
                    std::getenv("STRATA_TSUM") ? std::getenv("STRATA_TSUM") : "unset");
        return bad ? 1 : 0;
    }

    // ---- timing: the decode window's dense projections (the IQ3_S pack's types, then the Q4_0 pack's), a different weight copy every call
    const Shape all[] = {{14, 2560, 10240, "GDN qkv"},   {13, 2560, 10240, "GDN qkv"},  {14, 2560, 6144, "GDN gate"},
                         {12, 2560, 6144, "GDN gate"},   {23, 2560, 6144, "GDN gate"},  {8, 6144, 2560, "out"},
                         {14, 2560, 12288, "QSA q"},     {14, 2560, 512, "QSA k/v"},    {12, 2560, 640, "shexp gate/up"},
                         {23, 2560, 640, "shexp gate/up"}, {8, 640, 2560, "shexp down"}, {14, 2560, 248320, "head"},
                         {2, 2560, 10240, "GDN qkv"},    {2, 2560, 6144, "GDN gate"},   {2, 6144, 2560, "out"}};
    const int reps = argc > 3 ? std::atoi(argv[3]) : 300;
    std::vector<int> ncs;
    if (argc > 2) for (const char* p = argv[2]; *p; ++p) if (*p >= '1' && *p <= '8') ncs.push_back(*p - '0');
    if (ncs.empty()) ncs = {1, 2, 4, 6};
    const char* only = argc > 4 ? argv[4] : nullptr;   // a substring of "<type> <what>"
    for (const Shape& sh : all) {
        const Type& t = type_of(sh.type);
        if (only != nullptr && (std::string(t.name) + " " + sh.what).find(only) == std::string::npos) continue;
        const std::vector<uint8_t> hw = make_weights(t, sh.n_in, sh.n_out);
        // enough copies that a call never finds its weights in a cache (an RX 6000's 128 MB third-level cache holds six
        // copies of an out-projection): 1.2 GB of them, as a decode window reads between two uses of one matrix
        const int copies = (int) std::min<size_t>(256, std::max<size_t>(2, ((size_t) 1200 << 20) / hw.size()));
        std::vector<void*> dw;
        for (int c = 0; c < copies; ++c) dw.push_back(to_device(hw.data(), hw.size()));
        std::vector<float> hx((size_t) 8 * sh.n_in);
        std::normal_distribution<float> nd(0.0f, 1.5f);
        for (auto& v : hx) v = nd(rng);
        void* dx = to_device(hx.data(), hx.size() * 4);
        void* dq = nullptr;
        CHECK(hipMalloc(&dq, native_q8_1_bytes(sh.n_in, 8)));
        native_quantize_q8_1((const float*) dx, dq, sh.n_in, 8, stream);
        float* dy = nullptr;
        CHECK(hipMalloc((void**) &dy, (size_t) 8 * sh.n_out * 4));
        const int n = hw.size() > (size_t) 100 << 20 ? reps / 10 + 1 : reps;
        auto time_us = [&](int nc) {
            for (int i = 0; i < 20; ++i) native_mmvq(sh.type, dw[(size_t) i % copies], dq, dy, sh.n_in, sh.n_out, nc, stream);
            CHECK(hipStreamSynchronize(stream));
            double best = 1e30;
            for (int pass = 0; pass < 3; ++pass) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < n; ++i) native_mmvq(sh.type, dw[(size_t) i % copies], dq, dy, sh.n_in, sh.n_out, nc, stream);
                CHECK(hipStreamSynchronize(stream));
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
                if (us < best) best = us;
            }
            return best;
        };
        for (int nc : ncs) {
            native_mmvq_row_tune(0, 0);
            const double ref = time_us(nc);
            std::printf("%-6s %5d -> %6d %-14s x%d  default %7.1f us %5.0f GB/s |", t.name, sh.n_in, sh.n_out, sh.what, nc,
                        ref, hw.size() / ref / 1e3);
            double best = 1e30;
            int br = 0, bw = 0;
            for (int r : kR)
                for (int w : kW) {
                    if (r == 4 && nc > 4) continue;
                    native_mmvq_row_tune(r, w);
                    const double us = time_us(nc);
                    std::printf(" %dx%d %6.1f", r, w, us);
                    if (us < best) { best = us; br = r; bw = w; }
                }
            std::printf(" | best %dx%d %7.1f us %5.0f GB/s (%+.0f%%)\n", br, bw, best, hw.size() / best / 1e3,
                        100.0 * (ref / best - 1.0));
        }
        for (void* p : dw) hipFree(p);
        hipFree(dx); hipFree(dq); hipFree(dy);
    }
    native_mmvq_row_tune(-1, -1);
    return 0;
}
