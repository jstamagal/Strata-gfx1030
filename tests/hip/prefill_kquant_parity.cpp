// tests/hip/prefill_kquant_parity.cpp - llama.cpp's MMQ kernels for Q4_K / Q5_K / Q6_K on HIP, through Strata's
// wrapper (mmq::quantize + mmq::Context::run), with several experts of uneven row counts (zero rows, counts that
// are not a multiple of the tile).  Shapes: the experts' gate+up [1280 x 2560]; a dense projection [2560 x 2560]
// (one matrix, Gemm::native's STRATA_DENSE_MMQ use); and [2560 x 768] in place of down - a K-quant block is 256
// values, so the real down [2560 x 640] cannot be a K-quant (the GGUFs store it as Q5_1 / Q5_0 / Q8_0), and 768 is
// the nearest width that can.
//
// The reference is the kernel's own arithmetic in FP64: the weights' codes and scales are decoded here from the GGUF
// blocks (and checked against ggml's dequantizer), the activations are the q8_1 blocks the device quantizer wrote
// (read back; checked against the floats they came from).  So only the FP32 accumulation separates the two and the
// tolerance is tight: a wrong sub-block scale, nibble order, row bound, tile edge or expert stride shows.
// One case uses raw-byte weight blocks and saturated activations, the largest scale * dot-sum products the formats
// allow (the bound the FP32 scale multiply relies on).
//
// Every case prints a hash of the output bytes, so two builds can be compared bit for bit, and the last part times
// each product: 16 experts x 320 rows and one matrix x 512 rows (64-column tiles), the same 5120 rows spread unevenly
// over the 16 experts, and 16 experts x 48 ... 8 rows (the narrower tiles; 50 launches per timed region).  Best of
// `reps` (default 20).  Exit 77 when the build has no K-quant MMQ instances.
//   hip_prefill_kquant_parity [reps]
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata::prefill;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ok(hipMalloc(&p, n ? n : 16), "hipMalloc"); }
    ~Dev() { if (p) (void) hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
double f16(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return ggml_fp16_to_fp32(h);
}

// one weight row as the kernels see it: code q per value, scale per 16 values, offset per 32 values
// (w[k] = ws[k / 16] * q[k] - mo[k / 32])
struct Row {
    std::vector<int8_t> q;
    std::vector<double> ws, mo;
};
void scale_min_k4(int j, const uint8_t* s, int& sc, int& m) {   // ggml-quants.c get_scale_min_k4
    if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
    else { sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
}
void decode_row(ggml_type t, const uint8_t* w, int64_t cols, Row& r) {
    r.q.assign((size_t) cols, 0); r.ws.assign((size_t) cols / 16, 0.0); r.mo.assign((size_t) cols / 32, 0.0);
    for (int64_t b = 0; b < cols / 256; ++b) {
        int8_t* q = r.q.data() + b * 256;
        double* ws = r.ws.data() + b * 16;
        double* mo = r.mo.data() + b * 8;
        if (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K) {
            const bool q5 = t == GGML_TYPE_Q5_K;
            const uint8_t* blk = w + b * (q5 ? 176 : 144);
            const double d = f16(blk), dmin = f16(blk + 2);
            const uint8_t* scales = blk + 4;
            const uint8_t* qh = blk + 16;
            const uint8_t* ql = blk + (q5 ? 48 : 16);
            for (int j = 0; j < 8; ++j) {
                int sc, m;
                scale_min_k4(j, scales, sc, m);
                ws[2 * j] = ws[2 * j + 1] = d * sc;
                mo[j] = dmin * m;
                const uint8_t* l = ql + (j / 2) * 32;
                for (int k = 0; k < 32; ++k) {
                    int v = (j & 1) ? (l[k] >> 4) : (l[k] & 0xF);
                    if (q5 && (qh[k] >> j & 1)) v += 16;
                    q[j * 32 + k] = (int8_t) v;
                }
            }
        } else {   // Q6_K: 16 sub-blocks of 16, signed scales, no offset
            const uint8_t* blk = w + b * 210;
            const double d = f16(blk + 208);
            const int8_t* sc = (const int8_t*) blk + 192;
            for (int h = 0; h < 2; ++h) {
                const uint8_t* ql = blk + h * 64;
                const uint8_t* qh = blk + 128 + h * 32;
                for (int l = 0; l < 32; ++l) {
                    q[h * 128 + l]      = (int8_t) (((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                    q[h * 128 + l + 32] = (int8_t) (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    q[h * 128 + l + 64] = (int8_t) (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                    q[h * 128 + l + 96] = (int8_t) (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                }
            }
            for (int j = 0; j < 16; ++j) ws[j] = d * sc[j];
        }
    }
}

std::vector<uint8_t> quantize_rows(ggml_type type, int64_t rows, int64_t cols, std::mt19937& rng) {
    const auto* tr = ggml_get_type_traits(type);
    if (!tr || !tr->from_float_ref) throw std::runtime_error("no host quantizer for the type");
    const size_t rb = ggml_row_size(type, cols);
    std::vector<uint8_t> out((size_t) rows * rb);
    std::vector<float> row((size_t) cols);
    std::normal_distribution<float> g(0.0f, 0.03f);
    for (int64_t r = 0; r < rows; ++r) {
        for (auto& v : row) v = g(rng);
        if (r % 97 == 5) std::fill(row.begin(), row.begin() + 256, 0.0f);   // some all-zero blocks
        tr->from_float_ref(row.data(), out.data() + (size_t) r * rb, cols);
    }
    return out;
}
// raw-byte blocks: every code and scale value the format can hold, and blocks of the largest ones
std::vector<uint8_t> raw_rows(ggml_type type, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(type), nb = (size_t) rows * (size_t) (cols / 256);
    std::vector<uint8_t> out(nb * bs);
    std::uniform_real_distribution<float> sd(0.5e-3f, 1.5e-3f);
    for (size_t b = 0; b < nb; ++b) {
        uint8_t* blk = out.data() + b * bs;
        const int kind = (int) (rng() % 4);   // 0, 1: random bytes; 2: largest positive codes and scales; 3: most negative
        for (size_t i = 0; i < bs; ++i) blk[i] = kind < 2 ? (uint8_t) rng() : 0xFF;
        const uint16_t d = ggml_fp32_to_fp16(sd(rng)), dmin = ggml_fp32_to_fp16(sd(rng));
        if (type == GGML_TYPE_Q6_K) {
            if (kind == 2) std::memset(blk + 192, 0x7F, 16);                      // q = 31, scale 127
            if (kind == 3) { std::memset(blk, 0, 192); std::memset(blk + 192, 0x80, 16); }   // q = -32, scale -128
            std::memcpy(blk + 208, &d, 2);
        } else {
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &dmin, 2);
        }
    }
    return out;
}

struct Case {
    ggml_type type;
    int64_t w_rows, cols;
    std::vector<int> counts;
    unsigned seed;
    bool raw;   // raw-byte weights and saturated activations
};

bool check(const Case& c) {
    std::mt19937 rng(c.seed);
    const ggml_type type = c.type;
    const int64_t w_rows = c.w_rows, cols = c.cols;
    if (cols % 256 != 0) throw std::runtime_error("a K-quant row is whole blocks of 256 values");
    const int n = (int) c.counts.size();
    int total = 0, max_rows = 0;
    std::vector<int32_t> bounds(1, 0);
    for (int k : c.counts) { total += k; max_rows = std::max(max_rows, k); bounds.push_back(total); }
    const size_t eb = mmq::matrix_bytes(type, w_rows, cols), rb = ggml_row_size(type, cols);
    // two different matrices; every further expert is one of them with its rows rotated (so all experts differ)
    std::vector<uint8_t> w((size_t) n * eb);
    const auto m0 = c.raw ? raw_rows(type, w_rows, cols, rng) : quantize_rows(type, w_rows, cols, rng);
    const auto m1 = c.raw ? raw_rows(type, w_rows, cols, rng) : quantize_rows(type, w_rows, cols, rng);
    for (int e = 0; e < n; ++e) {
        const auto& m = (e & 1) ? m1 : m0;
        const int64_t rot = (e / 2) * 13 % w_rows;
        for (int64_t i = 0; i < w_rows; ++i)
            std::memcpy(w.data() + (size_t) e * eb + (size_t) i * rb, m.data() + (size_t) ((i + rot) % w_rows) * rb, rb);
    }
    std::vector<float> x((size_t) total * cols);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (auto& v : x) v = g(rng);
    if (total > 2) std::fill(x.begin() + cols, x.begin() + 2 * cols, 0.0f);   // one all-zero token row
    if (c.raw)
        for (int r = 0; r < total; ++r) {                                      // saturated rows: every value +-max
            if (r % 3 == 0) std::fill(x.begin() + (size_t) r * cols, x.begin() + (size_t) (r + 1) * cols, 3.0f);
            if (r % 3 == 2) std::fill(x.begin() + (size_t) r * cols, x.begin() + (size_t) (r + 1) * cols, -3.0f);
        }
    std::vector<int32_t> ids((size_t) total);
    for (int i = 0; i < total; ++i) ids[(size_t) i] = total - 1 - i;           // dst rows reversed

    hipStream_t s = nullptr;
    const size_t qb = mmq::q8_bytes(total, cols);
    Dev dw(w.size()), dx(x.size() * 4), dq(qb), db(bounds.size() * 4), di(ids.size() * 4), dd((size_t) total * w_rows * 4);
    ok(hipMemcpy(dw.p, w.data(), w.size(), hipMemcpyHostToDevice), "w");
    ok(hipMemcpy(dx.p, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
    ok(hipMemcpy(db.p, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice), "bounds");
    ok(hipMemcpy(di.p, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "ids");
    ok(hipMemset(dq.p, 0, qb), "xq");
    ok(hipMemset(dd.p, 0xff, (size_t) total * w_rows * 4), "dst");            // NaN bits: an unwritten output shows
    mmq::Context ctx;
    mmq::quantize((const float*) dx.p, nullptr, dq.p, type, cols, cols, total, s);
    mmq::Product p;
    p.w = dw.p; p.type = type; p.w_rows = w_rows; p.w_cols = cols; p.expert_bytes = eb; p.n = n;
    p.xq = dq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) di.p;
    p.total_rows = total; p.max_rows = max_rows; p.dst = (float*) dd.p; p.ld_dst = w_rows;
    ctx.run(p, s);
    ok(hipDeviceSynchronize(), "sync");
    std::vector<float> y((size_t) total * w_rows);
    ok(hipMemcpy(y.data(), dd.p, y.size() * 4, hipMemcpyDeviceToHost), "y");
    std::vector<uint8_t> xq(qb);
    ok(hipMemcpy(xq.data(), dq.p, qb, hipMemcpyDeviceToHost), "xq back");

    // the activations as the device quantizer stored them: 144-byte blocks {16 bytes of scales / sums; 128 codes},
    // block (k / 128, row) at index (k / 128) * total + row.  Q4_K / Q5_K: {f16 d, f16 sum} per 32; Q6_K: f32 d per 32.
    // Q6_K normally has D4 and can use the fused D4 producer; the opt-in DK
    // layout cannot. Query the actual compiled/running layout, not just getenv.
    const bool q8k = !mmq::swiglu_quant_ok(GGML_TYPE_Q6_K);
    const bool ds4 = !q8k && type != GGML_TYPE_Q6_K;
    std::vector<int8_t> qa((size_t) total * cols);
    std::vector<double> da((size_t) total * (cols / 32)), sa((size_t) total * (cols / 32), 0.0);
    long qbad = 0;
    double q_worst = 0;
    for (int r = 0; r < total; ++r)
        for (int64_t b = 0; b < cols / 128; ++b) {
            const uint8_t* blk = xq.data() + ((size_t) b * total + r) * 144;
            std::memcpy(qa.data() + (size_t) r * cols + b * 128, blk + 16, 128);
            for (int gq = 0; gq < 4; ++gq) {
                double d, sum = 0;
                if (q8k) {
                    float f; int16_t bs;
                    std::memcpy(&f, blk, 4); std::memcpy(&bs, blk+4+2*gq, 2);
                    d = f; sum = d*bs;
                    int want = 0;
                    for (int k = 0; k < 32; ++k) want += (int8_t) blk[16+gq*32+k];
                    if (bs != want) ++qbad;
                } else if (ds4) { d = f16(blk + 4 * gq); sum = f16(blk + 4 * gq + 2); }
                else { float f; std::memcpy(&f, blk + 4 * gq, 4); d = f; }
                const size_t gi = (size_t) r * (cols / 32) + b * 4 + gq;
                da[gi] = d; sa[gi] = sum;
                if (!std::isfinite(d) || !std::isfinite(sum)) { ++qbad; continue; }
                // the quantizer against its input: |d * q - x| <= d / 2 (d is a rounded f16 on DS4), the sum of the 32
                double xs = 0, xabs = 0;
                for (int k = 0; k < 32; ++k) {
                    const double xv = x[(size_t) r * cols + b * 128 + gq * 32 + k];
                    const int8_t qv = (int8_t) blk[16 + gq * 32 + k];
                    const double er = std::fabs(d * qv - xv);
                    if (er > 0.51 * d + 2e-3 * std::fabs(xv)) ++qbad;
                    q_worst = std::max(q_worst, d > 0 ? er / d : er);
                    xs += xv; xabs += std::fabs(xv);
                }
                if (ds4 && std::fabs(sum - xs) > 1e-3 * xabs + 1e-6) ++qbad;
            }
        }

    double err2 = 0, ref2 = 0, worst = 0, zero_row = 0, dec_worst = 0;
    long bad = 0;
    Row row;
    std::vector<float> wf((size_t) cols);
    const auto* tr = ggml_get_type_traits(type);
    for (int e = 0; e < n; ++e) {
        if (bounds[(size_t) e] == bounds[(size_t) e + 1]) continue;
        for (int64_t i = 0; i < w_rows; ++i) {
            const uint8_t* wr = w.data() + (size_t) e * eb + (size_t) i * rb;
            decode_row(type, wr, cols, row);
            if (i % 61 == 0) {   // this file's decode against ggml's dequantizer
                tr->to_float(wr, wf.data(), cols);
                for (int64_t k = 0; k < cols; ++k) {
                    const double wv = row.ws[(size_t) k / 16] * row.q[(size_t) k] - row.mo[(size_t) k / 32];
                    dec_worst = std::max(dec_worst, std::fabs(wv - wf[(size_t) k]) / (std::fabs(wv) + 1e-6));
                }
            }
            for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
                const int8_t* a = qa.data() + (size_t) r * cols;
                double ref = 0;
                for (int64_t g32 = 0; g32 < cols / 32; ++g32) {
                    double acc = 0;
                    for (int h = 0; h < 2; ++h) {
                        int si = 0;
                        const int8_t* qq = row.q.data() + g32 * 32 + h * 16;
                        const int8_t* aa = a + g32 * 32 + h * 16;
                        for (int k = 0; k < 16; ++k) si += (int) qq[k] * (int) aa[k];
                        acc += row.ws[(size_t) g32 * 2 + h] * si;
                    }
                    const size_t gi = (size_t) r * (cols / 32) + g32;
                    ref += da[gi] * acc - row.mo[(size_t) g32] * sa[gi];
                }
                const float got = y[(size_t) ids[(size_t) r] * w_rows + (size_t) i];
                if (!std::isfinite(got) || !std::isfinite(ref)) { ++bad; continue; }
                const double d = (double) got - ref;
                err2 += d * d; ref2 += ref * ref; worst = std::max(worst, std::fabs(d));
                if (r == 1) zero_row = std::max(zero_row, std::fabs((double) got));
            }
        }
    }
    const double rel = ref2 > 0 ? std::sqrt(err2 / ref2) : 0;
    const bool pass = bad == 0 && qbad == 0 && rel < 2e-5 && zero_row == 0.0 && dec_worst < 1e-4;
    std::string cs;
    for (int k : c.counts) cs += (cs.empty() ? "" : ",") + std::to_string(k);
    std::printf("%s %-4s W[%lld x %lld]%s rows {%s}: rel L2 %.2e, worst |diff| %.2e, zero row max %.1e, unwritten %ld, "
                "quantizer bad %ld (worst %.3f d), decode %.1e, hash %016llx\n",
                pass ? "PASS" : "FAIL", ggml_type_name(type), (long long) w_rows, (long long) cols, c.raw ? " raw" : "",
                cs.c_str(), rel, worst, zero_row, bad, qbad, q_worst, dec_worst,
                (unsigned long long) fnv(y.data(), y.size() * 4));
    return pass;
}

// `burst` products per timed region (a short product is launched several times back to back, one wait at the end)
bool time_shape(ggml_type type, int64_t w_rows, int64_t cols, const std::vector<int>& counts, int reps, const char* label,
                int burst = 1) {
    std::mt19937 rng(7);
    const int n = (int) counts.size();
    int total = 0, max_rows = 0;
    std::vector<int32_t> bounds(1, 0);
    for (int k : counts) { total += k; max_rows = std::max(max_rows, k); bounds.push_back(total); }
    const size_t eb = mmq::matrix_bytes(type, w_rows, cols);
    std::vector<uint8_t> w((size_t) n * eb);
    const auto q = quantize_rows(type, w_rows, cols, rng);
    for (int e = 0; e < n; ++e) std::memcpy(w.data() + (size_t) e * eb, q.data(), eb);
    std::vector<float> x((size_t) total * cols);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (auto& v : x) v = g(rng);
    std::vector<int32_t> ids((size_t) total);
    for (int i = 0; i < total; ++i) ids[(size_t) i] = i;
    Dev dw(w.size()), dx(x.size() * 4), dq(mmq::q8_bytes(total, cols)), db(bounds.size() * 4), di(ids.size() * 4),
        dd((size_t) total * w_rows * 4);
    ok(hipMemcpy(dw.p, w.data(), w.size(), hipMemcpyHostToDevice), "w");
    ok(hipMemcpy(dx.p, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
    ok(hipMemcpy(db.p, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice), "bounds");
    ok(hipMemcpy(di.p, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "ids");
    ok(hipMemset(dd.p, 0xff, (size_t) total * w_rows * 4), "dst");
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw.p; p.type = type; p.w_rows = w_rows; p.w_cols = cols; p.expert_bytes = eb; p.n = n;
    p.xq = dq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) di.p;
    p.total_rows = total; p.max_rows = max_rows; p.dst = (float*) dd.p; p.ld_dst = w_rows;
    double best_q = 1e30, best_p = 1e30;
    for (int rep = 0; rep < reps + 2; ++rep) {
        ok(hipDeviceSynchronize(), "sync");
        auto t0 = std::chrono::steady_clock::now();
        mmq::quantize((const float*) dx.p, nullptr, dq.p, type, cols, cols, total, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        auto t1 = std::chrono::steady_clock::now();
        for (int b = 0; b < burst; ++b) ctx.run(p, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        auto t2 = std::chrono::steady_clock::now();
        if (rep >= 2) {   // two warm-up rounds
            best_q = std::min(best_q, std::chrono::duration<double, std::milli>(t1 - t0).count());
            best_p = std::min(best_p, std::chrono::duration<double, std::milli>(t2 - t1).count() / burst);
        }
    }
    std::vector<float> y((size_t) total * w_rows);
    ok(hipMemcpy(y.data(), dd.p, y.size() * 4, hipMemcpyDeviceToHost), "y");
    long bad = 0;
    for (float v : y) bad += !std::isfinite(v);
    const double mac = (double) total * (double) w_rows * (double) cols;
    std::printf("time %-4s W[%lld x %lld] %s (%d rows, max %d): quantize %.3f ms, product %.3f ms (%.0f G MAC/s), "
                "unwritten %ld, hash %016llx\n",
                ggml_type_name(type), (long long) w_rows, (long long) cols, label, total, max_rows, best_q, best_p,
                mac / best_p / 1e6, bad, (unsigned long long) fnv(y.data(), y.size() * 4));
    return bad == 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        // since the merge of upstream 0.2.0 the K-quant instances run only with STRATA_MMQ_KQ=1 (the gfx103x engine
        // defaults set it); without it a KQUANTS build skipped here
        setenv("STRATA_MMQ_KQ", "1", 0);
        if (!mmq::supported(GGML_TYPE_Q4_K) || !mmq::supported(GGML_TYPE_Q5_K) || !mmq::supported(GGML_TYPE_Q6_K)) {
            std::printf("SKIP: this build has no K-quant MMQ instances (-DSTRATA_MMQ_KQUANTS=ON)\n");
            return 77;
        }
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        const int reps = argc > 1 ? std::max(1, std::atoi(argv[1])) : 20;
        const ggml_type types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
        bool all = true;
        unsigned seed = 1;
        for (ggml_type t : types) {
            all &= check({t, 128, 512, {33, 0, 1, 70}, seed++, false});            // small
            all &= check({t, 100, 512, {40, 9}, seed++, false});                   // rows not a multiple of 128: the fallback tiles
            all &= check({t, 256, 768, {5, 64, 0, 31, 130}, seed++, true});        // raw bytes, saturated: the largest products
            all &= check({t, 1280, 2560, {37, 0, 70, 32, 1, 129}, seed++, false}); // gate + up
            all &= check({t, 2560, 768, {37, 0, 70, 32, 1, 129}, seed++, false});  // in place of down
            all &= check({t, 2560, 768, {0, 320, 64, 3, 0, 200, 8}, seed++, false});
            all &= check({t, 2560, 2560, {203}, seed++, false});                   // a dense projection
            for (int j : {80, 96, 112, 128})
                all &= check({t, 128, 512, {j}, seed++, true});                   // owner's wider J tiles
        }
        const std::vector<int> even(16, 320);
        const std::vector<int> uneven = {1100, 0, 37, 640, 0, 0, 500, 12, 900, 0, 64, 700, 5, 0, 1000, 162};   // 5120 rows too
        for (ggml_type t : types) {
            all &= time_shape(t, 1280, 2560, even, reps, "16 experts x 320 rows");
            all &= time_shape(t, 2560, 768, even, reps, "16 experts x 320 rows");
            all &= time_shape(t, 2560, 2560, {512}, reps, "1 matrix x 512 rows");
        }
        for (ggml_type t : types) all &= time_shape(t, 1280, 2560, uneven, reps, "16 experts uneven");
        // few rows per expert: the narrower tiles (J = 48 ... 8 columns)
        for (ggml_type t : types)
            for (int rows : {48, 40, 32, 24, 16, 8})
                all &= time_shape(t, 1280, 2560, std::vector<int>(16, rows), reps,
                                  ("16 experts x " + std::to_string(rows) + " rows").c_str(), 50);
        std::printf(all ? "ALL PASS\n" : "FAILED\n");
        return all ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
