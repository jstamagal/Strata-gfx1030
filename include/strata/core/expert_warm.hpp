#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace strata::core {

// This router is a predictor only. Its results never enter the expert plan.
class NextExpertWarm {
public:
    struct Layer {
        std::vector<int8_t> q;
        std::vector<float> scales;
        size_t bytes = 0, down = 0;
    };
    NextExpertWarm(int nl, int ne, int dim) : layers((size_t) nl), ne_(ne), dim_(dim) {}
    bool set_router(int l, const uint16_t* bf16, size_t bytes, size_t down) {
        if (dim_ <= 0 || dim_ % 32 || ne_ <= 0 || down > bytes || l < 0 || l >= (int) layers.size()) return false;
        Layer& dst = layers[(size_t) l];
        dst.bytes = bytes; dst.down = down;
        dst.q.resize((size_t) ne_ * dim_); dst.scales.resize((size_t) ne_ * dim_ / 32);
        for (size_t b = 0; b < dst.scales.size(); ++b) {
            float f[32], max = 0;
            for (int i = 0; i < 32; ++i) {
                const uint32_t bits = (uint32_t) bf16[b * 32 + i] << 16;
                std::memcpy(f + i, &bits, 4);
                if (!std::isfinite(f[i])) return false;
                max = std::max(max, std::abs(f[i]));
            }
            const float scale = max / 127.0f;
            dst.scales[b] = scale;
            for (int i = 0; i < 32; ++i) dst.q[b * 32 + i] = (int8_t) (scale > 0 ? std::lrintf(f[i] / scale) : 0);
        }
        return true;
    }
    bool prepare(int l, const float* x, int nt, int k, const int32_t* res, std::vector<const uint8_t*> blobs) {
        if (l < 0 || l >= (int) layers.size() || layers[(size_t) l].q.empty() || nt <= 0 || nt > 16) return false;
        layer = l; nt_ = nt; k_ = std::clamp(k, 1, std::min(ne_, 16));
        res_.assign(res, res + ne_); blobs_ = std::move(blobs);
        if (blobs_.size() != (size_t) ne_) return false;
        x_.assign(x, x + (size_t) nt * dim_);
        xq_.resize((size_t) nt * dim_); xs_.resize((size_t) nt * dim_ / 32);
        logits_.resize((size_t) nt * ne_);
        scores_.resize((size_t) ne_);
        best = -1; arrive_ = 0; ready_ = false;
        x_arrive_ = 0; x_ready_ = false; invalid_ = false;
        return true;
    }
    static void job(void* ctx, int worker, int workers, const std::atomic<bool>& stop) {
        ((NextExpertWarm*) ctx)->run(worker, workers, stop);
    }
    std::vector<Layer> layers;
    int layer = -1, best = -1;
private:
    void select() {
        std::fill(scores_.begin(), scores_.end(), 0.0f);
        for (int t = 0; t < nt_; ++t) {
            int top[16]; float val[16]; int m = 0;
            for (int e = 0; e < ne_; ++e) {
                const float v = logits_[(size_t) t * ne_ + e];
                if (m == k_ && v <= val[k_ - 1]) continue;
                int j = m < k_ ? m++ : k_ - 1;
                while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; top[j] = top[j - 1]; --j; }
                val[j] = v; top[j] = e;
            }
            float w[16], sum = 0;
            for (int j = 0; j < m; ++j) sum += (w[j] = std::exp(val[j] - val[0]));
            for (int j = 0; j < m; ++j)
                if (res_[top[j]] < 0 && blobs_[top[j]]) scores_[top[j]] += w[j] / sum;
        }
        for (int e = 0; e < ne_; ++e)
            if (scores_[e] > 0 && (best < 0 || scores_[e] > scores_[best])) best = e;
    }
    void run(int worker, int workers, const std::atomic<bool>& stop) {
        for (size_t b = xs_.size() * worker / workers; b < xs_.size() * (worker + 1) / workers; ++b) {
            if (stop.load(std::memory_order_relaxed)) break;
            float max = 0;
            bool valid = true;
            for (int i = 0; i < 32; ++i) {
                valid = valid && std::isfinite(x_[b * 32 + i]);
                max = std::max(max, std::abs(x_[b * 32 + i]));
            }
            if (!valid) { invalid_.store(true, std::memory_order_relaxed); continue; }
            const float scale = max / 127.0f; xs_[b] = scale;
            for (int i = 0; i < 32; ++i) xq_[b * 32 + i] = (int8_t) (scale > 0 ? std::lrintf(x_[b * 32 + i] / scale) : 0);
        }
        if (x_arrive_.fetch_add(1, std::memory_order_acq_rel) + 1 == workers) x_ready_.store(true, std::memory_order_release);
        else while (!x_ready_.load(std::memory_order_acquire)) {
            if (stop.load(std::memory_order_relaxed)) return;
            std::this_thread::yield();
        }
        if (stop.load(std::memory_order_relaxed) || invalid_.load(std::memory_order_relaxed)) return;
        const Layer& r = layers[(size_t) layer];
        const int nb = dim_ / 32;
        for (int e = ne_ * worker / workers; e < ne_ * (worker + 1) / workers; ++e) {
            if (stop.load(std::memory_order_relaxed)) break;
            for (int t = 0; t < nt_; ++t) {
                float score = 0;
                for (int b = 0; b < nb; ++b) {
                    int dot = 0;
                    for (int i = 0; i < 32; ++i)
                        dot += (int) r.q[((size_t) e * nb + b) * 32 + i] * (int) xq_[((size_t) t * nb + b) * 32 + i];
                    score += (float) dot * r.scales[(size_t) e * nb + b] * xs_[(size_t) t * nb + b];
                }
                logits_[(size_t) t * ne_ + e] = score;
            }
        }
        if (arrive_.fetch_add(1, std::memory_order_acq_rel) + 1 == workers) {
            if (!stop.load(std::memory_order_relaxed)) select();
            ready_.store(true, std::memory_order_release);
        } else while (!ready_.load(std::memory_order_acquire)) {
            if (stop.load(std::memory_order_relaxed)) return;
            std::this_thread::yield();
        }
        if (best < 0) return;
        // Prefetches do not dereference mutable RAM homes; a concurrent swap may overwrite their contents.
        for (int part = 0; part < 2; ++part) {
            const size_t first = part == 0 ? 0 : r.down, end = part == 0 ? r.down : r.bytes;
            for (size_t off = first + (size_t) worker * 64; off < end; off += (size_t) workers * 64) {
                if (stop.load(std::memory_order_relaxed)) return;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(blobs_[(size_t) best] + off, 0, 2);
#endif
            }
        }
    }
    int ne_, dim_, nt_ = 0, k_ = 0;
    std::vector<int8_t> xq_;
    std::vector<float> x_, xs_, logits_, scores_;
    std::vector<int32_t> res_;
    std::vector<const uint8_t*> blobs_;
    alignas(64) std::atomic<int> arrive_{0}, x_arrive_{0};
    alignas(64) std::atomic<bool> ready_{false}, x_ready_{false}, invalid_{false};
};

} // namespace strata::core
