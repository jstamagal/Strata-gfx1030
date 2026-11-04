#include "strata/core/draft_cum.hpp"
#include <cstdio>
#include <limits>
#include <random>

using strata::core::DraftCumAdapt;
static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
int main() {
    DraftCumAdapt rule(0.7f);
    CHECK(std::abs(rule.threshold() - 0.49f) < 1e-6f);
    float p[]{0.8f, 0.8f, 0.8f, 0.8f};
    CHECK(rule.rows(p, 4, 5) == 4);
    CHECK(rule.rows(p, 0, 5) == 1);
    CHECK(rule.rows(p, 4, 1) == 1);
    float bad[]{0.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f};
    CHECK(rule.rows(bad, 3, 4) == 1);
    CHECK(!rule.keep(bad + 1, 0));
    float censored[]{0.81f, 0.21f, 0.61f};
    rule.observe(censored, 3, 1);
    CHECK(rule.seen(8) == 1 && rule.accepted(8) == 1);
    CHECK(rule.seen(2) == 1 && rule.accepted(2) == 0);
    CHECK(rule.seen(6) == 0);
    DraftCumAdapt good(0.7f), poor(0.7f);
    float q[]{0.71f, 0.71f, 0.71f};
    for (int i = 0; i < 200; ++i) { good.observe(q, 3, 3); poor.observe(q, 3, 0); }
    CHECK(good.rate(q[0]) > 0.99f && poor.rate(q[0]) < 0.05f);
    CHECK(good.rows(q, 3, 4) == 4 && poor.rows(q, 3, 4) == 1);
    const float cold = DraftCumAdapt(0.7f).threshold();
    CHECK(good.threshold() <= cold && poor.threshold() >= cold);
    DraftCumAdapt explicit_th(0.7f, 0.6f);
    CHECK(explicit_th.threshold() == 0.6f);
    CHECK(DraftCumAdapt(0.0f).threshold() == 0.5f);
    CHECK(DraftCumAdapt(0.7f, std::numeric_limits<float>::quiet_NaN()).threshold() == DraftCumAdapt(0.7f).threshold());
    DraftCumAdapt ram(0.7f);
    ram.observe_cpu(5, 100);
    CHECK(std::abs(ram.cpu_pressure() - 0.05f) < 1e-6f && ram.threshold() > cold);
    const float before = ram.threshold(); ram.observe_cpu(1, 0); CHECK(ram.threshold() == before);
    for (int i = 0; i < 200; ++i) ram.observe_cpu(100, 100);
    CHECK(ram.cpu_pressure() > 0.99f && ram.threshold() <= 0.99f);
    std::mt19937 gen(638);
    for (int k = 0; k < 10000; ++k) {
        float probs[16];
        for (auto& x : probs) x = (float) (gen() % 10001) / 10000.0f;
        const int cap = 1 + gen() % 16;
        float joint = 1;
        int t = 1;
        for (int j = 0; j < cap - 1; ++j) {
            joint *= rule.rate(probs[j]);
            CHECK((joint >= rule.threshold()) == rule.keep(probs, j));
            if (t == j + 1 && joint >= rule.threshold()) ++t;
        }
        CHECK(t == rule.rows(probs, 16, cap));
        rule.observe(probs, cap - 1, gen() % cap);
        CHECK(rule.threshold() >= 0.01f && rule.threshold() <= 0.99f);
    }
    std::printf("draft_cum_test: %s (cold start, censoring, calibration, host-chain product agreement)\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
