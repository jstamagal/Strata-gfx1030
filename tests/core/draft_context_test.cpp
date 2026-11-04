#include "strata/core/draft_context.hpp"
#include <cstdio>
#include <vector>
using strata::core::DraftContextRows;
static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
int main() {
    DraftContextRows r;
    r.init(4096, {0,1,2});
    int64_t prompt[]{0,7,8,7,1,-1,5000};
    r.begin(prompt, 7);
    CHECK((r.ids() == std::vector<int32_t>{8,7}));
    int32_t picks[]{8,9,10,1,-1,5000};
    r.note(picks, 6, false);
    CHECK((r.ids() == std::vector<int32_t>{8,7,9,10}));
    CHECK(r.flushed() == 0); r.commit(); CHECK(r.flushed() == 4);
    std::vector<int32_t> candidates(2048);
    for (int i = 0; i < 2048; ++i) candidates[i] = 100 + i;
    r.note(candidates.data(), 2048, true);
    CHECK(r.ids().size() == DraftContextRows::capacity - DraftContextRows::pick_reserve);
    int32_t tail[64]; for (int i = 0; i < 64; ++i) tail[i] = 3000 + i;
    r.note(tail, 64, false); CHECK(r.ids().size() == DraftContextRows::capacity);
    r.begin(nullptr, 0); CHECK(r.ids().empty() && r.flushed() == 0);
    r.note(tail, 64, false); CHECK(r.ids().size() == 64);
    std::vector<int64_t> long_prompt(600);
    for (int i = 0; i < 600; ++i) long_prompt[i] = 100 + i;
    r.begin(long_prompt.data(), 600);
    CHECK(r.ids().size() == DraftContextRows::prompt_share);
    CHECK(r.ids().front() == 444 && r.ids().back() == 699);
    std::printf("draft_context_test: %s (subset exclusion, deduplication, committed picks, reserved rows, reset)\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
