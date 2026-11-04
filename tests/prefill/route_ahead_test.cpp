#include "strata/prefill/route_ahead.hpp"
#include <cstdio>
#include <cstdlib>

using namespace strata::prefill::route_ahead;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s\n", #x); std::exit(1); } } while (0)
int main() {
    CHECK(requested_from(nullptr, 0) == 0);
    for (const char* s : {"", "-1", "65", "1x", "1,", "1,,2", "x,16", "999999999999999999999999999"})
        CHECK(requested_from(s, 0) == 0);
    CHECK(requested_from("16", 0) == 16); CHECK(requested_from("16", 3) == 16);
    CHECK(requested_from("0,16", 0) == 0); CHECK(requested_from("0,16", 1) == 16);
    CHECK(requested_from("0,16", 2) == 0); CHECK(requested_from("8,32,4", 2) == 4);
    CHECK(bank_slots(16, 96, 16) == 16); CHECK(bank_slots(64, 48, 16) == 16);
    CHECK(bank_slots(16, 32, 16) == 0); CHECK(bank_slots(0, 96, 16) == 0);
    History h;
    auto all = [](int32_t) { return true; };
    h.prepare(3, 8, false);
    CHECK(!h.any()); CHECK(h.predict(0, 16, all).empty());
    const int32_t counts[] = {0, 2, 7, 7, 1, 3, 0, 9};
    h.remember(1, counts);
    CHECK(h.any());
    CHECK(h.predict(0, 16, all).empty());
    CHECK(h.predict(1, 4, all) == std::vector<int32_t>({7, 2, 3, 5}));
    CHECK(h.predict(1, 4, [](int32_t e) { return e != 7 && e != 2; }) == std::vector<int32_t>({3, 5, 1, 4}));
    h.prepare(3, 8, false);
    CHECK(h.predict(1, 2, all) == std::vector<int32_t>({7, 2}));
    // Current routing may disagree completely: it remains the authoritative walk, not this hint.
    const int32_t different[] = {10, 0, 0, 0, 0, 0, 9, 0};
    h.remember(1, different);
    CHECK(h.predict(1, 8, all) == std::vector<int32_t>({0, 6}));
    h.prepare(3, 8, true); CHECK(!h.any());
    h.remember(1, counts); h.prepare(2, 4, false); CHECK(!h.any());
    std::puts("PASS: default-off/device selection, bounded bank, route ranking, cache filter, history reset");
}
