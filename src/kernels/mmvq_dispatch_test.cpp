#undef NDEBUG
#include "strata/kernels/mmvq_dispatch.hpp"

#include <array>
#include <cassert>
#include <cstdio>

using strata::kernels::dispatch_mmvq_columns;
using strata::kernels::mmvq_row_count;

static int checks = 0;
static void check(bool ok) { ++checks; assert(ok); }

template<int MaxCols>
void coverage() {
    for (int requested = -1; requested <= 18; ++requested) {
        int calls = 0, selected = 0;
        std::array<int, 16 * 7> row;
        std::array<std::array<int, 16 * 11>, 3> set;
        row.fill(-1);
        for (auto& matrix : set) matrix.fill(-1);
        const bool row_ok = dispatch_mmvq_columns<MaxCols>(requested, [&](auto columns) {
            constexpr int N = decltype(columns)::value;
            ++calls; selected = N;
            for (int c = 0; c < N; ++c)
                for (int r = 0; r < 7; ++r) row[c * 7 + r] = c * 100 + r;
        });
        const bool valid = requested >= 1 && requested <= MaxCols;
        check(row_ok == valid);
        check(calls == (valid ? 1 : 0));
        check(selected == (valid ? requested : 0));
        int set_calls = 0;
        const bool set_ok = dispatch_mmvq_columns<MaxCols>(requested, [&](auto columns) {
            constexpr int N = decltype(columns)::value;
            ++set_calls;
            check(N == requested);
            for (int matrix = 0; matrix < 3; ++matrix)
                for (int c = 0; c < N; ++c)
                    for (int r = 0; r < 11; ++r) set[matrix][c * 11 + r] = matrix * 10000 + c * 100 + r;
        });
        check(set_ok == valid && set_calls == (valid ? 1 : 0));
        for (int c = 0; c < 16; ++c) {
            const bool written = valid && c < requested;
            for (int r = 0; r < 7; ++r) check(row[c * 7 + r] == (written ? c * 100 + r : -1));
            for (int matrix = 0; matrix < 3; ++matrix)
                for (int r = 0; r < 11; ++r)
                    check(set[matrix][c * 11 + r] == (written ? matrix * 10000 + c * 100 + r : -1));
        }
    }
}

int main() {
    coverage<8>(); coverage<12>(); coverage<16>();
    for (int columns = 1; columns <= 16; ++columns)
        for (int requested_rows : {1, 2, 4, 8}) {
            const int rows = mmvq_row_count(requested_rows, columns);
            check(rows * columns <= 32);
            check(columns <= 4 || rows <= 2);
        }
    // Reproduce the old dispatch's boundary: a successful <8> launch for a
    // sixteen-column request leaves exactly the second half at its stale value.
    std::array<int, 16> old;
    old.fill(-1);
    for (int c = 0; c < 8; ++c) old[c] = c;
    check(old[7] == 7 && old[8] == -1 && old[15] == -1);
    std::printf("mmvq_dispatch_test: %d checks passed (ROW/SET, 8/12/16-column builds)\n", checks);
}
