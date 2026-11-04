// Exact column-count dispatch shared by the HIP row and grouped-matrix MMVQ paths.
#pragma once

#include <type_traits>

namespace strata::kernels {

constexpr int mmvq_row_count(int requested, int columns) {
    return requested >= 4 ? (columns <= 4 ? 4 : 2) : requested >= 2 ? 2 : 1;
}

template<int MaxCols, class Launch>
bool dispatch_mmvq_columns(int columns, Launch&& launch) {
    static_assert(MaxCols == 8 || MaxCols == 12 || MaxCols == 16, "MMVQ supports 8, 12 or 16 columns");
    // A wide count must never fall through to an 8-column kernel and report
    // success: that leaves the remaining output columns stale on every replay.
#define STRATA_MMVQ_COLUMNS(N) \
    case N: \
        if constexpr (N <= MaxCols) { launch(std::integral_constant<int, N>{}); return true; } \
        else return false;
    switch (columns) {
        STRATA_MMVQ_COLUMNS(1) STRATA_MMVQ_COLUMNS(2) STRATA_MMVQ_COLUMNS(3) STRATA_MMVQ_COLUMNS(4)
        STRATA_MMVQ_COLUMNS(5) STRATA_MMVQ_COLUMNS(6) STRATA_MMVQ_COLUMNS(7) STRATA_MMVQ_COLUMNS(8)
        STRATA_MMVQ_COLUMNS(9) STRATA_MMVQ_COLUMNS(10) STRATA_MMVQ_COLUMNS(11) STRATA_MMVQ_COLUMNS(12)
        STRATA_MMVQ_COLUMNS(13) STRATA_MMVQ_COLUMNS(14) STRATA_MMVQ_COLUMNS(15) STRATA_MMVQ_COLUMNS(16)
        default: return false;
    }
#undef STRATA_MMVQ_COLUMNS
}

}  // namespace strata::kernels
