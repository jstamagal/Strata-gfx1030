// include/strata/kernels/dp4a.hpp - the two instructions the kernels use that Pascal (sm_60) does not have.
//
// The engine is written for compute capability 7.5 and newer and refuses older cards at CMake configure time
// (a Pascal build passes -DSTRATA_EXPERIMENTAL_SM60=ON).  A GP100-class card is compute capability 6.0, which is
// missing exactly two things the i-quant kernels rely on:
//
//   * `__dp4a`, a byte-wise dot product, available from 6.1 (Pascal GP10x/GV11x).  The reference for its
//     fallback is llama.cpp's own (`ggml/src/ggml-cuda/common.cuh`), since the kernels in this directory are
//     transcribed from llama.cpp's vecdotq.cuh and that wrapper's operands are the same sites: it reinterprets
//     the operands as SIGNED bytes and accumulates in int32, which is what `__dp4a` does for these calls.  The
//     fallback below computes exactly that with four PTX `vmad` byte-select instructions, so it is bit-exact
//     rather than merely close (tools/sm60_dp4a_check.cu checks it against the byte-wise form).
//   * `__nanosleep`, available from 7.0 (Volta).  It only paces single-thread doorbell waits, so a loop that
//     spins without it is correct, merely busier.
//
// Both are compile-time selections on `__CUDA_ARCH__`, so one source tree builds for Pascal and for RTX
// 20/30/40/50 without a #define at the call sites.  A HIP build never takes either fallback: `__CUDA_ARCH__` is
// undefined there, so `STRATA_DP4A` is `__dp4a` (which `hip_compat/intrinsics.hpp` provides) and the pause is
// HIP's own `__nanosleep`.
#pragma once

#include <cstdint>
#if defined(STRATA_HIP_GFX906)
#include <cuda_runtime.h>   // gfx906: the compat layer (__forceinline__, __nanosleep, __dp4a)
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
__device__ __forceinline__ int strata_dp4a(const int a, const int b, const int c) {
    // PTX vmad with byte selectors: compiles to four VMAD.S8.S8 (cuobjdump -sass), no byte extraction.
    // Bit-exact with the byte-wise form it replaces: 1.07e9 cases (edge bytes, every per-lane byte pair, random),
    // 0 mismatches, ~2.2x faster in isolation on a Tesla P100 (tools/sm60_dp4a_check.cu).
    int r = c;
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;"
        : "+r"(r) : "r"(a), "r"(b));
    return r;
}
#define STRATA_DP4A(a, b, c) strata_dp4a((a), (b), (c))
#else
#define STRATA_DP4A(a, b, c) __dp4a((a), (b), (c))
#endif

#ifdef STRATA_RDNA2_DOT_Z
[[maybe_unused]] static constexpr bool strata_rdna2_dot_z_build = true;
#else
[[maybe_unused]] static constexpr bool strata_rdna2_dot_z_build = false;
#endif
// Owner dfb9b770b / a508a95b9 / 4d623c1f6: a literal-zero VOP3P chain head
// avoids v_mov_b32 0 + v_dot4c on gfx103x. Source-scoped build controls let the
// expert, dense/rowset, head and attention schedules be measured independently.
// Nonzero links, other HIP architectures and CUDA retain their original dot.
__device__ __forceinline__ int strata_dp4a_z(int a, int b, int c) {
#if defined(STRATA_RDNA2_DOT_Z) && defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__))
    if (__builtin_constant_p(c) && c == 0) {
        asm("v_dot4_i32_i8 %0, %1, %2, 0" : "=v"(c) : "v"(a), "v"(b));
        return c;
    }
#endif
    return STRATA_DP4A(a, b, c);
}
#if defined(STRATA_RDNA2_DOT_Z)
#undef STRATA_DP4A
#define STRATA_DP4A(a, b, c) strata_dp4a_z((a), (b), (c))
#endif

// Two integer forms that the AMDGPU backend compiles to v_mul_lo_u32, which runs at quarter rate on gfx10 (RDNA2):
//
//   * a per-byte mask: one bit per byte (b has bits only at 0x01010101) to 0x00 / 0xFF bytes.  (b << 8) - b and
//     b * 255 both fold to a 32-bit multiply.  v_perm_b32 with selector byte 12 + bit gives 0x00 (12) or 0xFF
//     (13) per byte - the same bits in one full-rate instruction.
//   * a byte broadcast: v * 0x01010101 for an 8-bit v.  v_perm_b32 with selector 0 copies byte 0 to all four.
//   * a scale multiply where both operands fit in 24 signed bits and the product fits in 32 (a q8 dot sum times a
//     6-bit K-quant scale): __mul24 is v_mul_i32_i24, full rate, the same integer.
//
// Only the HIP device pass takes the new forms; CUDA and the host pass keep the original expressions, so their
// code is unchanged.  Same integers everywhere, so the outputs are bitwise the old ones.
#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ uint32_t strata_bytemask(uint32_t b) { return __builtin_amdgcn_perm(0u, 0u, 0x0C0C0C0Cu + b); }
__device__ __forceinline__ uint32_t strata_bytebcast(uint32_t v) { return __builtin_amdgcn_perm(v, v, 0u); }
__device__ __forceinline__ int strata_smul(int a, int b) { return __mul24(a, b); }
#else
__device__ __forceinline__ uint32_t strata_bytemask(uint32_t b) { return (b << 8) - b; }
__device__ __forceinline__ uint32_t strata_bytebcast(uint32_t v) { return v * 0x01010101u; }
__device__ __forceinline__ int strata_smul(int a, int b) { return a * b; }
#endif

/// Yield the thread while a doorbell flag is being polled.  The length of the pause is a backoff hint, not a
/// contract, and only pre-Volta CUDA has no `__nanosleep` at all.
__device__ __forceinline__ void strata_spin_pause() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    // sm_6x: the loop spins.  Every call site is a single-thread doorbell wait, so nothing else is delayed.
#else
    __nanosleep(100);
#endif
}
