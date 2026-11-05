// include/strata/kernels/route_plan.hpp - STRATA_ROUTE_PLAN=1 (opt-in): the verify window's routing top-10 and the
// all-resident plan in ONE launch.  A node of the window's graph costs about 4 us on a Radeon PRO V620 before it does
// anything (router_chain_bench: an empty kernel 2.6 us, a one-row GEMV 3.9 us), and these two are small: 48 layers a
// window each run them back to back.  src/kernels/cuda/route_plan.cu has the kernel; hip_route_plan_parity checks
// ids, weights and the plan's bytes against native_router_top10_multi + resident_plan.
#pragma once

#include <cstdint>

namespace strata::kernels {

// STRATA_ROUTE_PLAN=1: the top-10 and the plan in one launch; =2 (HIP): the router's BF16 projection in it too
// (native_route_fused); anything else, or unset: off
bool route_plan_enabled();
int route_plan_mode();
// STRATA_T1_DIET (8; HIP gfx10+): both launches above with DPP wave reductions and the residency loaded ahead (the
// same ids, weights and plan bytes).  route_plan_fast_tune (tests): 1 / 0 forces it on / off, -1 the environment again
bool route_plan_fast();
void route_plan_fast_tune(int on);

// logits [n_tok][512] -> ids / weights [n_tok][10], exactly native_router_top10_multi's (native_router_top10's for one
// token), then resident_plan(ids, n_tok * 10, 10, res_layer, n_expert, cache_base, slot_off, blob, plan, capx,
// nullptr, 0, stream, plan_err)'s plan: the all-resident form (no skip word; an expert that is not resident leaves an
// empty plan and raises *plan_err).  1 <= n_tok <= 8.  Device pointers (plan_err may be mapped host memory, or null);
// capturable.
void native_route_plan(const float* logits, int32_t* ids, float* weights, int n_tok, const int32_t* res_layer,
                       int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                       int32_t* plan, long long capx, uint32_t* plan_err, void* stream);

// STRATA_ROUTE_PLAN=2 (HIP): bf16_gemv_fp32_mmvf_multi(x, ldx, w_router, logits, 512, 2560, 512, n_tok) and
// native_route_plan in ONE launch: the same logits [n_tok][512] (ldy 512), ids, weights and plan.  `done` is one
// zeroed device word of the caller's that the launch uses to count its finished rows (it leaves it zero).  2560-wide
// router rows, 512 experts, top 10.  False (nothing launched) where it cannot run (CUDA).
bool native_route_fused(const float* x, long long ldx, const uint16_t* w_router, float* logits, uint32_t* done,
                        int32_t* ids, float* weights, int n_tok, const int32_t* res_layer, int n_expert,
                        const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                        long long capx, uint32_t* plan_err, void* stream);

}  // namespace strata::kernels
