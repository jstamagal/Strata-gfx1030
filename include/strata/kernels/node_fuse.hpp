#pragma once
// STRATA_NODE_FUSE (opt-in, off by default): fewer nodes in the verifier's window graph, every value bitwise the
// launches each fusion replaces.  =1 turns all of them on; any other value is a bit mask of the single fusions (for
// measuring one at a time):
//   2  the MoE combine of a layer inside the next layer's first hyper-connection norm (fused_gr.cu)
//   4  the QSA layer's three per-head RMSNorm + RoPE launches (k, q, indexer q) as one (native_rope.cu)
//   8  the GDN conv + L2 norm and the a/b gates inside the recurrence kernel (verify_kernels.cu)
//  16  the shared expert reads the MoE input's q8_1 image the hyper-connection read already wrote (no quantize)
//  32  the shared expert's gate and up projections in one launch (native_mmvq_set, STRATA_MMVQ_SET's kernel)
//  64  the QSA indexer tail's window snapshot made by the indexer append (no copy launch before it)
// and two existing one-launch forms the window graph did not take on HIP (their own switches turn them on as well):
// 128  the n-gram block's post-projection steps for the window in one pass (STRATA_DF_PLE=1's path)
// 256  the head's hyper-connection read for the window's tokens in one launch set (STRATA_HEAD_MIX_MULTI=1's path),
//      with STRATA_QFUSE also writing the head projection's q8_1 input
// STRATA_T1_DIET (opt-in, off by default; HIP): more of the same, chiefly for the one-row (T=1) window.  =1 is every
// bit; any other value a bit mask:
//   2  the hyper-connection read's split register norm made by its wide down launch (1-4 tokens; fused_gr.cu)
//   4  a one-row window's QSA layer takes the batched forms (STRATA_IDX_PAIR's pair, the norm + RoPE x3, the gate q8_1)
//   8  the routing top-10 + plan with DPP wave reductions and the residency loads ahead (route_plan.cu)
#include <cstdlib>

namespace strata::kernels {
enum : int { kNodeFuseCombine = 2, kNodeFuseNormRope = 4, kNodeFuseGdn = 8, kNodeFuseShQ8 = 16, kNodeFuseShSet = 32, kNodeFuseIdxSnap = 64,
             kNodeFusePle = 128, kNodeFuseHeadMix = 256 };
inline int node_fuse_mask() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_NODE_FUSE");
        const int n = v != nullptr ? std::atoi(v) : 0;
        return n == 1 ? 0x7fffffff : n < 0 ? 0 : n;
    }();
    return m;
}
inline bool node_fuse(int bit) { return (node_fuse_mask() & bit) != 0; }
enum : int { kT1NormDown = 2, kT1QsaRow = 4, kT1Route = 8 };
inline int t1_diet_mask() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_T1_DIET");
        const int n = v != nullptr ? std::atoi(v) : 0;
        return n == 1 ? 0x7fffffff : n < 0 ? 0 : n;
    }();
    return m;
}
inline bool t1_diet(int bit) { return (t1_diet_mask() & bit) != 0; }
}  // namespace strata::kernels
