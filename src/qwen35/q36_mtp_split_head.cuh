#pragma once

// Rank 1 normally owns the upper half of the target vocabulary. A draft prefix
// lives in rank 0's half, so keep a second copy of the prefix's upper half on
// rank 1. Both ranks use the same Q8 row dot as the old short-head path.
void Engine::prepare_mtp_split_head() {
    use_dev(home);
    const Mat& hd = m->head;
    const int N = (int) m->g.n_embd;
    const int nb = N / 32;
    if (!mtp_short_head || !m->has_mtp || m->mtp_head.d || hd.type != 8 || hd.n_in != N ||
        draft_n <= 0 || draft_n >= m->g.n_vocab || draft_n > hd.n_out || draft_n % 16 ||
        N % 32 || nb > 128 || !fast_k) {
        std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_SPLIT_HEAD=1 inactive (needs eligible short Q8 draft head)\n");
        return;
    }
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) hd.type, be, bb) || hd.n_in % be) return;
    const size_t row_bytes = (size_t) hd.n_in / be * bb;
    const int rows = draft_n / 2;
    use_dev(1);
    const cudaError_t alloc = cudaMalloc((void**) &mtp_head_peer, (size_t) rows * row_bytes);
    if (alloc != cudaSuccess) {
        mtp_head_peer = nullptr;
        (void) cudaGetLastError();
        use_dev(home);
        std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_SPLIT_HEAD=1 inactive (rank 1 has insufficient VRAM)\n");
        return;
    }
    CK(cudaMemcpyPeerAsync(mtp_head_peer, 1, hd.d + (size_t) rows * row_bytes, home,
                           (size_t) rows * row_bytes, st));
    CK(cudaStreamSynchronize(st));
    mtp_head_peer_rows = rows;
    use_dev(home);
    std::fprintf(stderr, "strata-q36: STRATA_Q36_MTP_SPLIT_HEAD=1 (%d Q8 draft rows per rank, %.1f MiB copied to rank 1)\n",
                 rows, (double) rows * row_bytes / 1048576.0);
}

void Engine::tp_split_draft_head(int T) {
    const int N = (int) m->g.n_embd, nb = N / 32, rows = mtp_head_peer_rows;
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        const float* src = hmtp + (size_t) (T - 1) * N;
        quant(src, N, 1, q8);
        const uint8_t* weights = d ? mtp_head_peer : m->head.d;
        const uint8_t* packed = nullptr;
        if (!d) for (const auto& p : m->packed)
            if (std::get<0>(p) == m->head.d) packed = std::get<1>(p);
        const half* scales = packed ? reinterpret_cast<const half*>(packed + (size_t) m->head.n_in * m->head.n_out) : nullptr;
        float* out = d ? tp_logits[1] : logits_m;
#define Q36_MTP_SPLIT_ROW(NJ) do { \
    if (packed) q36k::launch_q8_rows<1, NJ, true>(packed, scales, q8, out, nb, rows, st); \
    else q36k::launch_q8_rows<1, NJ, false>(weights, nullptr, q8, out, nb, rows, st); \
} while (0)
        if (nb <= 32) Q36_MTP_SPLIT_ROW(1);
        else if (nb <= 64) Q36_MTP_SPLIT_ROW(2);
        else if (nb <= 96) Q36_MTP_SPLIT_ROW(3);
        else Q36_MTP_SPLIT_ROW(4);
#undef Q36_MTP_SPLIT_ROW
        CK(cudaEventRecord(tp_event[d], st));
    }
    use_dev(home);
    CK(cudaStreamWaitEvent(st, tp_event[1], 0));
    if (tp_check) tp_check_residual(T, true);
    CK(cudaMemcpyPeerAsync(logits_m + rows, home, tp_logits[1], 1, (size_t) rows * sizeof(float), st));
    if (verify_k) {
        float* reference = nullptr;
        CK(cudaMalloc((void**) &reference, (size_t) draft_n * sizeof(float)));
        q36k::launch_q8_rows<1, 4, false>(m->head.d, nullptr, q8, reference, nb, draft_n, st);
        verify_cmp("MTP split draft head", logits_m, reference, (size_t) draft_n);
        CK(cudaFree(reference));
    }
    draft_hooks(draft_n);
    k_argmax<<<dim3(kArgBlocks, 1), kThreads, 0, st>>>(logits_m, draft_n, pick_m, amv, ami, false);
    k_argmax<<<dim3(1, 1), kThreads, 0, st>>>(logits_m, kArgBlocks, pick_m, amv, ami, true);
    draft_picked_prob(draft_n);
}
