#pragma once

// The MTP input projection is mirrored in TP. For a small multi-row MTP call,
// compute half its output rows on each card using the same RDNA2 Q8 dot kernel,
// then exchange the small FP32 result. T=1 keeps the existing graph path.
bool Engine::tp_split_eh_eligible(int T) const {
    return mtp_split_eh && tp && T >= 2 && T <= 4 && !map_on && rdna2_mmvq &&
           decode_rdna2.size() == 2 && decode_rdna2[0] && decode_rdna2[1] &&
           m->eh_proj.type == 8 && m->eh_proj.n_in == 2 * m->g.n_embd &&
           m->eh_proj.n_out == m->g.n_embd && m->g.n_embd % 2 == 0 &&
           m->eh_proj.n_in % 32 == 0;
}

bool Engine::tp_split_eh_proj(int T, const float* seeds) {
    const int N = (int) m->g.n_embd, half = N / 2;
    float* compact[2] = {};
    float* result[2] = {};
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        if (mtp_cost && mtp_input_detail) CK(cudaEventRecord(mtp_input_events[d][0], st));
        const float* local_seeds = d == home ? seeds : h;
        K::iq_embed_rows(m->embd_type, embd_d, m->embd_row, tok_m, T, N, y, st);
        if (mtp_fuse_norm)
            k_mtp_pair_rms<<<dim3(T, 2), kThreads, 0, st>>>(y, local_seeds, m->mtp_enorm,
                                                               m->mtp_hnorm, mtp_cat, N, m->eps);
        else {
            k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2*N, N, m->eps);
            k_rms_ld<<<T, kThreads, 0, st>>>(local_seeds, N, m->mtp_hnorm, mtp_cat+N, 2*N, N, m->eps);
        }
        if (mtp_cost && mtp_input_detail) CK(cudaEventRecord(mtp_input_events[d][1], st));
        quant(mtp_cat, 2 * N, T, q8);
        q36decode::Q8Mat w = decode_mat(m->eh_proj, y);
        const size_t blocks = (size_t) d * half * w.nb;
        w.w += blocks * (w.scales ? 32 : 34);
        if (w.scales) w.scales += blocks;
        w.rows = half;
        w.tiles = std::min((half + 3) / 4, decode_tiles[d]);
        (void) q36decode::project(w, {}, {}, q8, {}, T, st);
        compact[d] = y;
        result[d] = x;
        CK(cudaEventRecord(tp_event[d], st));
    }
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        CK(cudaStreamWaitEvent(st, tp_event[1-d], 0));
        for (int r = 0; r < T; ++r) {
            CK(cudaMemcpyAsync(result[d] + (size_t) r * N + (size_t) d * half,
                               compact[d] + (size_t) r * half, (size_t) half * sizeof(float),
                               cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyPeerAsync(result[d] + (size_t) r * N + (size_t) (1-d) * half, d,
                                   compact[1-d] + (size_t) r * half, 1-d,
                                   (size_t) half * sizeof(float), st));
        }
        CK(cudaEventRecord(mtp_eh_done[d], st));
    }
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        CK(cudaStreamWaitEvent(st, mtp_eh_done[1-d], 0));
        if (mtp_cost && mtp_input_detail) CK(cudaEventRecord(mtp_input_events[d][2], st));
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
        if (mtp_cost && mtp_input_detail) CK(cudaEventRecord(mtp_input_events[d][3], st));
    }
    use_dev(home);
    if (verify_k) {
        float* reference = nullptr;
        CK(cudaMalloc((void**) &reference, (size_t) T * N * sizeof(float)));
        mm(m->eh_proj, mtp_cat, reference, T);
        verify_cmp("MTP split eh_proj", x, reference, (size_t) T * N);
        CK(cudaFree(reference));
    }
    return true;
}
