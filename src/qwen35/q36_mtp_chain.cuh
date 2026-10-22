#pragma once

// The first MTP draft has already been submitted. Each later T=1 TP MTP graph
// reads its predecessor's device pick directly. All picks and their proposal
// probabilities are returned together; the ordinary confidence/cumulative
// rule then shortens the target window to exactly the same offered prefix.
// Repetition penalties and lookup continuations retain the host path.
int Engine::device_draft_chain(int D, int64_t p, float* seeds, int last_row, float floor,
                               float& cumulative_prob, std::vector<int32_t>& drafts) {
    use_dev(home);
    if (!mtp_chain_picks) {
        CK(cudaMalloc((void**) &mtp_chain_picks, (size_t) kVerify * sizeof(int32_t)));
        CK(cudaMalloc((void**) &mtp_chain_probs, (size_t) kVerify * sizeof(float)));
    }
    const int N = (int) m->g.n_embd;
    const bool need_prob = floor > 0.f || mtp_cum_active;
    CK(cudaMemcpyAsync(mtp_chain_picks, pick_m, sizeof(int32_t), cudaMemcpyDeviceToDevice, st));
    if (need_prob) CK(cudaMemcpyAsync(mtp_chain_probs, d_prob, sizeof(float), cudaMemcpyDeviceToDevice, st));
    for (int d = 1; d < D; ++d) hpin[32 + d] = (int32_t) (p + d);
    for (int d = 1; d < D; ++d) {
        CK(cudaMemcpyAsync(tok_m, pick_m, sizeof(int32_t), cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(pos0_m, hpin + 32 + d, sizeof(int32_t), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * sizeof(float),
                           cudaMemcpyDeviceToDevice, st));
        last_row = 0;
        forward_mtp(1, p + d, seeds, true, false);
        CK(cudaMemcpyAsync(mtp_chain_picks + d, pick_m, sizeof(int32_t), cudaMemcpyDeviceToDevice, st));
        if (need_prob) CK(cudaMemcpyAsync(mtp_chain_probs + d, d_prob, sizeof(float), cudaMemcpyDeviceToDevice, st));
    }
    CK(cudaMemcpyAsync(hpin + 40, mtp_chain_picks, (size_t) D * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
    if (need_prob)
        CK(cudaMemcpyAsync(hpin + 48, mtp_chain_probs, (size_t) D * sizeof(float), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    for (int d = 0; d < D; ++d) {
        drafts[(size_t) d] = hpin[40 + d];
        if (need_prob) {
            float pr;
            std::memcpy(&pr, hpin + 48 + d, sizeof(float));
            cumulative_prob *= pr;
            if (pr < floor || (mtp_cum_active && cumulative_prob < mtp_cum_p)) {
                drafts.resize((size_t) d);
                return d;
            }
        }
    }
    return D;
}
