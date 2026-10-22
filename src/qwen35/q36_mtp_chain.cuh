#pragma once

// TP's next MTP input can come directly from the preceding draft pick. Queue
// the remaining drafts, then read all picks and confidence values back once.
// Confidence gates and repetition penalties use the ordinary host path, where
// each candidate must be inspected before deciding whether to draft deeper.
void Engine::device_draft_chain(int D, int64_t p, float* seeds, int last_row,
                                std::vector<int32_t>& drafts) {
    use_dev(home);
    if (!mtp_chain_picks) {
        CK(cudaMalloc((void**) &mtp_chain_picks, (size_t) kVerify * sizeof(int32_t)));
    }
    const int N = (int) m->g.n_embd;
    for (int d = 1; d < D; ++d) hpin[32 + d] = (int32_t) (p + d);
    for (int d = 1; d < D; ++d) {
        CK(cudaMemcpyAsync(tok_m, pick_m, sizeof(int32_t), cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(pos0_m, hpin + 32 + d, sizeof(int32_t), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * sizeof(float),
                           cudaMemcpyDeviceToDevice, st));
        last_row = 0;
        forward_mtp(1, p + d, seeds, true, false);
        CK(cudaMemcpyAsync(mtp_chain_picks + d, pick_m, sizeof(int32_t), cudaMemcpyDeviceToDevice, st));
    }
    CK(cudaMemcpyAsync(hpin + 40, mtp_chain_picks + 1, (size_t) (D - 1) * sizeof(int32_t),
                       cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    for (int d = 1; d < D; ++d) drafts[(size_t) d] = hpin[40 + d - 1];
}
