// Whole-device decode graphs. Call tokens are device memory, so capture/replay cannot reuse a host sequence number.
long long Engine::tp_graph_key(int T, bool verify, bool mtp) const {
    return (((((long long)cur * 2 + (int)rows_mode) * (kVerify+1) + (rows_mode ? rows_G : 0)) *
             (kVerify+1) + T) * 2 + (int)verify) * 2 + (int)mtp;
}
bool Engine::tp_graph_run(int T, int64_t p0, bool verify, bool mtp, const float* seeds) {
    if (!tp_graph || layer_dump.enabled() || verify_k || T > kVerify || map_on || mtab_now ||
        (tp_attn_rows && p0 + T <= kChunkCells * 4)) return false;
    // Main callers always stage MTP seeds here. Other seed addresses use eager execution, never a stale capture.
    use_dev(home);
    if (mtp && seeds != mtp_seed()) return false;
    const long long key = tp_graph_key(T, verify, mtp);
    const int N = (int)m->g.n_embd;
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        cudaGraphExec_t& gx = seg_graphs[key];
        if (gx) continue;
        cudaGraph_t graph = nullptr;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        if (mtp) {
            K::iq_embed_rows(m->embd_type, embd_d, m->embd_row, tok_m, T, N, y, st);
            k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2*N, N, m->eps);
            k_rms_ld<<<T, kThreads, 0, st>>>(d == home ? seeds : h, N, m->mtp_hnorm, mtp_cat+N, 2*N, N, m->eps);
            mm(m->eh_proj, mtp_cat, x, T);
            k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
            layer(m->mtp, m->layers.size(), T, p0, pos0_m, true, false, m->mtp_shnorm, hmtp);
        } else {
            K::iq_embed_rows(m->embd_type, embd_d, m->embd_row, tok, T, N, x, st);
            k_rms<<<T, kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);
            for (size_t l = 0; l < m->layers.size(); ++l) {
                const bool last = l+1 == m->layers.size();
                layer(m->layers[l], l, T, p0, pos0, true, verify,
                      last ? m->out_norm : m->layers[l+1].attn_norm, last ? hmain : h);
            }
        }
        CK(cudaStreamEndCapture(st, &graph));
        CK(cudaGraphInstantiate(&gx, graph, 0));
        CK(cudaGraphDestroy(graph));
    }
    // Capture both first: no running rendezvous kernel can block allocations needed to instantiate its peer.
    for (int d = 0; d < 2; ++d) { use_dev(d); CK(cudaGraphLaunch(seg_graphs[key], st)); }
    return true;
}
