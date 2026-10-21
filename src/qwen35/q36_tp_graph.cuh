// Whole-device decode graphs. Call tokens are device memory, so capture/replay cannot reuse a host sequence number.
long long Engine::tp_graph_key(int T, bool verify, bool mtp) const {
    return (((((long long)cur * 2 + (int)rows_mode) * (kVerify+1) + (rows_mode ? rows_G : 0)) *
             (kVerify+1) + T) * 2 + (int)verify) * 2 + (int)mtp;
}
bool Engine::tp_graph_run(int T, int64_t p0, bool verify, bool mtp, const float* seeds, bool capture_only) {
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
        if (tp_launch_stats) {
            size_t count = 0;
            CK(cudaGraphGetNodes(graph, nullptr, &count));
#if defined(__HIPCC__)
            std::vector<hipGraphNode_t> nodes(count);
            if (count) CK(hipGraphGetNodes(graph, nodes.data(), &count));
#else
            std::vector<cudaGraphNode_t> nodes(count);
            if (count) CK(cudaGraphGetNodes(graph, nodes.data(), &count));
#endif
            size_t kernels = 0, copies = 0;
            for (auto node : nodes) {
#if defined(__HIPCC__)
                hipGraphNodeType type;
                CK(hipGraphNodeGetType(node, &type));
                kernels += type == hipGraphNodeTypeKernel;
                copies += type == hipGraphNodeTypeMemcpy;
#else
                cudaGraphNodeType type;
                CK(cudaGraphNodeGetType(node, &type));
                kernels += type == cudaGraphNodeTypeKernel;
                copies += type == cudaGraphNodeTypeMemcpy;
#endif
            }
            if (tp_stats_capturing && T == 1) {
                const int saved = tp_stats_conv_ab[d] + tp_stats_gdn_q8[d] + tp_stats_router[d] +
                                  tp_stats_join_norm[d] + tp_stats_moe_norm[d];
                std::fprintf(stderr,
                    "strata-q36 TP_LAUNCH_STATS %s T=1 rank %d: graph kernels %zu active, %zu baseline; "
                    "saved %d (conv+alpha/beta %d, GDN+Q8 %d, router+group %d, mixer join+norm %d, MoE join+norm %d); "
                    "graph copies %zu; T=1 heads=1 forward has %zu explicit kernels including head "
                    "(input and split-head copies run outside graph)\n",
                    mtp ? "MTP" : "main", d, kernels, kernels + saved, saved,
                    tp_stats_conv_ab[d], tp_stats_gdn_q8[d], tp_stats_router[d],
                    tp_stats_join_norm[d], tp_stats_moe_norm[d], copies,
                    kernels + (d == home ? 4 : 2));
            } else {
                std::fprintf(stderr, "strata-q36 TP_LAUNCH_STATS captured %s T=%d verify=%d rank=%d: "
                                     "%zu kernels, %zu copies\n", mtp ? "MTP" : "main", T,
                                     (int) verify, d, kernels, copies);
            }
        }
        CK(cudaGraphInstantiate(&gx, graph, 0));
        CK(cudaGraphDestroy(graph));
    }
    // Capture both first: no running rendezvous kernel can block allocations needed to instantiate its peer.
    if (!capture_only)
        for (int d = 0; d < 2; ++d) { use_dev(d); CK(cudaGraphLaunch(seg_graphs[key], st)); }
    else use_dev(home);
    return true;
}
