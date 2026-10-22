// Included after Engine; TP-only orchestration. The original layer-split driver stays intact.
void Engine::tp_swap(int from, int to) {
    auto swap = [&](TpState& s) {
        std::swap(conv, s.conv); std::swap(S, s.S); std::swap(kc, s.kc); std::swap(vc, s.vc);
        std::swap(v_qkv, s.v_qkv); std::swap(v_hc, s.v_hc);
        std::swap(v_alpha, s.v_alpha); std::swap(v_beta, s.v_beta); std::swap(hlast, s.hlast);
        s.sessions.resize(sess.size());
        for (size_t i = 0; i < sess.size(); ++i) {
            std::swap(sess[i].conv, s.sessions[i].conv); std::swap(sess[i].S, s.sessions[i].S);
            std::swap(sess[i].kc, s.sessions[i].kc); std::swap(sess[i].vc, s.sessions[i].vc);
            std::swap(sess[i].hlast, s.sessions[i].hlast);
        }
    };
    swap(tp_state[from]); swap(tp_state[to]);
    m = tp_models[to];
}

void Engine::tp_init(const Model& a, const Model& b, int context, int chunk, int sessions, bool pictures) {
    tp = true; tp_check = q36tp::enabled("STRATA_Q36_TP_CHECK"); tp_models[0] = &a; tp_models[1] = &b; m = &a;
    n_dev = 2; home = cur_dev = 0; CK(cudaSetDevice(0));
    ctx = context; tmax = std::max(chunk, kVerify); n_sess = std::max(1, std::min(sessions, kVerify)); vision = pictures;
    use_graph = false; // Each rank has its own stream; ordinary single-device capture must never surround TP.
    dev_of.assign(a.layers.size() + 1, 0); devs.resize(2);
    const auto& g = a.g;
    n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
    att_rows = (int) (g.n_head * std::min(tmax, 64));
    CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocPortable));
    if (n_sess > 1) CK(cudaHostAlloc((void**) &hrow, 128 * sizeof(int32_t), cudaHostAllocPortable));
    if (vision) CK(cudaHostAlloc((void**) &h_map, (size_t) tmax * sizeof(int32_t), cudaHostAllocPortable));
    sess.resize(n_sess);
    for (int d = 0; d < 2; ++d) {
        use_dev(d); alloc_dev();
        for (const Layer& L : m->layers) {
            v_qkv.push_back(L.attn ? nullptr : dalloc((size_t) kVerify * g.ssm_conv_channels));
            v_hc.push_back(L.attn ? nullptr : dalloc((size_t) kVerify * g.ssm_conv_channels));
            v_alpha.push_back(L.attn ? nullptr : dalloc((size_t) kVerify * g.ssm_v_heads));
            v_beta.push_back(L.attn ? nullptr : dalloc((size_t) kVerify * g.ssm_v_heads));
        }
        for (Sess& ss : sess) alloc_session(ss);
        conv = sess[cur].conv; S = sess[cur].S; kc = sess[cur].kc; vc = sess[cur].vc; hlast = sess[cur].hlast;
        if (n_sess > 1) {
            const size_t count = conv.size() * n_sess;
            for (int kind = 0; kind < 4; ++kind) {
                std::vector<float*> table(count);
                for (size_t li = 0; li < conv.size(); ++li) for (int s = 0; s < n_sess; ++s) {
                    const Sess& ss = sess[s];
                    table[li * n_sess + s] = (kind == 0 ? ss.conv : kind == 1 ? ss.S : kind == 2 ? ss.kc : ss.vc)[li];
                }
                float*** out = kind == 0 ? &d_conv_tab : kind == 1 ? &d_S_tab : kind == 2 ? &d_kc_tab : &d_vc_tab;
                CK(cudaMalloc((void**)out, count * sizeof(float*)));
                CK(cudaMemcpy(*out, table.data(), count * sizeof(float*), cudaMemcpyHostToDevice));
            }
            d_rcell = dalloc_t<int32_t>(kVerify); d_rslot = dalloc_t<int32_t>(kVerify); d_rdelta = dalloc_t<int32_t>(kVerify);
            d_gslot = dalloc_t<int32_t>(kVerify); d_grow0 = dalloc_t<int32_t>(kVerify); d_gcnt = dalloc_t<int32_t>(kVerify);
        }
        tp_token[d] = dalloc_t<unsigned>(q36tp::join_blocks);
        if (tp_partial_split) tp_partial_done[d] = dalloc_t<unsigned>(kVerify);
        if (tp_fuse2_router) tp_route_done[d] = dalloc_t<unsigned>(1);
        tp_remote[d] = dalloc((size_t)tmax * g.n_embd);
        CK(cudaEventCreateWithFlags(&tp_copy_ready[d], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&tp_copy_done[d], cudaEventDisableTiming));
        tp_logits[d] = dalloc((size_t) kVerify * m->head.n_out);
        CK(cudaEventCreateWithFlags(&tp_event[d], cudaEventDisableTiming));
        if (mtp_split_eh) CK(cudaEventCreateWithFlags(&mtp_eh_done[d], cudaEventDisableTiming));
        if (mtp_input_detail)
            for (auto& ev : mtp_input_events[d]) CK(cudaEventCreate(&ev));
        CK(cudaHostAlloc((void**)&tp_host[d], (size_t) tmax * g.n_embd * 4, cudaHostAllocPortable | cudaHostAllocMapped));
        CK(cudaHostAlloc((void**)&tp_flags_host[d], q36tp::join_blocks * q36tp::flag_stride * sizeof(unsigned),
                         cudaHostAllocPortable | cudaHostAllocMapped));
        std::memset(tp_flags_host[d], 0, q36tp::join_blocks * q36tp::flag_stride * sizeof(unsigned));
        if (tp_join_row && g.n_embd % 32 == 0) {
            const size_t stride = (size_t) g.n_embd + 16;
            CK(cudaHostAlloc((void**)&tp_row_host[d], (size_t) kVerify * stride * sizeof(float),
                             cudaHostAllocPortable | cudaHostAllocMapped));
            for (int row = 0; row < kVerify; ++row) {
                unsigned* flag = (unsigned*)(tp_row_host[d] + (size_t) row * stride);
                flag[0] = flag[1] = 0;
            }
        }
    }
    for_devs([&](int d) {
        for (int r = 0; r < 2; ++r) {
            CK(cudaHostGetDevicePointer((void**)&tp_mapped[d][r], tp_host[r], 0));
            CK(cudaHostGetDevicePointer((void**)&tp_flags[d][r], tp_flags_host[r], 0));
            if (tp_row_host[r]) CK(cudaHostGetDevicePointer((void**)&tp_row_mapped[d][r], tp_row_host[r], 0));
        }
        CK(cudaDeviceSynchronize());
    });
    use_mmq = dev_mmq[0] && dev_mmq[1];
    if (tp_check) std::fprintf(stderr, "strata-q36: STRATA_Q36_TP_CHECK=1: compare both residual replicas at each forward boundary (host readback)\n");
    for (const char* flag : {"STRATA_Q36_TP_ATTN_ROWS", "STRATA_Q36_TP_ATTN_GRAPH_ROWS",
                             "STRATA_Q36_TP_JOIN_ROW", "STRATA_Q36_TP_JOIN_PEER_FIRST", "STRATA_Q36_TP_JOIN_FAST_SUBMIT",
                             "STRATA_Q36_TP_JOIN_MOE_PUBLISH",
                             "STRATA_Q36_TP_JOIN_DEFER_ACK",
                             "STRATA_Q36_TP_PARTIAL_SPLIT",
                             "STRATA_Q36_TP_HEAD_ROWS", "STRATA_Q36_TP_SMALL_ROWS", "STRATA_Q36_TP_LOGITS_PREFETCH",
                             "STRATA_Q36_TP_ROUTER_FENCE",
                             "STRATA_Q36_TP_ROUTER_PAIR",
                             "STRATA_Q36_TP_SMALL_TRACE",
                             "STRATA_Q36_TP_HEAD_Q8", "STRATA_Q36_TP_MOE_JOIN",
                             "STRATA_Q36_TP_FUSE2_CONV_AB", "STRATA_Q36_TP_FUSE2_GDN_Q8", "STRATA_Q36_TP_FUSE2_ROUTER",
                             "STRATA_Q36_TP_FUSE2_JOIN_NORM", "STRATA_Q36_TP_FUSE2_MOE_NORM",
                             "STRATA_Q36_TP_FUSE3_GDN_ALL", "STRATA_Q36_TP_FUSE3_SHARED",
                             "STRATA_Q36_TP_LAUNCH_STATS"})
        if (q36tp::enabled(flag)) std::fprintf(stderr, "strata-q36: %s=1 (experimental TP decode)\n", flag);
    if (q36tp::enabled("STRATA_Q36_TP_SMALL_TRACE")) {
        for (int d = 0; d < 2; ++d) {
            const Model& tm = *tp_models[d];
            auto log_mat = [&](const char* label, const Mat& w, int layer) {
                if (w.type == 8 && w.d)
                    std::fprintf(stderr, "strata-q36 TP_SMALL_TRACE rank %d layer %d %s: n_in=%d n_out=%d, weights=%p\n",
                                 d, layer, label, w.n_in, w.n_out, w.d);
            };
            log_mat("head", tm.head, -1);
            log_mat("mtp_head", tm.mtp_head, -1);
            log_mat("eh_proj", tm.eh_proj, -1);
            auto log_layer = [&](const Layer& L, int li) {
                log_mat("qkv", L.qkv, li); log_mat("zg", L.zg, li); log_mat("ssm_out", L.ssm_out, li);
                log_mat("wq", L.wq, li); log_mat("wk", L.wk, li); log_mat("wv", L.wv, li); log_mat("wo", L.wo, li);
                log_mat("sh_gate", L.sh_gate, li); log_mat("sh_up", L.sh_up, li); log_mat("sh_down", L.sh_down, li);
            };
            for (size_t li = 0; li < tm.layers.size(); ++li) log_layer(tm.layers[li], (int)li);
            if (tm.has_mtp) log_layer(tm.mtp, (int)tm.layers.size());
        }
    }
    if (tp_launch_stats)
        std::fprintf(stderr, "strata-q36 TP_LAUNCH_STATS: T=1 heads=1 submits 4 transfer calls "
                             "(token, position, two vocab halves), or 5 for MTP with peer seed; "
                             "MTP seed staging adds device copies; ordinary TP forward has no host stream wait\n");
    std::fprintf(stderr, "strata-q36: STRATA_Q36_TP=1: two equal tensor shards, all experts on both GPUs; "
                         "mapped-host decode joins, copy-engine prompt joins, replicated residual, %s; CPU/hybrid experts unsupported\n",
                         tp_graph && !verify_k ? "per-device graphs" : "eager streams");
}

void Engine::tp_prepare_launch_stats(int requested_mtp) {
    if (!tp_launch_stats) return;
#if defined(STRATA_Q36_HIP_LAUNCH_WRAP)
    const bool graph_was_on = tp_graph;
    tp_graph = false;
    tp_stats_warmup = true;
    const bool active[7] = {tp_fuse2_conv_ab, tp_fuse2_gdn_q8, tp_fuse2_router,
                            tp_fuse2_join_norm, tp_fuse2_moe_norm, tp_fuse3_gdn_all, tp_fuse3_shared};
    auto set_fusions = [&](bool enabled) {
        tp_fuse2_conv_ab = enabled && active[0];
        tp_fuse2_gdn_q8 = enabled && active[1];
        tp_fuse2_router = enabled && active[2];
        tp_fuse2_join_norm = enabled && active[3];
        tp_fuse2_moe_norm = enabled && active[4];
        tp_fuse3_gdn_all = enabled && active[5];
        tp_fuse3_shared = enabled && active[6];
    };
    auto count_one = [&](bool baseline) {
        tp_stats_baseline = baseline;
        set_fusions(!baseline);
        use_dev(home);
        CK(cudaMemsetAsync(tok, 0, sizeof(int32_t), st));
        CK(cudaMemsetAsync(pos0, 0, sizeof(int32_t), st));
        long long main_count[2] = {};
        q36tp_stats::eager = main_count;
        tp_forward(1, 0, 1, false);
        q36tp_stats::eager = nullptr;
        for_devs([&](int) { CK(cudaStreamSynchronize(st)); });
        std::fprintf(stderr, "strata-q36 TP_LAUNCH_STATS direct eager main T=1 %s: rank 0 %lld kernels, rank 1 %lld "
                             "(includes head; excludes async copies)\n", baseline ? "FUSE2/3 off" : "active",
                             main_count[0], main_count[1]);
        if (m->has_mtp && requested_mtp > 0) {
            use_dev(home);
            CK(cudaMemsetAsync(tok_m, 0, sizeof(int32_t), st));
            CK(cudaMemsetAsync(pos0_m, 0, sizeof(int32_t), st));
            CK(cudaMemsetAsync(mtp_seed(), 0, (size_t)m->g.n_embd * sizeof(float), st));
            long long mtp_count[2] = {};
            q36tp_stats::eager = mtp_count;
            tp_forward_mtp(1, 0, mtp_seed(), true);
            q36tp_stats::eager = nullptr;
            for_devs([&](int) { CK(cudaStreamSynchronize(st)); });
            std::fprintf(stderr, "strata-q36 TP_LAUNCH_STATS direct eager MTP T=1 %s: rank 0 %lld kernels, rank 1 %lld "
                                 "(includes draft head; excludes async copies)\n", baseline ? "FUSE2/3 off" : "active",
                                 mtp_count[0], mtp_count[1]);
        }
        reset();
    };
    count_one(true);
    count_one(false);
    set_fusions(true);
    for (int kind = 0; kind < 3; ++kind) {
        const char* name = kind == 0 ? "GDN" : kind == 1 ? "attention" : "MTP";
        for (int which = 0; which < 2; ++which) if (tp_layer_seen[which][kind])
            std::fprintf(stderr, "strata-q36 TP_LAUNCH_STATS direct T=1 %s layer %s: %d layers; "
                                 "rank 0 %.2f, rank 1 %.2f kernels/layer\n", name,
                         which ? "active" : "FUSE2/3 off", tp_layer_seen[which][kind],
                         (double)tp_layer_kernels[which][kind][0] / tp_layer_seen[which][kind],
                         (double)tp_layer_kernels[which][kind][1] / tp_layer_seen[which][kind]);
    }
    tp_graph = graph_was_on;
    tp_stats_warmup = false;
#else
    std::fprintf(stderr, "strata-q36 TP_LAUNCH_STATS: direct eager counts require a HIP linker with --wrap\n");
#endif
    if (!tp_graph || verify_k) return;
    tp_stats_capturing = true;
    tp_graph_run(1, kChunkCells * 4, false, false, nullptr, true);
    if (m->has_mtp && requested_mtp > 0) {
        tp_stats_conv_ab[0] = tp_stats_conv_ab[1] = 0;
        tp_stats_gdn_q8[0] = tp_stats_gdn_q8[1] = 0;
        tp_stats_gdn_all[0] = tp_stats_gdn_all[1] = 0;
        tp_stats_shared[0] = tp_stats_shared[1] = 0;
        tp_stats_router[0] = tp_stats_router[1] = 0;
        tp_stats_join_norm[0] = tp_stats_join_norm[1] = 0;
        tp_stats_moe_norm[0] = tp_stats_moe_norm[1] = 0;
        tp_graph_run(1, kChunkCells * 4, false, true, mtp_seed(), true);
    }
    tp_stats_capturing = false;
}

// All callers submit inputs on home. A cross-device event orders their copies without a host wait.
void Engine::tp_enter(int T, bool mtp, const float* seeds) {
    use_dev(home);
    CK(cudaEventRecord(tp_event[home], st));
    const auto& peer = devs[1];
    CK(cudaSetDevice(1));
    CK(cudaStreamWaitEvent(peer.st, tp_event[home], 0));
    auto cp = [&](void* dst, const void* src, size_t bytes) {
        if (dst && src && bytes) CK(cudaMemcpyPeerAsync(dst, 1, src, home, bytes, peer.st));
    };
    cp(mtp ? peer.tok_m : peer.tok, mtp ? tok_m : tok, (size_t) T * 4);
    cp(mtp ? peer.pos0_m : peer.pos0, mtp ? pos0_m : pos0, 4);
    if (seeds) cp(peer.h, seeds, (size_t) T * m->g.n_embd * 4);
    if (rows_mode) {
        for (auto pr : {std::make_pair(peer.d_rcell, d_rcell), std::make_pair(peer.d_rslot, d_rslot),
                        std::make_pair(peer.d_rdelta, d_rdelta), std::make_pair(peer.d_gslot, d_gslot),
                        std::make_pair(peer.d_grow0, d_grow0), std::make_pair(peer.d_gcnt, d_gcnt)})
            cp(pr.first, pr.second, (size_t) kVerify * 4);
    }
    if (map_on) cp(peer.d_map, d_map, (size_t) T * 4);
    CK(cudaSetDevice(home));
}

void Engine::tp_join(float* partial, int T) {
    if (T > kVerify) return; // the coordinator submits both copy-engine transfers after producing both partials
    q36tp::join<<<q36tp::join_blocks, kThreads, 0, st>>>(partial, tp_mapped[cur_dev][cur_dev],
        tp_mapped[cur_dev][1-cur_dev], (size_t) T * m->g.n_embd, tp_flags[cur_dev][cur_dev],
        tp_flags[cur_dev][1-cur_dev], tp_token[cur_dev], cur_dev);
}
bool Engine::tp_row_defer_ack(int T, int n) const {
    return tp_join_defer_ack && tp_join_row && tp_join_moe_publish && tp_partial_split &&
           tp_fuse2_join_norm && tp_fuse2_moe_norm && !tp_moe_join && !tp_fuse3_gdn_all &&
           fast_k && !verify_k &&
           T >= 1 && T <= 4 && n % 32 == 0;
}
void Engine::tp_join_norm(float* partial, const float* w, float* hout, int T, bool quantized, bool published) {
    const int N = (int) m->g.n_embd;
    if (tp_join_row && T <= 4 && N % 32 == 0) {
        float* mine = tp_row_mapped[cur_dev][cur_dev];
        const float* peer = tp_row_mapped[cur_dev][1-cur_dev];
        unsigned* token = tp_token[cur_dev];
        const bool defer_ack = tp_row_defer_ack(T, N);
        if (quantized && defer_ack)
            q36tp_row::join_norm<true, false, true><<<T, kThreads, 0, st>>>(
                partial, mine, peer, token, cur_dev, x, w, hout, (q36k::QB*) xq, N, m->eps);
        else if (quantized)
            q36tp_row::join_norm<true, false, false><<<T, kThreads, 0, st>>>(
                partial, mine, peer, token, cur_dev, x, w, hout, (q36k::QB*) xq, N, m->eps);
        else if (published && defer_ack)
            q36tp_row::join_norm<false, true, true><<<T, kThreads, 0, st>>>(
                partial, mine, peer, token, cur_dev, x, w, hout, nullptr, N, m->eps);
        else if (published)
            q36tp_row::join_norm<false, true, false><<<T, kThreads, 0, st>>>(
                partial, mine, peer, token, cur_dev, x, w, hout, nullptr, N, m->eps);
        else
            q36tp_row::join_norm<false, false, false><<<T, kThreads, 0, st>>>(
                partial, mine, peer, token, cur_dev, x, w, hout, nullptr, N, m->eps);
    } else if (quantized)
        q36tp_fuse2::join_norm<true><<<q36tp::join_blocks, kThreads, 0, st>>>(
            partial, tp_mapped[cur_dev][cur_dev], tp_mapped[cur_dev][1-cur_dev], (size_t) T * N,
            tp_flags[cur_dev][cur_dev], tp_flags[cur_dev][1-cur_dev], tp_token[cur_dev], cur_dev,
            x, w, hout, (q36k::QB*) xq, N, m->eps, T);
    else
        q36tp_fuse2::join_norm<false><<<q36tp::join_blocks, kThreads, 0, st>>>(
            partial, tp_mapped[cur_dev][cur_dev], tp_mapped[cur_dev][1-cur_dev], (size_t) T * N,
            tp_flags[cur_dev][cur_dev], tp_flags[cur_dev][1-cur_dev], tp_token[cur_dev], cur_dev,
            x, w, hout, nullptr, N, m->eps, T);
}
void Engine::tp_moe_end(int T, const int32_t* prow, const float* shared, const float* sgate,
                         const float* wn, float* hout, int sgs) {
    const int N = (int)m->g.n_embd;
    const bool fused_norm = tp_fuse2_moe_norm && !tp_moe_join && !verify_k && wn && T <= 4 && N % 32 == 0;
    const bool direct_publish = fused_norm && tp_join_row && tp_join_moe_publish && !tp_gdn_defer_hist[cur_dev];
    const bool defer_ack = direct_publish && tp_row_defer_ack(T, N);
    const bool split_partial = tp_partial_split && T <= 4 && !tp_moe_join && !tp_gdn_defer_hist[cur_dev];
    if (tp_moe_join && T <= kVerify)
        q36tp::moe_join<<<q36tp::join_blocks, kThreads, 0, st>>>(moe_out, eparts, prow, ewts, (int)m->top_k,
            shared, sgate, sgs, N, T, tp_mapped[cur_dev][cur_dev], tp_mapped[cur_dev][1-cur_dev],
            tp_flags[cur_dev][cur_dev], tp_flags[cur_dev][1-cur_dev], tp_token[cur_dev], cur_dev);
    else if (tp_gdn_defer_hist[cur_dev]) {
        q36tp_fuse3::partial_moe_hist<<<T, kThreads, 0, st>>>(moe_out, eparts, prow, ewts, (int)m->top_k,
            shared, sgate, sgs, N, tp_gdn_defer_hist[cur_dev], tp_gdn_defer_qkv[cur_dev],
            tp_gdn_defer_C[cur_dev], 2 * tp_gdn_defer_HK[cur_dev] * (int)m->g.ssm_state_size, T);
        if (verify_k && tp_gdn_ref_alloc[cur_dev]) {
            verify_cmp("fuse3 conv history", tp_gdn_defer_hist[cur_dev], tp_gdn_ref_hist[cur_dev],
                (size_t) 3 * tp_gdn_defer_C[cur_dev]);
            CK(cudaFree(tp_gdn_ref_alloc[cur_dev]));
            tp_gdn_ref_alloc[cur_dev] = tp_gdn_ref_hist[cur_dev] = nullptr;
        }
        tp_gdn_defer_hist[cur_dev] = nullptr;
    } else if (split_partial) {
        const int slices = (N + kThreads - 1) / kThreads;
        if (direct_publish && defer_ack)
            q36tp_partial::combine<true, true><<<dim3(slices, T), kThreads, 0, st>>>(
                moe_out, tp_row_mapped[cur_dev][cur_dev], tp_row_mapped[cur_dev][1-cur_dev], eparts, prow, ewts, (int)m->top_k,
                shared, sgate, sgs, N, tp_token[cur_dev], tp_partial_done[cur_dev]);
        else if (direct_publish)
            q36tp_partial::combine<true, false><<<dim3(slices, T), kThreads, 0, st>>>(
                moe_out, tp_row_mapped[cur_dev][cur_dev], tp_row_mapped[cur_dev][1-cur_dev], eparts, prow, ewts, (int)m->top_k,
                shared, sgate, sgs, N, tp_token[cur_dev], tp_partial_done[cur_dev]);
        else
            q36tp_partial::combine<false, false><<<dim3(slices, T), kThreads, 0, st>>>(
                moe_out, nullptr, nullptr, eparts, prow, ewts, (int)m->top_k,
                shared, sgate, sgs, N, nullptr, nullptr);
    } else if (direct_publish)
        q36tp_row::partial_moe_publish<<<T, kThreads, 0, st>>>(moe_out, tp_row_mapped[cur_dev][cur_dev],
            eparts, prow, ewts, (int)m->top_k, shared, sgate, sgs, N,
            tp_token[cur_dev]);
    else q36tp::partial_moe<<<T, kThreads, 0, st>>>(moe_out, eparts, prow, ewts, (int)m->top_k, shared, sgate, sgs, N);
    if (split_partial && verify_k) {
        float* reference = nullptr;
        CK(cudaMalloc((void**)&reference, (size_t) T * N * sizeof(float)));
        q36tp::partial_moe<<<T, kThreads, 0, st>>>(reference, eparts, prow, ewts, (int)m->top_k, shared, sgate, sgs, N);
        verify_cmp("tp partial split", moe_out, reference, (size_t) T * N);
        CK(cudaFree(reference));
    }
    xq_ready = false;
    if (T > kVerify) return; // prompt epilogue follows the copy-engine join
    if (fused_norm) {
        if (tp_stats_capturing) ++tp_stats_moe_norm[cur_dev];
        tp_join_norm(moe_out, wn, hout, T, false, direct_publish);
        return;
    }
    if (!tp_moe_join) tp_join(moe_out, T);
    if (wn) k_add_rms<<<T, kThreads, 0, st>>>(x, moe_out, wn, hout, N, m->eps);
    else k_add<<<grid1((int64_t)T * N), kThreads, 0, st>>>(x, moe_out, (int64_t)T * N);
}

void Engine::tp_forward(int T, int64_t p0, int heads, bool verify) {
    dump_begin(T, p0, verify);
    tp_enter(T);
    if (tp_graph_run(T, p0, verify)) { tp_head(T, heads); return; }
    if (tp_launch_stats && !tp_stats_warmup && T <= kVerify) ++tp_eager_calls[verify ? 1 : 0][T];
    const int N = (int)m->g.n_embd;
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        K::iq_embed_rows(m->embd_type, embd_d, m->embd_row, tok, T, N, x, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(x, d_img, d_map, N);
        k_rms<<<T, kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);
    }
    for (size_t l = 0; l < m->layers.size(); ++l) tp_layer_pair(l, T, p0, verify);
    tp_head(T, heads);
}

void Engine::tp_head(int T, int heads, bool mtp, bool draft, float* output, int32_t* picks) {
    const int N = (int)m->g.n_embd, V = (int)m->g.n_vocab, half = m->head.n_out;
    if (mtp && draft && heads == 1 && mtp_head_peer) {
        tp_split_draft_head(T);
        return;
    }
    // A limited draft vocabulary entirely on rank zero needs neither rank one's
    // head projection nor a full-vocabulary peer copy. The target still verifies it.
    const bool short_draft = mtp && draft && mtp_short_head && draft_n > 0 && draft_n < V &&
        draft_n <= half && !m->mtp_head.d && m->head.type == 8 && m->head.n_in == N &&
        draft_n % 8 == 0 && N % 32 == 0 && (N / 32 == 64 || N / 32 == 96) && fast_k;
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        if (heads && !short_draft) {
            const float* src = (mtp ? hmtp : hmain) + (size_t)(T-heads) * N;
            const int nb = m->head.n_in / 32;
            if (tp_head_rows && heads == 1 && m->head.type == 8 && m->head.n_in % 32 == 0 && nb <= 128) {
                prep(src, m->head.n_in, 1, q8);
                const uint8_t* pk = nullptr;
                for (const auto& p : m->packed) if (std::get<0>(p) == m->head.d) pk = std::get<1>(p);
                const __half* scales = pk ? (const __half*)(pk + (size_t) m->head.n_in * m->head.n_out) : nullptr;
#define Q36_TP_HEAD_ROW(NJ) do { \
    if (pk) q36k::launch_q8_rows<1, NJ, true>(pk, scales, q8, tp_logits[d], nb, half, st); \
    else q36k::launch_q8_rows<1, NJ, false>(m->head.d, nullptr, q8, tp_logits[d], nb, half, st); \
} while (0)
                if (nb <= 32) Q36_TP_HEAD_ROW(1);
                else if (nb <= 64) Q36_TP_HEAD_ROW(2);
                else if (nb <= 96) Q36_TP_HEAD_ROW(3);
                else Q36_TP_HEAD_ROW(4);
#undef Q36_TP_HEAD_ROW
                decode_check_mat("tp head rows", m->head, q8, tp_logits[d], 1);
            } else if (tp_head_q8 && decode_q8_ok(m->head, heads)) {
                prep(src, m->head.n_in, heads, q8);
                q36decode::project(decode_mat(m->head, tp_logits[d]), {}, {}, q8, {}, heads, st);
                decode_check_mat("tp head q8", m->head, q8, tp_logits[d], heads);
            } else mm(m->head, src, tp_logits[d], heads);
        }
        CK(cudaEventRecord(tp_event[d], st));
    }
    use_dev(home);
    CK(cudaStreamWaitEvent(st, tp_event[1], 0));
    if (tp_check) tp_check_residual(T, mtp);
    if (!heads) return;
    if (short_draft) { head_draft(hmtp + (size_t)(T - 1) * N); return; }
    float* dst = output ? output : mtp ? logits_m : logits;
    int32_t* pk = picks ? picks : mtp ? pick_m : pick;
    for (int r = 0; r < heads; ++r) {
        CK(cudaMemcpyAsync(dst + (size_t)r * V, tp_logits[0] + (size_t)r * half, (size_t)half * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyPeerAsync(dst + (size_t)r * V + half, home, tp_logits[1] + (size_t)r * half, 1, (size_t)half * 4, st));
    }
    const int sample_v = draft && draft_n > 0 ? std::min(draft_n, V) : V;
    if (draft) draft_hooks(sample_v);
    // Sampled verification consumes the complete logits on the CPU. Its argmax
    // result is never read, so the two reduction launches can be omitted.
    if (tp_sampled_verify) return;
    k_argmax<<<dim3(kArgBlocks, heads), kThreads, 0, st>>>(dst, sample_v, pk, amv, ami, false);
    k_argmax<<<dim3(1, heads), kThreads, 0, st>>>(dst, kArgBlocks, pk, amv, ami, true);
    if (draft) draft_picked_prob(sample_v);
}

bool place_experts_tp(Model& a, Model& b, Engine& E, std::string& err) {
    Model* models[2] = {&a, &b};
    // Preflight BOTH cards before allocating any expert, so failure is clean and never starts hybrid offload.
    for (int d = 0; d < 2; ++d) {
        E.use_dev(d);
        Model& m = *models[d];
        size_t need = 0, fr = 0, total = 0;
        for (const Layer& L : m.layers) need += L.stride * m.g.n_expert + kMmqTail;
        if (m.has_mtp) need += m.mtp.stride * m.g.n_expert + kMmqTail;
        CK(cudaMemGetInfo(&fr, &total));
        const size_t reserve = (size_t)256 << 20;
        const bool drop_packed = q36tp::enabled("STRATA_Q36_DROP_PACKED");
        if ((drop_packed || need + reserve > fr) && !m.packed.empty()) {
            for (const auto& p : m.packed) { K::native_q8_0_packed_unregister(std::get<0>(p)); CK(cudaFree(std::get<1>(p))); }
            m.packed.clear(); m.packed_dev.clear();
            CK(cudaMemGetInfo(&fr, &total));
            std::fprintf(stderr, "strata-q36: TP device %d: released packed dense copies (%s)\n", d,
                         drop_packed ? "STRATA_Q36_DROP_PACKED=1" : "to fit expert shards");
        }
        if (need + reserve > fr) {
            err = "TP device " + std::to_string(d) + " needs " + std::to_string(need >> 20) +
                  " MiB for expert shards plus 256 MiB reserve; free " + std::to_string(fr >> 20) +
                  " MiB. CPU/hybrid experts are unsupported in TP; reduce --chunk, --max-context or --batch.";
            E.use_dev(E.home); return false;
        }
    }
    for (int d = 0; d < 2; ++d) {
        E.use_dev(d);
        Model& m = *models[d];
        std::vector<Layer*> layers;
        for (Layer& L : m.layers) layers.push_back(&L);
        if (m.has_mtp) layers.push_back(&m.mtp);
        size_t bytes = 0;
        for (Layer* L : layers) {
            const size_t nb = L->stride * m.g.n_expert + kMmqTail;
            CK(cudaMalloc((void**)&L->blobs, nb)); CK(cudaMemset(L->blobs, 0, nb));
            std::vector<uint8_t> blob(L->stride, 0);
            for (int64_t e = 0; e < m.g.n_expert; ++e) {
                expert_blob(*L, e, blob.data());
                CK(cudaMemcpy(L->blobs + e * L->stride, blob.data(), L->stride, cudaMemcpyHostToDevice));
            }
            L->n_slots = (int)m.g.n_expert; bytes += nb;
        }
        std::fprintf(stderr, "strata-q36: TP device %d: %.2f GiB expert shards, half of every expert resident\n", d, bytes / double(1ull << 30));
    }
    E.use_dev(E.home); return true;
}

// Diagnostic only, at call boundaries: never inserts a host wait between layers.
void Engine::tp_check_residual(int T, bool mtp) {
    CK(cudaStreamSynchronize(st)); // home's event dependency also covers the peer
    const size_t n = (size_t)T * m->g.n_embd;
    std::vector<float> a(n), b(n);
    for (int kind = 0; kind < 2; ++kind) {
        const float* root = kind == 0 ? x : mtp ? hmtp : hmain;
        const float* peer = kind == 0 ? devs[1].x : mtp ? devs[1].hmtp : devs[1].hmain;
        CK(cudaMemcpy(a.data(), root, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaSetDevice(1));
        CK(cudaMemcpy(b.data(), peer, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaSetDevice(home));
        if (std::memcmp(a.data(), b.data(), n * 4)) {
            size_t i = 0;
            while (i < n && std::memcmp(&a[i], &b[i], sizeof(float)) == 0) ++i;
            std::fprintf(stderr, "strata-q36: TP replica mismatch (%s %s, T=%d) at %zu: %.9g / %.9g\n",
                         mtp ? "MTP" : "main", kind ? "hidden" : "residual", T, i, a[i], b[i]);
            std::exit(1);
        }
        ++tp_checks;
    }
}
