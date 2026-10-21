// MTP uses mirrored eh_proj and hidden seeds, followed by the same two sharded layer joins as the main model.
void Engine::tp_forward_mtp(int T, int64_t p0, const float* seeds, bool draft) {
    dump_begin(T, p0, false, true);
    const auto cost_start = mtp_cost ? Clock::now() : Clock::time_point{};
    tp_enter(T, true, seeds);
    draft_pos = p0 + T;
    if (!mtp_cost && tp_graph_run(T, p0, false, true, seeds)) { tp_head(T, draft ? 1 : 0, true, draft); return; }
    if (tp_launch_stats && !tp_stats_warmup && T <= kVerify) ++tp_eager_calls[2][T];
    const int N = (int)m->g.n_embd;
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        const float* local_seeds = d == home ? seeds : h; // tp_enter copied these before h is reused
        K::iq_embed_rows(m->embd_type, embd_d, m->embd_row, tok_m, T, N, y, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(y, d_img, d_map, N);
        k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2*N, N, m->eps);
        k_rms_ld<<<T, kThreads, 0, st>>>(local_seeds, N, m->mtp_hnorm, mtp_cat+N, 2*N, N, m->eps);
        mm(m->eh_proj, mtp_cat, x, T);
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
    }
    tp_layer_pair(m->layers.size(), T, p0, false, true);
    draft_pos = p0 + T;
    Clock::time_point head_start{};
    if (mtp_cost) {
        for_devs([&](int) { CK(cudaStreamSynchronize(st)); });
        head_start = Clock::now();
    }
    tp_head(T, draft ? 1 : 0, true, draft);
    if (mtp_cost) {
        CK(cudaStreamSynchronize(st));
        const double head_ms = ms_since(head_start);
        mtp_block_ms += std::chrono::duration<double, std::milli>(head_start - cost_start).count() +
                        (draft ? 0. : head_ms);
        if (draft) { mtp_head_ms += head_ms; ++mtp_head_calls; }
        ++mtp_cost_calls;
    }
}
