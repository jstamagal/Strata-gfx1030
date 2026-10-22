// MTP uses mirrored eh_proj and hidden seeds, followed by the same two sharded layer joins as the main model.
void Engine::tp_forward_mtp(int T, int64_t p0, const float* seeds, bool draft) {
    dump_begin(T, p0, false, true);
    const auto cost_start = mtp_cost ? Clock::now() : Clock::time_point{};
    tp_enter(T, true, seeds);
    draft_pos = p0 + T;
    const bool split_eh = tp_split_eh_eligible(T);
    if (!mtp_cost && !split_eh && tp_graph_run(T, p0, false, true, seeds)) {
        tp_head(T, draft ? 1 : 0, true, draft);
        return;
    }
    if (tp_launch_stats && !tp_stats_warmup && T <= kVerify) ++tp_eager_calls[2][T];
    const int N = (int)m->g.n_embd;
    if (split_eh) tp_split_eh_proj(T, seeds);
    else for (int d = 0; d < 2; ++d) {
        use_dev(d);
        if (mtp_cost && mtp_input_detail) CK(cudaEventRecord(mtp_input_events[d][0], st));
        const float* local_seeds = d == home ? seeds : h; // tp_enter copied these before h is reused
        K::iq_embed_rows(m->embd_type, embd_d, m->embd_row, tok_m, T, N, y, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(y, d_img, d_map, N);
        if (mtp_fuse_norm) {
            k_mtp_pair_rms<<<dim3(T, 2), kThreads, 0, st>>>(y, local_seeds, m->mtp_enorm,
                                                               m->mtp_hnorm, mtp_cat, N, m->eps);
            if (verify_k) {
                float* ref = nullptr;
                CK(cudaMalloc(&ref, (size_t) T * 2 * N * sizeof(float)));
                k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, ref, 2*N, N, m->eps);
                k_rms_ld<<<T, kThreads, 0, st>>>(local_seeds, N, m->mtp_hnorm, ref+N, 2*N, N, m->eps);
                verify_cmp("MTP paired RMS", mtp_cat, ref, (size_t) T * 2 * N);
                CK(cudaFree(ref));
            }
        } else {
            k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2*N, N, m->eps);
            k_rms_ld<<<T, kThreads, 0, st>>>(local_seeds, N, m->mtp_hnorm, mtp_cat+N, 2*N, N, m->eps);
        }
        if (mtp_cost && mtp_input_detail) CK(cudaEventRecord(mtp_input_events[d][1], st));
        mm(m->eh_proj, mtp_cat, x, T);
        if (mtp_cost && mtp_input_detail) CK(cudaEventRecord(mtp_input_events[d][2], st));
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
        if (mtp_cost && mtp_input_detail) CK(cudaEventRecord(mtp_input_events[d][3], st));
    }
    if (mtp_cost) {
        for_devs([&](int) { CK(cudaStreamSynchronize(st)); });
        if (mtp_input_detail) {
            const int bucket = T == 1 ? 0 : 1;
            ++mtp_input_phase_calls[bucket];
            for (int d = 0; d < 2; ++d) {
                use_dev(d);
                for (int phase = 0; phase < 3; ++phase) {
                    float ms = 0.f;
                    CK(cudaEventElapsedTime(&ms, mtp_input_events[d][phase], mtp_input_events[d][phase+1]));
                    mtp_input_phase_ms[bucket][d][phase] += ms;
                }
            }
            use_dev(home);
        }
        const double t = ms_since(cost_start);
        mtp_input_ms += t;
        if (T == 1) mtp_t1_input_ms += t;
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
        if (T == 1) {
            ++mtp_t1_calls;
            if (draft) { mtp_t1_head_ms += head_ms; ++mtp_t1_head_calls; }
        }
    }
}
