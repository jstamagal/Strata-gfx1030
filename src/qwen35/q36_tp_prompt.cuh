// Large chunks use the copy engine and event ordering. No host readback or per-layer stream synchronization.
void Engine::tp_copy_join(int T, bool moe) {
    const size_t n = (size_t)T * m->g.n_embd;
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        // The peer's previous H2D must finish before this device reuses its pinned send buffer.
        if (tp_copy_used) CK(cudaStreamWaitEvent(st, tp_copy_done[1-d], 0));
        CK(cudaMemcpyAsync(tp_host[d], moe ? moe_out : y, n * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(tp_copy_ready[d], st));
    }
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        CK(cudaStreamWaitEvent(st, tp_copy_ready[1-d], 0));
        CK(cudaMemcpyAsync(tp_remote[d], tp_host[1-d], n * 4, cudaMemcpyHostToDevice, st));
        q36tp::ordered_sum<<<grid1(n), kThreads, 0, st>>>(moe ? moe_out : y, tp_remote[d], n, d);
        CK(cudaEventRecord(tp_copy_done[d], st));
    }
    tp_copy_used = true;
}

void Engine::tp_layer_pair(size_t li, int T, int64_t p0, bool verify, bool mtp) {
#if defined(STRATA_Q36_HIP_LAUNCH_WRAP)
    const bool count_layer = tp_stats_warmup && q36tp_stats::eager && T == 1;
    const long long before0 = count_layer ? q36tp_stats::eager[0] : 0;
    const long long before1 = count_layer ? q36tp_stats::eager[1] : 0;
#endif
    auto run = [&](int phase) {
        for (int i = 0; i < 2; ++i) {
            const int d = tp_join_peer_first && T <= kVerify ? 1 - i : i;
            use_dev(d);
            const bool last = li + 1 == m->layers.size();
            const float* wn = mtp ? m->mtp_shnorm : last ? m->out_norm : m->layers[li+1].attn_norm;
            float* hout = mtp ? hmtp : last ? hmain : h;
            layer(mtp ? m->mtp : m->layers[li], li, T, p0, mtp ? pos0_m : pos0, false, verify, wn, hout, phase);
        }
    };
    const auto mtp_phase_start = mtp && mtp_cost ? Clock::now() : Clock::time_point{};
    run(1);
    Clock::time_point mtp_moe_start{};
    if (mtp && mtp_cost) {
        for_devs([&](int) { CK(cudaStreamSynchronize(st)); });
        const double t = ms_since(mtp_phase_start);
        mtp_mixer_ms += t;
        if (T == 1) mtp_t1_mixer_ms += t;
        mtp_moe_start = Clock::now();
    }
    if (T > kVerify) tp_copy_join(T, false);
    run(2);
    if (mtp && mtp_cost) {
        for_devs([&](int) { CK(cudaStreamSynchronize(st)); });
        const double t = ms_since(mtp_moe_start);
        mtp_moe_ms += t;
        if (T == 1) mtp_t1_moe_ms += t;
    }
    if (T > kVerify) {
        tp_copy_join(T, true);
        for (int d = 0; d < 2; ++d) {
            use_dev(d);
            const bool last = li + 1 == m->layers.size();
            const float* wn = mtp ? m->mtp_shnorm : last ? m->out_norm : m->layers[li+1].attn_norm;
            k_add_rms<<<T, kThreads, 0, st>>>(x, moe_out, wn, mtp ? hmtp : last ? hmain : h, (int)m->g.n_embd, m->eps);
        }
    }
    // A host wait inside rank zero's MoE epilogue would deadlock its join: submit rank one first.
    if (layer_dump.active()) {
        use_dev(home);
        layer_dump.finish(li, x, st);
    }
#if defined(STRATA_Q36_HIP_LAUNCH_WRAP)
    if (count_layer) {
        const int which = tp_stats_baseline ? 0 : 1;
        const int kind = mtp ? 2 : m->layers[li].attn ? 1 : 0;
        ++tp_layer_seen[which][kind];
        tp_layer_kernels[which][kind][0] += q36tp_stats::eager[0] - before0;
        tp_layer_kernels[which][kind][1] += q36tp_stats::eager[1] - before1;
    }
#endif
}
