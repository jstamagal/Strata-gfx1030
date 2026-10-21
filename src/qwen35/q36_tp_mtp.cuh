// MTP uses mirrored eh_proj and hidden seeds, followed by the same two sharded layer joins as the main model.
void Engine::tp_forward_mtp(int T, int64_t p0, const float* seeds, bool draft) {
    tp_enter(T, true, seeds);
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
    for (int phase : {1, 2}) for (int d = 0; d < 2; ++d) {
        use_dev(d);
        layer(m->mtp, m->layers.size(), T, p0, pos0_m, false, false, m->mtp_shnorm, hmtp, phase);
    }
    draft_pos = p0 + T;
    tp_head(T, draft ? 1 : 0, true, draft);
}
