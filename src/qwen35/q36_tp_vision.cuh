// Image allocations can grow between requests. Keep their addresses OUT of cached
// graphs: stage x (main) / y (MTP) first, then capture from normalization onward.
// d_mtab is allocated once per rank at init; its contents and the device position
// scalar change each request. use_dev remaps mtab_now to the local table.
// tp_enter has already ordered the peer's row map, tokens, position and MTP seeds.
void Engine::tp_picture_input(int T, bool mtp) {
    const int N = (int)m->g.n_embd;
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        float* out = mtp ? y : x;
        K::iq_embed_rows(m->embd_type, embd_d, m->embd_row, mtp ? tok_m : tok, T, N, out, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(out, d_img, d_map, N);
    }
    use_dev(home);
}
