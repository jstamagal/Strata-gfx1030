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
        tp_remote[d] = dalloc((size_t)tmax * g.n_embd);
        CK(cudaEventCreateWithFlags(&tp_copy_ready[d], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&tp_copy_done[d], cudaEventDisableTiming));
        tp_logits[d] = dalloc((size_t) kVerify * m->head.n_out);
        CK(cudaEventCreateWithFlags(&tp_event[d], cudaEventDisableTiming));
        CK(cudaHostAlloc((void**)&tp_host[d], (size_t) tmax * g.n_embd * 4, cudaHostAllocPortable | cudaHostAllocMapped));
        CK(cudaHostAlloc((void**)&tp_flags_host[d], q36tp::join_blocks * q36tp::flag_stride * sizeof(unsigned),
                         cudaHostAllocPortable | cudaHostAllocMapped));
        std::memset(tp_flags_host[d], 0, q36tp::join_blocks * q36tp::flag_stride * sizeof(unsigned));
    }
    for_devs([&](int d) {
        for (int r = 0; r < 2; ++r) {
            CK(cudaHostGetDevicePointer((void**)&tp_mapped[d][r], tp_host[r], 0));
            CK(cudaHostGetDevicePointer((void**)&tp_flags[d][r], tp_flags_host[r], 0));
        }
        CK(cudaDeviceSynchronize());
    });
    use_mmq = dev_mmq[0] && dev_mmq[1];
    if (tp_check) std::fprintf(stderr, "strata-q36: STRATA_Q36_TP_CHECK=1: compare both residual replicas at each forward boundary (host readback)\n");
    std::fprintf(stderr, "strata-q36: STRATA_Q36_TP=1: two equal tensor shards, all experts on both GPUs; "
                         "mapped-host decode joins, copy-engine prompt joins, replicated residual, %s; CPU/hybrid experts unsupported\n",
                         tp_graph && !verify_k ? "per-device graphs" : "eager streams");
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
void Engine::tp_moe_end(int T, const int32_t* prow, const float* shared, const float* sgate,
                         const float* wn, float* hout, int sgs) {
    const int N = (int)m->g.n_embd;
    q36tp::partial_moe<<<T, kThreads, 0, st>>>(moe_out, eparts, prow, ewts, (int)m->top_k, shared, sgate, sgs, N);
    xq_ready = false;
    if (T > kVerify) return; // prompt epilogue follows the copy-engine join
    tp_join(moe_out, T);
    if (wn) k_add_rms<<<T, kThreads, 0, st>>>(x, moe_out, wn, hout, N, m->eps);
    else k_add<<<grid1((int64_t)T * N), kThreads, 0, st>>>(x, moe_out, (int64_t)T * N);
}

void Engine::tp_forward(int T, int64_t p0, int heads, bool verify) {
    tp_enter(T);
    if (tp_graph_run(T, p0, verify)) { tp_head(T, heads); return; }
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
    for (int d = 0; d < 2; ++d) {
        use_dev(d);
        if (heads) mm(m->head, (mtp ? hmtp : hmain) + (size_t)(T-heads) * N, tp_logits[d], heads);
        CK(cudaEventRecord(tp_event[d], st));
    }
    use_dev(home);
    CK(cudaStreamWaitEvent(st, tp_event[1], 0));
    if (tp_check) tp_check_residual(T, mtp);
    if (!heads) return;
    float* dst = output ? output : mtp ? logits_m : logits;
    int32_t* pk = picks ? picks : mtp ? pick_m : pick;
    for (int r = 0; r < heads; ++r) {
        CK(cudaMemcpyAsync(dst + (size_t)r * V, tp_logits[0] + (size_t)r * half, (size_t)half * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyPeerAsync(dst + (size_t)r * V + half, home, tp_logits[1] + (size_t)r * half, 1, (size_t)half * 4, st));
    }
    const int sample_v = draft && draft_n > 0 ? std::min(draft_n, V) : V;
    if (draft) draft_hooks(sample_v);
    k_argmax<<<dim3(kArgBlocks, heads), kThreads, 0, st>>>(dst, sample_v, pk, amv, ami, false);
    k_argmax<<<dim3(1, heads), kThreads, 0, st>>>(dst, kArgBlocks, pk, amv, ami, true);
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
        if (need + reserve > fr && !m.packed.empty()) {
            for (const auto& p : m.packed) { K::native_q8_0_packed_unregister(std::get<0>(p)); CK(cudaFree(std::get<1>(p))); }
            m.packed.clear(); m.packed_dev.clear();
            CK(cudaMemGetInfo(&fr, &total));
            std::fprintf(stderr, "strata-q36: TP device %d: released packed dense copies to fit expert shards\n", d);
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
