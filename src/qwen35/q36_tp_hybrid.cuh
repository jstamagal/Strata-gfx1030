// Included after Engine. TP's rank-local half-expert cache uses the layer-split CPU pool and slot tables.
namespace q36tp_profile {
// STRP v1 stores ranked (layer, expert) pairs after its 24-byte header. The table following the pairs is
// redundant for placement. The MTP block has no profile entry and keeps expert-id order.
inline bool profile_order(const char* path, int layers, int experts, std::vector<std::vector<int>>& order,
                          std::string& err) {
    order.assign((size_t) layers + 1, {});
    if (path && *path) {
        FILE* f = std::fopen(path, "rb");
        if (!f) { err = std::string("cannot open TP expert profile: ") + path; return false; }
        uint8_t h[24];
        const bool got = std::fread(h, 1, sizeof(h), f) == sizeof(h);
        auto u32 = [](const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                                                   ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); };
        if (!got || std::memcmp(h, "STRP", 4) || u32(h + 4) != 1 || u32(h + 8) != (uint32_t)layers ||
            u32(h + 12) != (uint32_t)experts || u32(h + 20) > (uint64_t)layers * experts) {
            std::fclose(f); err = "TP expert profile has an invalid STRP v1 header or model shape"; return false;
        }
        const uint32_t count = u32(h + 20);
        std::vector<std::vector<char>> seen((size_t) layers, std::vector<char>((size_t) experts, 0));
        for (uint32_t i = 0; i < count; ++i) {
            uint8_t p[4];
            if (std::fread(p, 1, 4, f) != 4) { std::fclose(f); err = "TP expert profile is truncated"; return false; }
            const int l = p[0] | (p[1] << 8), e = p[2] | (p[3] << 8);
            if (l >= layers || e >= experts || seen[(size_t)l][(size_t)e]) {
                std::fclose(f); err = "TP expert profile has an invalid or duplicate pair"; return false;
            }
            seen[(size_t)l][(size_t)e] = 1;
            order[(size_t)l].push_back(e);
        }
        std::fclose(f);
    }
    for (auto& row : order) {
        std::vector<char> seen((size_t) experts, 0);
        for (int e : row) seen[(size_t)e] = 1;
        for (int e = 0; e < experts; ++e) if (!seen[(size_t)e]) row.push_back(e);
    }
    return true;
}
} // namespace q36tp_profile

bool place_experts_tp(Model& a, Model& b, Engine& E, int threads, std::string& err) {
    Model* models[2] = {&a, &b};
    const int ne = (int)a.g.n_expert;
    // A streamed KV request has already opted into host residency. If the Q4
    // shards do not fit after its bounded KV allocation, use the CPU expert
    // pool rather than refusing the owner's unchanged serve config.
    const bool hybrid_forced = q36tp::enabled("STRATA_Q36_TP_HYBRID");
    const bool hybrid_on = hybrid_forced || (E.kv_int8 && E.kv_resident > 0);
    const char* hybrid_name = hybrid_forced ? "STRATA_Q36_TP_HYBRID=1" : "--kv-resident TP fallback";
    std::vector<std::vector<int>> order;
    const char* profile = hybrid_on ? std::getenv("STRATA_Q36_TP_PROFILE") : nullptr;
    if (!q36tp_profile::profile_order(profile, (int)a.layers.size(), ne, order, err)) return false;
    for (size_t li=0; li<a.layers.size(); ++li)
        if (!a.layers[li].initial_rank.empty()) order[li] = a.layers[li].initial_rank;
    if (a.has_mtp && !a.mtp.initial_rank.empty()) order[a.layers.size()] = a.mtp.initial_rank;
    struct Budget { int main = 0, mtp = 0; size_t free = 0, bytes = 0; } budget[2];
    const size_t reserve = (size_t)(hybrid_on ? 768 : 256) << 20;
    size_t host_bytes = 0;
    bool any_hybrid = false;
    // Budget each card after its dense tensors, KV state and work buffers have already been allocated.
    for (int d = 0; d < 2; ++d) {
        E.use_dev(d);
        Model& m = *models[d];
        size_t fr = 0, total = 0, all = 0, main_slot = 0;
        CK(cudaMemGetInfo(&fr, &total));
        for (const Layer& L : m.layers) { all += L.stride * (size_t)ne + kMmqTail; main_slot += L.stride; }
        const size_t mtp_full = m.has_mtp ? m.mtp.stride * (size_t)ne + kMmqTail : 0;
        all += mtp_full;
        const bool drop = q36tp::enabled("STRATA_Q36_DROP_PACKED");
        if ((drop || all + reserve > fr) && !m.packed.empty()) {
            for (const auto& p : m.packed) {
                K::native_q8_0_packed_unregister(std::get<0>(p)); CK(cudaFree(std::get<1>(p)));
            }
            m.packed.clear(); m.packed_dev.clear();
            CK(cudaMemGetInfo(&fr, &total));
            std::fprintf(stderr, "strata-q36: TP rank %d released packed dense copies\n", d);
        }
        budget[d].free = fr;
        if (all + reserve <= fr) { budget[d].main = ne; budget[d].mtp = m.has_mtp ? ne : 0; continue; }
        if (!hybrid_on) {
            err = "TP rank " + std::to_string(d) + " needs " + std::to_string(all >> 20) +
                  " MiB of half-experts; free " + std::to_string(fr >> 20) +
                  " MiB. Set STRATA_Q36_TP_HYBRID=1 to place the rest in host memory.";
            E.use_dev(E.home); return false;
        }
#if !defined(STRATA_Q36_CPU_POOL)
        err = "STRATA_Q36_TP_HYBRID=1 requires the CPU expert pool (build with STRATA_NATIVE_EXPERTS=ON)";
        E.use_dev(E.home); return false;
#else
        any_hybrid = true;
        const size_t work = (size_t)E.tmax * m.top_k * m.g.n_embd * sizeof(float) +
                            (size_t)E.tmax * m.top_k * sizeof(int32_t);
        // Drafts visit the MTP block repeatedly; keep it resident when the rank can afford it.
        const bool mtp_whole = mtp_full && fr > reserve + work + mtp_full;
        const size_t room = fr > reserve + work + (mtp_whole ? mtp_full : 0) ?
                            fr - reserve - work - (mtp_whole ? mtp_full : 0) : 0;
        const size_t cost = main_slot + (mtp_whole ? 0 : m.has_mtp ? m.mtp.stride : 0);
        const size_t tails = (m.layers.size() + (m.has_mtp && !mtp_whole ? 1 : 0)) * kMmqTail;
        const int slots = (int)std::min<size_t>((size_t)ne, room > tails ? (room - tails) / std::max<size_t>(cost, 1) : 0);
        budget[d].main = slots;
        budget[d].mtp = m.has_mtp ? (mtp_whole ? ne : slots) : 0;
#endif
    }
    for (int d = 0; d < 2; ++d) {
        Model& m = *models[d];
        size_t rank_host = 0;
        for (const Layer& L : m.layers) rank_host += L.stride * (size_t)(ne - budget[d].main) +
                                                   (budget[d].main < ne ? kMmqTail : 0);
        if (m.has_mtp && budget[d].mtp < ne)
            rank_host += m.mtp.stride * (size_t)(ne - budget[d].mtp) + kMmqTail;
        host_bytes += rank_host;
    }
    uint8_t* arena = nullptr;
    if (host_bytes && cudaHostAlloc((void**)&arena, host_bytes, cudaHostAllocPortable) != cudaSuccess) {
        err = "cannot allocate " + std::to_string(host_bytes >> 20) + " MiB of pinned TP half-experts";
        E.use_dev(E.home); return false;
    }
    size_t host_off = 0;
    for (int d = 0; d < 2; ++d) {
        E.use_dev(d);
        Model& m = *models[d];
        E.hy.resize(m.layers.size() + (m.has_mtp ? 1 : 0));
        std::vector<Layer*> layers;
        for (Layer& L : m.layers) layers.push_back(&L);
        if (m.has_mtp) layers.push_back(&m.mtp);
        for (size_t li = 0; li < layers.size(); ++li) {
            Layer& L = *layers[li];
            Engine::Hy& H = E.hy[li];
            const int S = li == m.layers.size() ? budget[d].mtp : budget[d].main;
            const auto& rank = order[li];
            L.n_slots = S;
            const size_t nb = L.stride * (size_t)S + kMmqTail;
            CK(cudaMalloc((void**)&L.blobs, nb)); CK(cudaMemset(L.blobs, 0, nb));
            budget[d].bytes += nb;
            H.slot_of.assign((size_t)ne, -1);
            H.exp_in_slot.resize((size_t)S);
            H.hpos.assign((size_t)ne, -1);
            H.host_exp.resize((size_t)(ne - S));
            H.score.assign((size_t)ne, 0.0);
            if (S < ne) {
                L.hblobs = arena + host_off;
                host_off += L.stride * (size_t)(ne - S) + kMmqTail;
#if defined(STRATA_Q36_CPU_POOL)
                if (!strata::kernels::cpu::native_fmt(L.L.gu_type, L.L.d_type, m.g.n_embd, m.g.n_ff, H.fmt, err)) return false;
                if (H.fmt.up_off != L.L.up_off || H.fmt.down_off != L.L.down_off || H.fmt.bytes != L.L.bytes) {
                    err = "TP CPU and GPU half-expert layouts differ"; return false;
                }
#endif
            }
            std::vector<uint8_t> blob(L.stride, 0);
            for (int s = 0; s < ne; ++s) {
                const int e = S == ne ? s : rank[(size_t)s]; // full layers use the direct expert-id addressing path
                if (L.cache) cached_experts(L, e, 1, blob.data());
                else expert_blob(L, e, blob.data());
                if (s < S) {
                    H.slot_of[(size_t)e] = s; H.exp_in_slot[(size_t)s] = e;
                    CK(cudaMemcpy(L.blobs + (size_t)s * L.stride, blob.data(), L.stride, cudaMemcpyHostToDevice));
                } else {
                    const int h = s - S;
                    H.hpos[(size_t)e] = h; H.host_exp[(size_t)h] = e;
                    std::memcpy(L.hblobs + (size_t)h * L.stride, blob.data(), L.stride);
                }
            }
            if (S < ne) {
                CK(cudaMalloc(&H.slot_of_d, (size_t)ne * 4));
                CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t)ne * 4, cudaMemcpyHostToDevice));
            }
            if (hybrid_on)
                std::fprintf(stderr, "strata-q36: TP rank %d layer %zu: %d/%d half-experts in VRAM\n", d, li, S, ne);
        }
        if (budget[d].main < ne || (m.has_mtp && budget[d].mtp < ne)) {
            const size_t entries = (size_t)E.tmax * m.top_k;
            CK(cudaMalloc(&E.ecpu, entries * m.g.n_embd * 4));
            CK(cudaMalloc(&E.d_miss, entries * 4));
            CK(cudaEventCreateWithFlags(&E.ev_route, cudaEventDisableTiming));
        }
        if (hybrid_on)
            std::fprintf(stderr, "strata-q36: TP rank %d: %.2f GiB expert shards in VRAM, %.2f GiB free before placement\n",
                         d, budget[d].bytes / double(1ull << 30), budget[d].free / double(1ull << 30));
        else
            std::fprintf(stderr, "strata-q36: TP device %d: %.2f GiB expert shards, half of every expert resident\n",
                         d, budget[d].bytes / double(1ull << 30));
    }
    E.use_dev(E.home);
    if (any_hybrid) {
#if defined(STRATA_Q36_CPU_POOL)
        const size_t entries = (size_t)E.tmax * a.top_k, N = (size_t)a.g.n_embd;
        CK(cudaHostAlloc((void**)&E.h_h, (size_t)E.tmax * N * 4, cudaHostAllocPortable | cudaHostAllocMapped));
        CK(cudaHostAlloc((void**)&E.cpu_out, entries * N * 4, cudaHostAllocPortable | cudaHostAllocMapped));
        CK(cudaHostAlloc((void**)&E.h_ids, entries * 4, cudaHostAllocPortable | cudaHostAllocMapped));
        CK(cudaHostAlloc((void**)&E.h_miss, entries * 4, cudaHostAllocPortable | cudaHostAllocMapped));
        E.acts = (uint8_t*)std::malloc((size_t)E.tmax * strata::kernels::cpu::kNativeActBytes + 64);
        size_t wide = 0;
        for (const Model* m : models) {
            for (const Layer& L : m->layers) wide = std::max(wide, L.stride);
            if (m->has_mtp) wide = std::max(wide, m->mtp.stride);
        }
        E.bounce_bytes = wide * 64;
        CK(cudaHostAlloc((void**)&E.bounce, E.bounce_bytes, cudaHostAllocPortable));
        E.pool = std::make_unique<strata::kernels::cpu::ExpertPool>(threads);
        E.hybrid_t = E.tmax; // prompt chunks also use resident GPU experts plus the CPU pool
        E.main_hybrid = budget[0].main < ne || budget[1].main < ne;
        E.tp_graph = false;
        a.hybrid = b.hybrid = true;
        std::fprintf(stderr, "strata-q36: %s: %.2f GiB pinned host half-experts, "
                             "%d CPU threads; eager TP layers\n", hybrid_name, host_bytes / double(1ull << 30),
                     E.pool->workers() + (E.pool->host_works() ? 1 : 0));
        if (profile && *profile) std::fprintf(stderr, "strata-q36: STRATA_Q36_TP_PROFILE=%s (STRP v1 initial slots)\n", profile);
#endif
    }
    else if (hybrid_on) {
        std::fprintf(stderr, "strata-q36: %s: all half-experts fit in VRAM; 0 GiB host experts\n", hybrid_name);
    }
    return true;
}
