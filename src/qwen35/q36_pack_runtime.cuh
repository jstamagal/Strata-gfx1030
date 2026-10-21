// Included after Model/expert_blob: q36-only placement hooks, kept out of the engine driver.
void prepare_expert_storage(Model& m, const std::string& pack, const std::string& profile) {
    const size_t nl = m.layers.size() + size_t(m.has_mtp), ne = size_t(m.g.n_expert);
    // Artifacts include the model's MTP blocks even when this run disables MTP.
    const size_t file_nl = size_t(m.g.n_layers + m.g.n_mtp_layers);
    std::vector<std::vector<int>> ranks;
    if (!profile.empty()) ranks = q36pack::profile(profile, file_nl, ne);
    if (!pack.empty()) {
        m.expert_cache = std::make_shared<q36pack::Cache>();
        std::vector<std::string> sources;
        for (size_t s=0; s<m.file->size(); ++s) sources.push_back(m.file->shard(s).path());
        m.expert_cache->open(pack, sources, file_nl, ne, size_t(m.g.n_embd), size_t(m.g.n_ff));
    }
    for (size_t li=0; li<nl; ++li) {
        Layer& L = li<m.layers.size() ? m.layers[li] : m.mtp;
        L.cache_layer=li;
        if (!ranks.empty()) L.initial_rank=std::move(ranks[li]);
        if (m.expert_cache) {
            const auto& x=m.expert_cache->layers.at(li);
            q36pack::require(x.gu==L.L.gu_type && x.dt==L.L.d_type && x.stride==L.stride &&
                             x.up==L.L.up_off && x.down==L.L.down_off && x.bytes==L.L.bytes,
                             "cached expert layout mismatch at layer " + std::to_string(li));
            L.cache=m.expert_cache.get();
        }
    }
    if (!pack.empty()) std::fprintf(stderr,"strata-q36: --pack %s: contiguous cached expert reads enabled\n",pack.c_str());
    if (!profile.empty()) std::fprintf(stderr,"strata-q36: --expert-profile %s: ranked startup slots; rebalancer unchanged\n",profile.c_str());
}
// Only hybrid layers need a permutation. Fully resident kernels index experts by their actual ID.
void select_initial_order(Layer& L, int slots, int64_t ne) {
    L.use_initial_rank = slots < ne && !L.initial_rank.empty();
}
void cached_experts(const Layer& L, size_t first, size_t count, uint8_t* dst) {
    size_t i=0;
    while (i<count) {
        const size_t e=initial_expert(L, first+i);
        size_t n=1;
        while (i+n<count && initial_expert(L,first+i+n)==int(e+n)) ++n;
        L.cache->read(L.cache_layer,e,n,dst+i*L.stride);
        i+=n;
    }
}
void upload_initial_experts(const Layer& L, size_t count) {
    if (L.cache) {
        // One contiguous host staging allocation and one H2D copy for the layer's slots.
        uint8_t* stage=nullptr;
        CK(cudaHostAlloc(reinterpret_cast<void**>(&stage),count*L.stride,cudaHostAllocPortable));
        cached_experts(L,0,count,stage);
        CK(cudaMemcpy(L.blobs,stage,count*L.stride,cudaMemcpyHostToDevice));
        CK(cudaFreeHost(stage));
    } else {
        std::vector<uint8_t> blob(L.L.bytes,0);
        for (size_t e=0;e<count;++e) {
            expert_blob(L,e,blob.data());
            CK(cudaMemcpy(L.blobs+e*L.stride,blob.data(),L.L.bytes,cudaMemcpyHostToDevice));
        }
    }
}
template<class Hy> void initial_tables(const Layer& L, Hy& H, size_t ne, int slots) {
    H.slot_of.assign(ne,-1); H.hpos.assign(ne,-1);
    H.exp_in_slot.resize(slots); H.host_exp.resize(ne-slots);
    for (size_t i=0;i<ne;++i) {
        const int e=initial_expert(L,i);
        if (i<size_t(slots)) { H.slot_of[e]=int(i); H.exp_in_slot[i]=e; }
        else { H.hpos[e]=int(i)-slots; H.host_exp[i-slots]=e; }
    }
    H.score.assign(ne,0.0); // No invented prior counts: online routing still controls rebalancing.
}
inline void trace_experts(q36pack::Trace& trace, size_t li, const int32_t* ids, size_t n, cudaStream_t st) {
    std::vector<int32_t> host(n);
    CK(cudaMemcpyAsync(host.data(),ids,n*sizeof(int32_t),cudaMemcpyDeviceToHost,st));
    CK(cudaStreamSynchronize(st));
    trace.add(li,host.data(),n);
}
