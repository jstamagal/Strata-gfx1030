// Members of q36 Engine. Kept separate to minimize conflicts between lanes.
bool rdna2_mmvq = q36decode::enabled("STRATA_Q36_RDNA2_MMVQ");
bool moe_out_wide = q36decode::enabled("STRATA_Q36_MOE_OUT_WIDE");
bool decode_fuse = q36decode::enabled("STRATA_Q36_FUSE");
bool tpx_q8_t1_fuse = q36decode::enabled("STRATA_Q36_TPX_Q8_T1_FUSE");
bool expert_fuse = q36decode::enabled("STRATA_Q36_FUSE_EXPERT");
std::vector<bool> decode_rdna2;
std::vector<size_t> decode_lds;
std::vector<int> decode_tiles;
void decode_init() {
    if (!(rdna2_mmvq || moe_out_wide || decode_fuse || expert_fuse || tp_head_q8 || tp_fuse3_shared || (tp && tpx_q8_t1_fuse))) return;
    for (const char* flag : {"STRATA_Q36_RDNA2_MMVQ", "STRATA_Q36_MOE_OUT_WIDE", "STRATA_Q36_FUSE", "STRATA_Q36_FUSE_EXPERT"})
        if (q36decode::enabled(flag)) std::fprintf(stderr, "strata-q36: %s=1 (experimental, verify on device)\n", flag);
    decode_rdna2.resize(n_dev, false);
    decode_lds.resize(n_dev, 0);
    decode_tiles.resize(n_dev, 1);
    for (int d = 0; d < n_dev; ++d) {
        cudaDeviceProp p{};
        CK(cudaGetDeviceProperties(&p, d));
        decode_tiles[d] = std::max(1, p.multiProcessorCount * 8);
#if defined(__HIPCC__)
        decode_rdna2[d] = std::strncmp(p.gcnArchName, "gfx1030", 7) == 0 && (p.gcnArchName[7] == '\0' || p.gcnArchName[7] == ':');
#endif
        if (p.maxThreadsPerBlock >= 1024) decode_lds[d] = p.sharedMemPerBlock;
        std::fprintf(stderr, "strata-q36: decode device %d: RDNA2 kernels %s, MoE LDS budget %zu bytes\n",
                     d, decode_rdna2[d] ? "available" : "fallback", decode_lds[d]);
    }
}
bool decode_gpu() const { return !decode_rdna2.empty() && decode_rdna2[cur_dev]; }
bool decode_q8_ok(const Mat& w, int T) const {
    return decode_gpu() && T >= 1 && T <= 4 && w.type == 8 && w.n_in > 0 && w.n_in % 32 == 0 && w.n_out > 0;
}
q36decode::Q8Mat decode_mat(const Mat& w, float* out) const {
    q36decode::Q8Mat a{w.d, nullptr, out, w.n_in / 32, w.n_out, std::min((w.n_out + 3) / 4, decode_tiles[cur_dev])};
    // Look up the actual surviving copy: placement may have released the packed
    // weights to make room for experts. No stale pointer cached in a Mat.
    for (const auto& p : m->packed) if (std::get<0>(p) == w.d) {
        a.w = std::get<1>(p);
        a.scales = (const half*) (a.w + (size_t) w.n_in * w.n_out);
        break;
    }
    return a;
}
void decode_check_mat(const char* name, const Mat& w, const void* qin, const float* result, int T) {
    if (!verify_k) return;
    float* ref = nullptr;
    CK(cudaMalloc((void**) &ref, (size_t) T * w.n_out * sizeof(float)));
    mmq_ref(w, qin, ref, T);
    verify_cmp(name, result, ref, (size_t) T * w.n_out,
               " K " + std::to_string(w.n_in) + " M " + std::to_string(w.n_out) + " T " + std::to_string(T));
    CK(cudaFree(ref));
}
bool decode_mmvq(const Mat& w, const void* qin, float* out, int T) {
    if (!rdna2_mmvq || !decode_q8_ok(w, T)) return false;
    q36decode::project(decode_mat(w, out), {}, {}, qin, {}, T, st);
    decode_check_mat("rdna2 q8", w, qin, out, T);
    return true;
}
bool decode_projections(const Layer& L, int T, float* qkv_b = nullptr, float* al_b = nullptr, float* be_b = nullptr) {
    if (!decode_fuse && !(tp && tpx_q8_t1_fuse && T == 1)) return false;
    if (L.attn) {
        if (!decode_q8_ok(L.wq, T) || !decode_q8_ok(L.wk, T) || !decode_q8_ok(L.wv, T)) return false;
        q36tpx::dense_trace(L.wq.name, L.wq.type, L.wq.n_in, L.wq.n_out, "fused_qkv", T, L.wq.d);
        q36tpx::dense_trace(L.wk.name, L.wk.type, L.wk.n_in, L.wk.n_out, "fused_qkv", T, L.wk.d);
        q36tpx::dense_trace(L.wv.name, L.wv.type, L.wv.n_in, L.wv.n_out, "fused_qkv", T, L.wv.d);
        q36decode::project(decode_mat(L.wq, qg), decode_mat(L.wk, kx), decode_mat(L.wv, vx), xq, {}, T, st);
        decode_check_mat("fuse Q", L.wq, xq, qg, T);
        decode_check_mat("fuse K", L.wk, xq, kx, T);
        decode_check_mat("fuse V", L.wv, xq, vx, T);
    } else {
        if (!decode_q8_ok(L.qkv, T) || !decode_q8_ok(L.zg, T)) return false;
        q36tpx::dense_trace(L.qkv.name, L.qkv.type, L.qkv.n_in, L.qkv.n_out, "fused_qkv", T, L.qkv.d);
        q36tpx::dense_trace(L.zg.name, L.zg.type, L.zg.n_in, L.zg.n_out, "fused_qkv", T, L.zg.d);
        q36decode::AB ab{L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, (int) m->g.n_embd, (int) m->g.ssm_v_heads};
        q36decode::project(decode_mat(L.qkv, qkv_b), decode_mat(L.zg, z), {}, xq, ab, T, st);
        decode_check_mat("fuse QKV", L.qkv, xq, qkv_b, T);
        decode_check_mat("fuse Z", L.zg, xq, z, T);
        if (verify_k) {
            float* ref = nullptr;
            const size_t n = (size_t) T * ab.heads;
            CK(cudaMalloc((void**) &ref, 2 * n * sizeof(float)));
            k_gdn_ab<<<dim3(rows8(2 * ab.heads), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a,
                ref, ref + n, ab.n, ab.heads);
            verify_cmp("fuse alpha", al_b, ref, n);
            verify_cmp("fuse beta", be_b, ref + n, n);
            CK(cudaFree(ref));
        }
    }
    return true;
}
bool decode_shared(const Layer& L, int T) {
    if (!(decode_fuse || (tp && tp_fuse3_shared)) || !decode_q8_ok(L.sh_gate, T) || !decode_q8_ok(L.sh_up, T) ||
        L.sh_gate.n_in != L.sh_up.n_in || L.sh_gate.n_out != L.sh_up.n_out || L.sh_gate.n_out % 32) return false;
    if (tp_stats_capturing && tp && tp_fuse3_shared && !decode_fuse) tp_stats_shared[cur_dev] += 2;
    q36tpx::dense_trace(L.sh_gate.name, L.sh_gate.type, L.sh_gate.n_in, L.sh_gate.n_out, "fused_shared", T, L.sh_gate.d);
    q36tpx::dense_trace(L.sh_up.name, L.sh_up.type, L.sh_up.n_in, L.sh_up.n_out, "fused_shared", T, L.sh_up.d);
    q36decode::shared(decode_mat(L.sh_gate, nullptr), decode_mat(L.sh_up, nullptr), xq, q8, T, st);
    if (verify_k) {
        float* ref = nullptr;
        void* qr = nullptr;
        const int FS = L.sh_gate.n_out;
        const size_t n = (size_t) T * FS, nb = K::native_q8_1_bytes(FS, T);
        CK(cudaMalloc((void**) &ref, 2 * n * sizeof(float)));
        CK(cudaMalloc(&qr, nb));
        mmq_ref(L.sh_gate, xq, ref, T);
        mmq_ref(L.sh_up, xq, ref + n, T);
        K::native_swiglu_quantize_q8_1(ref, ref + n, qr, FS, T, st);
        verify_cmp("fuse shared q8", (const float*) q8, (const float*) qr, nb / sizeof(float));
        CK(cudaFree(qr)); CK(cudaFree(ref));
    }
    return true;
}
bool decode_experts(const Layer& L, int T, int64_t cap_groups, int64_t cap_entries) {
    if (!(rdna2_mmvq || expert_fuse) || !decode_gpu() || T < 1 || T > 4) return false;
    if (!K::q36_expert_decode(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
            cap_groups, cap_entries, xq, escr, eparts, st, rdna2_mmvq, expert_fuse)) return false;
    if (verify_k) {
        float* ref = nullptr;
        const size_t n = (size_t) cap_entries * L.L.n_embd;
        CK(cudaMalloc((void**) &ref, n * sizeof(float)));
        if (m->hybrid) CK(cudaMemsetAsync(ref, 0, n * sizeof(float), st));
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
            cap_groups, cap_entries, xq, escr, ref, st);
        verify_cmp("rdna2 experts", eparts, ref, n,
            " gu " + std::to_string(L.L.gu_type) + " down " + std::to_string(L.L.d_type) + " T " + std::to_string(T));
        CK(cudaFree(ref));
    }
    return true;
}
void verify_tpx_experts(const Layer& L, int64_t cap_groups, int64_t cap_entries) {
    if (!verify_k || !(q36tp::enabled("STRATA_Q36_TPX_GU16") || q36tp::enabled("STRATA_Q36_TPX_DOWN16"))) return;
    const size_t n = (size_t)cap_entries * L.L.n_embd;
    float* ref = nullptr;
    CK(cudaMalloc((void**)&ref, n * sizeof(float)));
    if (m->hybrid) CK(cudaMemsetAsync(ref, 0, n * sizeof(float), st));
    K::native_expert_tpx_tune(0, 0);
    K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                             cap_groups, cap_entries, xq, escr, ref, st);
    K::native_expert_tpx_tune(-1, -1);
    verify_cmp("tpx experts", eparts, ref, n,
               " gu " + std::to_string(L.L.gu_type) + " down " + std::to_string(L.L.d_type));
    CK(cudaFree(ref));
}
bool decode_wide_ok(int T, int N, int Kp) const {
    return moe_out_wide && T >= 1 && T <= kGemmT && N > 0 && N % 32 == 0 && Kp > 0 &&
        !decode_lds.empty() && q36decode::moe_lds(N, Kp) <= decode_lds[cur_dev];
}

void input_norm(const float* w, int T) {
    const int N = m->g.n_embd;
    if (!decode_fuse || T < 1 || T > kGemmT || N % 32) {
        k_rms<<<T, kThreads, 0, st>>>(x, w, h, N, m->eps);
        return;
    }
    q36decode::rms_q<<<T, kThreads, 0, st>>>(x, w, h, (q36k::QB*) xq, N, m->eps);
    xq_ready = true;
    if (verify_k) {
        float* ref = nullptr;
        CK(cudaMalloc((void**) &ref, (size_t) T * N * sizeof(float)));
        k_rms<<<T, kThreads, 0, st>>>(x, w, ref, N, m->eps);
        verify_cmp("fuse input norm", h, ref, (size_t) T * N);
        verify_q8("fuse input q8", h, T, "");
        CK(cudaFree(ref));
    }
}
