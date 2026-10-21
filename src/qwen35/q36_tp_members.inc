    // TP storage is separate from the existing layer-split DevSet and from session metadata (position/LRU/tokens).
    bool tp = false;
    const Model* tp_models[2] = {};
    struct TpState {
        std::vector<Sess> sessions;
        std::vector<float*> conv, S, kc, vc, v_qkv, v_hc, v_alpha, v_beta;
        float* hlast = nullptr;
    } tp_state[2];
    float *tp_host[2] = {}, *tp_mapped[2][2] = {}, *tp_logits[2] = {};
    unsigned *tp_flags_host[2] = {}, *tp_flags[2][2] = {}, *tp_token[2] = {};
    cudaEvent_t tp_event[2] = {};
    void tp_swap(int from, int to);
    void tp_init(const Model& a, const Model& b, int context, int chunk, int sessions, bool pictures);
    void tp_enter(int T, bool mtp = false, const float* seeds = nullptr);
    void tp_join(float* partial, int T);
    void tp_moe_end(int T, const int32_t* prow, const float* shared, const float* sgate,
                    const float* wn, float* hout, int sgs);
    void tp_forward(int T, int64_t p0, int heads, bool verify);
    void tp_head(int T, int heads, bool mtp = false, bool draft = false, float* output = nullptr, int32_t* picks = nullptr);
    void tp_forward_mtp(int T, int64_t p0, const float* seeds, bool draft);
    float* tp_remote[2] = {};
    cudaEvent_t tp_copy_ready[2] = {}, tp_copy_done[2] = {};
    bool tp_copy_used = false, tp_graph = true;
    void tp_copy_join(int T, bool moe);
    void tp_layer_pair(size_t li, int T, int64_t p0, bool verify, bool mtp = false);
    long long tp_graph_key(int T, bool verify, bool mtp) const;
    bool tp_graph_run(int T, int64_t p0, bool verify, bool mtp = false, const float* seeds = nullptr);
    bool tp_check = false;
    bool tp_attn_rows = q36tp::enabled("STRATA_Q36_TP_ATTN_ROWS");
    bool tp_head_q8 = q36tp::enabled("STRATA_Q36_TP_HEAD_Q8");
    bool tp_moe_join = q36tp::enabled("STRATA_Q36_TP_MOE_JOIN");
    long long tp_checks = 0;
    void tp_check_residual(int T, bool mtp);
