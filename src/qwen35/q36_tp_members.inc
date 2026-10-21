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
