// Foreign prompt layers on the first card; decode keeps its original session addresses and split.
#pragma once
#include "strata/core/session.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace strata::program {
class PromptSplitState {
public:
    PromptSplitState() = default;
    ~PromptSplitState();
    PromptSplitState(const PromptSplitState&) = delete;
    PromptSplitState& operator=(const PromptSplitState&) = delete;
    bool init(const core::ModelGeometry& g, core::SessionState& first, core::SessionState& owner,
              int first_device, int owner_device, int64_t lo, int64_t hi,
              cudaStream_t first_stream, cudaStream_t owner_stream, std::string& err);
    // Called before a segment, and after the preceding later-stage chunk has drained but before its next launch.
    bool pull(int64_t position, int64_t end, std::string& err);
    bool push(int64_t position, std::string& err);
    core::SessionState& foreign() { return foreign_; }
    uint64_t device_bytes() const { return device_bytes_; }
    uint64_t transfer_bytes() const { return transfer_bytes_; }
private:
    bool move(bool to_owner, int64_t position, std::string& err);
    void release();
    core::ModelGeometry g_;
    core::SessionState foreign_;
    core::SessionState* owner_ = nullptr;
    int first_device_ = 0, owner_device_ = 1;
    cudaStream_t first_stream_ = nullptr, owner_stream_ = nullptr;
    std::vector<core::QsaState> qsa_;
    std::vector<void*> buffers_;
    void* transfer_ = nullptr;
    uint64_t device_bytes_ = 0, transfer_bytes_ = 0;
};
}  // namespace strata::program
