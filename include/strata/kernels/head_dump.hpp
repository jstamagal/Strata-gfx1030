#pragma once
#include <cstdint>

namespace strata::kernels {
bool head_dump_enabled();
// Before graph capture, on each participating device. Process-lifetime, bounded mapped storage.
void head_dump_prepare(int n_in);
// Capturable diagnostic node. Logits must contain the exact greedy winner (dumping disables the draft proxy).
void head_dump_record(int which, int type, int n_vocab, int T, int cols, int column0, int step,
                      const float* mixed, const void* q8_1, const float* logits, const int32_t* map,
                      const int32_t* positions, int position_stride, uint64_t vocab_hash, void* stream);
// Host only; writes complete records without waiting on GPU work. Safe after a window/chain completes.
void head_dump_flush();

// Little-endian binary file: FileHeader, then fixed-size records (RecordHeader, FP32[n_in], Q8_1[n_in/32]).
// Q8_1 is the exact input image read by the head; FP32 is the post-final-norm vector before quantization.
struct HeadDumpFileHeader {
    uint32_t magic, version, n_in, q8_bytes, record_header_bytes, max_rows, record_bytes, reserved;
};
struct HeadDumpRecordHeader {
    uint32_t magic, which, T, column, type, n_vocab;
    int32_t subset_index, token, position, device, step;
    uint32_t reserved;
    uint64_t vocab_hash;
    uint32_t record_bytes, reserved2;
};
inline constexpr uint32_t kHeadDumpMagic = 0x31444853u; // SHD1
static_assert(sizeof(HeadDumpFileHeader)==32 && sizeof(HeadDumpRecordHeader)==64);
} // namespace strata::kernels
