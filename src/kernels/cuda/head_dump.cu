// Bounded, graph-safe diagnostic capture; no file I/O or runtime callbacks inside a graph.
#include "strata/kernels/head_dump.hpp"
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#if defined(__HIPCC__)
#include "head_exact.cuh"
#endif

namespace strata::kernels {
namespace {
struct SlotBuffer {
    uint8_t* host=nullptr; uint8_t* device=nullptr; unsigned* count=nullptr; unsigned read=0;
};
struct DumpState {
    std::mutex mutex;
    std::array<SlotBuffer,64> buffers{};
    FILE* file=nullptr;
    int cap=0,n_in=0,qbytes=0,record_bytes=0,written=0;
    bool failed=false;
    ~DumpState() { if(file) std::fclose(file); }
};
DumpState& state() { static DumpState s;return s; }
void check_dump(cudaError_t e,const char* where) {
    if(e!=cudaSuccess)throw std::runtime_error(std::string("head dump ")+where+": "+cudaGetErrorString(e));
}
#if defined(__HIPCC__)
__launch_bounds__(256)
__global__ void record_kernel(uint8_t* storage,unsigned* count,int cap,int rb,int n,int qb,
                               HeadDumpRecordHeader header,int cols,const float* mixed,const uint8_t* q8,
                               const float* logits,const int32_t* map,const int32_t* positions,int pstride) {
    using namespace hdetail;
    const int col=int(blockIdx.x),tid=int(threadIdx.x),lane=tid&31,wave=tid>>5;
    __shared__ unsigned slot;
    __shared__ float bv[8];__shared__ int bi[8];
    if(!tid)slot=atomicAdd(count,0u)>=unsigned(cap)?unsigned(cap):atomicAdd(count,1u);
    __syncthreads();
    if(slot>=unsigned(cap)||col>=cols)return;
    float v=ninf();int id=header.n_vocab;
    const float* l=logits+size_t(col)*header.n_vocab;
    for(int r=tid;r<int(header.n_vocab);r+=256)
        if(l[r]>v||(l[r]==v&&r<id)){v=l[r];id=r;}
    for(int o=16;o;o>>=1){
        const float ov=__shfl_xor_sync(0xffffffffu,v,o);const int oi=__shfl_xor_sync(0xffffffffu,id,o);
        if(ov>v||(ov==v&&oi<id)){v=ov;id=oi;}
    }
    if(!lane){bv[wave]=v;bi[wave]=id;}
    uint8_t* dst=storage+size_t(slot)*rb;
    float* fp=reinterpret_cast<float*>(dst+sizeof(HeadDumpRecordHeader));
    uint32_t* qp=reinterpret_cast<uint32_t*>(fp+n);
    for(int i=tid;i<n;i+=256)fp[i]=mixed[size_t(col)*n+i];
    for(int i=tid;i<qb/4;i+=256)qp[i]=reinterpret_cast<const uint32_t*>(q8+size_t(col)*qb)[i];
    __syncthreads();
    if(!tid){
        for(int w=1;w<8;++w)if(bv[w]>v||(bv[w]==v&&bi[w]<id)){v=bv[w];id=bi[w];}
        if(id>=int(header.n_vocab))id=0;
        header.magic=0;header.column+=col;header.subset_index=id;header.token=map?map[id]:id;
        header.position=positions?positions[size_t(col)*pstride]:-1;
        *reinterpret_cast<HeadDumpRecordHeader*>(dst)=header;
    }
    __threadfence_system();
    __syncthreads();
    if(!tid){reinterpret_cast<volatile HeadDumpRecordHeader*>(dst)->magic=kHeadDumpMagic;__threadfence_system();}
}
#endif
} // namespace

bool head_dump_enabled() {
#if defined(__HIPCC__)
    static const bool on=[] {const char* e=std::getenv("STRATA_HEAD_DUMP");return e&&*e;}();
    return on;
#else
    return false;
#endif
}
void head_dump_prepare(int n_in) {
    if(!head_dump_enabled())return;
    if(n_in!=2560)throw std::invalid_argument("head dump requires a 2560-wide head");
    auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
    if(!s.file){
        const char* c=std::getenv("STRATA_HEAD_DUMP_MAX");
        const long long cap=c?std::strtoll(c,nullptr,10):20000;
        if(cap<1||cap>1000000)throw std::invalid_argument("STRATA_HEAD_DUMP_MAX must be 1..1000000");
        s.cap=int(cap);s.n_in=n_in;s.qbytes=(n_in/32)*36;
        s.record_bytes=int(sizeof(HeadDumpRecordHeader))+n_in*4+s.qbytes;
        s.file=std::fopen(std::getenv("STRATA_HEAD_DUMP"),"wb");
        if(!s.file)throw std::runtime_error("cannot create STRATA_HEAD_DUMP file");
        const HeadDumpFileHeader h{kHeadDumpMagic,1,uint32_t(n_in),uint32_t(s.qbytes),sizeof(HeadDumpRecordHeader),uint32_t(s.cap),uint32_t(s.record_bytes),0};
        if(std::fwrite(&h,sizeof h,1,s.file)!=1)throw std::runtime_error("cannot write head dump header");
        std::fflush(s.file);
        std::fprintf(stderr,"strata head dump: %s, at most %d rows; exact draft head forced; FP32 and Q8_1 inputs\n",std::getenv("STRATA_HEAD_DUMP"),s.cap);
    }
    int dev=0;check_dump(cudaGetDevice(&dev),"device");
    if(dev<0||dev>=int(s.buffers.size()))throw std::runtime_error("head dump device index out of range");
    auto& b=s.buffers[dev];if(b.host)return;
    const size_t bytes=size_t(s.cap)*s.record_bytes;
#if defined(__HIPCC__)
    check_dump(hipHostMalloc(reinterpret_cast<void**>(&b.host),bytes,hipHostMallocMapped|hipHostMallocCoherent|hipHostMallocPortable),"mapped allocation");
#else
    check_dump(cudaHostAlloc(reinterpret_cast<void**>(&b.host),bytes,cudaHostAllocMapped|cudaHostAllocPortable),"mapped allocation");
#endif
    std::memset(b.host,0,bytes);
    check_dump(cudaHostGetDevicePointer(reinterpret_cast<void**>(&b.device),b.host,0),"mapped alias");
    check_dump(cudaMalloc(reinterpret_cast<void**>(&b.count),sizeof(unsigned)),"counter");
    check_dump(cudaMemset(b.count,0,sizeof(unsigned)),"counter clear");
}
void head_dump_record(int which,int type,int n_vocab,int T,int cols,int column0,int step,
                      const float* mixed,const void* q8,const float* logits,const int32_t* map,
                      const int32_t* positions,int pstride,uint64_t vocab_hash,void* stream) {
    if(!head_dump_enabled())return;
#if defined(__HIPCC__)
    auto& s=state();int dev=0;check_dump(cudaGetDevice(&dev),"device");
    if(dev<0||dev>=int(s.buffers.size())||!s.buffers[dev].device)throw std::runtime_error("head dump was not prepared before capture");
    if(!mixed||!q8||!logits||cols<1||n_vocab<1)throw std::invalid_argument("invalid head dump buffers");
    const auto& b=s.buffers[dev];
    HeadDumpRecordHeader h{};h.which=uint32_t(which);h.T=uint32_t(T);h.column=uint32_t(column0);
    h.type=uint32_t(type);h.n_vocab=uint32_t(n_vocab);h.device=dev;h.step=step;h.vocab_hash=vocab_hash;h.record_bytes=uint32_t(s.record_bytes);
    record_kernel<<<cols,256,0,static_cast<cudaStream_t>(stream)>>>(b.device,b.count,s.cap,s.record_bytes,s.n_in,s.qbytes,h,cols,mixed,static_cast<const uint8_t*>(q8),logits,map,positions,pstride);
    check_dump(cudaGetLastError(),"record launch");
#endif
}
void head_dump_flush() {
    if(!head_dump_enabled())return;
    auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
    if(!s.file||s.failed||s.written>=s.cap)return;
    for(auto& b:s.buffers){
        if(!b.host)continue;
        while(b.read<unsigned(s.cap)&&s.written<s.cap){
            const uint8_t* p=b.host+size_t(b.read)*s.record_bytes;
            if(reinterpret_cast<const volatile HeadDumpRecordHeader*>(p)->magic!=kHeadDumpMagic)break;
            std::atomic_thread_fence(std::memory_order_acquire);
            if(std::fwrite(p,s.record_bytes,1,s.file)!=1){s.failed=true;std::fprintf(stderr,"strata head dump: write failed\n");return;}
            ++b.read;++s.written;
        }
    }
    std::fflush(s.file);
    if(s.written==s.cap)std::fprintf(stderr,"strata head dump: row cap reached (%d); no more records written\n",s.cap);
}
} // namespace strata::kernels
