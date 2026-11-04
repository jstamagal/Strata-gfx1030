// CPU integration harness for the actual loader and placement helpers (GPU copies stubbed).
#include "src/qwen35/q36_pack.hpp"
#include <cassert>
#include <cstdlib>
#include <memory>
#include <tuple>

// A minimal shape-compatible fixture lets the real runtime placement code run on CPU.
using cudaStream_t = int;
constexpr int cudaHostAllocPortable=0, cudaMemcpyHostToDevice=0, cudaMemcpyDeviceToHost=1;
int cudaHostAlloc(void** p, size_t n, int) { *p=std::malloc(n); return *p ? 0 : 1; }
int cudaFreeHost(void* p) { std::free(p); return 0; }
int cudaMemcpy(void* d,const void* s,size_t n,int) { std::memcpy(d,s,n); return 0; }
int cudaMemcpyAsync(void* d,const void* s,size_t n,int k,int) { return cudaMemcpy(d,s,n,k); }
int cudaStreamSynchronize(int) { return 0; }
#define CK(x) q36pack::require((x)==0,"CPU fixture copy failed")
struct Layout { int gu_type, d_type; size_t up_off, down_off, bytes; };
struct Layer {
    Layout L; size_t stride=0, cache_layer=0; q36pack::Cache* cache=nullptr;
    uint8_t* blobs=nullptr; std::vector<int> initial_rank; bool use_initial_rank=false;
};
struct File {
    std::string name;
    std::string path() const { return name; }
    size_t size() const { return 0; }
    const File& shard(size_t) const { return *this; }
};
struct Model {
    struct { size_t n_expert, n_layers, n_mtp_layers, n_embd, n_ff; } g;
    std::vector<Layer> layers; bool has_mtp=false; Layer mtp;
    std::shared_ptr<File> file; std::shared_ptr<q36pack::Cache> expert_cache;
};
int initial_expert(const Layer& L,size_t i) { return L.use_initial_rank ? L.initial_rank.at(i) : int(i); }
void expert_blob(const Layer&,int64_t,uint8_t*) { throw std::runtime_error("unexpected GGUF fallback"); }
#include "src/qwen35/q36_pack_runtime.cuh"
struct Hy { std::vector<int> slot_of,hpos,exp_in_slot,host_exp; std::vector<double> score; };
int main(int argc,char** argv) {
    try {
        q36pack::require(argc>=4,"arguments");
        const std::string mode=argv[1];
        if(mode=="profile") {
            auto ranks=q36pack::profile(argv[2],3,5);
            assert(ranks[0][0]==4 && ranks[1][0]==3);
            q36pack::Trace t; t.open(argv[3],3,5);
            int32_t ids[]={4,4,1}; t.add(0,ids,3); t.save();
            return 0;
        }
        q36pack::Cache c;
        c.open(argv[2],{argv[3],argv[4]},3,5,512,256);
        for(size_t li=0;li<3;++li) {
            const auto& x=c.layers[li];
            std::vector<uint8_t> raw(5*x.stride), slots(5*x.stride), host(5*x.stride);
            c.read(li,0,5,raw.data());
            Layer L{{x.gu,x.dt,x.up,x.down,x.bytes},size_t(x.stride),li,&c,slots.data(),{4,1,3,0,2}};
            for(int s=0;s<=5;++s) {
                select_initial_order(L,s,5);
                Hy H; initial_tables(L,H,5,s);
                if(s) upload_initial_experts(L,s);
                cached_experts(L,s,5-s,host.data());
                for(int e=0;e<5;++e) {
                    const uint8_t* ptr=H.slot_of[e]>=0 ? slots.data()+H.slot_of[e]*x.stride
                                                      : host.data()+H.hpos[e]*x.stride;
                    assert(!std::memcmp(ptr,raw.data()+e*x.stride,x.stride));
                }
                if(s==5) assert(H.exp_in_slot==std::vector<int>({0,1,2,3,4}));
            }
        }
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
