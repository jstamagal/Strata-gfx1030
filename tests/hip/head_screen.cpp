// Synthetic and head-only GGUF parity/latency; never loads the model or its experts.
#include "strata/kernels/head_screen.hpp"
#include "strata/kernels/head_dump.hpp"
#include <array>
#include "strata/kernels/head_wide.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

using namespace strata::kernels;
static void ck(cudaError_t e,const char* what) { if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e)); std::exit(e==cudaErrorMemoryAllocation?75:1); } }
struct Buffer {
    void* p=nullptr;
    explicit Buffer(size_t n) { ck(cudaMalloc(&p,n),"allocate"); }
    ~Buffer() { if(p) cudaFree(p); }
    template<class T> T* as() { return static_cast<T*>(p); }
};
__global__ void flush_kernel(const uint32_t* p,size_t n,uint32_t* sink) {
    uint32_t v=0;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) v+=p[i];
    atomicAdd(sink,v);
}
static double timed(cudaStream_t s,const std::function<void()>& f,Buffer& flush,Buffer& sink,bool cold) {
    cudaEvent_t a,b; ck(cudaEventCreate(&a),"event"); ck(cudaEventCreate(&b),"event");
    std::vector<float> v;
    for(int i=0;i<13;++i) {
        if(cold) flush_kernel<<<1024,256,0,s>>>(flush.as<uint32_t>(),(384ull<<20)/4,sink.as<uint32_t>());
        ck(cudaEventRecord(a,s),"record"); f(); ck(cudaEventRecord(b,s),"record"); ck(cudaEventSynchronize(b),"time");
        float ms; ck(cudaEventElapsedTime(&ms,a,b),"elapsed"); if(i>=4) v.push_back(ms*1000);
    }
    cudaEventDestroy(a);cudaEventDestroy(b);std::sort(v.begin(),v.end()); return v[v.size()/2];
}
static std::vector<int> top(const float* p,int n,int k) {
    std::vector<int> ids(n);std::iota(ids.begin(),ids.end(),0);
    auto cmp=[&](int a,int b) {
        if(std::isnan(p[a])) return !std::isnan(p[b])?false:a<b;
        if(std::isnan(p[b])) return true;
        return p[a]>p[b]||(p[a]==p[b]&&a<b);
    };
    k=std::min(k,n); std::partial_sort(ids.begin(),ids.begin()+k,ids.end(),cmp); ids.resize(k); return ids;
}
static void set_env(const char* name,const char* value) {
#if defined(_WIN32)
    _putenv_s(name,value?value:"");
#else
    if(value)setenv(name,value,1);else unsetenv(name);
#endif
}
static void proxy_policy() {
    const char* e=std::getenv("STRATA_MTP_HEAD_PROXY");const bool had=e!=nullptr;const std::string saved=e?e:"";
    cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0),"policy device");
    const bool gfx1030=std::strncmp(p.gcnArchName,"gfx1030",7)==0&&(p.gcnArchName[7]=='\0'||p.gcnArchName[7]==':');
    set_env("STRATA_MTP_HEAD_PROXY",nullptr);
    const bool automatic=head_proxy_enabled(8,2560,0);
    if(automatic!=gfx1030||head_proxy_enabled(13,2560,0)||head_proxy_enabled(14,2560,0)||head_proxy_enabled(8,2304,0))std::abort();
    const size_t owned=head_proxy_reserve_bytes(8,2560,106299,0),borrowed=head_proxy_reserve_bytes(8,2560,106299,0,false);
    if(owned!=(gfx1030?85176080u:0u)||borrowed!=(gfx1030?132880u:0u)||head_proxy_reserve_bytes(14,2560,106299,0))std::abort();
    set_env("STRATA_MTP_HEAD_PROXY","0");
    if(head_proxy_enabled(8,2560,0)||head_proxy_reserve_bytes(8,2560,106299,0))std::abort();
    set_env("STRATA_MTP_HEAD_PROXY","1");
    for(int type:{8,13,14})if(!head_proxy_enabled(type,2560,0)||head_proxy_reserve_bytes(type,2560,106299,0)!=85176080u)std::abort();
    if(head_proxy_enabled(23,2560,0)||head_proxy_enabled(8,2304,0))std::abort();
    set_env("STRATA_MTP_HEAD_PROXY","invalid");if(head_proxy_enabled(8,2560,0))std::abort();
    set_env("STRATA_MTP_HEAD_PROXY",had?saved.c_str():nullptr);
    std::printf("head proxy policy: unset Q8_0=%s Q6_K=off Q5_K=off, owned reserve=%zu shared scratch=%zu bytes; explicit0/1 PASS\n",automatic?"on":"off",owned,borrowed);
}
static void policy() {
    SamplerParams p; p.greedy=true;
    if(head_screen_keep(p,true)!=1) std::abort();
    p.greedy=false;p.top_k=20; if(head_screen_keep(p,true)!=20) std::abort();
    for(int k:{0,65,100}) {p.top_k=k;if(head_screen_keep(p,true)!=0)std::abort();}
    p.top_k=64;if(head_screen_keep(p,true)!=64)std::abort();
    p.penalty_last_n=64;p.penalty_freq=0.2f;if(head_screen_keep(p,true)!=0)std::abort();
    p.penalty_freq=0;p.penalty_repeat=1.1f;if(head_screen_keep(p,true)!=0)std::abort();
    p.penalty_repeat=1;p.penalty_present=0.1f;if(head_screen_keep(p,true)!=0)std::abort();
    p.penalty_present=0;if(head_screen_keep(p,true)!=64)std::abort();
    p.logit_bias=reinterpret_cast<const float*>(uintptr_t(1));if(head_screen_keep(p,true)!=0)std::abort();
    p.logit_bias=nullptr;if(head_screen_keep(p,true,true)!=0)std::abort();
}
static std::vector<uint8_t> synthetic(int type,int rows,std::mt19937_64& rng) {
    std::vector<uint8_t> w(native_mmvq_weight_bytes(type,2560,rows));
    for(auto& b:w)b=uint8_t(rng());
    const int block=type==8?34:type==14?210:176;
    for(size_t at=0;at<w.size();at+=block) {
        uint16_t d=uint16_t(0x1000+(rng()&0x3ff));
        if(rng()%7==0)d|=0x8000;
        if(rng()%97==0)d=uint16_t(rng()%2?0x8000:0);
        const int off=type==14?208:0;std::memcpy(w.data()+at+off,&d,2);
        if(type==13) {const uint16_t m=0x0800;std::memcpy(w.data()+at+2,&m,2);}
    }
    const size_t rb=w.size()/rows;
    // All duplicated top values must use the lower row id, including the final partial tile.
    for(int r=1;r<rows;r+=37)std::memcpy(w.data()+size_t(r)*rb,w.data(),rb);
    return w;
}
struct InputRow {
    HeadDumpRecordHeader header;
    std::array<float,2560> x;
    std::array<uint8_t,2880> q;
};
static_assert(sizeof(InputRow)==13184);
struct DumpRows { std::vector<InputRow> verify,draft; };
static DumpRows read_inputs(const char* path,int limit) {
    std::unique_ptr<FILE,decltype(&std::fclose)> f(std::fopen(path,"rb"),&std::fclose);
    if(!f)throw std::runtime_error("cannot open input dump");
    HeadDumpFileHeader h;
    if(std::fread(&h,sizeof h,1,f.get())!=1||h.magic!=kHeadDumpMagic||h.version!=1||h.n_in!=2560||
       h.q8_bytes!=2880||h.record_header_bytes!=64||h.record_bytes!=sizeof(InputRow))throw std::runtime_error("invalid SHD1 file header");
    DumpRows out;uint64_t seen[2]={};std::mt19937_64 rng(17);
    while(true) {
        InputRow r;const size_t n=std::fread(&r,1,sizeof r,f.get());
        if(!n&&std::feof(f.get()))break;
        if(n!=sizeof r||r.header.magic!=kHeadDumpMagic||r.header.record_bytes!=sizeof r||r.header.which>1)
            throw std::runtime_error("truncated/invalid SHD1 record");
        auto& v=r.header.which?out.draft:out.verify;const auto at=seen[r.header.which]++;
        if(v.size()<size_t(limit))v.push_back(r);
        else {const auto j=rng()%(at+1);if(j<uint64_t(limit))v[size_t(j)]=r;}
    }
    std::printf("input dump %s: verify=%llu draft=%llu, retained %zu/%zu rows (deterministic reservoir)\n",path,
        (unsigned long long)seen[0],(unsigned long long)seen[1],out.verify.size(),out.draft.size());
    if(out.verify.empty()&&out.draft.empty())throw std::runtime_error("empty input dump");
    return out;
}
static int dump_smoke(const char* path) {
    set_env("STRATA_HEAD_DUMP",path);set_env("STRATA_HEAD_DUMP_MAX","6");head_dump_prepare(2560);
    cudaStream_t s;ck(cudaStreamCreate(&s),"dump stream");
    Buffer x(2*2560*4),q(native_q8_1_bytes(2560,2)),y(8*4),map(4*4),pos(2*4);
    std::vector<float> hx(5120);for(size_t i=0;i<hx.size();++i)hx[i]=float(int(i%37)-18)*0.125f;
    const float hy[8]={1,2,2,-1,0,-3,0,0};const int hm[4]={11,42,7,100},hp[2]={123,124};
    ck(cudaMemcpy(x.p,hx.data(),hx.size()*4,cudaMemcpyHostToDevice),"dump x");
    ck(cudaMemcpy(y.p,hy,sizeof hy,cudaMemcpyHostToDevice),"dump logits");
    ck(cudaMemcpy(map.p,hm,sizeof hm,cudaMemcpyHostToDevice),"dump map");ck(cudaMemcpy(pos.p,hp,sizeof hp,cudaMemcpyHostToDevice),"dump positions");
    native_quantize_q8_1(x.as<float>(),q.p,2560,2,s);ck(cudaStreamSynchronize(s),"dump q");
    std::array<uint8_t,5760> hq;ck(cudaMemcpy(hq.data(),q.p,hq.size(),cudaMemcpyDeviceToHost),"dump reference q");
    ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"dump capture");
    head_dump_record(0,8,4,2,2,0,-1,x.as<float>(),q.p,y.as<float>(),nullptr,pos.as<int32_t>(),1,0,s);
    head_dump_record(1,8,4,1,1,0,2,x.as<float>(),q.p,y.as<float>(),map.as<int32_t>(),pos.as<int32_t>(),1,1234,s);
    cudaGraph_t g;cudaGraphExec_t ex;ck(cudaStreamEndCapture(s,&g),"dump end");ck(cudaGraphInstantiate(&ex,g,0),"dump instantiate");
    for(int i=0;i<4;++i){ck(cudaGraphLaunch(ex,s),"dump replay");ck(cudaStreamSynchronize(s),"dump sync");head_dump_flush();}
    const auto data=read_inputs(path,64);
    if(data.verify.size()+data.draft.size()!=6||data.verify.empty()||data.draft.empty())return 1;
    for(const auto* list:{&data.verify,&data.draft})for(const auto& r:*list) {
        const bool draft=r.header.which==1;const int col=int(r.header.column),pick=col==0?1:0;
        if(col<0||col>1||r.header.T!=unsigned(draft?1:2)||r.header.subset_index!=pick||
           r.header.token!=(draft?hm[pick]:pick)||r.header.position!=hp[col]||r.header.vocab_hash!=(draft?1234u:0u)||
           std::memcmp(r.x.data(),hx.data()+col*2560,2560*4)||std::memcmp(r.q.data(),hq.data()+col*2880,2880))return 1;
    }
    cudaGraphExecDestroy(ex);cudaGraphDestroy(g);ck(cudaStreamDestroy(s),"dump destroy");
    std::printf("head dump: exact tie picks, mapped tokens, FP32/Q8_1 bytes, graph replay, six-row cap PASS\n");return 0;
}
static int exercise(int type,std::vector<uint8_t> hw,int rows,bool bench,bool real,const char* vocab,const DumpRows* inputs=nullptr) {
    constexpr int N=2560,MT=8;
    if(!head_proxy_supported(type,N))return 77;
    cudaStream_t s;ck(cudaStreamCreate(&s),"stream");
    const size_t rb=hw.size()/rows;
    Buffer w(hw.size()),x(N*MT*4),xq(native_q8_1_bytes(N,MT)),ya(size_t(rows)*MT*4),yb(size_t(rows)*MT*4),keep(4),scratch(head_screen_scratch_bytes());
    Buffer wide(head_wide_scratch_bytes()),oa(MT*4),ob(MT*4);
    ck(cudaMemcpy(w.p,hw.data(),hw.size(),cudaMemcpyHostToDevice),"upload weights");
    std::mt19937_64 rng(719);std::normal_distribution<float> nd(0,1);
    std::vector<float> hx(N*MT);for(auto& v:hx)v=nd(rng);
    for(int i=0;i<N;++i) {hx[2*N+i]=0;hx[5*N+i]*=1e-30f;if(i%3==0)hx[6*N+i]=0;}
    std::vector<uint8_t> captured_q;
    std::vector<const InputRow*> verify_inputs;
    if(inputs&&!inputs->verify.empty()) {
        captured_q.resize(MT*2880);
        for(int j=0;j<MT;++j) {
            const auto& r=inputs->verify[size_t(j)*(inputs->verify.size()-1)/(MT-1)];
            if(int(r.header.type)!=type||int(r.header.n_vocab)!=rows)throw std::runtime_error("verify dump does not match head GGUF");
            std::copy(r.x.begin(),r.x.end(),hx.begin()+j*N);std::copy(r.q.begin(),r.q.end(),captured_q.begin()+j*2880);
            verify_inputs.push_back(&r);
        }
        std::printf("verify parity/bench uses captured post-norm inputs and their actual Q8_1 bytes\n");
    }
    ck(cudaMemcpy(x.p,hx.data(),hx.size()*4,cudaMemcpyHostToDevice),"upload x");
    // A strongly aligned column exercises certified proxy pruning, not just its all-exact fallback.
    if(!real)iq_dequant_f32(type,static_cast<const uint8_t*>(w.p),N,x.as<float>()+7*N,s);
    std::vector<float> a(size_t(rows)*MT),b(a.size());
    long cases=0,errors=0,tail=0;
    if(head_screen_supported(type,N,rows,1)) {
        Buffer copy(head_screen_copy_bytes(type,rows));head_screen_pack(type,w.p,rows,copy.p,s);
        for(int cols=1;cols<=MT;++cols) {
            native_quantize_q8_1(x.as<float>(),xq.p,N,cols,s);
            if(!captured_q.empty())ck(cudaMemcpyAsync(xq.p,captured_q.data(),size_t(cols)*2880,cudaMemcpyHostToDevice,s),"captured verify Q8_1");
            native_mmvq(type,w.p,xq.p,ya.as<float>(),N,rows,cols,s);ck(cudaStreamSynchronize(s),"reference");
            ck(cudaMemcpy(a.data(),ya.p,size_t(rows)*cols*4,cudaMemcpyDeviceToHost),"reference logits");
            // Replay the same graph with different K, including full exact and a return to greedy.
            ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"capture");
            head_screen(type,copy.p,w.p,xq.p,yb.as<float>(),rows,cols,keep.as<int32_t>(),scratch.p,s);
            cudaGraph_t g;cudaGraphExec_t ex;ck(cudaStreamEndCapture(s,&g),"end capture");ck(cudaGraphInstantiate(&ex,g,0),"instantiate");
            for(int k:{0,1,20,64,1}) {
                ck(cudaMemcpy(keep.p,&k,4,cudaMemcpyHostToDevice),"K");ck(cudaGraphLaunch(ex,s),"replay");ck(cudaStreamSynchronize(s),"screen");
                ck(cudaMemcpy(b.data(),yb.p,size_t(rows)*cols*4,cudaMemcpyDeviceToHost),"screen logits");
                for(int j=0;j<cols;++j) {
                    const float* ar=a.data()+size_t(j)*rows;const float* br=b.data()+size_t(j)*rows;
                    const auto ta=top(ar,rows,k?k:64),tb=top(br,rows,k?k:64);
                    bool same=ta==tb;
                    if(!verify_inputs.empty()&&ta[0]!=verify_inputs[j]->header.token) {
                        ++errors;std::printf("CAPTURE PICK MISMATCH col=%d position=%d saved=%d replay=%d\n",j,
                            verify_inputs[j]->header.position,verify_inputs[j]->header.token,ta[0]);
                    }
                    for(int id:ta)same=same&&std::memcmp(ar+id,br+id,4)==0;
                    if(!k)same=same&&std::memcmp(ar,br,size_t(rows)*4)==0;
                    if(!same){++errors;std::printf("MISMATCH type=%d T=%d K=%d col=%d exact-pick=%d screen-pick=%d\n",type,cols,k,j,ta[0],tb[0]);}
                    ++cases;
                    if(k==20)for(int r=0;r<rows;++r)tail+=std::memcmp(ar+r,br+r,4)!=0;
                }
                SamplerParams sp;sp.top_k=k?k:20;sp.seed=123;sp.gumbel=true;sp.temperature=0.7f;
                sample_tokens(ya.as<float>(),cols,rows,nullptr,0,sp,oa.as<int>(),s);
                sample_tokens(yb.as<float>(),cols,rows,nullptr,0,sp,ob.as<int>(),s);
                ck(cudaStreamSynchronize(s),"sample");
                int pa[8],pb[8];ck(cudaMemcpy(pa,oa.p,cols*4,cudaMemcpyDeviceToHost),"draw A");ck(cudaMemcpy(pb,ob.p,cols*4,cudaMemcpyDeviceToHost),"draw B");
                // K=1 preserves only greedy, not a top-20 sampled draw.
                if(k!=1&&std::memcmp(pa,pb,cols*4)){++errors;std::printf("DRAW MISMATCH type=%d T=%d K=%d\n",type,cols,k);}
            }
            cudaGraphExecDestroy(ex);cudaGraphDestroy(g);
        }
        if(!real) {
            // Nonfinite original scales must survive the screen and exact rescore.
            const int off=type==8?0:208;uint16_t inf=0x7c00,nan=0x7e00;
            ck(cudaMemcpy(static_cast<char*>(w.p)+off,&inf,2,cudaMemcpyHostToDevice),"inf scale");
            if(rows>1)ck(cudaMemcpy(static_cast<char*>(w.p)+rb+off,&nan,2,cudaMemcpyHostToDevice),"nan scale");
            head_screen_pack(type,w.p,rows,copy.p,s);int k=20;ck(cudaMemcpy(keep.p,&k,4,cudaMemcpyHostToDevice),"K");
            native_mmvq(type,w.p,xq.p,ya.as<float>(),N,rows,8,s);
            head_screen(type,copy.p,w.p,xq.p,yb.as<float>(),rows,8,keep.as<int32_t>(),scratch.p,s);ck(cudaStreamSynchronize(s),"nonfinite");
            ck(cudaMemcpy(a.data(),ya.p,size_t(rows)*8*4,cudaMemcpyDeviceToHost),"nf A");ck(cudaMemcpy(b.data(),yb.p,size_t(rows)*8*4,cudaMemcpyDeviceToHost),"nf B");
            for(int j=0;j<8;++j)for(int r=0;r<std::min(rows,2);++r)
                if(std::memcmp(&a[size_t(j)*rows+r],&b[size_t(j)*rows+r],4)){++errors;std::printf("NONFINITE MISMATCH type=%d col=%d row=%d\n",type,j,r);}
            ck(cudaMemcpy(w.p,hw.data(),hw.size(),cudaMemcpyHostToDevice),"restore weights");head_screen_pack(type,w.p,rows,copy.p,s);
            // One bad activation scale invalidates the whole column's bounds, not just a shortlist.
            ck(cudaMemcpy(static_cast<char*>(xq.p)+(7*80+3)*36,&nan,2,cudaMemcpyHostToDevice),"nan x scale");
            native_mmvq(type,w.p,xq.p,ya.as<float>(),N,rows,8,s);
            head_screen(type,copy.p,w.p,xq.p,yb.as<float>(),rows,8,keep.as<int32_t>(),scratch.p,s);ck(cudaStreamSynchronize(s),"bad activation column");
            ck(cudaMemcpy(a.data(),static_cast<float*>(ya.p)+size_t(7)*rows,rows*4,cudaMemcpyDeviceToHost),"bad x A");
            ck(cudaMemcpy(b.data(),static_cast<float*>(yb.p)+size_t(7)*rows,rows*4,cudaMemcpyDeviceToHost),"bad x B");
            if(std::memcmp(a.data(),b.data(),rows*4)){++errors;std::printf("BAD ACTIVATION MISMATCH type=%d\n",type);}
            native_quantize_q8_1(x.as<float>(),xq.p,N,8,s);
        }
        if(inputs&&!inputs->verify.empty()) {
            size_t at=0;int batch=0;long checks=0;
            while(at<inputs->verify.size()) {
                const int cols=std::min<int>(1+batch%8,int(inputs->verify.size()-at));
                std::array<uint8_t,MT*2880> qbatch;
                for(int j=0;j<cols;++j) {
                    const auto& r=inputs->verify[at+j];
                    if(int(r.header.type)!=type||int(r.header.n_vocab)!=rows)throw std::runtime_error("captured verify metadata mismatch");
                    std::copy(r.q.begin(),r.q.end(),qbatch.begin()+j*2880);
                }
                ck(cudaMemcpyAsync(xq.p,qbatch.data(),size_t(cols)*2880,cudaMemcpyHostToDevice,s),"replay batch Q8_1");
                native_mmvq(type,w.p,xq.p,ya.as<float>(),N,rows,cols,s);ck(cudaStreamSynchronize(s),"replay reference");
                ck(cudaMemcpy(a.data(),ya.p,size_t(cols)*rows*4,cudaMemcpyDeviceToHost),"replay reference logits");
                std::vector<std::vector<int>> reference(cols);
                for(int j=0;j<cols;++j) {
                    reference[j]=top(a.data()+size_t(j)*rows,rows,64);
                    if(reference[j][0]!=inputs->verify[at+j].header.token){++errors;std::printf("SAVED VERIFY PICK MISMATCH input=%zu\n",at+j);}
                }
                for(int k:{1,20,64}) {
                    ck(cudaMemcpy(keep.p,&k,4,cudaMemcpyHostToDevice),"replay K");
                    head_screen(type,copy.p,w.p,xq.p,yb.as<float>(),rows,cols,keep.as<int32_t>(),scratch.p,s);ck(cudaStreamSynchronize(s),"replay screen");
                    ck(cudaMemcpy(b.data(),yb.p,size_t(cols)*rows*4,cudaMemcpyDeviceToHost),"replay screened logits");
                    for(int j=0;j<cols;++j) {
                        const float* ar=a.data()+size_t(j)*rows;const float* br=b.data()+size_t(j)*rows;
                        const auto ids=top(br,rows,k);bool same=std::equal(ids.begin(),ids.end(),reference[j].begin());
                        for(int id:ids)same=same&&std::memcmp(ar+id,br+id,4)==0;
                        if(!same){++errors;std::printf("CAPTURE TOPK MISMATCH input=%zu T=%d K=%d\n",at+j,cols,k);}
                        ++checks;
                    }
                }
                at+=cols;++batch;
            }
            std::printf("captured verify sweep: %zu inputs, %ld exact top-K column checks (varying T1..8), cumulative errors=%ld\n",inputs->verify.size(),checks,errors);
        }
        std::printf("screen parity type=%d rows=%d: %ld columns, %ld mismatches, %ld differing tail entries at K20\n",type,rows,cases,errors,tail);
        if(bench) {
            Buffer flush(384ull<<20),sink(4);ck(cudaMemset(flush.p,1,384ull<<20),"flush init");ck(cudaMemset(sink.p,0,4),"sink");
            // Timings use independent nonzero Gaussian columns, not the parity degeneracies.
            if(captured_q.empty())for(auto& v:hx)v=nd(rng);
            ck(cudaMemcpy(x.p,hx.data(),hx.size()*4,cudaMemcpyHostToDevice),"bench x");
            std::printf("BENCH type=%d rows=%d K=20 old=HEAD_WIDE(Q6)/native_mmvq(Q8) us\n",type,rows);
            std::printf("T,old_warm,new_warm,old_cold,new_cold,equal_logit_percent\n");
            for(int cols=1;cols<=8;++cols) {
                native_quantize_q8_1(x.as<float>(),xq.p,N,cols,s);
                if(!captured_q.empty())ck(cudaMemcpyAsync(xq.p,captured_q.data(),size_t(cols)*2880,cudaMemcpyHostToDevice,s),"bench captured Q8_1");
                int k=20;ck(cudaMemcpy(keep.p,&k,4,cudaMemcpyHostToDevice),"bench K");
                auto old=[&]{if(type==14)head_wide(w.p,xq.p,ya.as<float>(),N,rows,cols,wide.p,nullptr,nullptr,s);else native_mmvq(type,w.p,xq.p,ya.as<float>(),N,rows,cols,s);};
                auto screen=[&]{head_screen(type,copy.p,w.p,xq.p,yb.as<float>(),rows,cols,keep.as<int32_t>(),scratch.p,s);};
                const double ow=timed(s,old,flush,sink,false),nw=timed(s,screen,flush,sink,false);
                const double oc=timed(s,old,flush,sink,true),nc=timed(s,screen,flush,sink,true);
                ck(cudaMemcpy(a.data(),ya.p,size_t(cols)*rows*4,cudaMemcpyDeviceToHost),"bench exact logits");
                ck(cudaMemcpy(b.data(),yb.p,size_t(cols)*rows*4,cudaMemcpyDeviceToHost),"bench screened logits");
                size_t equal=0;for(size_t i=0;i<size_t(cols)*rows;++i)equal+=std::memcmp(&a[i],&b[i],4)==0;
                std::printf("%d,%.1f,%.1f,%.1f,%.1f,%.3f\n",cols,ow,nw,oc,nc,100.0*double(equal)/double(size_t(cols)*rows));std::fflush(stdout);
            }
        }
    }
    // Use the actual MTP subset when supplied, not an arbitrary prefix of the head.
    std::vector<int32_t> map;
    if(vocab) {
        FILE* f=std::fopen(vocab,"rb");if(!f){std::perror(vocab);return 1;}std::fseek(f,0,SEEK_END);long n=std::ftell(f);std::rewind(f);
        if(n<=0||n%4){std::fclose(f);return 1;}map.resize(n/4);if(std::fread(map.data(),1,n,f)!=size_t(n)){std::fclose(f);return 1;}std::fclose(f);
    } else {
        if(inputs&&!inputs->draft.empty()&&inputs->draft[0].header.vocab_hash)throw std::runtime_error("captured MTP subset requires --draft-vocab");
        map.resize(inputs&&!inputs->draft.empty()?rows:std::min(rows,106299));std::iota(map.begin(),map.end(),0);
    }
    uint64_t vocab_hash=0;
    if(vocab) {vocab_hash=14695981039346656037ull;const auto* p=reinterpret_cast<const uint8_t*>(map.data());
        for(size_t i=0;i<map.size()*4;++i){vocab_hash^=p[i];vocab_hash*=1099511628211ull;}}
    std::vector<uint8_t> sub(map.size()*rb);
    for(size_t i=0;i<map.size();++i){if(map[i]<0||map[i]>=rows)return 1;std::memcpy(sub.data()+i*rb,hw.data()+size_t(map[i])*rb,rb);}
    const int nr=int(map.size());
    Buffer dw(sub.size()),proxy(head_proxy_bytes(nr)),ps(head_proxy_scratch_bytes(nr)),pr(4);
    ck(cudaMemcpy(dw.p,sub.data(),sub.size(),cudaMemcpyHostToDevice),"draft rows");head_proxy_pack(type,dw.p,nr,proxy.p,s);
    int changed=0;
    for(int trial=0;trial<12;++trial) {
        const InputRow* captured=nullptr;
        if(inputs&&!inputs->draft.empty()) {
            captured=&inputs->draft[size_t(trial)*(inputs->draft.size()-1)/11];
            if(int(captured->header.type)!=type||int(captured->header.n_vocab)!=nr||captured->header.vocab_hash!=vocab_hash)
                throw std::runtime_error("draft dump does not match head GGUF/vocabulary hash");
            std::copy(captured->x.begin(),captured->x.end(),hx.begin());
        } else for(int i=0;i<N;++i)hx[i]=trial==0?0.0f:nd(rng);
        ck(cudaMemcpy(x.p,hx.data(),N*4,cudaMemcpyHostToDevice),"draft x");native_quantize_q8_1(x.as<float>(),xq.p,N,1,s);
        if(captured)ck(cudaMemcpyAsync(xq.p,captured->q.data(),2880,cudaMemcpyHostToDevice,s),"captured draft Q8_1");
        native_mmvq(type,dw.p,xq.p,ya.as<float>(),N,nr,1,s);
        head_proxy_pick(type,proxy.p,dw.p,xq.p,nr,yb.as<float>(),ps.p,ob.as<int32_t>(),pr.as<float>(),s);ck(cudaStreamSynchronize(s),"draft proxy");
        ck(cudaMemcpy(a.data(),ya.p,nr*4,cudaMemcpyDeviceToHost),"draft A");ck(cudaMemcpy(b.data(),yb.p,nr*4,cudaMemcpyDeviceToHost),"draft B");
        int pick;float p;ck(cudaMemcpy(&pick,ob.p,4,cudaMemcpyDeviceToHost),"pick");ck(cudaMemcpy(&p,pr.p,4,cudaMemcpyDeviceToHost),"probability");
        const int exact=top(a.data(),nr,1)[0];changed+=pick!=exact;
        if(captured&&(exact!=captured->header.subset_index||map[exact]!=captured->header.token)) {
            ++errors;std::printf("CAPTURE DRAFT PICK MISMATCH position=%d saved=%d replay=%d\n",captured->header.position,captured->header.token,map[exact]);
        }
        if(pick<0||pick>=nr||!std::isfinite(p)||p<=0||p>1||std::memcmp(&a[pick],&b[pick],4)){++errors;std::printf("PROXY MISMATCH type=%d trial=%d pick=%d exact=%d p=%g\n",type,trial,pick,exact,p);}
        head_proxy_pick(type,proxy.p,dw.p,xq.p,nr,yb.as<float>(),ps.p,oa.as<int32_t>(),pr.as<float>(),s);ck(cudaStreamSynchronize(s),"draft repeat");
        int repeat;float rp;ck(cudaMemcpy(&repeat,oa.p,4,cudaMemcpyDeviceToHost),"repeat pick");ck(cudaMemcpy(&rp,pr.p,4,cudaMemcpyDeviceToHost),"repeat p");
        if(repeat!=pick||std::memcmp(&p,&rp,4)){++errors;std::printf("PROXY NONDETERMINISTIC type=%d trial=%d\n",type,trial);}
    }
    if(inputs&&!inputs->draft.empty()) {
        size_t changed_all=0;
        for(size_t i=0;i<inputs->draft.size();++i) {
            const auto& r=inputs->draft[i];
            if(int(r.header.type)!=type||int(r.header.n_vocab)!=nr||r.header.vocab_hash!=vocab_hash)
                throw std::runtime_error("captured draft metadata mismatch");
            ck(cudaMemcpyAsync(xq.p,r.q.data(),2880,cudaMemcpyHostToDevice,s),"replay draft Q8_1");
            native_mmvq(type,dw.p,xq.p,ya.as<float>(),N,nr,1,s);
            head_proxy_pick(type,proxy.p,dw.p,xq.p,nr,yb.as<float>(),ps.p,ob.as<int32_t>(),pr.as<float>(),s);ck(cudaStreamSynchronize(s),"replay draft");
            ck(cudaMemcpy(a.data(),ya.p,nr*4,cudaMemcpyDeviceToHost),"replay draft exact");
            ck(cudaMemcpy(b.data(),yb.p,nr*4,cudaMemcpyDeviceToHost),"replay draft approximate");
            int pick;float p;ck(cudaMemcpy(&pick,ob.p,4,cudaMemcpyDeviceToHost),"replay draft pick");ck(cudaMemcpy(&p,pr.p,4,cudaMemcpyDeviceToHost),"replay draft probability");
            const int exact=top(a.data(),nr,1)[0];changed_all+=pick!=exact;
            if(exact!=r.header.subset_index||map[exact]!=r.header.token) {++errors;std::printf("SAVED DRAFT PICK MISMATCH input=%zu\n",i);}
            if(pick<0||pick>=nr||!std::isfinite(p)||p<=0||p>1||std::memcmp(&a[pick],&b[pick],4)) {
                ++errors;std::printf("CAPTURE PROXY MISMATCH input=%zu\n",i);
            }
        }
        std::printf("captured draft sweep: %zu inputs, changed proposals=%zu, chosen-row bits exact, cumulative errors=%ld\n",inputs->draft.size(),changed_all,errors);
    }
    std::printf("proxy parity type=%d rows=%d: 12 cases, exact chosen-row bits checked, changed drafts=%d, total errors=%ld\n",type,nr,changed,errors);
    if(bench) {
        Buffer flush(384ull<<20),sink(4);ck(cudaMemset(flush.p,1,384ull<<20),"flush init");ck(cudaMemset(sink.p,0,4),"sink");
        auto old=[&]{native_mmvq(type,dw.p,xq.p,ya.as<float>(),N,nr,1,s);row_top_prob(ya.as<float>(),1,nr,oa.as<int>(),pr.as<float>(),s);};
        auto now=[&]{head_proxy_pick(type,proxy.p,dw.p,xq.p,nr,yb.as<float>(),ps.p,ob.as<int32_t>(),pr.as<float>(),s);};
        std::printf("DRAFT type=%d rows=%d old_warm=%.1f proxy_warm=%.1f old_cold=%.1f proxy_cold=%.1f us\n",type,nr,
            timed(s,old,flush,sink,false),timed(s,now,flush,sink,false),timed(s,old,flush,sink,true),timed(s,now,flush,sink,true));
    }
    ck(cudaStreamDestroy(s),"destroy stream");return errors?1:0;
}
int main(int argc,char** argv) {
    policy();bool bench=false;int rows=8192,input_limit=256;
    const char *gguf=nullptr,*vocab=nullptr,*input_path=nullptr,*dump_path=nullptr;
    for(int i=1;i<argc;++i) {
        if(!std::strcmp(argv[i],"--bench"))bench=true;
        else if(!std::strcmp(argv[i],"--rows")&&i+1<argc)rows=std::atoi(argv[++i]);
        else if(!std::strcmp(argv[i],"--gguf")&&i+1<argc)gguf=argv[++i];
        else if(!std::strcmp(argv[i],"--draft-vocab")&&i+1<argc)vocab=argv[++i];
        else if(!std::strcmp(argv[i],"--inputs")&&i+1<argc)input_path=argv[++i];
        else if(!std::strcmp(argv[i],"--input-max")&&i+1<argc)input_limit=std::atoi(argv[++i]);
        else if(!std::strcmp(argv[i],"--dump-test")&&i+1<argc)dump_path=argv[++i];
        else {std::fprintf(stderr,"usage: hip_head_screen_parity [--rows N] [--gguf shard] [--draft-vocab file] [--inputs SHD1] [--input-max N] [--bench] [--dump-test path]\n");return 1;}
    }
    if(rows<1||rows>1048576||input_limit<1||input_limit>20000||(input_path&&!gguf))return 1;
    int count=0;ck(cudaGetDeviceCount(&count),"devices");if(!count)return 77;ck(cudaSetDevice(0),"device 0");
    proxy_policy();
    if(dump_path)return dump_smoke(dump_path);
    DumpRows inputs;if(input_path)inputs=read_inputs(input_path,input_limit);
    if(gguf) {
        strata::GgufFile f(gguf);const auto* t=f.find("output.weight");
        if(!t||t->shape.size()!=2||t->shape[0]!=2560||t->shape[1]>1048576) {std::fprintf(stderr,"not a supported output.weight shard\n");return 1;}
        rows=int(t->shape[1]);const auto n=strata::tensor_payload_bytes(*t);
        if(t->offset>f.file_size()-f.data_start()||n>f.file_size()-f.data_start()-t->offset)return 1;
        std::printf("head-only GGUF %s: type=%u shape=%llu x %llu bytes=%llu\n",gguf,t->type,(unsigned long long)t->shape[1],(unsigned long long)t->shape[0],(unsigned long long)n);
        return exercise(int(t->type),std::vector<uint8_t>(f.tensor_data(*t),f.tensor_data(*t)+n),rows,bench,true,vocab,input_path?&inputs:nullptr);
    }
    std::mt19937_64 rng(3327);int result=0;
    for(int type:{8,14,13})for(int n:{1,3,577,rows}) {
        result|=exercise(type,synthetic(type,n,rng),n,bench&&n==rows,false,nullptr);
    }
    return result;
}
