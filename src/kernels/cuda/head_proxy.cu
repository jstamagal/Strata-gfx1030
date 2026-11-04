// Basalt's fitted two-bit proposal head, using gfx1030 dot4 instead of tensor cores.
#include "strata/kernels/head_screen.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#if defined(__HIPCC__)
#include "head_exact.cuh"
#endif

namespace strata::kernels {
namespace {
int tiles(int n) { return (n+31)/32; }
#if defined(__HIPCC__)
using namespace hdetail;
constexpr int kPer = 4, kCap = 64;
__device__ __forceinline__ int level(float v, float step) { return min(3,max(0,int(floorf(v/step+2.0f)))); }

// The same six trial steps and half-way norm correction as Basalt's fit_group<true>.
__device__ half fit(const float* w, float& step) {
    float am = 1e-9f;
#pragma unroll
    for (int i=0;i<32;++i) am=fmaxf(am,fabsf(w[i]));
    float best=am/1.5f, be=pinf();
    for (int t=0;t<6;++t) {
        const float s=am/1.5f*(1.0f-0.1f*float(t));
        const float sh=__half2float(__float2half(s));
        float e=0.0f;
#pragma unroll
        for (int i=0;i<32;++i) { const float r=(float(level(w[i],sh))-1.5f)*sh-w[i]; e+=r*r; }
        if (e<be) { be=e; best=s; }
    }
    half hs=__float2half(best); step=__half2float(hs);
    float ww=0.0f,wr=0.0f;
#pragma unroll
    for (int i=0;i<32;++i) { ww+=w[i]*w[i]; wr+=w[i]*(float(level(w[i],step))-1.5f)*step; }
    if (wr>0.0f) hs=__float2half(step*sqrtf(ww/wr));
    return hs;
}

template<int TYPE>
__global__ void proxy_pack_kernel(const void* weights,int rows,uint2* codes,half* scales) {
    const int i=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    const int cap=((rows+31)/32)*32;
    if (i>=cap*80) return;
    const int row=i/80,g=i%80;
    const size_t at=(size_t(row/32)*80+g)*32+row%32;
    if (row>=rows) { codes[at]=make_uint2(0,0); scales[at]=__float2half(0); return; }
    float w[32];
#pragma unroll
    for (int l=0;l<32;++l) {
        if constexpr (TYPE==8) {
            const Q80& b=static_cast<const Q80*>(weights)[size_t(row)*80+g];
            w[l]=__half2float(b.d)*float(b.qs[l]);
        } else if constexpr (TYPE==14) {
            const Q6& b=static_cast<const Q6*>(weights)[size_t(row)*10+g/8];
            const int k=(g%8)*32+l;
            w[l]=__half2float(b.d)*float(q6scale(b,k))*float(q6code(b,k)-32);
        } else {
            const Q5& b=static_cast<const Q5*>(weights)[size_t(row)*10+g/8];
            const int gg=g%8; int sc,mn; q5scale(b,gg,sc,mn);
            const int low=(b.qs[(gg/2)*32+l]>>(4*(gg&1)))&15;
            const int high=(b.qh[l]>>gg)&1;
            const float2 dm=__half22float2(b.dm);
            w[l]=dm.x*float(sc)*float(low|(high<<4))-dm.y*float(mn);
        }
    }
    float step; const half scale=fit(w,step);
    uint32_t v[2]={0,0};
#pragma unroll
    for (int c=0;c<2;++c)
#pragma unroll
        for (int t=0;t<4;++t)
#pragma unroll
            for (int b=0;b<4;++b) v[c]|=uint32_t(level(w[c*16+t*4+b],step))<<(8*b+2*t);
    codes[at]=make_uint2(v[0],v[1]); scales[at]=scale;
}

__device__ __forceinline__ float proxy_dot(uint2 v,half scale,const X81& a) {
    int dot=0,sx=0;
#pragma unroll
    for (int h=0;h<2;++h) {
        const uint32_t q=h?v.y:v.x;
#pragma unroll
        for (int k=0;k<4;++k) {
            const int x=word(a.qs+h*16+k*4);
            dot=STRATA_DP4A(int((q>>(2*k))&0x03030303u),x,dot);
            sx=STRATA_DP4A(0x01010101,x,sx);
        }
    }
    return __half2float(scale)*(__low2float(a.ds)*float(dot)-1.5f*(__low2float(a.ds)*float(sx)));
}

__launch_bounds__(128)
__global__ void score_kernel(const uint2* codes,const half* scales,const X81* x,int rows,
                              float* logits,float2* cand,float2* part) {
    __shared__ X81 ax[80];
    __shared__ float red[4][32];
    const int tid=int(threadIdx.x),wave=tid>>5,lane=tid&31,tile=int(blockIdx.x),row=tile*32+lane;
    for (int i=tid;i<80*9;i+=128) reinterpret_cast<int*>(ax)[i]=reinterpret_cast<const int*>(x)[i];
    __syncthreads();
    float acc=0.0f;
#pragma unroll
    for (int t=0;t<20;++t) {
        const int g=20*wave+t;
        const size_t at=(size_t(tile)*80+g)*32+lane;
        acc+=proxy_dot(codes[at],scales[at],ax[g]);
    }
    red[wave][lane]=acc;
    __syncthreads();
    if (wave) return;
    float v=((red[0][lane]+red[1][lane])+red[2][lane])+red[3][lane];
    if (row>=rows || !finite(v)) v=ninf();
    if (row<rows) logits[row]=v;
    float m=v;
    for (int o=16;o;o>>=1) m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,o));
    float e=v>ninf()?__expf(v-m):0.0f; e=sum32(e);
    if (!lane) part[tile]=make_float2(m,e);
#pragma unroll
    for (int k=0;k<kPer;++k) {
        float bv=v; int bi=row;
        for (int o=16;o;o>>=1) {
            const float ov=__shfl_xor_sync(0xffffffffu,bv,o); const int oi=__shfl_xor_sync(0xffffffffu,bi,o);
            if (ov>bv||(ov==bv&&oi<bi)) { bv=ov; bi=oi; }
        }
        if (!lane) cand[tile*kPer+k]=make_float2(bv,__int_as_float(bi));
        if (row==bi) v=ninf();
    }
}
__device__ unsigned long long key(float2 c) {
    // Treat both signed zeros as equal, then prefer the lower subset row.
    const unsigned u=c.x==0.0f?0u:__float_as_uint(c.x);
    const unsigned o=(u&0x80000000u)?~u:(u|0x80000000u);
    return (static_cast<unsigned long long>(o)<<32)|(0xffffffffu-unsigned(__float_as_int(c.y)));
}
struct Final { unsigned hist[256]; unsigned long long want; int need,count,list[kCap]; float approx[kCap],exact[kCap],max[8]; };

template<int TYPE>
__launch_bounds__(256)
__global__ void final_kernel(const void* weights,const X81* x,int nt,int rows,float* logits,
                              const float2* cand,const float2* part,int32_t* pick,float* probability) {
    __shared__ X81 ax[80]; __shared__ Final f;
    const int tid=int(threadIdx.x),wave=tid>>5,lane=tid&31,nc=nt*kPer;
    float m=ninf();
    for (int i=tid;i<nc;i+=256) m=fmaxf(m,cand[i].x);
    for (int o=16;o;o>>=1) m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,o));
    if (!lane) f.max[wave]=m;
    if (!tid) { f.want=0; f.need=kCap; f.count=0; }
    for (int i=tid;i<80*9;i+=256) reinterpret_cast<int*>(ax)[i]=reinterpret_cast<const int*>(x)[i];
    __syncthreads();
    m=f.max[0]; for (int i=1;i<8;++i) m=fmaxf(m,f.max[i]);
    if (m==ninf()) { if (!tid) { *pick=0; *probability=1.0f; } return; }
    for (int i=tid;i<nc;i+=256) if (cand[i].x>=m-5.0f && cand[i].x>ninf()) atomicAdd(&f.count,1);
    __syncthreads();
    const int total=f.count;
    if (total>kCap) {
        unsigned long long mask=0;
        for (int shift=56;shift>=0;shift-=8) {
            f.hist[tid]=0; __syncthreads();
            const auto want=f.want;
            for (int i=tid;i<nc;i+=256) {
                const float2 c=cand[i]; const auto k=key(c);
                if (c.x>=m-5.0f && c.x>ninf() && (k&mask)==want) atomicAdd(&f.hist[(k>>shift)&255],1u);
            }
            __syncthreads();
            if (!tid) {
                int need=f.need,b=255;
                for (;b>0;--b) { if (need<=int(f.hist[b])) break; need-=int(f.hist[b]); }
                f.want=want|(static_cast<unsigned long long>(b)<<shift); f.need=need;
            }
            mask|=255ull<<shift; __syncthreads();
        }
    }
    if (!tid) f.count=0;
    __syncthreads();
    for (int i=tid;i<nc;i+=256) {
        const float2 c=cand[i];
        if (c.x>=m-5.0f && c.x>ninf() && (total<=kCap||key(c)>=f.want)) {
            const int at=atomicAdd(&f.count,1); if (at<kCap) { f.list[at]=__float_as_int(c.y); f.approx[at]=c.x; }
        }
    }
    __syncthreads();
    const int n=min(total,kCap);
    int row=0,rank=0; float approx=0.0f;
    if (tid<n) { row=f.list[tid]; approx=f.approx[tid]; for (int j=0;j<n;++j) rank+=f.list[j]<row; }
    __syncthreads();
    if (tid<n) { f.list[rank]=row; f.approx[rank]=approx; }
    __syncthreads();
    for (int i=wave;i<n;i+=8) {
        float v[1]; exact_row<TYPE,1>(weights,ax,f.list[i],lane,v);
        if (!lane) { f.exact[i]=v[0]; logits[f.list[i]]=v[0]; }
    }
    __syncthreads();
    if (wave) return;
    float bv=ninf(); int bi=0x7fffffff;
    for (int i=lane;i<n;i+=32)
        if (f.exact[i]>bv||(f.exact[i]==bv&&f.list[i]<bi)) { bv=f.exact[i]; bi=f.list[i]; }
    for (int o=16;o;o>>=1) {
        const float ov=__shfl_xor_sync(0xffffffffu,bv,o); const int oi=__shfl_xor_sync(0xffffffffu,bi,o);
        if (ov>bv||(ov==bv&&oi<bi)) { bv=ov; bi=oi; }
    }
    float sum=0.0f;
    for (int t=lane;t<nt;t+=32) { const float2 p=part[t]; if (p.y>0.0f) sum+=p.y*__expf(p.x-bv); }
    for (int i=lane;i<n;i+=32) sum+=__expf(f.exact[i]-bv)-__expf(f.approx[i]-bv);
    sum=sum32(sum);
    if (!lane) { *pick=bi<rows?bi:0; *probability=finite(sum)&&sum>1.0f?1.0f/sum:1.0f; }
}
template<int TYPE>
void launch_proxy(const void* proxy,const void* w,const void* x,int rows,float* y,void* scratch,int32_t* pick,float* p,cudaStream_t s) {
    const int nt=tiles(rows); const size_t groups=size_t(nt)*32*80;
    const auto* codes=static_cast<const uint2*>(proxy); const auto* scales=reinterpret_cast<const half*>(codes+groups);
    auto* cand=static_cast<float2*>(scratch); float2* part=cand+nt*kPer;
    score_kernel<<<nt,128,0,s>>>(codes,scales,static_cast<const X81*>(x),rows,y,cand,part);
    final_kernel<TYPE><<<1,256,0,s>>>(w,static_cast<const X81*>(x),nt,rows,y,cand,part,pick,p);
}
__launch_bounds__(128)
__global__ void norm_kernel(const Q6* weights,const uint2* codes,const half* scales,float2* norms,int rows) {
    const int row=int(blockIdx.x),g=int(threadIdx.x);
    if (row>=rows) return;
    float re=0.0f,ww=0.0f,pp=0.0f;
    if (g<80) {
        const size_t at=(size_t(row/32)*80+g)*32+row%32;
        const uint2 v=codes[at]; const float scale=__half2float(scales[at]);
        const Q6& b=weights[size_t(row)*10+g/8];
#pragma unroll
        for (int l=0;l<32;++l) {
            const int k=(g%8)*32+l;
            const float w=__half2float(b.d)*float(q6scale(b,k))*float(q6code(b,k)-32);
            const int q=int(((l<16?v.x:v.y)>>(8*(l%4)+2*((l%16)/4)))&3u);
            const float p=scale*(float(q)-1.5f),e=w-p;
            re+=e*e; ww+=w*w; pp+=p*p;
        }
    }
    __shared__ float4 part[128]; part[g]=make_float4(re,ww,pp,0.0f);
    __syncthreads();
    for (int o=64;o;o>>=1) {
        if (g<o) { part[g].x+=part[g+o].x;part[g].y+=part[g+o].y;part[g].z+=part[g+o].z; }
        __syncthreads();
    }
    if (!g) norms[row]=make_float2(sqrtf(part[0].x)*1.001f,(sqrtf(part[0].y)+sqrtf(part[0].z))*1.001f);
}

template<int NC>
__launch_bounds__(128)
__global__ void bound_screen_kernel(const uint2* codes,const half* scales,const float2* norms,
                                     const X81* x,int rows,float* logits,float* lower,const int32_t* keep) {
    if (*keep<=0 || *keep>64) return;
    __shared__ X81 ax[NC][80];
    __shared__ float xx[NC][80],xn[NC],red[4][NC][32];
    __shared__ float maxima[NC][32];
    const int tid=int(threadIdx.x),wave=tid>>5,lane=tid&31;
    for (int i=tid;i<NC*80;i+=128) {
        const X81 a=x[i];ax[i/80][i%80]=a;
        int sq=0;
#pragma unroll
        for (int k=0;k<8;++k) { const int q=word(a.qs+4*k);sq=STRATA_DP4A(q,q,sq); }
        const float d=__low2float(a.ds);xx[i/80][i%80]=d*d*float(sq);
    }
    __syncthreads();
    if (tid<NC) {float v=0.0f;for(int g=0;g<80;++g)v+=xx[tid][g];xn[tid]=sqrtf(v)*1.001f;}
    __syncthreads();
    float best[NC];
#pragma unroll
    for (int j=0;j<NC;++j)best[j]=ninf();
    const int nt=(rows+31)/32;
    for (int tile=int(blockIdx.x);tile<nt;tile+=kHeadScreenGrid) {
        float c[NC]={};
        // A lane owns a row; four waves divide its groups. Never unroll the whole head into registers.
#pragma unroll 1
        for (int t=0;t<20;++t) {
            const int g=20*wave+t;const size_t at=(size_t(tile)*80+g)*32+lane;
            const uint2 v=codes[at];const half d=scales[at];
#pragma unroll
            for (int j=0;j<NC;++j)c[j]+=proxy_dot(v,d,ax[j][g]);
        }
#pragma unroll
        for (int j=0;j<NC;++j)red[wave][j][lane]=c[j];
        __syncthreads();
        const int row=tile*32+lane;
        if (!wave && row<rows) {
            const float2 n=norms[row];
#pragma unroll
            for (int j=0;j<NC;++j) {
                const float center=((red[0][j][lane]+red[1][j][lane])+red[2][j][lane])+red[3][j][lane];
                // Cauchy bounds the residual; 2^-8 exceeds 7*gamma_4096, including the proxy's
                // unsigned-dot centering. Inflated norms and the final margin cover their reductions/endpoints.
                const float radius=fmaf((n.x+0x1p-8f*n.y)*xn[j],1.0625f,1e-6f);
                float lo=center-radius,hi=center+radius;
                if (!finite(lo)||!finite(hi)) {lo=ninf();hi=pinf();}
                logits[size_t(j)*rows+row]=hi;best[j]=fmaxf(best[j],lo);
            }
        }
        __syncthreads();
    }
    if (!wave) {
#pragma unroll
        for(int j=0;j<NC;++j)maxima[j][lane]=best[j];
    }
    __syncthreads();
    if(tid<NC) {
        float v=ninf();for(int l=0;l<32;++l)v=fmaxf(v,maxima[tid][l]);
        lower[size_t(tid)*kHeadScreenGrid+blockIdx.x]=v;
    }
}
void check_proxy() { const auto e=cudaGetLastError(); if (e!=cudaSuccess) throw std::runtime_error(std::string("head proxy: ")+cudaGetErrorString(e)); }
#endif
} // namespace
bool head_proxy_enabled(int type,int n_in,int device) {
    if(!head_proxy_supported(type,n_in))return false;
    if(const char* e=std::getenv("STRATA_MTP_HEAD_PROXY"))return e[0]=='1'&&!e[1];
#if defined(__HIPCC__)
    if(type!=8&&type!=14)return false;
    if(device<0&&cudaGetDevice(&device)!=cudaSuccess){(void)cudaGetLastError();return false;}
    cudaDeviceProp p{};
    if(cudaGetDeviceProperties(&p,device)!=cudaSuccess){(void)cudaGetLastError();return false;}
    return std::strncmp(p.gcnArchName,"gfx1030",7)==0&&(p.gcnArchName[7]=='\0'||p.gcnArchName[7]==':');
#else
    (void)device;return false;
#endif
}
std::size_t head_proxy_reserve_bytes(int type,int n_in,int rows,int device,bool owns_proxy) {
    if(rows<=0||rows>1048576||!head_proxy_enabled(type,n_in,device))return 0;
    return (owns_proxy?head_proxy_bytes(rows):0)+head_proxy_scratch_bytes(rows);
}
bool head_proxy_supported(int type,int n_in) {
#if defined(__HIPCC__)
    return (type==8||type==13||type==14)&&n_in==2560;
#else
    (void)type; (void)n_in; return false;
#endif
}
std::size_t head_proxy_bytes(int rows) { return rows>0?size_t(tiles(rows))*32*80*10:0; }
std::size_t head_proxy_scratch_bytes(int rows) { return rows>0?size_t(tiles(rows))*5*8:0; }
void head_proxy_pack(int type,const void* weights,int rows,void* proxy,void* stream) {
    if (!head_proxy_supported(type,2560)||rows<=0||rows>1048576||!weights||!proxy) throw std::invalid_argument("head proxy pack: unsupported head");
#if defined(__HIPCC__)
    const size_t groups=size_t(tiles(rows))*32*80; auto* q=static_cast<uint2*>(proxy); auto* sc=reinterpret_cast<half*>(q+groups);
    const auto s=static_cast<cudaStream_t>(stream);
#define PACK(T) proxy_pack_kernel<T><<<(groups+127)/128,128,0,s>>>(weights,rows,q,sc)
    if (type==8) { PACK(8); } else if (type==14) { PACK(14); } else { PACK(13); }
#undef PACK
    check_proxy();
#endif
}
void head_proxy_pick(int type,const void* proxy,const void* w,const void* x,int rows,float* y,void* scratch,int32_t* pick,float* p,void* stream) {
    if (!head_proxy_supported(type,2560)||rows<=0||rows>1048576||!proxy||!w||!x||!y||!scratch||!pick||!p) throw std::invalid_argument("head proxy: unsupported head");
#if defined(__HIPCC__)
    const auto s=static_cast<cudaStream_t>(stream);
    if (type==8) launch_proxy<8>(proxy,w,x,rows,y,scratch,pick,p,s);
    else if (type==14) launch_proxy<14>(proxy,w,x,rows,y,scratch,pick,p,s);
    else launch_proxy<13>(proxy,w,x,rows,y,scratch,pick,p,s);
    check_proxy();
#endif
}
std::size_t head_proxy_bound_bytes(int rows) { return head_proxy_bytes(rows)+(rows>0?size_t(rows)*8:0); }
void head_proxy_bound_pack(const void* weights,int rows,void* proxy,void* stream) {
    head_proxy_pack(14,weights,rows,proxy,stream);
#if defined(__HIPCC__)
    const size_t groups=size_t(tiles(rows))*32*80;
    auto* q=static_cast<uint2*>(proxy);const auto* sc=reinterpret_cast<const half*>(q+groups);
    auto* norms=reinterpret_cast<float2*>(static_cast<char*>(proxy)+head_proxy_bytes(rows));
    norm_kernel<<<rows,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const Q6*>(weights),q,sc,norms,rows);
    check_proxy();
#endif
}
void head_proxy_screen(const void* proxy,const void* x,float* y,float* lower,int rows,int cols,const int32_t* keep,void* stream) {
    if(!proxy||!x||!y||!lower||!keep||rows<=0||cols<1||cols>8)throw std::invalid_argument("head proxy screen: invalid buffers/shape");
#if defined(__HIPCC__)
    const size_t groups=size_t(tiles(rows))*32*80;
    const auto* q=static_cast<const uint2*>(proxy);const auto* sc=reinterpret_cast<const half*>(q+groups);
    const auto* norms=reinterpret_cast<const float2*>(static_cast<const char*>(proxy)+head_proxy_bytes(rows));
    const auto s=static_cast<cudaStream_t>(stream);
#define BS(N) case N: bound_screen_kernel<N><<<kHeadScreenGrid,128,0,s>>>(q,sc,norms,static_cast<const X81*>(x),rows,y,lower,keep);break
    switch(cols){BS(1);BS(2);BS(3);BS(4);BS(5);BS(6);BS(7);BS(8);}
#undef BS
    check_proxy();
#endif
}
} // namespace strata::kernels
