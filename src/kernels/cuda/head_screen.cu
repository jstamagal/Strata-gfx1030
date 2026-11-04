// Certified omitted-bit screening; original GGUF rows supply the exact rescore.
#include "strata/kernels/head_screen.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#if defined(__HIPCC__)
#include "head_exact.cuh"
#endif

namespace strata::kernels {
namespace {
constexpr int kGrid = kHeadScreenGrid, kHist = 1024, kSlice = 128;
struct Scratch { float lower[8][kGrid], threshold[8]; };
int rowbytes(int type) { return type == 8 ? 2080 : 1792; }
bool env_on(const char* name) { const char* e = std::getenv(name); return e && e[0] == '1' && !e[1]; }
bool fitted_q6() { const char* e = std::getenv("STRATA_HEAD_Q6_BITS"); return e && e[0] == '2' && !e[1]; }
#if defined(__HIPCC__)
using namespace hdetail;

// Q8: Basalt's scale/nibble/top planes, without the low plane (exact rows stay in GGUF).
// Q6: per group, 16 nibble bytes + 4 top-bit bytes; then signed scales and block d's.
template<int TYPE>
__global__ void pack_kernel(const void* weights, uint8_t* copy, int rows) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= rows * 80) return;
    const int r = i / 80, g = i % 80;
    uint8_t* dst = copy + size_t(r) * (TYPE == 8 ? 2080 : 1792);
    uint8_t h[32];
    if constexpr (TYPE == 8) {
        const Q80& b = static_cast<const Q80*>(weights)[size_t(r) * 80 + g];
        reinterpret_cast<half*>(dst)[g] = b.d;
#pragma unroll
        for (int l = 0; l < 32; ++l) h[l] = (uint8_t(b.qs[l]) ^ 128) >> 2;
    } else {
        const Q6& b = static_cast<const Q6*>(weights)[size_t(r) * 10 + g / 8];
#pragma unroll
        for (int l = 0; l < 32; ++l) h[l] = q6code(b, (g % 8) * 32 + l) >> 1;
        reinterpret_cast<int8_t*>(dst + 1600)[2*g] = q6scale(b, (g % 8) * 32);
        reinterpret_cast<int8_t*>(dst + 1600)[2*g+1] = q6scale(b, (g % 8) * 32 + 16);
        if (g % 8 == 0) reinterpret_cast<half*>(dst + 1760)[g / 8] = b.d;
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint32_t n = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) n |= uint32_t((h[8*p+j] & 15) | ((h[8*p+4+j] & 15) << 4)) << (8*j);
        reinterpret_cast<uint32_t*>(dst + (TYPE == 8 ? 160 + g*16 : g*20))[p] = n;
    }
    if constexpr (TYPE == 8) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            uint32_t t = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) t |= uint32_t(h[16*p+4*q+j] >> 4) << (8*j+2*q);
            reinterpret_cast<uint32_t*>(dst + 1440 + g*8)[p] = t;
        }
    } else {
        uint32_t t = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int q = 0; q < 8; ++q) t |= uint32_t(h[4*q+j] >> 4) << (8*j+q);
        *reinterpret_cast<uint32_t*>(dst + g*20 + 16) = t;
    }
}

struct Activation { int q[8]; float d; int sum[2], abs[2]; };
__device__ __forceinline__ Activation load_activation(const X81& a) {
    Activation z;z.d=__low2float(a.ds);
#pragma unroll
    for(int h=0;h<2;++h) {
        int sx=0,sa=0;
#pragma unroll
        for(int t=0;t<4;++t) {
            const int v=word(a.qs+h*16+t*4);z.q[h*4+t]=v;sx=STRATA_DP4A(v,0x01010101,sx);
#pragma unroll
            for(int b=0;b<4;++b)sa+=abs(int(int8_t(uint32_t(v)>>(8*b))));
        }
        z.sum[h]=sx;z.abs[h]=sa;
    }
    return z;
}
template<int TYPE>
__device__ __forceinline__ void screen_group(const uint8_t* row,int g,const Activation& a,
                                             float& center,float& width,unsigned& bad) {
    const uint32_t* nb=reinterpret_cast<const uint32_t*>(row+(TYPE==8?160+g*16:g*20));
    int dot[2]={};
#pragma unroll
    for(int h=0;h<2;++h) {
#pragma unroll
        for(int p=0;p<2;++p) {
            const int q=h*2+p;const uint32_t n=nb[q];
            const uint32_t t=TYPE==8?reinterpret_cast<const uint32_t*>(row+1440+g*8)[q/2]:nb[4];
            const int sh=TYPE==8?4*(q&1):2*q;const uint32_t mask=TYPE==8?0x03030303u:0x01010101u;
            const int v0=int((n&0x0f0f0f0fu)|(((t>>sh)&mask)<<4));
            const int v1=int(((n>>4)&0x0f0f0f0fu)|(((t>>(sh+(TYPE==8?2:1)))&mask)<<4));
            dot[h]=STRATA_DP4A(v0,a.q[h*4+p*2],dot[h]);dot[h]=STRATA_DP4A(v1,a.q[h*4+p*2+1],dot[h]);
        }
    }
    if constexpr(TYPE==8) {
        const float scale=__half2float(reinterpret_cast<const half*>(row)[g])*a.d;
        const float mid=4.0f*float(dot[0]+dot[1])-126.5f*float(a.sum[0]+a.sum[1]);
        center+=scale*mid;width+=fabsf(scale)*(1.5f*float(a.abs[0]+a.abs[1]));bad|=!finite(scale);
    } else {
        const float d=__half2float(reinterpret_cast<const half*>(row+1760)[g/8]);
#pragma unroll
        for(int h=0;h<2;++h) {
            const float scale=(d*reinterpret_cast<const int8_t*>(row+1600)[2*g+h])*a.d;
            const float mid=2.0f*float(dot[h])-31.5f*float(a.sum[h]);
            center+=scale*mid;width+=fabsf(scale)*(0.5f*float(a.abs[h]));bad|=!finite(scale);
        }
    }
}
template<int TYPE, int NC>
__launch_bounds__(128)
__global__ void screen_kernel(const uint8_t* copy, const X81* x, float* y, int rows,
                               const int32_t* keep, Scratch* sc) {
    if (*keep <= 0 || *keep > 64) return;
    constexpr int P = NC <= 1 ? 1 : NC <= 2 ? 2 : NC <= 4 ? 4 : 8, G = 32/P;
    __shared__ Activation ax[NC][80];
    __shared__ float wm[4][NC];
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5;
    const int col = lane/G, glane = lane%G;
    for (int i = tid; i < NC * 80; i += 128) ax[i/80][i%80]=load_activation(x[i]);
    __syncthreads();
    float best = ninf();
    for (int r = int(blockIdx.x)*4+wave; r < rows; r += kGrid*4) {
        const uint8_t* row = copy+size_t(r)*(TYPE==8?2080:1792);
        float center = 0.0f, width = 0.0f; unsigned bad = 0;
        // Columns occupy subwaves; repeated weight addresses coalesce, without NC copies in VGPRs.
        for (int g = glane; g < 80; g += G) screen_group<TYPE>(row,g,ax[col < NC ? col : 0][g],center,width,bad);
#pragma unroll
        for (int o = G/2; o; o >>= 1) {
            center += __shfl_xor_sync(0xffffffffu,center,o);
            width += __shfl_xor_sync(0xffffffffu,width,o);
            bad |= __shfl_xor_sync(0xffffffffu,bad,o);
        }
        // The half-integer midpoints are exact in FP32 before scaling. Magnitude sums then stay
        // below 86*w; even 4096 combined roundings at u=2^-24 consume < 1/32*w.
        const float margin = fmaf(width,1.0625f,1e-6f);
        float lo = center-margin, hi = center+margin;
        if (bad || !finite(lo) || !finite(hi)) { lo=ninf(); hi=pinf(); }
        if (!glane && col < NC) y[size_t(col)*rows+r] = hi;
        best = fmaxf(best,lo);
    }
    if (!glane && col < NC) wm[wave][col] = best;
    __syncthreads();
    if (tid < NC) sc->lower[tid][blockIdx.x] = fmaxf(fmaxf(wm[0][tid],wm[1][tid]),fmaxf(wm[2][tid],wm[3][tid]));
}

// At wider T, columns use whole subwaves of the workgroup. Activation groups stay in registers:
// no large LDS image limits residency, and the four waves reuse each row's weight cache lines.
template<int TYPE,int NC>
__launch_bounds__(128)
__global__ void screen_wide_kernel(const uint8_t* copy,const X81* x,float* y,int rows,
                                    const int32_t* keep,Scratch* sc) {
    if(*keep<=0||*keep>64)return;
    const int tid=int(threadIdx.x),col=tid/16,lane=tid%16;
    Activation a[5];
#pragma unroll
    for(int s=0;s<5;++s)a[s]=load_activation(x[(col<NC?col:0)*80+lane+s*16]);
    float best=ninf();
    for(int r=int(blockIdx.x);r<rows;r+=kGrid) {
        const uint8_t* row=copy+size_t(r)*(TYPE==8?2080:1792);
        float center=0.0f,width=0.0f;unsigned bad=0;
#pragma unroll
        for(int s=0;s<5;++s)screen_group<TYPE>(row,lane+s*16,a[s],center,width,bad);
#pragma unroll
        for(int o=8;o;o>>=1) {
            center+=__shfl_xor_sync(0xffffffffu,center,o);width+=__shfl_xor_sync(0xffffffffu,width,o);
            bad|=__shfl_xor_sync(0xffffffffu,bad,o);
        }
        const float margin=fmaf(width,1.0625f,1e-6f);float lo=center-margin,hi=center+margin;
        if(bad||!finite(lo)||!finite(hi)){lo=ninf();hi=pinf();}
        if(!lane&&col<NC)y[size_t(col)*rows+r]=hi;
        best=fmaxf(best,lo);
    }
    if(!lane&&col<NC)sc->lower[col][blockIdx.x]=best;
}

// Maxima refer to disjoint rows. The K-th maximum's bin lower edge cannot exceed the K-th exact logit.
__global__ void threshold_kernel(const int32_t* keep, Scratch* sc) {
    const int k = *keep;
    if (k <= 0 || k > 64) return;
    __shared__ unsigned hist[kHist];
    const int j = int(blockIdx.x), tid = int(threadIdx.x);
    for (int b = tid; b < kHist; b += 256) hist[b] = 0;
    __syncthreads();
    for (int g = tid; g < kGrid; g += 256) {
        float v = sc->lower[j][g];
        if (finite(v)) {
            v = fminf(1000.0f, fmaxf(-1000.0f, v));
            const int b = min(kHist-1, max(0, int(floorf(v*8.0f)) + kHist/2));
            atomicAdd(hist+b, 1u);
        }
    }
    __syncthreads();
    if (tid == 0) {
        unsigned count = 0; int b = kHist-1;
        for (; b > 0; --b) { count += hist[b]; if (count >= unsigned(k)) break; }
        sc->threshold[j] = b > 0 ? float(b-kHist/2)/8.0f : ninf();
    }
}

template<int TYPE, int NC>
__launch_bounds__(256)
__global__ void rescore_kernel(const void* weights, const X81* x, float* y, int rows,
                                const int32_t* keep, const Scratch* sc) {
    __shared__ X81 ax[NC*80];
    __shared__ uint32_t list[kSlice*NC];
    __shared__ int count;
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5;
    const bool all = *keep <= 0 || *keep > 64;
    for (int i = tid; i < NC*80*9; i += 256)
        reinterpret_cast<int*>(ax)[i] = reinterpret_cast<const int*>(x)[i];
    if (tid == 0) count = 0;
    __syncthreads();
    const int r = int(blockIdx.x)*kSlice + tid;
    if (tid < kSlice && r < rows) {
#pragma unroll
        for (int j = 0; j < NC; ++j)
            if (all || !(y[size_t(j)*rows+r] < sc->threshold[j]))
                list[atomicAdd(&count,1)] = uint32_t(tid) | (uint32_t(j) << 16);
    }
    __syncthreads();
    for (int i = wave; i < count; i += 8) {
        const uint32_t entry = list[i];
        const int row = int(blockIdx.x)*kSlice + int(entry & 0xffffu), col = int(entry >> 16);
        float v[1]; exact_row<TYPE,1>(weights,ax+col*80,row,lane,v);
        if (!lane) y[size_t(col)*rows+row] = v[0];
    }
}
template<int TYPE, int NC>
void launch(const void* copy, const void* weights, const void* x, float* y, int rows,
            const int32_t* keep, void* scratch, cudaStream_t s) {
    auto* sc = static_cast<Scratch*>(scratch);
    if (TYPE == 14 && fitted_q6()) head_proxy_screen(copy,x,y,&sc->lower[0][0],rows,NC,keep,s);
    else if constexpr(NC>=5) screen_wide_kernel<TYPE,NC><<<kGrid,128,0,s>>>(static_cast<const uint8_t*>(copy),static_cast<const X81*>(x),y,rows,keep,sc);
    else screen_kernel<TYPE,NC><<<kGrid,128,0,s>>>(static_cast<const uint8_t*>(copy), static_cast<const X81*>(x), y, rows, keep, sc);
    threshold_kernel<<<NC,256,0,s>>>(keep, sc);
    rescore_kernel<TYPE,NC><<<(rows+kSlice-1)/kSlice,256,0,s>>>(weights, static_cast<const X81*>(x), y, rows, keep, sc);
}
template<int TYPE>
void dispatch(const void* copy, const void* weights, const void* x, float* y, int rows, int cols,
              const int32_t* keep, void* scratch, cudaStream_t s) {
#define HCASE(N) case N: launch<TYPE,N>(copy,weights,x,y,rows,keep,scratch,s); break
    switch (cols) { HCASE(1); HCASE(2); HCASE(3); HCASE(4); HCASE(5); HCASE(6); HCASE(7); HCASE(8); }
#undef HCASE
}
void check() {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("head screen: ") + cudaGetErrorString(e));
}
#endif
} // namespace

bool head_screen_supported(int type, int n_in, int n_out, int ncols) {
#if defined(__HIPCC__)
    return (type == 8 || type == 14) && n_in == 2560 && n_out > 0 && n_out <= 1048576 && ncols >= 1 && ncols <= 8;
#else
    (void)type; (void)n_in; (void)n_out; (void)ncols; return false;
#endif
}
bool head_screen_enabled(int type) {
    return env_on(type == 8 ? "STRATA_HEAD_Q8_SCREEN" : "STRATA_HEAD_Q6_SCREEN");
}
int head_screen_max_cols() {
    static const int value=[] {
        const char* e=std::getenv("STRATA_HEAD_SCREEN_MAX_T");const int n=e?std::atoi(e):8;
        return n>=1&&n<=8?n:8;
    }();
    return value;
}
std::size_t head_screen_copy_bytes(int type, int n_out) {
    if (!head_screen_supported(type,2560,n_out,1)) return 0;
    return type == 14 && fitted_q6() ? head_proxy_bound_bytes(n_out) : size_t(n_out)*rowbytes(type);
}
std::size_t head_screen_scratch_bytes() { return sizeof(Scratch); }
void head_screen_pack(int type, const void* weights, int n_out, void* copy, void* stream) {
    if (!head_screen_supported(type,2560,n_out,1) || !weights || !copy) throw std::invalid_argument("head screen pack: unsupported head");
    if (type == 14 && fitted_q6()) { head_proxy_bound_pack(weights,n_out,copy,stream); return; }
#if defined(__HIPCC__)
    const auto s = static_cast<cudaStream_t>(stream);
    if (type == 8) pack_kernel<8><<<(n_out*80+127)/128,128,0,s>>>(weights,static_cast<uint8_t*>(copy),n_out);
    else pack_kernel<14><<<(n_out*80+127)/128,128,0,s>>>(weights,static_cast<uint8_t*>(copy),n_out);
    check();
#endif
}
void head_screen(int type, const void* copy, const void* weights, const void* x, float* y,
                 int rows, int cols, const int32_t* keep, void* scratch, void* stream) {
    if (!head_screen_supported(type,2560,rows,cols) || !copy || !weights || !x || !y || !keep || !scratch)
        throw std::invalid_argument("head screen: unsupported head or null buffer");
#if defined(__HIPCC__)
    const auto s = static_cast<cudaStream_t>(stream);
    if (type == 8) dispatch<8>(copy,weights,x,y,rows,cols,keep,scratch,s);
    else dispatch<14>(copy,weights,x,y,rows,cols,keep,scratch,s);
    check();
#endif
}
} // namespace strata::kernels
