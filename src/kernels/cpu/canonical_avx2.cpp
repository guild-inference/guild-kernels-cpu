// Copyright (c) Guild Inference Contributors. MIT License.
// Canonical planar Q2 layout on AVX2/FMA/F16C, without VNNI or VBMI.
// Preserve the 16-lane accumulation shape of the default VNNI implementation.
#include "guild/kernels/cpu/expert.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstring>

namespace guild::kernels::cpu {
namespace {
float half(const uint8_t* p){uint16_t h;std::memcpy(&h,p,2);return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));}
float reduce16(const float* a){
    float eight[8],four[4];
    for(int i=0;i<8;++i)eight[i]=a[i]+a[i+8];
    for(int i=0;i<4;++i)four[i]=eight[i]+eight[i+4];
    return (four[0]+four[2])+(four[1]+four[3]);
}
float dot(const uint8_t* codes,const uint8_t* scales,const ActQ& a,int blocks){
    alignas(64) float accum[16]={},correction[16]={};
    for(int b0=0;b0<blocks;b0+=8){
        for(int i=0;i<8 && b0+i<blocks;++i){
            const int b=b0+i;const float d=half(scales+2*b);
            correction[2*i]=std::fma(d,a.hx[2*b],correction[2*i]);
            correction[2*i+1]=std::fma(d,a.hx[2*b+1],correction[2*i+1]);
            for(int lane=0;lane<16;++lane){int sum=0;
                for(int j=0;j<4;++j){const int q=4*lane+j;
                    sum+=int((codes[size_t(b)*16+q/4]>>(2*(q%4)))&3)*int(a.q[64*b+q]);}
                const float scale=d*a.scale[2*b+lane/8];
                accum[lane]=std::fma(scale,float(sum),accum[lane]);}
        }
    }
    return reduce16(accum)-reduce16(correction);
}
}
void canonical_gu_avx2(const uint8_t* b,const ActQ* const* a,int nt,float* const* out,int r0,int r1){
    for(int r=r0;r<r1;++r)for(int t=0;t<nt;++t){
        const float g=dot(b+O_GU_CODES+size_t(2*r)*ROW_GU,b+O_GU_SCALES+size_t(2*r)*SC_GU*2,*a[t],SC_GU);
        const float u=dot(b+O_GU_CODES+size_t(2*r+1)*ROW_GU,b+O_GU_SCALES+size_t(2*r+1)*SC_GU*2,*a[t],SC_GU);
        out[t][r]=(g/(1.f+std::exp(-g)))*u;
    }
}
void canonical_down_avx2(const uint8_t* b,const ActQ* const* a,int nt,float* const* out,int r0,int r1){
    for(int r=r0;r<r1;++r)for(int t=0;t<nt;++t)
        out[t][r]=dot(b+O_D_CODES+size_t(r)*ROW_D,b+O_D_SCALES+size_t(r)*SC_D*2,*a[t],SC_D);
}
}
