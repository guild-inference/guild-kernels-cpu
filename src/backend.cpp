// Copyright (c) Guild Inference Contributors. MIT License.
// Public backend callbacks; no scheduling, model storage or Core state lives here.
#include <guild/abi/cpu_backend_abi.h>
#include "guild/kernels/cpu/features.hpp"
#include "guild/kernels/cpu/native_expert.hpp"
#include "guild/kernels/cpu/kq_avx2.hpp"
#include "guild/kernels/cpu/kq_avx1.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include "ggml-cpu.h"

namespace guild::kernels::cpu {
void canonical_gu_avx2(const uint8_t*, const ActQ* const*, int, float* const*, int, int);
void canonical_down_avx2(const uint8_t*, const ActQ* const*, int, float* const*, int, int);
}
namespace cpu=guild::kernels::cpu;
namespace {
guild_status_t format(int32_t gu,int32_t down,int64_t n,int64_t ff,
                      guild_cpu_expert_format_v1_t* out,guild_error_info_t* error) {
    if(!out || n<=0 || n>cpu::H || ff<=0 || ff>cpu::FF) return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    try {
        std::string message;
        if(cpu::native_fmt(gu,down,n,ff,*out,message)) return GUILD_STATUS_OK;
        if(error) {error->struct_size=sizeof(*error);error->code=GUILD_STATUS_ERR_INVALID_TENSOR;
            std::snprintf(error->message,sizeof(error->message),"%s",message.c_str());}
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    } catch(const std::exception& e) {
        if(error)std::snprintf(error->message,sizeof(error->message),"%s",e.what());
        return GUILD_STATUS_ERR_EXECUTION_FAILED;
    } catch(...) {return GUILD_STATUS_ERR_EXECUTION_FAILED;}
}
const guild_cpu_backend_ext_v2_t table = {
    {sizeof(guild_cpu_backend_ext_v2_t),GUILD_CPU_EXPERT_TAG_V1,1,
     +[](const cpu::NativeFmt* f,const float* x,void* y){cpu::native_quant_act(*f,x,y);},
     +[](const cpu::NativeFmt* f,const float* x,void* y){cpu::native_quant_h(*f,x,y);},
     +[](const cpu::NativeFmt* f,const uint8_t* b,const void* const* x,int n,float* const* y,int a,int z){cpu::native_gu_rows(*f,b,x,n,y,a,z);},
     +[](const cpu::NativeFmt* f,const uint8_t* b,const void* const* x,int n,float* const* y,int a,int z){cpu::native_down_rows(*f,b,x,n,y,a,z);}},
    GUILD_CPU_BACKEND_VERSION_V2,0,format,
    +[](const float* x,int n,cpu::ActQ* a){cpu::act_quant_any(x,n,*a);},
    +[](const uint8_t* b,const cpu::ActQ* const* a,int n,float* const* y,int r0,int r1){
        if(cpu::cpu_avx512_ok())cpu::s2_expert_gu_rows_multi(b,a,n,y,r0,r1);
        else cpu::canonical_gu_avx2(b,a,n,y,r0,r1);},
    +[](const uint8_t* b,const cpu::ActQ* const* a,int n,float* const* y,int r0,int r1){
        if(cpu::cpu_avx512_ok())cpu::s2_expert_down_rows_multi(b,a,n,y,r0,r1);
        else cpu::canonical_down_avx2(b,a,n,y,r0,r1);},
    +[](const uint8_t* w,uint64_t bytes,int blocks,const cpu::ActQ* const* a,int n,float* const* y,int r0,int r1){
        cpu::q2_rows_any(w,size_t(bytes),blocks,a,n,y,r0,r1);},
    +[](const uint16_t* w,int rows,int cols,const float* x,int nt,float* out){
        if(cpu::cpu_avx2_ok())cpu::bf16_rows_dot_multi(w,rows,cols,x,nt,out);
        else if(cpu::cpu_avx1_ok())cpu::bf16_rows_dot_multi_avx1(w,rows,cols,x,nt,out);
        else for(int t=0;t<nt;++t)for(int r=0;r<rows;++r){
            float sum=0;
            for(int c=0;c<cols;++c){uint32_t bits=uint32_t(w[size_t(r)*cols+c])<<16;float f;
                std::memcpy(&f,&bits,sizeof(f));sum+=f*x[size_t(t)*cols+c];}
            out[size_t(t)*rows+r]=sum;}},
    +[](const float* parts,const float* weights,float* out,int k,int n)->guild_status_t {
        uint32_t nonfinite=0;
        for(int d=0;d<n;++d){float acc=0;
            for(int r=0;r<k;++r){const float value=parts[size_t(r)*n+d];uint32_t bits;
                std::memcpy(&bits,&value,sizeof(bits));nonfinite|=(bits&0x7f800000u)==0x7f800000u;
                acc=std::fma(value,weights[r],acc);}
            out[d]=acc;}
        return nonfinite?GUILD_STATUS_ERR_NONFINITE_OUTPUT:GUILD_STATUS_OK;},
    cpu::swiglu
};
}

extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_backend_table_v2() {
    // Pure table discovery does not execute SIMD or initialize GGML. Plugin init
    // and native format construction initialize their own state after ISA gating.
    return &table;
}
