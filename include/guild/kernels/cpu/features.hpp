// Copyright (c) Guild Inference Contributors. MIT License.
#pragma once
#include "guild/kernels/cpu/expert.hpp"
#include <string>
namespace guild::kernels::cpu {
bool cpu_avx512_ok();
bool cpu_avx512bw_ok();
bool cpu_avx2_ok();
bool cpu_avx1_ok();
bool cpu_sse42_ok();
int cpu_isa_cap();
const char* isa_floor_build();
std::string cpu_name();
inline bool q2_native_kernels(int type){return type==42 && cpu_avx2_ok();}
void q2_rows_any(const uint8_t*,size_t,int,const ActQ* const*,int,float* const*,int,int);
void act_quant_any(const float*,int,ActQ&);
void swiglu(const float*,const float*,float*,int);
}
