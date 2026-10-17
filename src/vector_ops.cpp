// Copyright (c) Guild Inference Contributors. MIT License.
#include "guild/kernels/cpu/features.hpp"
#include <cmath>
namespace guild::kernels::cpu {
void swiglu(const float* gate,const float* up,float* out,int count){
    for(int i=0;i<count;++i)out[i]=(gate[i]/(1.f+std::exp(-gate[i])))*up[i];
}
}
