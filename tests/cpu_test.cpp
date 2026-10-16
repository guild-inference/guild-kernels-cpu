#include <guild/abi/cpu_expert_abi.h>
#include "guild/kernels/cpu/native_expert.hpp"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);std::abort();}}while(0)
extern "C" guild_status_t guild_plugin_entry_v1(uint32_t,guild_plugin_descriptor_v1_t*,guild_error_info_t*);
static guild_cpu_expert_ext_v1_t table{};
static guild_status_t capture(guild_core_host_t,const guild_kernel_op_desc_v1_t* d,guild_error_info_t*) {
    CHECK(d->next_ext);std::memcpy(&table,d->next_ext,sizeof(table));return GUILD_STATUS_OK;
}
int main() {
    guild_plugin_descriptor_v1_t plugin{};guild_host_api_v1_t host{};
    host.struct_size=sizeof(host);host.register_kernel_op=capture;
    CHECK(guild_plugin_entry_v1(GUILD_KERNEL_ABI_VERSION_V1,&plugin,nullptr)==GUILD_STATUS_OK);
    CHECK(plugin.init(nullptr,&host,nullptr)==GUILD_STATUS_OK);
    using namespace guild::kernels::cpu;
    NativeFmt f;std::string error;CHECK(native_fmt(12,8,256,256,f,error));
    std::vector<float> values(256*256),x(256),y(256),reference(256);
    for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*0.01f);
    for(size_t i=0;i<x.size();++i)x[i]=std::cos(float(i)*0.1f);
    std::vector<uint8_t> blob(f.bytes),activation(f.act_bytes),direct(f.act_bytes);
    const auto* traits=ggml_get_type_traits(GGML_TYPE_Q4_K);
    traits->from_float_ref(values.data(),blob.data(),int64_t(values.size()));
    traits->from_float_ref(values.data(),blob.data()+f.up_off,int64_t(values.size()));
    table.quant_act(&f,x.data(),activation.data());
    ggml_get_type_traits_cpu((ggml_type)f.gu_act)->from_float(x.data(),direct.data(),256);
    CHECK(activation==direct);
    const void* acts[]={activation.data()};float* out[]={y.data()};
    table.gate_up(&f,blob.data(),acts,1,out,0,256);
    const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_Q4_K)->vec_dot;
    for(int r=0;r<256;++r){float g=0,u=0;dot(256,&g,0,blob.data()+r*f.gu_row,0,acts[0],0,1);
        dot(256,&u,0,blob.data()+f.up_off+r*f.gu_row,0,acts[0],0,1);reference[r]=(g/(1.f+std::exp(-g)))*u;}
    CHECK(std::memcmp(y.data(),reference.data(),y.size()*sizeof(float))==0);
}
