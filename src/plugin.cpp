// Copyright (c) Guild Inference Contributors. MIT License.
#include <guild/abi/cpu_expert_abi.h>
#include "guild/kernels/cpu/native_expert.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"
#include <atomic>
#include "ggml-cpu.h"
namespace cpu=guild::kernels::cpu;
namespace {
std::atomic<uint64_t> calls{0};
const guild_cpu_expert_ext_v1_t table={sizeof(table),GUILD_CPU_EXPERT_TAG_V1,1,
    +[](const cpu::NativeFmt* f,const float* x,void* y){cpu::native_quant_act(*f,x,y);},
    +[](const cpu::NativeFmt* f,const float* x,void* y){cpu::native_quant_h(*f,x,y);},
    +[](const cpu::NativeFmt* f,const uint8_t* b,const void* const* x,int n,float* const* y,int a,int z){
        calls.fetch_add(1,std::memory_order_relaxed); cpu::native_gu_rows(*f,b,x,n,y,a,z);},
    +[](const cpu::NativeFmt* f,const uint8_t* b,const void* const* x,int n,float* const* y,int a,int z){cpu::native_down_rows(*f,b,x,n,y,a,z);}
};
guild_status_t init(guild_core_host_t host,const guild_host_api_v1_t* api,guild_error_info_t* err) {
    if(!api || api->struct_size<sizeof(*api) || !cpu::cpu_avx2_ok()) return GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE;
    // This DSO owns an isolated GGML instance. Core's initialized conversion
    // tables are not shared with it; initialize before publishing any callbacks.
    ggml_cpu_init();
    guild_kernel_op_desc_v1_t d{};d.struct_size=sizeof(d);d.abi_version=GUILD_KERNEL_ABI_VERSION_V1;
    d.op_kind=GUILD_OP_BATCHED_EXPERT_EXEC;d.op_name="batched_expert_exec";
    d.impl_id="external.cpu.native_expert.v1";d.package_id="org.guild.kernels.cpu";
    d.impl_version=GUILD_KERNEL_ABI_MAKE_VERSION(GUILD_CPU_VERSION_MAJOR,GUILD_CPU_VERSION_MINOR,GUILD_CPU_VERSION_PATCH);d.backend=GUILD_BACKEND_CPU;
    d.required_cpu_isa=CPU_REQUIRED_ISA;
    d.weight_dtype=GUILD_DTYPE_ANY;d.input_dtype=GUILD_DTYPE_ANY;d.output_dtype=GUILD_DTYPE_F32;d.accum_dtype=GUILD_DTYPE_F32;
    d.priority=300;d.next_ext=&table;d.supported_sync_modes=GUILD_SYNC_HOST_BLOCKING;
    return api->register_kernel_op(host,&d,err);
}
}
extern "C" GUILD_PLUGIN_EXPORT uint64_t guild_cpu_gate_up_calls(){return calls.load();}
extern "C" GUILD_PLUGIN_EXPORT guild_status_t guild_plugin_entry_v1(uint32_t abi,guild_plugin_descriptor_v1_t* d,guild_error_info_t*) {
    if(GUILD_ABI_VERSION_MAJOR(abi)!=1 || !d) return GUILD_STATUS_ERR_ABI_MISMATCH;
    *d={};d->struct_size=sizeof(*d);d->abi_version=GUILD_KERNEL_ABI_VERSION_V1;
    d->package_id="org.guild.kernels.cpu";d->package_version=GUILD_PACKAGE_VERSION;d->publisher_id="org.guild-inference";d->init=init;
    return GUILD_STATUS_OK;
}
