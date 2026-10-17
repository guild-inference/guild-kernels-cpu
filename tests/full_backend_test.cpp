// Independent contract/oracle test: no Core source headers or private C++ types.
#include <guild/abi/cpu_backend_abi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);std::abort();}}while(0)
extern "C" guild_status_t guild_plugin_entry_v1(uint32_t,guild_plugin_descriptor_v1_t*,guild_error_info_t*);
static guild_cpu_backend_ext_v2_t api{};
int main(){
    guild_plugin_descriptor_v1_t plugin{};guild_host_api_v1_t host{};host.struct_size=sizeof(host);
    host.register_kernel_op=+[](guild_core_host_t,const guild_kernel_op_desc_v1_t* d,guild_error_info_t*)->guild_status_t {
        CHECK(d->next_ext);uint32_t size=0;std::memcpy(&size,d->next_ext,4);CHECK(size>=sizeof(api));
        std::memcpy(&api,d->next_ext,sizeof(api));return GUILD_STATUS_OK;
    };
    CHECK(guild_plugin_entry_v1(GUILD_KERNEL_ABI_VERSION_V1,&plugin,nullptr)==GUILD_STATUS_OK);
    CHECK(plugin.init(nullptr,&host,nullptr)==GUILD_STATUS_OK);
    CHECK(api.backend_version==2 && api.native.tag==GUILD_CPU_EXPERT_TAG_V1);
    constexpr int n=2560,ff=640,nt=2;
    constexpr size_t row_gu=640,row_down=160,offset_down=2*ff*row_gu;
    constexpr size_t scales_gu=offset_down+n*row_down,scales_down=scales_gu+2*ff*40*2;
    constexpr size_t bytes=1382400;
    std::vector<uint8_t> blob(bytes,0xaa);
    for(size_t at=scales_gu;at<bytes;at+=2){blob[at]=0;blob[at+1]=0x3c;}
    std::vector<float> x(nt*n),hidden(nt*ff),result(nt*n);
    guild_cpu_planar_act_v1_t act[nt],hq[nt];
    const guild_cpu_planar_act_v1_t* ap[]={&act[0],&act[1]};
    const guild_cpu_planar_act_v1_t* hp[]={&hq[0],&hq[1]};
    float* h[]={hidden.data(),hidden.data()+ff};float* y[]={result.data(),result.data()+n};
    for(int t=0;t<nt;++t){std::fill(x.begin()+t*n,x.begin()+(t+1)*n,0.001f*(t+1));api.quant_planar(x.data()+t*n,n,&act[t]);}
    api.canonical_gate_up(blob.data(),ap,nt,h,0,ff);
    for(int t=0;t<nt;++t){
        double g=0;for(int i=0;i<n;++i)g+=double(act[t].q[i])*act[t].scale[i/32];
        const double expected=(g/(1+std::exp(-g)))*g;
        for(int r=0;r<ff;++r)CHECK(std::abs(double(h[t][r])-expected)<2e-5*std::max(1.0,std::abs(expected)));
        api.quant_planar(h[t],ff,&hq[t]);
    }
    api.canonical_down(blob.data(),hp,nt,y,0,n);
    for(int t=0;t<nt;++t){double expected=0;for(int i=0;i<ff;++i)expected+=double(hq[t].q[i])*hq[t].scale[i/32];
        for(int r=0;r<n;++r)CHECK(std::abs(double(y[t][r])-expected)<2e-5*std::max(1.0,std::abs(expected)));}
    std::vector<uint8_t> gguf(3*40*18,0xaa);
    for(size_t at=0;at<gguf.size();at+=18){gguf[at]=0;gguf[at+1]=0x3c;}
    float rows[nt][3]={{-999,-999,-999},{-999,-999,-999}};float* rp[]={rows[0],rows[1]};
    api.q2_gguf_rows(gguf.data(),40*18,40,ap,nt,rp,1,2);
    for(int t=0;t<nt;++t){double expected=0;for(int i=0;i<n;++i)expected+=double(act[t].q[i])*act[t].scale[i/32];
        CHECK(rows[t][0]==-999 && rows[t][2]==-999 && std::abs(rows[t][1]-expected)<2e-5*expected);}
    std::vector<uint16_t> router(n*3,0x3f80);float routed[nt*3]={};
    api.router_bf16(router.data(),3,n,x.data(),nt,routed);
    for(int t=0;t<nt;++t)for(int r=0;r<3;++r)CHECK(std::abs(routed[t*3+r]-n*double(x[t*n]))<1e-4);
    float parts[]={1,1,2,2},weights[]={.25f,.75f},combined[2]={};
    CHECK(api.combine(parts,weights,combined,2,2)==GUILD_STATUS_OK && combined[0]==1.75f && combined[1]==1.75f);
    parts[0]=std::numeric_limits<float>::quiet_NaN();CHECK(api.combine(parts,weights,combined,2,2)==GUILD_STATUS_ERR_NONFINITE_OUTPUT);
    CHECK(api.swiglu);float gate[]={0,1},up[]={2,3},activated[2]={};api.swiglu(gate,up,activated,2);
    CHECK(activated[0]==0 && std::abs(activated[1]-3.f/(1.f+std::exp(-1.f)))<1e-6f);
    guild_cpu_expert_format_v1_t format{};guild_error_info_t error{};
    CHECK(api.format(12,8,2048,512,&format,&error)==GUILD_STATUS_OK && format.n_embd==2048 && format.bytes>0);
    CHECK(api.format(12,8,-1,512,&format,&error)==GUILD_STATUS_ERR_INVALID_ARGUMENT);
    std::puts("CPU C ABI: canonical Q2, native Q2, quantization, router, combine and format oracles PASS");
}
