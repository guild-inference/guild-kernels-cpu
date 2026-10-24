// The repacked IQ3_S rows (iq_avx2.hpp, kIq3sNl4Type) against the kernels the IQ3_S rows take: ggml-cpu's
// single-token dot and iq256_gu_rows, bit for bit, and through native_gu_rows when iq3s_nl4_exact().
#include "guild/kernels/cpu/features.hpp"
#include "guild/kernels/cpu/iq_avx2.hpp"
#include "guild/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);std::abort();}}while(0)
using namespace guild::kernels::cpu;
int main() {
    if (!cpu_avx2_ok()) { std::puts("iq3s_nl4_test: no AVX2, skipped"); return 0; }
    ggml_cpu_init();
    constexpr int H = 1024, FF = 64, E = 6, NT = 8;
    NativeFmt f; std::string error;
    CHECK(native_fmt(GGML_TYPE_IQ3_S, GGML_TYPE_Q8_0, H, FF, f, error));
    NativeFmt p = f;
    p.gu_type = kIq3sNl4Type;
    p.gu_row = (H / 256) * kIq3sNl4Block;
    p.up_off = p.gu_row * FF;
    ggml_quantize_init(GGML_TYPE_IQ3_S);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> w(2 * FF * H), x(H);
    std::vector<uint8_t> act(NT * f.act_bytes);
    std::vector<const void*> a(NT);
    for (int t = 0; t < NT; ++t) {
        for (auto& v : x) v = nd(rng) * (1 + t);
        ggml_get_type_traits_cpu((ggml_type) f.gu_act)->from_float(x.data(), act.data() + t * f.act_bytes, H);
        a[t] = act.data() + t * f.act_bytes;
    }
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S)->vec_dot;
    std::vector<float> want(NT * FF), got(NT * FF);
    std::vector<float*> pw(NT), pg(NT);
    for (int t = 0; t < NT; ++t) { pw[t] = &want[t * FF]; pg[t] = &got[t * FF]; }
    for (int e = 0; e < E; ++e) {
        for (auto& v : w) v = nd(rng) * 0.02f * (1 + e);
        std::vector<uint8_t> blob(f.bytes), packed(2 * p.up_off);
        ggml_quantize_chunk(GGML_TYPE_IQ3_S, w.data(), blob.data(), 0, 2 * FF, H, nullptr);
        for (int r = 0; r < 2 * FF; ++r)
            CHECK(iq3s_nl4_repack(blob.data() + r * f.gu_row, H / 256, packed.data() + r * p.gu_row));
        for (int nt = 1; nt <= NT; ++nt) {
            // ggml order: every token alone through ggml-cpu's dot
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < FF; ++r) {
                    float g = 0, u = 0;
                    dot(H, &g, 0, blob.data() + r * f.gu_row, 0, a[t], 0, 1);
                    dot(H, &u, 0, blob.data() + f.up_off + r * f.gu_row, 0, a[t], 0, 1);
                    pw[t][r] = (g / (1.f + std::exp(-g))) * u;
                }
            iq3s_nl4_gu_rows(packed.data(), p.gu_row, p.up_off, H, a.data(), nt, pg.data(), 0, FF, true);
            CHECK(std::memcmp(want.data(), got.data(), nt * FF * sizeof(float)) == 0);
            // iq256 order, on a row range that does not start at 0
            iq256_gu_rows(GGML_TYPE_IQ3_S, blob.data(), f.gu_row, f.up_off, H, a.data(), nt, pw.data(), 3, FF - 1);
            iq3s_nl4_gu_rows(packed.data(), p.gu_row, p.up_off, H, a.data(), nt, pg.data(), 3, FF - 1, false);
            for (int t = 0; t < nt; ++t) CHECK(std::memcmp(pw[t] + 3, pg[t] + 3, (FF - 4) * sizeof(float)) == 0);
            // the dispatch: what the IQ3_S rows compute, the repacked rows compute
            if (iq3s_nl4_exact()) {
                native_gu_rows(f, blob.data(), a.data(), nt, pw.data(), 0, FF);
                native_gu_rows(p, packed.data(), a.data(), nt, pg.data(), 0, FF);
                CHECK(std::memcmp(want.data(), got.data(), nt * FF * sizeof(float)) == 0);
            }
        }
    }
    std::printf("iq3s_nl4_test: repacked IQ3_S rows bitwise equal (dispatch checked: %d)\n", (int) iq3s_nl4_exact());
}
