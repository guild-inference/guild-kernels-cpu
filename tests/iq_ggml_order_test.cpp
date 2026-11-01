// #152: the multi-token i-quant kernels (iq_avx512.cpp, iq_avx2.cpp) against ggml-cpu's single-token dot, bit for bit,
// for every group size 1..MAXT: a token's expert rows must not depend on how many tokens of a verify window share the
// expert.  Random blocks (every bit pattern of these formats is a valid block once its f16 scale is finite) and
// ggml-quantized activations; the gate/up form (silu(g) * u), the plain rows, and the IQ4_NL down rows.  Then the
// dispatch (native_gu_rows / native_down_rows): a group of nt tokens against each token alone.
#include "guild/kernels/cpu/expert.hpp"
#include "guild/kernels/cpu/features.hpp"
#include "guild/kernels/cpu/iq_avx2.hpp"
#include "guild/kernels/cpu/iq_avx512.hpp"
#include "guild/kernels/cpu/kq_avx2.hpp"
#include "guild/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);std::abort();}}while(0)
using namespace guild::kernels::cpu;

namespace {
std::mt19937 rng(152);

// n values of `type` with random bytes and a random finite f16 scale (bytes 0-1 of every block are `d`)
std::vector<uint8_t> random_rows(int type, int rows, int n) {
    const size_t rb = ggml_row_size((ggml_type) type, n), bb = ggml_type_size((ggml_type) type);
    std::vector<uint8_t> w(rb * rows);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(1e-4f, 3e-3f);
    for (auto& v : w) v = (uint8_t) byte(rng);
    for (size_t b = 0; b < w.size() / bb; ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(scale(rng) * (byte(rng) & 1 ? 1.f : -1.f));
        std::memcpy(w.data() + b * bb, &d, 2);
    }
    return w;
}

std::vector<std::vector<uint8_t>> random_acts(int type, int n, int nt) {
    const auto* tr = ggml_get_type_traits_cpu((ggml_type) type);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> a(nt, std::vector<uint8_t>(ggml_row_size(tr->vec_dot_type, n)));
    std::vector<float> x(n);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng) * (1.f + t);
        ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(x.data(), a[t].data(), n);
    }
    return a;
}

using Rows = void (*)(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
using GuRows = void (*)(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int);

// rows [3, rows - 1) of `w` for nt = 1..MAXT tokens through `rows_fn` / `gu_fn`, against ggml's dot per token
size_t check_iq(int type, const char* isa, Rows rows_fn, GuRows gu_fn) {
    constexpr int H = 1024, FF = 40;
    const size_t rb = ggml_row_size((ggml_type) type, H);
    const auto w = random_rows(type, 2 * FF, H);   // gate rows, then up rows
    const auto acts = random_acts(type, H, MAXT);
    const void* a[MAXT];
    for (int t = 0; t < MAXT; ++t) a[t] = acts[t].data();
    const auto dot = ggml_get_type_traits_cpu((ggml_type) type)->vec_dot;
    std::vector<float> ref((size_t) MAXT * 2 * FF), want_ff((size_t) MAXT * FF);
    for (int t = 0; t < MAXT; ++t) {
        for (int r = 0; r < 2 * FF; ++r) dot(H, &ref[(size_t) t * 2 * FF + r], 0, w.data() + r * rb, 0, a[t], 0, 1);
        for (int r = 0; r < FF; ++r) {
            const float g = ref[(size_t) t * 2 * FF + r], u = ref[(size_t) t * 2 * FF + FF + r];
            want_ff[(size_t) t * FF + r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
    size_t differ = 0;
    for (int nt = 1; nt <= MAXT; ++nt) {
        std::vector<float> got((size_t) nt * 2 * FF, NAN), gff((size_t) nt * FF, NAN);
        float* op[MAXT];
        float* fp[MAXT];
        for (int t = 0; t < nt; ++t) { op[t] = got.data() + (size_t) t * 2 * FF; fp[t] = gff.data() + (size_t) t * FF; }
        rows_fn(type, w.data(), rb, H, a, nt, op, 3, 2 * FF - 1);
        gu_fn(type, w.data(), rb, FF * rb, H, a, nt, fp, 3, FF - 1);
        for (int t = 0; t < nt; ++t) {
            differ += std::memcmp(op[t] + 3, &ref[(size_t) t * 2 * FF + 3], (2 * FF - 4) * sizeof(float)) != 0;
            differ += std::memcmp(fp[t] + 3, &want_ff[(size_t) t * FF + 3], (FF - 4) * sizeof(float)) != 0;
        }
    }
    std::printf("  %-8s %-7s rows and gate/up vs ggml's dot, 1..%d tokens: %zu of %d token-sets differ\n",
                ggml_type_name((ggml_type) type), isa, MAXT, differ, MAXT * (MAXT + 1));
    return differ;
}

size_t check_iq4nl(int n) {
    constexpr int ROWS = 48;
    const size_t rb = ggml_row_size(GGML_TYPE_IQ4_NL, n);
    const auto w = random_rows(GGML_TYPE_IQ4_NL, ROWS, n);
    const auto acts = random_acts(GGML_TYPE_IQ4_NL, n, MAXT);
    const void* a[MAXT];
    for (int t = 0; t < MAXT; ++t) a[t] = acts[t].data();
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL)->vec_dot;
    std::vector<float> ref((size_t) MAXT * ROWS);
    for (int t = 0; t < MAXT; ++t)
        for (int r = 0; r < ROWS; ++r) dot(n, &ref[(size_t) t * ROWS + r], 0, w.data() + r * rb, 0, a[t], 0, 1);
    size_t differ = 0;
    for (int nt = 1; nt <= MAXT; ++nt) {
        std::vector<float> got((size_t) nt * ROWS, NAN);
        float* op[MAXT];
        for (int t = 0; t < nt; ++t) op[t] = got.data() + (size_t) t * ROWS;
        iq4nl256_down_rows(w.data(), rb, n, a, nt, op, 1, ROWS);
        for (int t = 0; t < nt; ++t)
            differ += std::memcmp(op[t] + 1, &ref[(size_t) t * ROWS + 1], (ROWS - 1) * sizeof(float)) != 0;
    }
    std::printf("  iq4_nl   avx2    down rows (%d values) vs ggml's dot, 1..%d tokens: %zu of %d token-sets differ\n",
                n, MAXT, differ, MAXT * (MAXT + 1) / 2);
    return differ;
}

// native_gu_rows / native_down_rows: each token of an nt-token group against the same token alone
size_t check_dispatch(int gu, int down, int H, int FF) {
    NativeFmt f;
    std::string err;
    CHECK(native_fmt(gu, down, H, FF, f, err));
    auto blob = random_rows(gu, 2 * FF, H);   // gate + up
    const auto dn = random_rows(down, H, FF);
    blob.resize(f.bytes);
    std::memcpy(blob.data() + f.down_off, dn.data(), dn.size());
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> act(MAXT, std::vector<uint8_t>(f.act_bytes));
    std::vector<std::vector<uint8_t>> hq(MAXT, std::vector<uint8_t>(f.h_bytes));
    std::vector<float> x(H), ff((size_t) MAXT * FF), out((size_t) MAXT * H), one_ff(FF), one_out(H);
    const void* a[MAXT];
    const void* h[MAXT];
    float* fp[MAXT];
    float* op[MAXT];
    for (int t = 0; t < MAXT; ++t) {
        for (auto& v : x) v = nd(rng);
        native_quant_act(f, x.data(), act[t].data());
        a[t] = act[t].data();
        fp[t] = ff.data() + (size_t) t * FF;
        op[t] = out.data() + (size_t) t * H;
    }
    size_t differ = 0;
    for (int nt = 2; nt <= MAXT; ++nt) {
        native_gu_rows(f, blob.data(), a, nt, fp, 0, FF);
        for (int t = 0; t < nt; ++t) {
            native_quant_h(f, fp[t], hq[t].data());
            h[t] = hq[t].data();
        }
        native_down_rows(f, blob.data(), h, nt, op, 0, H);
        for (int t = 0; t < nt; ++t) {
            const void* a1[1] = {a[t]};
            const void* h1[1] = {h[t]};
            float* f1[1] = {one_ff.data()};
            float* o1[1] = {one_out.data()};
            native_gu_rows(f, blob.data(), a1, 1, f1, 0, FF);
            native_down_rows(f, blob.data(), h1, 1, o1, 0, H);
            differ += std::memcmp(one_ff.data(), fp[t], FF * sizeof(float)) != 0;
            differ += std::memcmp(one_out.data(), op[t], H * sizeof(float)) != 0;
        }
    }
    std::printf("  dispatch %s/%s: a group of 2..%d tokens against each token alone: %zu rows differ\n",
                ggml_type_name((ggml_type) gu), ggml_type_name((ggml_type) down), MAXT, differ);
    return differ;
}
}  // namespace

int main() {
    if (!cpu_avx2_ok()) { std::puts("iq_ggml_order_test: no AVX2, skipped"); return 0; }
    // GUILD_IQ_GGML_ORDER=0: the kernels' earlier order, reported (it differs) but not failed
    const char* order = std::getenv("GUILD_IQ_GGML_ORDER");
    const bool legacy = order != nullptr && std::atoi(order) == 0;
    // the AVX-512 kernels run where cpu_avx512_ok() (which also wants VNNI and VBMI); IQ512_TEST_BW=1 runs them on
    // any AVX-512 BW CPU (a Skylake-SP Xeon), for this test only - their code uses AVX-512 F and BW instructions
    const bool can512 = cpu_avx512_ok() || (std::getenv("IQ512_TEST_BW") != nullptr && cpu_avx512bw_ok());
    ggml_cpu_init();
    size_t differ = 0;
    for (int type : {16, 17, 18, 21, 22, 23}) {   // IQ2_XXS IQ2_XS IQ3_XXS IQ3_S IQ2_S IQ4_XS
        if (can512 && iq512_supported(type)) differ += check_iq(type, "avx512", iq512_rows, iq512_gu_rows);
        if (iq256_supported(type)) differ += check_iq(type, "avx2", iq256_rows, iq256_gu_rows);
    }
    for (int n : {512, 640, 768}) differ += check_iq4nl(n);
    // Q8_0 rows (kq_avx2.cpp): two rows to an AVX-512BW register where the CPU has it (GUILD_Q8_512=0: the AVX-2
    // kernel), one token or several, against ggml's dot; then the dispatch of Q8_0 experts
    differ += check_iq(GGML_TYPE_Q8_0, kq512_rows(GGML_TYPE_Q8_0, 1024) ? "avx512bw" : "avx2", kq256_rows,
                       kq256_gu_rows);
    differ += check_dispatch(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 1024, 512);
    // the formats of the shipped IQ packs: IQ3_S or IQ4_XS gate/up with IQ4_NL or Q8_0 down rows
    differ += check_dispatch(GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, 1024, 512);
    differ += check_dispatch(GGML_TYPE_IQ3_S, GGML_TYPE_Q8_0, 1024, 512);
    differ += check_dispatch(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, 1024, 512);
    differ += check_dispatch(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_NL, 1024, 512);
    if (legacy) {
        std::printf("iq_ggml_order_test: GUILD_IQ_GGML_ORDER=0, %zu token-sets differ (not a failure)\n", differ);
        return 0;
    }
    std::printf("iq_ggml_order_test: %s\n", differ ? "MISMATCH" : "every multi-token row bitwise equal to ggml-cpu's");
    return differ ? 1 : 0;
}
