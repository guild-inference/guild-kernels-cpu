// src/kernels/cpu/expert_layout.cpp - plan v0.3 P6: the per-layer expert table.  See the header.
#include "guild/kernels/cpu/features.hpp"
#include "guild/kernels/cpu/expert.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#endif
#include <fstream>
#include <sstream>

namespace guild::kernels::cpu {
int cpu_isa_cap() {
    // GUILD_FORCE_ISA (tests, the experimental older-CPU builds): the engine's own dispatch acts as if this CPU
    // stopped at that level.  ggml-cpu is not affected: it runs what the build compiled it for.
    static const int cap = [] {
        const char* f = std::getenv("GUILD_FORCE_ISA");
        if (f == nullptr || f[0] == '\0') return 3;
        const std::string v(f);
        if (v == "avx2") return 2;
        if (v == "avx") return 1;
        if (v == "sse" || v == "sse4.2" || v == "none") return 0;
        std::fprintf(stderr, "guild: GUILD_FORCE_ISA=%s is not avx2, avx or sse; ignored\n", f);
        return 3;
    }();
    return cap;
}

bool cpu_avx512_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("GUILD_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        if (cpu_isa_cap() < 3) return false;
        return cpu_features().usable();
    }();
    return ok;
}

bool cpu_avx512bw_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("GUILD_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1];
        return ((ebx >> 16) & 1u) && ((ebx >> 17) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u);
    }();
    return ok;
}

bool cpu_avx2_ok() {
    static const bool ok = [] {
        if (cpu_isa_cap() < 2) return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        const unsigned ecx1 = r[2];
        // FMA (12), OSXSAVE (27), AVX (28), F16C (29)
        if (!((ecx1 >> 12) & 1u) || !((ecx1 >> 27) & 1u) || !((ecx1 >> 28) & 1u) || !((ecx1 >> 29) & 1u)) return false;
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0x6) != 0x6) return false;            // the OS saves the SSE and AVX state
        cpuid(7, 0);
        return ((r[1] >> 5) & 1u) != 0;                    // AVX2
    }();
    return ok;
}

bool cpu_avx1_ok() {
    // AVX (Sandy Bridge, 2011): AVX + OSXSAVE with the OS saving the YMM state; FMA, F16C and AVX2 not needed.
    // From the Strata_Dirigo fork (rwkeyes).
    static const bool ok = [] {
        if (cpu_isa_cap() < 1) return false;
        unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, 1, 0);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
        if (!((r[2] >> 27) & 1u) || !((r[2] >> 28) & 1u)) return false;   // OSXSAVE, AVX
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        return (xcr0 & 0x6) == 0x6;
    }();
    return ok;
}

bool cpu_sse42_ok() {
    static const bool ok = [] {
        unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, 1, 0);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
        return ((r[2] >> 20) & 1u) && ((r[2] >> 23) & 1u);   // SSE4.2, POPCNT
    }();
    return ok;
}

const char* isa_floor_build() {
#if defined(GUILD_ISA_FLOOR_AVX)
    return "avx";
#elif defined(GUILD_ISA_FLOOR_NONE)
    return "sse4.2";
#else
    return "";
#endif
}

std::string cpu_name() {
    unsigned r[12] = {};
#if defined(_MSC_VER)
    int x[4];
    __cpuid(x, (int) 0x80000000u);
    if ((unsigned) x[0] < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(x, (int) (0x80000002u + i));
        for (int j = 0; j < 4; ++j) r[i * 4 + j] = (unsigned) x[j];
    }
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid(0x80000000u, a, b, c, d);
    if (a < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) __cpuid(0x80000002u + i, r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]);
#endif
    char s[49] = {};
    std::memcpy(s, r, 48);
    std::string name(s);
    const size_t b0 = name.find_first_not_of(' '), b1 = name.find_last_not_of(' ');
    return b0 == std::string::npos ? std::string("unknown") : name.substr(b0, b1 - b0 + 1);
}

#if !defined(GUILD_CPU_PROBE_ONLY)
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1) {
    if (cpu_avx512_ok()) q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_any(const float* x, int n, ActQ& a) {
    if (cpu_avx512_ok()) act_quant_q8_1(x, n, a);
    else act_quant_q8_1_avx2(x, n, a);
}
#endif


} // namespace guild::kernels::cpu
