// include/guild/kernels/cpu/iq_avx2.hpp - AVX-2 multi-token dot products for the i-quant expert
// formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS) against Q8_K activations (ggml's block_q8_K).
//
// The AVX-512 kernel's scheme on the CPUs without it (AMD Zen 2/3, Intel Core 12th-14th gen): the weights of a
// 32-value chunk are decoded once per verify window and every token applies them with five instructions.
#pragma once

#include <cstddef>
#include <cstdint>

namespace guild::kernels::cpu {

bool iq256_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// out[t][r] = w_r . h[t] for IQ4_NL (type 20) rows against Q8_0 activations (ggml's block_q8_0).
/// IQ4_NL is a 32-value-block format, so this does not go through iq256_rows (QK_K blocks, Q8_K acts).
void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                        float* const* out, int r0, int r1);

/// IQ3_S repacked for the CPU: the grid lookup done once at load.  Each 256-value block becomes 134 bytes - the
/// f16 scale, the four scale bytes, and one 4-bit code per value, c = (w + 15) / 2 for the signed grid value w
/// (an odd number in -15..15).  Code bytes come in four 32-byte groups of 64 values: byte k of group j holds
/// value 64j + k in its low nibble and value 64j + 32 + k in its high one.  Decoding is two `pshufb` per 32 values
/// instead of eight grid loads and a sign expansion.
#define GUILD_CPU_HAS_IQ3S_NL4 1 ///< for a host built against older kernel sources
inline constexpr int kIq3sNl4Type = 1021;     ///< NativeFmt::gu_type of a repacked layer (not a ggml type)
inline constexpr size_t kIq3sNl4Block = 134;  ///< bytes per 256 values
/// Repacks `nblocks` IQ3_S blocks; false (and nothing usable written) if a grid value is outside the code range.
bool iq3s_nl4_repack(const uint8_t* iq3s, int nblocks, uint8_t* out) noexcept;
/// iq256_gu_rows for repacked rows, bitwise equal to the kernel the IQ3_S rows would have taken: `ggml_order`
/// reproduces ggml-cpu's single-token `ggml_vec_dot_iq3_s_q8_K` (its final 8-lane sum), otherwise iq256_gu_rows.
void iq3s_nl4_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                      float* const* ff, int r0, int r1, bool ggml_order);

}  // namespace guild::kernels::cpu
