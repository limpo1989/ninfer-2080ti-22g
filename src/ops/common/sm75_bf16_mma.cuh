#pragma once

#include "ops/common/mma.cuh"

namespace ninfer::ops {

// Scale both BF16 panels by 2^8. Within [2^-22, 255], every nonzero
// operand becomes a normal FP16 value without losing a single significand bit.
// Out-of-range panels use the original FP32 path. Accumulation stays FP32 and
// the caller scales the partial sum back by 2^-16 before adding it.
template <int M, int N>
__device__ __forceinline__ bool scale_bf16_mma_fragments(
    unsigned (&a)[M][4], unsigned (&b)[N][2], float& inverse_scale) {
    const auto exact_range = [](unsigned pair) {
        const unsigned lo = pair & 0x7fffU, hi = (pair >> 16) & 0x7fffU;
        return (lo == 0 || (lo >= 0x3480U && lo <= 0x437fU)) &&
               (hi == 0 || (hi >= 0x3480U && hi <= 0x437fU));
    };
    bool exact = true;
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { exact = exact && exact_range(a[m][i]); }
    }
#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
        for (int i = 0; i < 2; ++i) { exact = exact && exact_range(b[n][i]); }
    }
    if (!__all_sync(0xffffffff, exact)) { return false; }
    inverse_scale = 0x1p-16F;
    // Adding eight to each BF16 exponent multiplies by 256. The admitted
    // magnitudes cannot carry into the adjacent packed half. Signed zeros
    // become tiny signed BF16 values that convert back to the same FP16 zero.
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { a[m][i] = bf162_to_f162(a[m][i] + 0x04000400U); }
    }
#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
        for (int i = 0; i < 2; ++i) { b[n][i] = bf162_to_f162(b[n][i] + 0x04000400U); }
    }
    return true;
}

// Retain the represented BF16 operands exactly. All lanes must take the same
// MMA instruction, so any nonrepresentable pair selects the FP32 fallback for
// the entire warp. Accumulation remains FP32 in both paths.
__device__ __forceinline__ bool bf16_pair_is_exact_half(unsigned bf16, unsigned half_bits) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bf16));
    const float2 b = __half22float2(*reinterpret_cast<const half2*>(&half_bits));
    return a.x == b.x && a.y == b.y;
}

__device__ __forceinline__ void mma_bf16_exact_fp16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    const unsigned ah0 = bf162_to_f162(a0), ah1 = bf162_to_f162(a1);
    const unsigned ah2 = bf162_to_f162(a2), ah3 = bf162_to_f162(a3);
    const unsigned bh0 = bf162_to_f162(b0), bh1 = bf162_to_f162(b1);
    const bool exact = bf16_pair_is_exact_half(a0, ah0) && bf16_pair_is_exact_half(a1, ah1) &&
                       bf16_pair_is_exact_half(a2, ah2) && bf16_pair_is_exact_half(a3, ah3) &&
                       bf16_pair_is_exact_half(b0, bh0) && bf16_pair_is_exact_half(b1, bh1);
    if (__all_sync(0xffffffff, exact)) {
        mma_f16(c0, c1, c2, c3, ah0, ah1, ah2, ah3, bh0, bh1);
        return;
    }
#endif
    mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}

} // namespace ninfer::ops
