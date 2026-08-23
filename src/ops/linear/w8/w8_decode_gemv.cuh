#pragma once

// W8G32 small-T decode GEMV.
//
// The exact-small-T MMA core owns one 16-row output tile per CTA and refills its shared staging
// through two barriers at every K group. At T=1..8 that structure issues only a handful of
// global loads between barriers, so a 1.27 GB vocabulary head runs at a fraction of the device's
// streaming read rate even though the MMA itself is nearly free.
//
// This core drops the MMA and the staging entirely: one warp owns one output row, each lane
// streams a private 16-byte code vector per K phase, and the activations are re-read from L1
// (they are a few KiB and shared by every warp on the SM). There is no shared memory and no
// barrier, so the loop is a pure dependent-free load stream and reaches the device's streaming
// read rate. Products are formed in FP32 from exactly-represented int8 codes and bf16
// activations, which is at least as accurate as the MMA route it replaces.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/w8/w8_rowsplit_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One lane owns 16 contiguous codes, so a warp covers 512 K values per phase and a lane's codes
// always sit inside a single 32-wide quantization group.
inline constexpr int kW8DecodeGemvValuesPerLane = 16;
inline constexpr int kW8DecodeGemvPhaseValues   = 32 * kW8DecodeGemvValuesPerLane;

template <int Hidden, int ActiveTokens, int RowsPerCta, class Output>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w8_decode_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output) {
    static_assert(Hidden > 0 && (Hidden % kW8DecodeGemvPhaseValues) == 0);
    static_assert(ActiveTokens >= 1);
    static_assert(RowsPerCta > 0 && RowsPerCta * 32 <= 1024);

    constexpr int kGroup          = 32;
    constexpr int kGroupsPerRow   = Hidden / kGroup;
    constexpr int kPhases         = Hidden / kW8DecodeGemvPhaseValues;
    constexpr int kGroupsPerPhase = kW8DecodeGemvPhaseValues / kGroup;

    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int cta_row0 = static_cast<int>(blockIdx.x) * RowsPerCta;
    const int row      = cta_row0 + warp;

    const std::uint8_t* __restrict__ code_row = codes + static_cast<std::int64_t>(row) * Hidden;
    const auto* __restrict__ scale_row = reinterpret_cast<const std::uint16_t*>(scales) +
                                         static_cast<std::int64_t>(row) * kGroupsPerRow;

    float acc[ActiveTokens];
#pragma unroll
    for (int token = 0; token < ActiveTokens; ++token) { acc[token] = 0.0F; }

#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        const int lane_k0  = phase * kW8DecodeGemvPhaseValues + lane * kW8DecodeGemvValuesPerLane;
        const uint4 packed = load_vec<uint4>(code_row + lane_k0);

        // Two lanes share a group; one half-warp load plus a broadcast covers the phase.
        unsigned scale_bits = 0;
        if (lane < kGroupsPerPhase) { scale_bits = scale_row[phase * kGroupsPerPhase + lane]; }
        scale_bits = __shfl_sync(kFullWarpMask, scale_bits, lane >> 1);
        const float scale = __half2float(__ushort_as_half(scale_bits));

        float weights[kW8DecodeGemvValuesPerLane];
#pragma unroll
        for (int word_index = 0; word_index < 4; ++word_index) {
            const std::uint32_t word = (&packed.x)[word_index];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                weights[word_index * 4 + byte] =
                    static_cast<float>(static_cast<std::int8_t>((word >> (byte * 8)) & 0xffu));
            }
        }

#pragma unroll
        for (int token = 0; token < ActiveTokens; ++token) {
            const __nv_bfloat16* __restrict__ x_row =
                x + static_cast<std::int64_t>(token) * Hidden + lane_k0;
            const uint4 lo = load_vec<uint4>(x_row);
            const uint4 hi = load_vec<uint4>(x_row + 8);
            float partial  = 0.0F;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const uint4 bits = half == 0 ? lo : hi;
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float2 values = bf16x2_bits_to_float2((&bits.x)[pair]);
                    partial = fmaf(weights[half * 8 + pair * 2], values.x, partial);
                    partial = fmaf(weights[half * 8 + pair * 2 + 1], values.y, partial);
                }
            }
            acc[token] = fmaf(partial, scale, acc[token]);
        }
    }

#pragma unroll
    for (int token = 0; token < ActiveTokens; ++token) {
        acc[token] = warp_reduce_sum(acc[token]);
    }
    if (lane == 0) {
        const W8OutputTile tile = output.tile(cta_row0);
#pragma unroll
        for (int token = 0; token < ActiveTokens; ++token) {
            *tile.at(row, token) = __float2bfloat16_rn(acc[token]);
        }
    }
}

} // namespace ninfer::ops::detail
