#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kN                 = 34816;
constexpr int kK                 = 5120;
constexpr int kIntermediate      = kN / 2;
constexpr int kGroupK            = 64;
constexpr int kGroups            = kK / kGroupK;
constexpr int kBytesPerGroup     = 32;
constexpr int kVecBytes          = 16;
constexpr int kGroupsPerWarpTile = 16;
constexpr int kVecsPerWarpTile   = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
// One warp owns one gate/up row pair. Eight warps per CTA keep two CTAs resident on a Turing SM
// and halve the per-row activation re-read against the four-warp shape this replaces.
constexpr int kWarpsPerBlock     = 8;
constexpr int kBlockThreads      = kWarpsPerBlock * 32;
constexpr int kPairsPerBlock     = kWarpsPerBlock;
constexpr int kTiles             = kGroups / kGroupsPerWarpTile;
constexpr int kTileK             = kGroupsPerWarpTile * kGroupK;
static_assert(kIntermediate % kPairsPerBlock == 0);
static_assert(kBytesPerGroup == 2 * kVecBytes);
static_assert(kGroups % kGroupsPerWarpTile == 0);
static_assert(kVecsPerWarpTile == 32);

struct Q4SwiGluSmallTGeometry {
    static constexpr int kInputRows    = kK;
    static constexpr int kGroupsPerRow = kK / kGroupK;
};

struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerCta = 8;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? kIntermediate : 0);
    }
};

struct Q4SwiGluSmallTEpilogue {
    __nv_bfloat16* out;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < ActiveCols) {
            out[static_cast<std::int64_t>(col0) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < ActiveCols) {
            out[static_cast<std::int64_t>(col0 + 1) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

using SmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_small_t_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    constexpr int kBlocks = kIntermediate / Q4SwiGluSmallTRows::kOutputRowsPerCta;
    const Q4SwiGluSmallTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data)};
    q4_small_t_mma_kernel<Q4SwiGluSmallTGeometry, TileCols, ActiveCols, Q4SwiGluSmallTEpilogue,
                          Q4SwiGluSmallTRows>
        <<<kBlocks, Q4DraftSmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            epilogue, Q4SwiGluSmallTRows{});
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_active<kQ4SwiGluLastGemvPair + 1 + static_cast<int>(Offsets)>...};
}

constexpr auto kSmallTLaunchers = make_small_t_launchers(
    std::make_index_sequence<32 - kQ4SwiGluLastGemvPair>{});

// Decode-shaped fused gate/up GEMV for T = 1..kQ4SwiGluLastGemvPair.
//
// The exact-small-T MMA core it replaces above T=1 stages a 16-row tile through shared memory for
// output columns that are almost all padding at these token counts, and lands near a third of the
// device's streaming read rate. Here a warp reads each weight row exactly once and carries all T
// tokens through it, so the token count costs arithmetic instead of DRAM.
//
// Two Turing-specific details carry the bandwidth. Turing has no async copy, so the cp.async
// pipeline this replaces degenerated to a load/store pair that stalls the warp at the shared
// store: the next tile's codes and scales are instead prefetched into registers and spilled to
// shared only after the current tile has been consumed, which keeps a warp's global loads in
// flight across the whole unpack. And the activations are staged one K tile at a time, which
// keeps the CTA's shared footprint independent of T and leaves the bf16x2 reads
// bank-conflict-free (lane L reads word L of its group).
template <int ActiveTokens>
__global__ __launch_bounds__(kBlockThreads, 2) void q4_linear_swiglu_gemv_pair_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    static_assert(ActiveTokens >= 1);
    constexpr int kStages    = 2;
    constexpr int kXTileVecs = ActiveTokens * kTileK / 8;

    __shared__ __align__(16) __nv_bfloat16 x_tile[ActiveTokens][kTileK];
    __shared__ uint4 code_tile[kWarpsPerBlock][kStages][2][kVecsPerWarpTile];
    __shared__ uint4 scale_tile[kWarpsPerBlock][kStages][2][2];

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * kGroups * kBytesPerGroup;
    const std::uint8_t* gate_scale_row = scales + static_cast<std::int64_t>(out_row) * kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * 2;

    uint4 gate_code_reg;
    uint4 up_code_reg;
    uint4 gate_scale_reg;
    uint4 up_scale_reg;

    const auto load_tile_regs = [&](int tile) {
        const int g0  = tile * kGroupsPerWarpTile;
        gate_code_reg = load_ldg<uint4>(
            reinterpret_cast<const uint4*>(gate_code_row + g0 * kBytesPerGroup) + lane);
        up_code_reg = load_ldg<uint4>(
            reinterpret_cast<const uint4*>(up_code_row + g0 * kBytesPerGroup) + lane);
        if (lane < 2) {
            gate_scale_reg =
                load_ldg<uint4>(reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
            up_scale_reg =
                load_ldg<uint4>(reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
        }
    };

    const auto spill_tile_regs = [&](int buf) {
        code_tile[warp][buf][0][lane] = gate_code_reg;
        code_tile[warp][buf][1][lane] = up_code_reg;
        if (lane < 2) {
            scale_tile[warp][buf][0][lane] = gate_scale_reg;
            scale_tile[warp][buf][1][lane] = up_scale_reg;
        }
    };

    // x is 16-byte aligned by construction (activations come from the 256-byte-aligned workspace
    // arena), so a K tile stages as uint4.
    const auto stage_x_tile = [&](int tile) {
        auto* x_tile_v = reinterpret_cast<uint4*>(&x_tile[0][0]);
        for (int i = static_cast<int>(threadIdx.x); i < kXTileVecs; i += kBlockThreads) {
            const int token = i / (kTileK / 8);
            const int slot  = i - token * (kTileK / 8);
            x_tile_v[i]     = load_ldg<uint4>(
                reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token) * kK +
                                               tile * kTileK) +
                slot);
        }
    };

    float gate_acc[ActiveTokens];
    float up_acc[ActiveTokens];
#pragma unroll
    for (int token = 0; token < ActiveTokens; ++token) {
        gate_acc[token] = 0.0f;
        up_acc[token]   = 0.0f;
    }

    load_tile_regs(0);
    spill_tile_regs(0);
    stage_x_tile(0);
    __syncthreads();

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const bool has_next = tile + 1 < kTiles;
        if (has_next) { load_tile_regs(tile + 1); }

        const int buf           = tile & 1;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales   = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));

            const int gate_packed =
                static_cast<int>(gate_codes[tile_group * kBytesPerGroup + lane]);
            const float gate_w0 =
                static_cast<float>(sign_extend<4>(gate_packed & 0x0f)) * gate_scale;
            const float gate_w1 = static_cast<float>(sign_extend<4>(gate_packed >> 4)) * gate_scale;
            const int up_packed = static_cast<int>(up_codes[tile_group * kBytesPerGroup + lane]);
            const float up_w0   = static_cast<float>(sign_extend<4>(up_packed & 0x0f)) * up_scale;
            const float up_w1   = static_cast<float>(sign_extend<4>(up_packed >> 4)) * up_scale;

            const int tile_k = tile_group * kGroupK + lane * 2;
#pragma unroll
            for (int token = 0; token < ActiveTokens; ++token) {
                const float2 xv = __bfloat1622float2(
                    reinterpret_cast<const __nv_bfloat162*>(&x_tile[token][0])[tile_k >> 1]);
                gate_acc[token] = fmaf(gate_w0, xv.x, gate_acc[token]);
                gate_acc[token] = fmaf(gate_w1, xv.y, gate_acc[token]);
                up_acc[token]   = fmaf(up_w0, xv.x, up_acc[token]);
                up_acc[token]   = fmaf(up_w1, xv.y, up_acc[token]);
            }
        }

        if (has_next) {
            __syncthreads();
            spill_tile_regs((tile + 1) & 1);
            stage_x_tile(tile + 1);
            __syncthreads();
        }
    }

#pragma unroll
    for (int token = 0; token < ActiveTokens; ++token) {
        const float gate = warp_reduce_sum(gate_acc[token]);
        const float up   = warp_reduce_sum(up_acc[token]);
        if (lane == 0) {
            out[static_cast<std::int64_t>(token) * kIntermediate + out_row] =
                __float2bfloat16(silu(gate) * up);
        }
    }
}

using GemvPairLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveTokens>
void launch_gemv_pair_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int kBlocks = kIntermediate / kPairsPerBlock;
    q4_linear_swiglu_gemv_pair_kernel<ActiveTokens><<<kBlocks, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_gemv_pair_launchers(std::index_sequence<Offsets...>) {
    return std::array<GemvPairLauncher, sizeof...(Offsets)>{
        &launch_gemv_pair_active<1 + static_cast<int>(Offsets)>...};
}

constexpr auto kGemvPairLaunchers =
    make_gemv_pair_launchers(std::make_index_sequence<kQ4SwiGluLastGemvPair>{});

} // namespace

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("q4 linear_swiglu GEMV requires weight [34816,5120]");
    }
    if (x.ne[1] < 1 || x.ne[1] > kQ4SwiGluLastGemvPair) {
        throw std::invalid_argument("q4 linear_swiglu GEMV pair requires T=1..6");
    }
    kGemvPairLaunchers[static_cast<std::size_t>(x.ne[1] - 1)](x, w, out, stream);
}

void q4_linear_swiglu_small_t_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (x.ne[1] <= kQ4SwiGluLastGemvPair || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=7..32");
    }
    kSmallTLaunchers[static_cast<std::size_t>(x.ne[1] - kQ4SwiGluLastGemvPair - 1)](x, w, out,
                                                                                    stream);
}

} // namespace ninfer::ops::detail
