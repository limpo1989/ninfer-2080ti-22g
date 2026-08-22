#pragma once

// ninfer::ops - KVarN record encode/decode kernels.
//
// One CTA owns one (tile, kv_head) record and runs HeadDim threads, so thread `d` owns
// channel `d` and holds that channel's Group token values in registers for the whole pass.
// Every statistic the codec needs is then either channel-local (a register loop) or
// token-wise across the CTA (one warp reduction plus an 8-warp combine).

#include "ops/common/warp.cuh"
#include "ops/kernel/kvarn_codec.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr float kKvarnStdMin    = 1e-3f;
inline constexpr float kKvarnStdMax    = 1e3f;
inline constexpr float kKvarnLogMin    = -0.3f;
inline constexpr float kKvarnLogMax    = 10.0f;
inline constexpr float kKvarnScaleMin  = 1e-10f;
inline constexpr int kKvarnRotateBatch = 8;

// Shared state of one record pass. Kept in one struct so the K and V phases reuse the same
// storage without a second allocation.
template <typename Spec>
struct KvarnCompressShared {
    static constexpr int D     = Spec::HeadDim;
    static constexpr int G     = Spec::Group;
    static constexpr int Warps = D / kWarpSize;

    float rotate[kKvarnRotateBatch * D];
    float partial_sum[Warps * G];
    float partial_sq[Warps * G];
    float token_std[G];
    float log_token[G];
    float best_token[G];
    float token_scale[G];
    float token_zero[G];
    float spread[Warps];
};

// Rotates every token's channel vector into the orthonormal Hadamard frame. `values` holds this
// thread's channel across all Group tokens on entry and on exit.
template <typename Spec>
__device__ __forceinline__ void kvarn_rotate_tile(float* values, float* scratch, int lane) {
    constexpr int D = Spec::HeadDim;
    constexpr int G = Spec::Group;
    const float norm = rsqrtf(static_cast<float>(D));

    for (int base = 0; base < G; base += kKvarnRotateBatch) {
#pragma unroll
        for (int b = 0; b < kKvarnRotateBatch; ++b) { scratch[b * D + lane] = values[base + b]; }
        __syncthreads();
#pragma unroll
        for (int stride = 1; stride < D; stride *= 2) {
            float mine[kKvarnRotateBatch];
            float other[kKvarnRotateBatch];
#pragma unroll
            for (int b = 0; b < kKvarnRotateBatch; ++b) {
                mine[b]  = scratch[b * D + lane];
                other[b] = scratch[b * D + (lane ^ stride)];
            }
            __syncthreads();
#pragma unroll
            for (int b = 0; b < kKvarnRotateBatch; ++b) {
                scratch[b * D + lane] =
                    (lane & stride) != 0 ? other[b] - mine[b] : mine[b] + other[b];
            }
            __syncthreads();
        }
#pragma unroll
        for (int b = 0; b < kKvarnRotateBatch; ++b) { values[base + b] = scratch[b * D + lane] * norm; }
        __syncthreads();
    }
}

// Standard deviation of every token column across all HeadDim channels, written to token_std.
template <typename Spec>
__device__ __forceinline__ void kvarn_token_std(const float* values, float inv_channel,
                                                KvarnCompressShared<Spec>& shared, int tid) {
    constexpr int D     = Spec::HeadDim;
    constexpr int G     = Spec::Group;
    constexpr int Warps = D / kWarpSize;
    const int warp      = tid >> 5;
    const int lane      = tid & (kWarpSize - 1);

    for (int t = 0; t < G; ++t) {
        const float x       = values[t] * inv_channel * __expf(-shared.log_token[t]);
        const float sum     = warp_sum(x);
        const float squares = warp_sum(x * x);
        if (lane == 0) {
            shared.partial_sum[warp * G + t] = sum;
            shared.partial_sq[warp * G + t]  = squares;
        }
    }
    __syncthreads();

    if (tid < G) {
        float sum     = 0.0f;
        float squares = 0.0f;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            sum += shared.partial_sum[w * G + tid];
            squares += shared.partial_sq[w * G + tid];
        }
        const float mean     = sum / D;
        const float variance = fmaxf(0.0f, (squares - D * mean * mean) / (D - 1));
        shared.token_std[tid] = sqrtf(variance);
    }
    __syncthreads();
}

// Standard deviation of this thread's channel row across the Group tokens.
template <typename Spec>
__device__ __forceinline__ float kvarn_channel_std(const float* values, float inv_channel,
                                                   const KvarnCompressShared<Spec>& shared) {
    constexpr int G = Spec::Group;
    float sum       = 0.0f;
    float squares   = 0.0f;
#pragma unroll 8
    for (int t = 0; t < G; ++t) {
        const float x = values[t] * inv_channel * __expf(-shared.log_token[t]);
        sum += x;
        squares += x * x;
    }
    const float mean     = sum / G;
    const float variance = fmaxf(0.0f, (squares - G * mean * mean) / (G - 1));
    return sqrtf(variance);
}

// Sum of the token-axis and channel-axis standard-deviation spreads. The normalization keeps
// the scales of the lowest-imbalance pass rather than the last one.
template <typename Spec>
__device__ __forceinline__ float kvarn_imbalance(const float* values, float inv_channel,
                                                 KvarnCompressShared<Spec>& shared, int tid) {
    constexpr int D     = Spec::HeadDim;
    constexpr int G     = Spec::Group;
    constexpr int Warps = D / kWarpSize;
    const int warp      = tid >> 5;
    const int lane      = tid & (kWarpSize - 1);

    kvarn_token_std<Spec>(values, inv_channel, shared, tid);

    const float channel = kvarn_channel_std<Spec>(values, inv_channel, shared);
    float channel_max   = warp_max(channel);
    float channel_min   = -warp_max(-channel);
    if (lane == 0) {
        shared.partial_sum[warp] = channel_max;
        shared.partial_sq[warp]  = channel_min;
    }
    __syncthreads();

    if (tid == 0) {
        float token_max = 0.0f;
        float token_min = CUDART_INF_F;
        for (int t = 0; t < G; ++t) {
            token_max = fmaxf(token_max, shared.token_std[t]);
            token_min = fminf(token_min, shared.token_std[t]);
        }
        channel_max = 0.0f;
        channel_min = CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            channel_max = fmaxf(channel_max, shared.partial_sum[w]);
            channel_min = fminf(channel_min, shared.partial_sq[w]);
        }
        shared.spread[0] = token_max / fmaxf(token_min, 1e-8f) +
                           channel_max / fmaxf(channel_min, 1e-8f);
    }
    __syncthreads();
    const float result = shared.spread[0];
    __syncthreads();
    return result;
}

// Alternating log-domain column/row standard-deviation normalization. TokenAxisFirst selects
// the tile orientation: a K tile has tokens on its column axis, a V tile has channels there.
template <typename Spec, bool TokenAxisFirst>
__device__ __forceinline__ void kvarn_variance_normalize(const float* values, int iterations,
                                                         KvarnCompressShared<Spec>& shared, int tid,
                                                         float& best_channel) {
    constexpr int G = Spec::Group;

    float log_channel = 0.0f;
    best_channel      = 0.0f;
    if (tid < G) {
        shared.log_token[tid]  = 0.0f;
        shared.best_token[tid] = 0.0f;
    }
    __syncthreads();

    float best = kvarn_imbalance<Spec>(values, 1.0f, shared, tid);

    const auto update_token = [&]() {
        kvarn_token_std<Spec>(values, __expf(-log_channel), shared, tid);
        if (tid < G) {
            const float deviation = fminf(fmaxf(shared.token_std[tid], kKvarnStdMin), kKvarnStdMax);
            shared.log_token[tid] = fminf(fmaxf(shared.log_token[tid] + __logf(deviation),
                                                kKvarnLogMin),
                                          kKvarnLogMax);
        }
        __syncthreads();
    };
    const auto update_channel = [&]() {
        const float deviation = fminf(
            fmaxf(kvarn_channel_std<Spec>(values, __expf(-log_channel), shared), kKvarnStdMin),
            kKvarnStdMax);
        log_channel = fminf(fmaxf(log_channel + __logf(deviation), kKvarnLogMin), kKvarnLogMax);
        __syncthreads();
    };

    for (int iteration = 0; iteration < iterations; ++iteration) {
        if constexpr (TokenAxisFirst) {
            update_token();
            update_channel();
        } else {
            update_channel();
            update_token();
        }

        const float imbalance = kvarn_imbalance<Spec>(values, __expf(-log_channel), shared, tid);
        if (imbalance <= best) {
            best         = imbalance;
            best_channel = log_channel;
            if (tid < G) { shared.best_token[tid] = shared.log_token[tid]; }
        }
        __syncthreads();
    }

    if (tid < G) { shared.log_token[tid] = shared.best_token[tid]; }
    __syncthreads();
}

template <typename Spec, int KVHeads>
__launch_bounds__(Spec::HeadDim) __global__
    void kvarn_compress_kernel(const __nv_bfloat16* __restrict__ k_src,
                               const __nv_bfloat16* __restrict__ v_src,
                               const std::int32_t* __restrict__ page_ids,
                               std::uint8_t* __restrict__ records, int iterations) {
    constexpr int D          = Spec::HeadDim;
    constexpr int G          = Spec::Group;
    constexpr int KeyMax     = (1 << Spec::KeyBits) - 1;
    constexpr int ValueMax   = (1 << Spec::ValueBits) - 1;
    constexpr int KeyPack    = Spec::KeyPack;
    constexpr int ValuePack  = Spec::ValuePack;

    __shared__ KvarnCompressShared<Spec> shared;

    const int kv_head = static_cast<int>(blockIdx.x);
    const int tile    = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & (kWarpSize - 1);

    std::uint8_t* record =
        records + kvarn_record_offset<Spec, KVHeads>(page_ids[tile], kv_head);

    const std::int64_t src_base =
        static_cast<std::int64_t>(tid) +
        static_cast<std::int64_t>(D) * kv_head +
        static_cast<std::int64_t>(D) * KVHeads * (static_cast<std::int64_t>(tile) * G);
    constexpr std::int64_t src_stride = static_cast<std::int64_t>(D) * KVHeads;

    float values[G];
    float best_channel = 0.0f;

    // ── K tile: [HeadDim, Group], round-to-nearest per channel ──────────────────────────────
    for (int t = 0; t < G; ++t) {
        values[t] = __bfloat162float(k_src[src_base + src_stride * t]);
    }
    kvarn_rotate_tile<Spec>(values, shared.rotate, tid);
    kvarn_variance_normalize<Spec, true>(values, iterations, shared, tid, best_channel);

    {
        const float channel_scale = __expf(best_channel);
        const float inv_channel   = 1.0f / channel_scale;
        float lo                  = CUDART_INF_F;
        float hi                  = -CUDART_INF_F;
#pragma unroll 8
        for (int t = 0; t < G; ++t) {
            const float balanced = values[t] * inv_channel * __expf(-shared.log_token[t]);
            lo                   = fminf(lo, balanced);
            hi                   = fmaxf(hi, balanced);
        }
        const float step = fmaxf((hi - lo) / KeyMax, kKvarnScaleMin);
        kvarn_store_f16(record, Spec::KScaleOff, tid, channel_scale * step);
        kvarn_store_f16(record, Spec::KZeroOff, tid, channel_scale * lo);
        if (tid < G) {
            kvarn_store_f16(record, Spec::KTokenOff, tid, __expf(shared.log_token[tid]));
        }

        // Channel `tid` owns Group/KeyPack contiguous payload bytes.
        std::uint8_t* payload =
            record + Spec::KPayloadOff + static_cast<std::int64_t>(tid) * (G / KeyPack);
        for (int byte = 0; byte < G / KeyPack; ++byte) {
            unsigned packed = 0;
#pragma unroll
            for (int j = 0; j < KeyPack; ++j) {
                const int t          = byte * KeyPack + j;
                const float balanced = values[t] * inv_channel * __expf(-shared.log_token[t]);
                const int code       = static_cast<int>(floorf((balanced - lo) / step + 0.5f));
                packed |= static_cast<unsigned>(min(max(code, 0), KeyMax)) << (j * Spec::KeyBits);
            }
            payload[byte] = static_cast<std::uint8_t>(packed);
        }
    }
    __syncthreads();

    // ── V tile: [Group, HeadDim], round-to-nearest per token ────────────────────────────────
    for (int t = 0; t < G; ++t) {
        values[t] = __bfloat162float(v_src[src_base + src_stride * t]);
    }
    kvarn_rotate_tile<Spec>(values, shared.rotate, tid);
    kvarn_variance_normalize<Spec, false>(values, iterations, shared, tid, best_channel);

    {
        const float channel_scale = __expf(best_channel);
        const float inv_channel   = 1.0f / channel_scale;
        kvarn_store_f16(record, Spec::VChannelOff, tid, channel_scale);

        // A V row spans every channel, so its range is a CTA-wide reduction per token.
        const int warp      = tid >> 5;
        constexpr int Warps = D / kWarpSize;
        for (int t = 0; t < G; ++t) {
            const float balanced = values[t] * inv_channel * __expf(-shared.log_token[t]);
            const float hi       = warp_max(balanced);
            const float lo       = -warp_max(-balanced);
            if (lane == 0) {
                shared.partial_sum[warp * G + t] = hi;
                shared.partial_sq[warp * G + t]  = lo;
            }
        }
        __syncthreads();
        if (tid < G) {
            float hi = -CUDART_INF_F;
            float lo = CUDART_INF_F;
#pragma unroll
            for (int w = 0; w < Warps; ++w) {
                hi = fmaxf(hi, shared.partial_sum[w * G + tid]);
                lo = fminf(lo, shared.partial_sq[w * G + tid]);
            }
            const float step          = fmaxf((hi - lo) / ValueMax, kKvarnScaleMin);
            const float token_scale   = __expf(shared.log_token[tid]);
            shared.token_scale[tid]   = step;
            shared.token_zero[tid]    = lo;
            kvarn_store_f16(record, Spec::VScaleOff, tid, token_scale * step);
            kvarn_store_f16(record, Spec::VZeroOff, tid, token_scale * lo);
        }
        __syncthreads();

        // A V row is channel-contiguous, so ValuePack neighbouring threads share one byte.
        std::uint8_t* payload = record + Spec::VPayloadOff;
        for (int t = 0; t < G; ++t) {
            const float balanced = values[t] * inv_channel * __expf(-shared.log_token[t]);
            const int code       = static_cast<int>(
                floorf((balanced - shared.token_zero[t]) / shared.token_scale[t] + 0.5f));
            unsigned packed = static_cast<unsigned>(min(max(code, 0), ValueMax))
                              << ((tid % ValuePack) * Spec::ValueBits);
#pragma unroll
            for (int offset = 1; offset < ValuePack; offset *= 2) {
                packed |= __shfl_down_sync(kFullWarpMask, packed, offset, kWarpSize);
            }
            if (tid % ValuePack == 0) {
                payload[(static_cast<std::int64_t>(t) * D + tid) / ValuePack] =
                    static_cast<std::uint8_t>(packed);
            }
        }
    }
}

template <typename Spec, int KVHeads>
__launch_bounds__(Spec::HeadDim) __global__
    void kvarn_decompress_kernel(const std::uint8_t* __restrict__ records,
                                 const std::int32_t* __restrict__ page_ids,
                                 __nv_bfloat16* __restrict__ k_dst,
                                 __nv_bfloat16* __restrict__ v_dst) {
    constexpr int D = Spec::HeadDim;
    constexpr int G = Spec::Group;

    __shared__ float scratch[kKvarnRotateBatch * D];

    const int kv_head = static_cast<int>(blockIdx.x);
    const int tile    = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);

    const std::uint8_t* record =
        records + kvarn_record_offset<Spec, KVHeads>(page_ids[tile], kv_head);

    const std::int64_t dst_base =
        static_cast<std::int64_t>(tid) +
        static_cast<std::int64_t>(D) * kv_head +
        static_cast<std::int64_t>(D) * KVHeads * (static_cast<std::int64_t>(tile) * G);
    constexpr std::int64_t dst_stride = static_cast<std::int64_t>(D) * KVHeads;

    float values[G];

    const float k_scale             = kvarn_load_f16(record, Spec::KScaleOff, tid);
    const float k_zero              = kvarn_load_f16(record, Spec::KZeroOff, tid);
    const std::uint8_t* k_payload   = record + Spec::KPayloadOff;
    for (int t = 0; t < G; ++t) {
        const int code = kvarn_unpack<Spec::KeyBits>(
            k_payload, static_cast<std::int64_t>(tid) * G + t);
        values[t] = (code * k_scale + k_zero) * kvarn_load_f16(record, Spec::KTokenOff, t);
    }
    kvarn_rotate_tile<Spec>(values, scratch, tid);
    for (int t = 0; t < G; ++t) { k_dst[dst_base + dst_stride * t] = __float2bfloat16(values[t]); }

    const float v_channel           = kvarn_load_f16(record, Spec::VChannelOff, tid);
    const std::uint8_t* v_payload   = record + Spec::VPayloadOff;
    for (int t = 0; t < G; ++t) {
        const int code = kvarn_unpack<Spec::ValueBits>(
            v_payload, static_cast<std::int64_t>(t) * D + tid);
        values[t] = (code * kvarn_load_f16(record, Spec::VScaleOff, t) +
                     kvarn_load_f16(record, Spec::VZeroOff, t)) *
                    v_channel;
    }
    kvarn_rotate_tile<Spec>(values, scratch, tid);
    for (int t = 0; t < G; ++t) { v_dst[dst_base + dst_stride * t] = __float2bfloat16(values[t]); }
}

} // namespace ninfer::ops
