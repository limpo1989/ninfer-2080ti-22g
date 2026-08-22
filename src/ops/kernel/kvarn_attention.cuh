#pragma once

// ninfer::ops - grouped-query attention over KVarN records.
//
// Scores and the value reduction are evaluated in the record's own Hadamard frame: the rotation
// is orthonormal, so rotating Q reproduces the original scores exactly, and one more rotation
// returns the accumulated output to the original frame. No dequantized K or V tile is ever
// materialized outside registers.
//
// One CTA owns one (kv_head, split, query token) and runs HeadDim threads. The score phase maps
// threads to (token-in-page, channel quarter) so that the packed key bytes are read in the order
// they are stored; the value phase maps threads to channels for the same reason.

#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/kvarn_codec.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Spec, typename Geometry>
struct KvarnAttentionShared {
    static constexpr int D     = Spec::HeadDim;
    static constexpr int G     = Spec::Group;
    static constexpr int Rows  = Geometry::GroupSize;
    static constexpr int Quads = D / G; // channel chunks the score phase splits over

    float query[Rows * D];
    float score[Quads * Rows * G];
    float probability[Rows * G];
    float key_scale[D];
    float key_zero[D];
    float key_token[G];
    float value_channel[D];
    float value_scale[G];
    float value_zero[G];
    float running_max[Rows];
    float running_sum[Rows];
    float rescale[Rows];
};

// Partial (max, sum, accumulator) layout shared with the reducer.
__device__ __forceinline__ std::int64_t kvarn_partial_stat_index(int q_head, int q_heads, int token,
                                                                 int split, int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(q_heads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

__device__ __forceinline__ std::int64_t kvarn_partial_acc_index(int q_head, int q_heads, int d,
                                                                int head_dim, int token, int split,
                                                                int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(head_dim) *
               kvarn_partial_stat_index(q_head, q_heads, token, split, tokens);
}

template <typename Spec, typename Geometry>
__launch_bounds__(Spec::HeadDim) __global__
    void kvarn_attention_partial_kernel(const __nv_bfloat16* __restrict__ q,
                                        const std::uint8_t* __restrict__ records,
                                        const std::int32_t* __restrict__ block_table,
                                        std::int32_t pages, std::int32_t tokens, float scale,
                                        __nv_bfloat16* __restrict__ partial_acc,
                                        float* __restrict__ partial_max,
                                        float* __restrict__ partial_sum) {
    constexpr int D       = Spec::HeadDim;
    constexpr int G       = Spec::Group;
    constexpr int Rows    = Geometry::GroupSize;
    constexpr int Quads   = D / G;
    constexpr float Log2E = 1.4426950408889634074f;

    __shared__ KvarnAttentionShared<Spec, Geometry> shared;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int token       = static_cast<int>(blockIdx.z);
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int slot        = tid % G;  // token inside the page
    const int quad        = tid / G;  // channel chunk

    const int pages_per_split = (pages + split_count - 1) / split_count;
    const int page_begin      = split * pages_per_split;
    const int page_end        = min(pages, page_begin + pages_per_split);

    // Rotate this token's query rows into the record frame.
    for (int row = 0; row < Rows; ++row) {
        const int q_head = kv_head * Rows + row;
        shared.query[row * D + tid] = __bfloat162float(
            q[static_cast<std::int64_t>(tid) +
              static_cast<std::int64_t>(D) * (q_head + Geometry::QHeads * token)]);
    }
    __syncthreads();
    for (int row = 0; row < Rows; ++row) { kvarn_hadamard_shared<D>(shared.query + row * D, tid); }

    if (tid < Rows) {
        shared.running_max[tid] = -CUDART_INF_F;
        shared.running_sum[tid] = 0.0f;
    }
    float accumulator[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) { accumulator[row] = 0.0f; }
    __syncthreads();

    for (int page = page_begin; page < page_end; ++page) {
        const std::uint8_t* record =
            records + kvarn_record_offset<Spec, Geometry::KVHeads>(block_table[page], kv_head);

        shared.key_scale[tid]     = kvarn_load_f16(record, Spec::KScaleOff, tid);
        shared.key_zero[tid]      = kvarn_load_f16(record, Spec::KZeroOff, tid);
        shared.value_channel[tid] = kvarn_load_f16(record, Spec::VChannelOff, tid);
        if (tid < G) {
            shared.key_token[tid]   = kvarn_load_f16(record, Spec::KTokenOff, tid);
            shared.value_scale[tid] = kvarn_load_f16(record, Spec::VScaleOff, tid);
            shared.value_zero[tid]  = kvarn_load_f16(record, Spec::VZeroOff, tid);
        }
        __syncthreads();

        // ── scores: thread `slot` owns one key token, `quad` owns a quarter of the channels ──
        const std::uint8_t* key_payload = record + Spec::KPayloadOff;
        const float token_scale         = shared.key_token[slot];
        float partial[Rows];
#pragma unroll
        for (int row = 0; row < Rows; ++row) { partial[row] = 0.0f; }
        for (int d = quad * G; d < (quad + 1) * G; ++d) {
            const int code =
                kvarn_unpack<Spec::KeyBits>(key_payload, static_cast<std::int64_t>(d) * G + slot);
            const float key = (code * shared.key_scale[d] + shared.key_zero[d]) * token_scale;
#pragma unroll
            for (int row = 0; row < Rows; ++row) { partial[row] += shared.query[row * D + d] * key; }
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            shared.score[(quad * Rows + row) * G + slot] = partial[row];
        }
        __syncthreads();

        // ── online softmax over this page's G keys ──────────────────────────────────────────
        for (int flat = tid; flat < Rows * G; flat += D) {
            const int row = flat / G;
            const int key = flat - row * G;
            float total   = 0.0f;
#pragma unroll
            for (int q = 0; q < Quads; ++q) { total += shared.score[(q * Rows + row) * G + key]; }
            shared.probability[flat] = total * scale;
        }
        __syncthreads();

        if (tid < Rows) {
            float page_max = -CUDART_INF_F;
            for (int key = 0; key < G; ++key) {
                page_max = fmaxf(page_max, shared.probability[tid * G + key]);
            }
            const float updated = fmaxf(shared.running_max[tid], page_max);
            float page_sum      = 0.0f;
            for (int key = 0; key < G; ++key) {
                const float weight = exp2f((shared.probability[tid * G + key] - updated) * Log2E);
                shared.probability[tid * G + key] = weight;
                page_sum += weight;
            }
            const float rescale = shared.running_max[tid] == -CUDART_INF_F
                                      ? 0.0f
                                      : exp2f((shared.running_max[tid] - updated) * Log2E);
            shared.running_sum[tid] = shared.running_sum[tid] * rescale + page_sum;
            shared.running_max[tid] = updated;
            shared.rescale[tid]     = rescale;
        }
        __syncthreads();

#pragma unroll
        for (int row = 0; row < Rows; ++row) { accumulator[row] *= shared.rescale[row]; }

        // ── values: thread `tid` owns one channel of the output ─────────────────────────────
        const std::uint8_t* value_payload = record + Spec::VPayloadOff;
        const float channel               = shared.value_channel[tid];
        for (int key = 0; key < G; ++key) {
            const int code = kvarn_unpack<Spec::ValueBits>(
                value_payload, static_cast<std::int64_t>(key) * D + tid);
            const float value =
                (code * shared.value_scale[key] + shared.value_zero[key]) * channel;
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                accumulator[row] += shared.probability[row * G + key] * value;
            }
        }
        __syncthreads();
    }

    // Return the split-local accumulator to the original frame before publishing it. The
    // rotation is linear, so it commutes with the reducer's weighted sum over splits.
    for (int row = 0; row < Rows; ++row) {
        shared.query[row * D + tid] = accumulator[row];
    }
    __syncthreads();
    for (int row = 0; row < Rows; ++row) { kvarn_hadamard_shared<D>(shared.query + row * D, tid); }

    for (int row = 0; row < Rows; ++row) {
        const int q_head = kv_head * Rows + row;
        partial_acc[kvarn_partial_acc_index(q_head, Geometry::QHeads, tid, D, token, split,
                                            tokens)] =
            __float2bfloat16(shared.query[row * D + tid]);
    }
    if (tid < Rows) {
        const int q_head = kv_head * Rows + tid;
        const std::int64_t index =
            kvarn_partial_stat_index(q_head, Geometry::QHeads, token, split, tokens);
        const bool empty     = page_begin >= page_end;
        partial_max[index]   = empty ? -CUDART_INF_F : shared.running_max[tid];
        partial_sum[index]   = empty ? 0.0f : shared.running_sum[tid];
    }
}

// Combines the split-local partials into the final BF16 output.
template <typename Spec, typename Geometry>
__launch_bounds__(Spec::HeadDim) __global__
    void kvarn_attention_reduce_kernel(const __nv_bfloat16* __restrict__ partial_acc,
                                       const float* __restrict__ partial_max,
                                       const float* __restrict__ partial_sum,
                                       std::int32_t split_count, std::int32_t tokens,
                                       __nv_bfloat16* __restrict__ out) {
    constexpr int D       = Spec::HeadDim;
    constexpr float Log2E = 1.4426950408889634074f;

    const int q_head = static_cast<int>(blockIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const int tid    = static_cast<int>(threadIdx.x);

    float total_max = -CUDART_INF_F;
    for (int split = 0; split < split_count; ++split) {
        total_max = fmaxf(total_max,
                          partial_max[kvarn_partial_stat_index(q_head, Geometry::QHeads, token,
                                                               split, tokens)]);
    }
    if (total_max == -CUDART_INF_F) {
        out[static_cast<std::int64_t>(tid) +
            static_cast<std::int64_t>(D) * (q_head + Geometry::QHeads * token)] =
            __float2bfloat16(0.0f);
        return;
    }

    float total_sum = 0.0f;
    float numerator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
        const std::int64_t index =
            kvarn_partial_stat_index(q_head, Geometry::QHeads, token, split, tokens);
        const float split_sum = partial_sum[index];
        if (split_sum <= 0.0f) { continue; }
        const float weight = exp2f((partial_max[index] - total_max) * Log2E);
        total_sum += split_sum * weight;
        numerator += weight * __bfloat162float(partial_acc[kvarn_partial_acc_index(
                                  q_head, Geometry::QHeads, tid, D, token, split, tokens)]);
    }
    out[static_cast<std::int64_t>(tid) +
        static_cast<std::int64_t>(D) * (q_head + Geometry::QHeads * token)] =
        __float2bfloat16(total_sum > 0.0f ? numerator / total_sum : 0.0f);
}

} // namespace ninfer::ops
