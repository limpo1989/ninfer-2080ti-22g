#pragma once

// SM75 KVarN prefill: evaluate the compressed prefix with native FP16 Tensor Cores,
// evaluate the small dense sink/tail/current region separately, then merge both
// online-softmax states. KVarN storage and state transitions remain unchanged.

#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/kvarn_gqa.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKvarnTcKeyTile = 32;

template <typename Geometry>
__device__ __forceinline__ std::int64_t kvarn_prefill_row_index(int q_head, int query,
                                                                int query_columns, int row) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (query + static_cast<std::int64_t>(query_columns) * row);
}

template <typename Spec, typename Geometry, int QueryRows, bool TensorPv = false>
struct KvarnTcPrefixShared {
    static constexpr int D       = Spec::HeadDim;
    static constexpr int Bc      = kKvarnTcKeyTile;
    static constexpr int QRows   = QueryRows;
    static constexpr int Threads = D;

    static_assert(QueryRows % 16 == 0);
    static_assert(D == kGqaPrefillHeadDim);

    __align__(16) __half query[QueryRows * D];

    union alignas(16) Scratch {
        __half key[Bc * D];
        float probability[Bc * QueryRows];
        float transform[D];
        __half tensor_pv[TensorPv ? QueryRows * Bc + Bc * D : 1];
    } scratch;

    float key_scale[D];
    float key_zero[D];
    float value_channel[D];
    float key_token[Spec::Group];
    float value_scale[Spec::Group];
    float value_zero[Spec::Group];
    float running_max[QueryRows];
    float running_sum[QueryRows];
    float rescale[QueryRows];
};

template <typename Spec, typename Geometry, int QueryRows, bool TensorPv>
__device__ __forceinline__ void
kvarn_stage_key_tile(KvarnTcPrefixShared<Spec, Geometry, QueryRows, TensorPv>& shared,
                     const std::uint8_t* __restrict__ record, int key_base, int tid) {
    using Shared          = KvarnTcPrefixShared<Spec, Geometry, QueryRows, TensorPv>;
    constexpr int D       = Shared::D;
    constexpr int Bc      = Shared::Bc;
    constexpr int G       = Spec::Group;
    constexpr int Threads = Shared::Threads;

    for (int element = tid; element < Bc * D; element += Threads) {
        const int key        = element / D;
        const int d          = element - key * D;
        const int record_key = key_base + key;
        const int code       = kvarn_unpack<Spec::KeyBits>(record + Spec::KPayloadOff,
                                                           static_cast<std::int64_t>(d) * G + record_key);
        const float decoded =
            (code * shared.key_scale[d] + shared.key_zero[d]) * shared.key_token[record_key];
        shared.scratch.key[key * D + gqa_prefill_swz(key, d)] = __float2half(decoded);
    }
}

// Heads query heads share one KV head; Tokens consecutive queries share each record decode.
// Padding belongs only to the MMA tile, never to the public query or partial-output layout.
template <typename Spec, typename Geometry, int QueryRows, int Heads = 1,
          int Tokens = QueryRows, bool TensorPv = false>
__launch_bounds__(Spec::HeadDim, (QueryRows <= 16 ? 2 : 1)) __global__ void kvarn_gqa_tc_prefix_kernel(
    const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ prefix,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ block_tables,
    std::int32_t table_stride, std::int32_t width, std::int32_t query_offset,
    std::int32_t query_columns, const std::uint8_t* __restrict__ records, float scale,
    float* __restrict__ prefix_max, float* __restrict__ prefix_sum) {
    using Shared                = KvarnTcPrefixShared<Spec, Geometry, QueryRows, TensorPv>;
    constexpr int D             = Shared::D;
    constexpr int Bc            = Shared::Bc;
    constexpr int Threads       = Shared::Threads;
    constexpr int QueryWarps    = QueryRows / 16;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int LogicalRows  = Heads * Tokens;
    constexpr int HeadTiles    = Geometry::QHeads / Heads;
    constexpr int PvDimWarps   = (Threads / 32) / QueryWarps;
    constexpr int PvNt         = D / (PvDimWarps * 8);
    static_assert(!TensorPv || (QueryRows == 32 && Heads == 1));
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    __shared__ Shared shared;

    const int query_block = static_cast<int>(blockIdx.x);
    static_assert(LogicalRows <= QueryRows);
    static_assert(LogicalRows % 4 == 0);
    static_assert(Heads == 1 || Heads == Geometry::GroupSize);
    const int head_begin  = static_cast<int>(blockIdx.y) % HeadTiles * Heads;
    const int split       = static_cast<int>(blockIdx.y) / HeadTiles;
    const int splits      = static_cast<int>(gridDim.y) / HeadTiles;
    const int row         = static_cast<int>(blockIdx.z);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int query_begin = query_block * Tokens;
    const int kv_head     = head_begin / Geometry::GroupSize;
    const int output_row  = row + static_cast<int>(gridDim.z) * split;
    const int columns     = kvarn_row_columns(valid_columns, row, width);
    const int query_tokens =
        max(0, min(query_columns - query_begin, columns - query_offset - query_begin));

    if (query_begin >= query_columns) { return; }

    const int first = positions[width * row];
    const int record_page_end =
        first >= kKvarnSinkPages * Spec::Group ? first / Spec::Group : kKvarnSinkPages;
    const int pages_per_split =
        (record_page_end - kKvarnSinkPages + splits - 1) / splits;
    const int page_begin = kKvarnSinkPages + split * pages_per_split;
    const int page_end   = min(record_page_end, page_begin + pages_per_split);
    if (query_tokens <= 0 || page_begin >= page_end) {
        for (int local_row = 0; local_row < LogicalRows; ++local_row) {
            const int query  = query_begin + local_row / Heads;
            const int q_head = head_begin + local_row % Heads;
            if (query >= query_columns) { continue; }
            const std::int64_t index =
                static_cast<std::int64_t>(tid) +
                static_cast<std::int64_t>(D) *
                    (q_head + Geometry::QHeads * (query + query_columns * output_row));
            prefix[index] = __float2bfloat16(0.0f);
            if (tid == 0) {
                const std::int64_t stat =
                    kvarn_prefill_row_index<Geometry>(q_head, query, query_columns, output_row);
                prefix_max[stat] = -CUDART_INF_F;
                prefix_sum[stat] = 0.0f;
            }
        }
        return;
    }

    for (int local_row = 0; local_row < QueryRows; ++local_row) {
        const int query          = query_begin + local_row / Heads;
        const int q_head         = head_begin + local_row % Heads;
        const std::int64_t index = static_cast<std::int64_t>(tid) +
                                   static_cast<std::int64_t>(D) *
                                       (q_head + Geometry::QHeads * (query + query_columns * row));
        shared.scratch.transform[tid] =
            local_row < LogicalRows && local_row / Heads < query_tokens
                ? __bfloat162float(q[index]) : 0.0f;
        kvarn_hadamard_shared<D>(shared.scratch.transform, tid);
        shared.query[local_row * D + gqa_prefill_swz(local_row, tid)] =
            __float2half(shared.scratch.transform[tid]);
    }
    if (tid < QueryRows) {
        shared.running_max[tid] = -CUDART_INF_F;
        shared.running_sum[tid] = 0.0f;
        shared.rescale[tid]     = 0.0f;
    }
    __syncthreads();

    float accumulator[TensorPv ? 1 : LogicalRows];
    float pv_acc[TensorPv ? PvNt : 1][4];
    if constexpr (TensorPv) {
#pragma unroll
        for (int n = 0; n < PvNt; ++n) {
#pragma unroll
            for (int c = 0; c < 4; ++c) { pv_acc[n][c] = 0.0f; }
        }
    } else {
#pragma unroll
        for (int local_row = 0; local_row < LogicalRows; ++local_row) { accumulator[local_row] = 0.0f; }
    }

    const int table_row       = table_rows[row];
    const std::int32_t* table = block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    for (int page = page_begin; page < page_end; ++page) {
        const std::uint8_t* record =
            records + kvarn_record_offset<Spec, Geometry::KVHeads>(table[page], kv_head);
        for (int d = tid; d < D; d += Threads) {
            shared.key_scale[d]     = kvarn_load_f16(record, Spec::KScaleOff, d);
            shared.key_zero[d]      = kvarn_load_f16(record, Spec::KZeroOff, d);
            shared.value_channel[d] = kvarn_load_f16(record, Spec::VChannelOff, d);
        }
        for (int key = tid; key < Spec::Group; key += Threads) {
            shared.key_token[key]   = kvarn_load_f16(record, Spec::KTokenOff, key);
            shared.value_scale[key] = kvarn_load_f16(record, Spec::VScaleOff, key);
            shared.value_zero[key]  = kvarn_load_f16(record, Spec::VZeroOff, key);
        }
        __syncthreads();

#pragma unroll
        for (int half = 0; half < Spec::Group / Bc; ++half) {
            kvarn_stage_key_tile(shared, record, half * Bc, tid);
            __syncthreads();

            float score[QKNt][4];
            if (warp < QueryWarps) {
                const int a_mat     = lane >> 3;
                const int a_rin     = lane & 7;
                const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
                const int b_rin     = lane & 7;
                const int b_koff    = ((lane >> 3) & 1) << 3;
                const int warp_row0 = warp * 16;

                const unsigned q_sbase = smem_addr(shared.query);
                const unsigned k_sbase = smem_addr(shared.scratch.key);
                const unsigned q_lane_base =
                    q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
                const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
                const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
                const unsigned k_lane_base = k_sbase + static_cast<unsigned>(b_rin * 512) +
                                             (static_cast<unsigned>(lane >> 4) << 12);
                const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
                const unsigned k_r  = static_cast<unsigned>(b_rin << 4);

#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
                }
                unsigned af[2][4];
                unsigned bf[2][QKNt][2];
                ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                            gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                        gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), 0u,
                                             k_as, k_r));
                }
#pragma unroll
                for (int k_step = 0; k_step < QKKs; ++k_step) {
                    const int cur = k_step & 1;
                    const int nxt = cur ^ 1;
                    if (k_step + 1 < QKKs) {
                        const unsigned ck = static_cast<unsigned>((k_step + 1) << 5);
                        ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                                    gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                            ldmatrix_x4(bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                                        bf[nxt][nt2 + 1][1],
                                        gqa_prefill_swz_addr(k_lane_base +
                                                                 static_cast<unsigned>(nt2 * 4096),
                                                             ck, k_as, k_r));
                        }
                    }
#pragma unroll
                    for (int nt = 0; nt < QKNt; ++nt) {
                        mma_f16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                                af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
                    }
                }
            }
            // The probability buffer aliases the staged key tile. All QK warps must finish
            // their ldmatrix loads before any lane starts overwriting it.
            __syncthreads();

            if (warp < QueryWarps) {
                const int gid        = lane >> 2;
                const int lid        = lane & 3;
                const int warp_row0  = warp * 16;
                const int local_row0 = warp_row0 + gid;
                const int local_row1 = local_row0 + 8;
                const bool valid0 = local_row0 < LogicalRows && local_row0 / Heads < query_tokens;
                const bool valid1 = local_row1 < LogicalRows && local_row1 / Heads < query_tokens;
                float bm0            = -CUDART_INF_F;
                float bm1            = -CUDART_INF_F;
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    score[nt][0] *= scale;
                    score[nt][1] *= scale;
                    score[nt][2] *= scale;
                    score[nt][3] *= scale;
                    if (!valid0) { score[nt][0] = score[nt][1] = -CUDART_INF_F; }
                    if (!valid1) { score[nt][2] = score[nt][3] = -CUDART_INF_F; }
                    bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                    bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
                }
                bm0 = warp_max<4>(bm0, FullMask);
                bm1 = warp_max<4>(bm1, FullMask);

                const float old_m0 = shared.running_max[local_row0];
                const float old_m1 = shared.running_max[local_row1];
                const float nm0    = fmaxf(old_m0, bm0);
                const float nm1    = fmaxf(old_m1, bm1);
                const float alpha0 =
                    old_m0 == -CUDART_INF_F ? 0.0f : exp2_approx((old_m0 - nm0) * Log2E);
                const float alpha1 =
                    old_m1 == -CUDART_INF_F ? 0.0f : exp2_approx((old_m1 - nm1) * Log2E);
                float bl0 = 0.0f;
                float bl1 = 0.0f;
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const float p00 = valid0 ? exp2_approx((score[nt][0] - nm0) * Log2E) : 0.0f;
                    const float p01 = valid0 ? exp2_approx((score[nt][1] - nm0) * Log2E) : 0.0f;
                    const float p10 = valid1 ? exp2_approx((score[nt][2] - nm1) * Log2E) : 0.0f;
                    const float p11 = valid1 ? exp2_approx((score[nt][3] - nm1) * Log2E) : 0.0f;
                    bl0 += p00 + p01;
                    bl1 += p10 + p11;
                    const int key                                            = nt * 8 + 2 * lid;
                    if constexpr (TensorPv) {
                        const int row0 = local_row0 * Bc;
                        const int row1 = local_row1 * Bc;
                        const int swz0 = (local_row0 & 3) << 3;
                        const int swz1 = (local_row1 & 3) << 3;
                        shared.scratch.tensor_pv[row0 + (key ^ swz0)] = __float2half(p00);
                        shared.scratch.tensor_pv[row0 + ((key + 1) ^ swz0)] = __float2half(p01);
                        shared.scratch.tensor_pv[row1 + (key ^ swz1)] = __float2half(p10);
                        shared.scratch.tensor_pv[row1 + ((key + 1) ^ swz1)] = __float2half(p11);
                    } else {
                        shared.scratch.probability[key * QueryRows + local_row0] = p00;
                        shared.scratch.probability[(key + 1) * QueryRows + local_row0] = p01;
                        shared.scratch.probability[key * QueryRows + local_row1]       = p10;
                        shared.scratch.probability[(key + 1) * QueryRows + local_row1] = p11;
                    }
                }
                bl0 = warp_sum<4>(bl0, FullMask);
                bl1 = warp_sum<4>(bl1, FullMask);
                if (lid == 0) {
                    shared.running_max[local_row0] = valid0 ? nm0 : -CUDART_INF_F;
                    shared.running_sum[local_row0] =
                        __fmaf_rn(shared.running_sum[local_row0], alpha0, bl0);
                    shared.rescale[local_row0]     = alpha0;
                    shared.running_max[local_row1] = valid1 ? nm1 : -CUDART_INF_F;
                    shared.running_sum[local_row1] =
                        __fmaf_rn(shared.running_sum[local_row1], alpha1, bl1);
                    shared.rescale[local_row1] = alpha1;
                }
            }
            __syncthreads();

            if constexpr (TensorPv) {
                // Eight warps cover 2 query groups x 4 dimension groups. Each warp owns
                // a 16x64 output tile (32 FP32 registers), not the whole 256-dimension row.
                const int pv_row = (warp / PvDimWarps) * 16;
                const int pv_dim = (warp % PvDimWarps) * (D / PvDimWarps);
                const int row0 = pv_row + (lane >> 2);
                const int row1 = row0 + 8;
                const float alpha0 = shared.rescale[row0];
                const float alpha1 = shared.rescale[row1];
#pragma unroll
                for (int n = 0; n < PvNt; ++n) {
                    pv_acc[n][0] *= alpha0;
                    pv_acc[n][1] *= alpha0;
                    pv_acc[n][2] *= alpha1;
                    pv_acc[n][3] *= alpha1;
                }
                __half* probability = shared.scratch.tensor_pv;
                __half* value_tile = probability + QueryRows * Bc;
                for (int key = 0; key < Bc; ++key) {
                    const int record_key = half * Bc + key;
                    const int code = kvarn_unpack<Spec::ValueBits>(
                        record + Spec::VPayloadOff, static_cast<std::int64_t>(record_key) * D + tid);
                    const float value = (code * shared.value_scale[record_key] +
                                         shared.value_zero[record_key]) * shared.value_channel[tid];
                    value_tile[key * D + gqa_prefill_swz(key, tid)] = __float2half(value);
                }
                __syncthreads();
#pragma unroll
                for (int k_step = 0; k_step < Bc; k_step += 16) {
                    unsigned af[4];
                    const int ar = pv_row + (lane & 7) + (((lane >> 3) & 1) << 3);
                    const int ac = k_step + ((lane >> 4) << 3);
                    ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(probability + ar * Bc + (ac ^ ((ar & 3) << 3))));
#pragma unroll
                    for (int n = 0; n < PvNt; ++n) {
                        unsigned bf0, bf1;
                        const int vr = k_step + (lane & 15);
                        const int vc = pv_dim + n * 8;
                        ldmatrix_x2_t(bf0, bf1,
                            smem_addr(value_tile + vr * D + gqa_prefill_swz(vr, vc)));
                        mma_f16(pv_acc[n][0], pv_acc[n][1], pv_acc[n][2], pv_acc[n][3],
                                af[0], af[1], af[2], af[3], bf0, bf1);
                    }
                }
            } else {
#pragma unroll
            for (int local_row = 0; local_row < LogicalRows; ++local_row) {
                accumulator[local_row] *= shared.rescale[local_row];
            }
            const float channel               = shared.value_channel[tid];
            const std::uint8_t* value_payload = record + Spec::VPayloadOff;
            for (int key = 0; key < Bc; ++key) {
                const int record_key = half * Bc + key;
                const int code       = kvarn_unpack<Spec::ValueBits>(
                    value_payload, static_cast<std::int64_t>(record_key) * D + tid);
                const float value =
                    (code * shared.value_scale[record_key] + shared.value_zero[record_key]) *
                    channel;
                const float4* weights =
                    reinterpret_cast<const float4*>(shared.scratch.probability + key * QueryRows);
#pragma unroll
                for (int quartet = 0; quartet < LogicalRows / 4; ++quartet) {
                    const float4 p = weights[quartet];
                    accumulator[quartet * 4 + 0] += p.x * value;
                    accumulator[quartet * 4 + 1] += p.y * value;
                    accumulator[quartet * 4 + 2] += p.z * value;
                    accumulator[quartet * 4 + 3] += p.w * value;
                }
            }
            }
            __syncthreads();
        }
    }

    for (int local_row = 0; local_row < LogicalRows; ++local_row) {
        const float sum               = shared.running_sum[local_row];
        if constexpr (TensorPv) {
            // Fragment owners differ from the final dimension owners. Finish every read of
            // the previous row before any warp overwrites the shared transform buffer.
            __syncthreads();
            const int row0 = (warp / PvDimWarps) * 16 + (lane >> 2);
            const int pv_dim = (warp % PvDimWarps) * (D / PvDimWarps);
            if (local_row == row0 || local_row == row0 + 8) {
#pragma unroll
                for (int n = 0; n < PvNt; ++n) {
                    const int dim = pv_dim + n * 8 + 2 * (lane & 3);
                    const float a = local_row == row0 ? pv_acc[n][0] : pv_acc[n][2];
                    const float b = local_row == row0 ? pv_acc[n][1] : pv_acc[n][3];
                    shared.scratch.transform[dim] = sum > 0.0f ? a / sum : 0.0f;
                    shared.scratch.transform[dim + 1] = sum > 0.0f ? b / sum : 0.0f;
                }
            }
        } else {
            shared.scratch.transform[tid] = sum > 0.0f ? accumulator[local_row] * __frcp_rn(sum) : 0.0f;
        }
        kvarn_hadamard_shared<D>(shared.scratch.transform, tid);
        const int query  = query_begin + local_row / Heads;
        const int q_head = head_begin + local_row % Heads;
        if (query < query_columns) {
            const std::int64_t output_index =
                static_cast<std::int64_t>(tid) +
                static_cast<std::int64_t>(D) *
                    (q_head + Geometry::QHeads * (query + query_columns * output_row));
            prefix[output_index] = __float2bfloat16(shared.scratch.transform[tid]);
            if (tid == 0) {
                const std::int64_t stat =
                    kvarn_prefill_row_index<Geometry>(q_head, query, query_columns, output_row);
                prefix_max[stat] = sum > 0.0f ? shared.running_max[local_row] : -CUDART_INF_F;
                prefix_sum[stat] = sum;
            }
        }
    }
}

template <typename Spec, typename Geometry, int QueryTile>
__launch_bounds__(Spec::HeadDim, (QueryTile * Geometry::GroupSize <= 24 ? 2 : 1)) __global__
    void kvarn_gqa_dense_prefill_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
        const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ positions,
        const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
        std::int32_t width, std::int32_t query_offset, std::int32_t query_columns,
        const __nv_bfloat16* __restrict__ stage_k, const __nv_bfloat16* __restrict__ stage_v,
        float scale, __nv_bfloat16* __restrict__ out, float* __restrict__ dense_max,
        float* __restrict__ dense_sum) {
    using Shared        = KvarnPrefillShared<Spec, Geometry, QueryTile>;
    constexpr int D     = Shared::D;
    constexpr int G     = Shared::G;
    constexpr int Rows  = Shared::Rows;
    constexpr int R     = Shared::R;
    constexpr int Warps = D / kWarpSize;

    __shared__ Shared shared;

    const int kv_head = static_cast<int>(blockIdx.x);
    const int tile    = static_cast<int>(blockIdx.y);
    const int row     = static_cast<int>(blockIdx.z);
    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & (kWarpSize - 1);
    const int warp    = tid >> 5;
    const int q_begin = tile * QueryTile;
    const int columns = kvarn_row_columns(valid_columns, row, width);

    if (tid < QueryTile) {
        const int query  = q_begin + tid;
        const int column = query_offset + query;
        shared.position[tid] =
            (query < query_columns && column < columns) ? positions[column + width * row] : -1;
    }
    if (tid < R) {
        shared.running_max[tid] = -CUDART_INF_F;
        shared.running_sum[tid] = 0.0f;
    }
    __syncthreads();

    int last_position = -1;
#pragma unroll
    for (int j = 0; j < QueryTile; ++j) { last_position = max(last_position, shared.position[j]); }

    for (int r = 0; r < R; ++r) {
        const int j      = r / Rows;
        const int query  = q_begin + j;
        const int q_head = kv_head * Rows + (r - j * Rows);
        const float value =
            shared.position[j] < 0
                ? 0.0f
                : __bfloat162float(
                      q[static_cast<std::int64_t>(tid) +
                        static_cast<std::int64_t>(D) *
                            (q_head + Geometry::QHeads * (query + query_columns * row))]);
        shared.query[tid * R + r] = __float2half(value);
    }
    __syncthreads();

    float accumulator[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { accumulator[r] = 0.0f; }

    const int first           = positions[width * row];
    const int table_row       = table_rows[row];
    const int record_page_end = first >= kKvarnSinkPages * G ? first / G : kKvarnSinkPages;
    const int last_page       = last_position >= 0 ? last_position / G : -1;
    for (int page = 0; page <= last_page; ++page) {
        if (page >= kKvarnSinkPages && page < record_page_end) { continue; }

        for (int key = warp; key < G; key += Warps) {
            const int absolute = page * G + key;
            if (absolute > last_position) {
                for (int r = lane; r < R; r += kWarpSize) {
                    shared.probability[key * R + r] = -CUDART_INF_F;
                }
                continue;
            }
            std::int64_t source;
            if (absolute >= first) {
                source =
                    kvarn_fresh_index<Spec, Geometry>(0, kv_head, (absolute - first) + width * row);
            } else {
                source = kvarn_stage_index<Spec, Geometry>(
                    0, kv_head, kvarn_stage_slot(absolute, G), table_row);
            }
            const __nv_bfloat16* keys = absolute >= first ? k + source : stage_k + source;
            float partial[R];
#pragma unroll
            for (int r = 0; r < R; ++r) { partial[r] = 0.0f; }
            for (int chunk = 0; chunk < D / kWarpSize; ++chunk) {
                const int d       = chunk * kWarpSize + lane;
                const float value = __bfloat162float(keys[d]);
#pragma unroll
                for (int octet = 0; octet < R / 8; ++octet) {
                    float tile_q[8];
                    kvarn_load_row_octet(shared.query + d * R + octet * 8, tile_q);
#pragma unroll
                    for (int i = 0; i < 8; ++i) { partial[octet * 8 + i] += tile_q[i] * value; }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float total = warp_sum(partial[r]);
                if (lane == 0) {
                    shared.probability[key * R + r] =
                        absolute > shared.position[r / Rows] ? -CUDART_INF_F : total * scale;
                }
            }
        }
        __syncthreads();

        kvarn_prefill_fold(shared, warp, lane);
#pragma unroll
        for (int r = 0; r < R; ++r) { accumulator[r] *= shared.rescale[r]; }

        for (int key = 0; key < G; ++key) {
            const int absolute = page * G + key;
            if (absolute > last_position) { break; }
            std::int64_t source;
            if (absolute >= first) {
                source = kvarn_fresh_index<Spec, Geometry>(tid, kv_head,
                                                           (absolute - first) + width * row);
            } else {
                source = kvarn_stage_index<Spec, Geometry>(
                    tid, kv_head, kvarn_stage_slot(absolute, G), table_row);
            }
            const float value = __bfloat162float(absolute >= first ? v[source] : stage_v[source]);
            const float4* weights = reinterpret_cast<const float4*>(shared.probability + key * R);
#pragma unroll
            for (int quartet = 0; quartet < R / 4; ++quartet) {
                const float4 weight = weights[quartet];
                accumulator[quartet * 4 + 0] += weight.x * value;
                accumulator[quartet * 4 + 1] += weight.y * value;
                accumulator[quartet * 4 + 2] += weight.z * value;
                accumulator[quartet * 4 + 3] += weight.w * value;
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < R; ++r) {
        const int j     = r / Rows;
        const int query = q_begin + j;
        if (query >= query_columns) { continue; }
        const int q_head  = kv_head * Rows + (r - j * Rows);
        const float sum   = shared.running_sum[r];
        const float value = shared.position[j] < 0 || !(sum > 0.0f) ? 0.0f : accumulator[r] / sum;
        out[static_cast<std::int64_t>(tid) +
            static_cast<std::int64_t>(D) *
                (q_head + Geometry::QHeads * (query + query_columns * row))] =
            __float2bfloat16(value);
    }
    if (tid < R) {
        const int j     = tid / Rows;
        const int query = q_begin + j;
        if (query < query_columns) {
            const int q_head = kv_head * Rows + (tid - j * Rows);
            const std::int64_t index =
                kvarn_prefill_row_index<Geometry>(q_head, query, query_columns, row);
            dense_max[index] = shared.running_max[tid];
            dense_sum[index] = shared.running_sum[tid];
        }
    }
}

template <typename Spec, typename Geometry>
__launch_bounds__(Spec::HeadDim) __global__
    void kvarn_merge_prefill_kernel(const __nv_bfloat16* __restrict__ prefix,
                                    const float* __restrict__ prefix_max,
                                    const float* __restrict__ prefix_sum,
                                    const float* __restrict__ dense_max,
                                    const float* __restrict__ dense_sum, std::int32_t query_columns,
                                    __nv_bfloat16* __restrict__ out) {
    constexpr int D         = Spec::HeadDim;
    constexpr float Log2E   = 1.4426950408889634074f;
    const int q_head        = static_cast<int>(blockIdx.x);
    const int query         = static_cast<int>(blockIdx.y);
    const int row           = static_cast<int>(blockIdx.z);
    const int d             = static_cast<int>(threadIdx.x);
    const std::int64_t stat = kvarn_prefill_row_index<Geometry>(q_head, query, query_columns, row);
    const std::int64_t element = static_cast<std::int64_t>(d) + stat * D;
    const float lp             = prefix_sum[stat];
    const float ld             = dense_sum[stat];
    if (!(lp > 0.0f)) { return; }
    if (!(ld > 0.0f)) {
        out[element] = prefix[element];
        return;
    }
    const float mp = prefix_max[stat];
    const float md = dense_max[stat];
    const float m  = fmaxf(mp, md);
    const float wp = lp * exp2_approx((mp - m) * Log2E);
    const float wd = ld * exp2_approx((md - m) * Log2E);
    const float numerator =
        __bfloat162float(prefix[element]) * wp + __bfloat162float(out[element]) * wd;
    out[element] = __float2bfloat16(numerator / (wp + wd));
}

// Prefix splits are independently normalized in the original (inverse-rotated) frame.
// Merge their LSE weights with the dense sink/tail/current result without re-reading KV.
template <typename Spec, typename Geometry>
__launch_bounds__(Spec::HeadDim) __global__
    void kvarn_merge_query_splits_kernel(const __nv_bfloat16* __restrict__ prefix,
                                         const float* __restrict__ prefix_max,
                                         const float* __restrict__ prefix_sum,
                                         const float* __restrict__ dense_max,
                                         const float* __restrict__ dense_sum,
                                         std::int32_t query_columns, std::int32_t splits,
                                         __nv_bfloat16* __restrict__ out) {
    constexpr int D       = Spec::HeadDim;
    constexpr float Log2E = 1.4426950408889634074f;
    __shared__ float weights[32];
    __shared__ float dense_weight;
    __shared__ float denominator;
    const int d = static_cast<int>(threadIdx.x);
    const std::int64_t stat = kvarn_prefill_row_index<Geometry>(
        static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y), query_columns,
        static_cast<int>(blockIdx.z));
    const std::int64_t stride =
        static_cast<std::int64_t>(Geometry::QHeads) * query_columns * gridDim.z;
    const std::int64_t element = stat * D + d;
    if (d < 32) {
        const std::int64_t partial_stat = stat + d * stride;
        const float l = d < splits ? prefix_sum[partial_stat] : 0.0f;
        const float m = l > 0.0f ? prefix_max[partial_stat] : -CUDART_INF_F;
        const float ld = dense_sum[stat];
        const float md = ld > 0.0f ? dense_max[stat] : -CUDART_INF_F;
        const float maximum = warp_max(fmaxf(m, md));
        const float w = l > 0.0f ? l * exp2_approx((m - maximum) * Log2E) : 0.0f;
        weights[d] = w;
        const float total = warp_sum(w);
        if (d == 0) {
            dense_weight = ld > 0.0f ? ld * exp2_approx((md - maximum) * Log2E) : 0.0f;
            denominator  = total + dense_weight;
        }
    }
    __syncthreads();
    float numerator = __bfloat162float(out[element]) * dense_weight;
    for (int split = 0; split < splits; ++split) {
        numerator += __bfloat162float(prefix[element + split * stride * D]) * weights[split];
    }
    out[element] = __float2bfloat16(denominator > 0.0f ? numerator / denominator : 0.0f);
}

} // namespace ninfer::ops
