#pragma once

// ninfer::ops - batched causal grouped-query attention over a KVarN-backed history.
//
// A sequence's KV lives in three places at once, because a tile cannot be quantized until all
// `Group` of its tokens exist:
//
//   positions [0, 128)              BF16 sink pages, never quantized (attention sinks carry
//                                   extreme score mass and dominate low-bit KV error);
//   positions [128, record_end)     KVarN records, one per complete page;
//   positions [record_end, frontier) BF16 tail page, still filling;
//   positions [frontier, ...)       this call's fresh K/V, not yet committed anywhere.
//
// The sink and tail share one per-row stage buffer of kKvarnStageTokens tokens. Scores over the
// record region are evaluated in the Hadamard-rotated frame and the other regions in the original
// frame; the split-local accumulator is rotated out of the record frame between the two passes,
// which is exact because the rotation is orthonormal and linear.

#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/kvarn_compress.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKvarnSinkPages  = 2;
inline constexpr int kKvarnStagePages = kKvarnSinkPages + 1;

// Stage token index of an absolute position: the sink pages are addressed directly and every
// later page reuses the single tail page.
__device__ __forceinline__ int kvarn_stage_slot(int position, int group) {
    const int sink = kKvarnSinkPages * group;
    return position < sink ? position : sink + (position % group);
}

template <typename Spec, typename Geometry>
__device__ __forceinline__ std::int64_t kvarn_stage_index(int channel, int kv_head, int slot,
                                                          int row) {
    return static_cast<std::int64_t>(channel) +
           static_cast<std::int64_t>(Spec::HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) *
                    (static_cast<std::int64_t>(slot) +
                     static_cast<std::int64_t>(kKvarnStagePages * Spec::Group) * row));
}

template <typename Spec, typename Geometry>
__device__ __forceinline__ std::int64_t kvarn_fresh_index(int channel, int kv_head, int column) {
    return static_cast<std::int64_t>(channel) +
           static_cast<std::int64_t>(Spec::HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * column);
}

// Number of valid columns of one batch row.
__device__ __forceinline__ int kvarn_row_columns(const std::int32_t* valid_columns, int row,
                                                 int width) {
    return valid_columns == nullptr ? width : valid_columns[row];
}

// ── stage append ────────────────────────────────────────────────────────────────────────────
//
// Only tokens the stage still owns after the call are written: the sink pages, and the trailing
// page the call leaves incomplete. Tokens of pages this call completed are already in a record,
// and writing them would race with the trailing page for the single tail slot they share.

template <typename Spec, typename Geometry>
__launch_bounds__(Spec::HeadDim) __global__
    void kvarn_stage_append_kernel(const __nv_bfloat16* __restrict__ k,
                                   const __nv_bfloat16* __restrict__ v,
                                   const std::int32_t* __restrict__ positions,
                                   const std::int32_t* __restrict__ valid_columns,
                                   const std::int32_t* __restrict__ table_rows,
                                   std::int32_t width, __nv_bfloat16* __restrict__ stage_k,
                                   __nv_bfloat16* __restrict__ stage_v) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int column  = static_cast<int>(blockIdx.y);
    const int row     = static_cast<int>(blockIdx.z);
    const int tid     = static_cast<int>(threadIdx.x);

    const int columns = kvarn_row_columns(valid_columns, row, width);
    if (column >= columns) { return; }

    const int position  = positions[column + width * row];
    const int sink_end  = kKvarnSinkPages * Spec::Group;
    const int tail_base = (positions[width * row] + columns) / Spec::Group * Spec::Group;
    if (position >= sink_end && position < tail_base) { return; }

    const int slot = kvarn_stage_slot(position, Spec::Group);
    const std::int64_t source =
        kvarn_fresh_index<Spec, Geometry>(tid, kv_head, column + width * row);
    const std::int64_t destination =
        kvarn_stage_index<Spec, Geometry>(tid, kv_head, slot, table_rows[row]);

    stage_k[destination] = k[source];
    stage_v[destination] = v[source];
}

// ── page completion ─────────────────────────────────────────────────────────────────────────
//
// One CTA per (kv_head, candidate page, row). The candidate is compressed only when this call
// is what completes it, so the launch shape is independent of the frontier and stays capturable.

template <typename Spec, typename Geometry>
__launch_bounds__(Spec::HeadDim) __global__
    void kvarn_stage_compress_kernel(const __nv_bfloat16* __restrict__ k,
                                     const __nv_bfloat16* __restrict__ v,
                                     const std::int32_t* __restrict__ positions,
                                     const std::int32_t* __restrict__ valid_columns,
                                     const std::int32_t* __restrict__ table_rows,
                                     const std::int32_t* __restrict__ block_tables,
                                     std::int32_t table_stride, std::int32_t width,
                                     const __nv_bfloat16* __restrict__ stage_k,
                                     const __nv_bfloat16* __restrict__ stage_v,
                                     std::uint8_t* __restrict__ records, int iterations) {
    constexpr int G = Spec::Group;

    __shared__ KvarnCompressShared<Spec> shared;

    const int kv_head   = static_cast<int>(blockIdx.x);
    const int candidate = static_cast<int>(blockIdx.y);
    const int row       = static_cast<int>(blockIdx.z);
    const int tid       = static_cast<int>(threadIdx.x);

    const int columns = kvarn_row_columns(valid_columns, row, width);
    if (columns <= 0) { return; }

    const int first    = positions[width * row];
    const int frontier = first + columns;
    const int page     = first / G + candidate;
    if (page < kKvarnSinkPages || (page + 1) * G > frontier) { return; }

    const int page_base   = page * G;
    const int stage_count = min(G, max(0, first - page_base));
    const int fresh_first = page_base + stage_count - first;

    const int table_row = table_rows[row];
    const std::int64_t stage_base =
        kvarn_stage_index<Spec, Geometry>(0, 0, 0, table_row);
    const std::int64_t fresh_base = kvarn_fresh_index<Spec, Geometry>(0, kv_head, width * row);
    const int stage_slot = kvarn_stage_slot(page_base, G);

    const KvarnSplitSource<Spec, Geometry::KVHeads> key_source{
        stage_k + stage_base + static_cast<std::int64_t>(Spec::HeadDim) * kv_head,
        k + fresh_base, stage_slot, stage_count, fresh_first};
    const KvarnSplitSource<Spec, Geometry::KVHeads> value_source{
        stage_v + stage_base + static_cast<std::int64_t>(Spec::HeadDim) * kv_head,
        v + fresh_base, stage_slot, stage_count, fresh_first};

    std::uint8_t* record =
        records + kvarn_record_offset<Spec, Geometry::KVHeads>(
                      block_tables[static_cast<std::int64_t>(table_row) * table_stride + page],
                      kv_head);
    kvarn_encode_record_device<Spec>(key_source, value_source, iterations, record, shared, tid);
}

// ── attention ───────────────────────────────────────────────────────────────────────────────

template <typename Spec, typename Geometry>
struct KvarnGqaShared {
    static constexpr int D     = Spec::HeadDim;
    static constexpr int G     = Spec::Group;
    static constexpr int Rows  = Geometry::GroupSize;
    static constexpr int Quads = D / G;

    float query[Rows * D];
    float rotated[Rows * D];
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

__device__ __forceinline__ std::int64_t kvarn_partial_stat_index(int q_head, int q_heads, int flat,
                                                                 int flat_count, int split) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(q_heads) *
               (static_cast<std::int64_t>(flat) +
                static_cast<std::int64_t>(flat_count) * static_cast<std::int64_t>(split));
}

__device__ __forceinline__ std::int64_t kvarn_partial_acc_index(int q_head, int q_heads, int channel,
                                                                int head_dim, int flat,
                                                                int flat_count, int split) {
    return static_cast<std::int64_t>(channel) +
           static_cast<std::int64_t>(head_dim) *
               kvarn_partial_stat_index(q_head, q_heads, flat, flat_count, split);
}

// Folds one page's Rows x Group scores into the running online-softmax state and leaves the
// per-key weights in shared.probability.
template <typename Spec, typename Geometry>
__device__ __forceinline__ void kvarn_fold_page(KvarnGqaShared<Spec, Geometry>& shared, int tid) {
    constexpr int Rows    = Geometry::GroupSize;
    constexpr int G       = Spec::Group;
    constexpr float Log2E = 1.4426950408889634074f;

    if (tid < Rows) {
        float page_max = -CUDART_INF_F;
        for (int key = 0; key < G; ++key) {
            page_max = fmaxf(page_max, shared.probability[tid * G + key]);
        }
        const float updated = fmaxf(shared.running_max[tid], page_max);
        float page_sum      = 0.0f;
        for (int key = 0; key < G; ++key) {
            const float weight = shared.probability[tid * G + key] == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2f((shared.probability[tid * G + key] - updated) * Log2E);
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
}

template <typename Spec, typename Geometry>
__launch_bounds__(Spec::HeadDim) __global__ void kvarn_gqa_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ block_tables, std::int32_t table_stride, std::int32_t width,
    std::int32_t query_offset, std::int32_t query_columns, std::int32_t chunk_columns,
    std::int32_t column_begin, std::int32_t flat_count, const __nv_bfloat16* __restrict__ stage_k,
    const __nv_bfloat16* __restrict__ stage_v, const std::uint8_t* __restrict__ records,
    float scale, __nv_bfloat16* __restrict__ partial_acc, float* __restrict__ partial_max,
    float* __restrict__ partial_sum) {
    constexpr int D     = Spec::HeadDim;
    constexpr int G     = Spec::Group;
    constexpr int Rows  = Geometry::GroupSize;
    constexpr int Quads = D / G;
    constexpr int Warps = D / kWarpSize;

    __shared__ KvarnGqaShared<Spec, Geometry> shared;

    const int kv_head = static_cast<int>(blockIdx.x);
    const int split   = static_cast<int>(blockIdx.y);
    const int splits  = static_cast<int>(gridDim.y);
    const int flat    = static_cast<int>(blockIdx.z);
    const int row     = flat / chunk_columns;
    const int local   = flat - row * chunk_columns;
    const int query   = column_begin + local;
    const int column  = query_offset + query;
    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & (kWarpSize - 1);
    const int warp    = tid >> 5;
    const int slot    = tid % G;
    const int quad    = tid / G;

    const int columns = kvarn_row_columns(valid_columns, row, width);
    const bool active = column < columns;

    // The reducer writes an inactive column's zeros itself, so it never reads these partials.
    if (!active) { return; }

    const int first     = positions[width * row];
    const int position  = positions[column + width * row];
    const int table_row = table_rows[row];
    const std::int32_t* table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;

    // Region boundaries. Records exist only for whole pages that are already committed.
    const int sink_end        = kKvarnSinkPages * G;
    const int record_page_end = first >= sink_end ? first / G : kKvarnSinkPages;
    const int record_end      = record_page_end * G;

    const int total_pages     = position / G + 1;
    const int pages_per_split = (total_pages + splits - 1) / splits;
    const int page_begin      = split * pages_per_split;
    const int page_end        = min(total_pages, page_begin + pages_per_split);

    for (int r = 0; r < Rows; ++r) {
        const int q_head = kv_head * Rows + r;
        const float value = __bfloat162float(
            q[static_cast<std::int64_t>(tid) +
              static_cast<std::int64_t>(D) *
                  (q_head + Geometry::QHeads * (query + query_columns * row))]);
        shared.query[r * D + tid]   = value;
        shared.rotated[r * D + tid] = value;
    }
    __syncthreads();
    for (int r = 0; r < Rows; ++r) { kvarn_hadamard_shared<D>(shared.rotated + r * D, tid); }

    if (tid < Rows) {
        shared.running_max[tid] = -CUDART_INF_F;
        shared.running_sum[tid] = 0.0f;
    }
    float accumulator[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) { accumulator[r] = 0.0f; }
    __syncthreads();

    // ── pass 1: record pages, evaluated in the rotated frame ────────────────────────────────
    const int record_begin = max(page_begin, kKvarnSinkPages);
    const int record_last  = min(page_end, record_page_end);
    for (int page = record_begin; page < record_last; ++page) {
        const std::uint8_t* record =
            records + kvarn_record_offset<Spec, Geometry::KVHeads>(table[page], kv_head);

        shared.key_scale[tid]     = kvarn_load_f16(record, Spec::KScaleOff, tid);
        shared.key_zero[tid]      = kvarn_load_f16(record, Spec::KZeroOff, tid);
        shared.value_channel[tid] = kvarn_load_f16(record, Spec::VChannelOff, tid);
        if (tid < G) {
            shared.key_token[tid]   = kvarn_load_f16(record, Spec::KTokenOff, tid);
            shared.value_scale[tid] = kvarn_load_f16(record, Spec::VScaleOff, tid);
            shared.value_zero[tid]  = kvarn_load_f16(record, Spec::VZeroOff, tid);
        }
        __syncthreads();

        const std::uint8_t* key_payload = record + Spec::KPayloadOff;
        const float token_scale         = shared.key_token[slot];
        float partial[Rows];
#pragma unroll
        for (int r = 0; r < Rows; ++r) { partial[r] = 0.0f; }
        for (int d = quad * G; d < (quad + 1) * G; ++d) {
            const int code =
                kvarn_unpack<Spec::KeyBits>(key_payload, static_cast<std::int64_t>(d) * G + slot);
            const float key = (code * shared.key_scale[d] + shared.key_zero[d]) * token_scale;
#pragma unroll
            for (int r = 0; r < Rows; ++r) { partial[r] += shared.rotated[r * D + d] * key; }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) { shared.score[(quad * Rows + r) * G + slot] = partial[r]; }
        __syncthreads();

        for (int index = tid; index < Rows * G; index += D) {
            const int r   = index / G;
            const int key = index - r * G;
            float total   = 0.0f;
#pragma unroll
            for (int qi = 0; qi < Quads; ++qi) { total += shared.score[(qi * Rows + r) * G + key]; }
            shared.probability[index] = total * scale;
        }
        __syncthreads();

        kvarn_fold_page<Spec, Geometry>(shared, tid);
#pragma unroll
        for (int r = 0; r < Rows; ++r) { accumulator[r] *= shared.rescale[r]; }

        const std::uint8_t* value_payload = record + Spec::VPayloadOff;
        const float channel               = shared.value_channel[tid];
        for (int key = 0; key < G; ++key) {
            const int code = kvarn_unpack<Spec::ValueBits>(
                value_payload, static_cast<std::int64_t>(key) * D + tid);
            const float value =
                (code * shared.value_scale[key] + shared.value_zero[key]) * channel;
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                accumulator[r] += shared.probability[r * G + key] * value;
            }
        }
        __syncthreads();
    }

    // Leave the record frame before the BF16 regions contribute.
    if (record_begin < record_last) {
        for (int r = 0; r < Rows; ++r) { shared.rotated[r * D + tid] = accumulator[r]; }
        __syncthreads();
        for (int r = 0; r < Rows; ++r) { kvarn_hadamard_shared<D>(shared.rotated + r * D, tid); }
        for (int r = 0; r < Rows; ++r) { accumulator[r] = shared.rotated[r * D + tid]; }
        __syncthreads();
    }

    // ── pass 2: sink, tail, and fresh pages, evaluated in the original frame ────────────────
    for (int page = page_begin; page < page_end; ++page) {
        if (page >= kKvarnSinkPages && page < record_page_end) { continue; }

        // Each warp owns Group/Warps keys of this page and reduces their scores itself.
        for (int key = warp; key < G; key += Warps) {
            const int absolute = page * G + key;
            if (absolute > position) {
                if (lane < Rows) { shared.probability[lane * G + key] = -CUDART_INF_F; }
                continue;
            }
            std::int64_t source = 0;
            if (absolute >= first) {
                source = kvarn_fresh_index<Spec, Geometry>(0, kv_head,
                                                           (absolute - first) + width * row);
            } else {
                source = kvarn_stage_index<Spec, Geometry>(
                    0, kv_head, kvarn_stage_slot(absolute, G), table_row);
            }
            const __nv_bfloat16* keys = absolute >= first ? k + source : stage_k + source;

            float partial[Rows];
#pragma unroll
            for (int r = 0; r < Rows; ++r) { partial[r] = 0.0f; }
            for (int chunk = 0; chunk < D / kWarpSize; ++chunk) {
                const int d       = chunk * kWarpSize + lane;
                const float value = __bfloat162float(keys[d]);
#pragma unroll
                for (int r = 0; r < Rows; ++r) { partial[r] += shared.query[r * D + d] * value; }
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const float total = warp_sum(partial[r]);
                if (lane == 0) { shared.probability[r * G + key] = total * scale; }
            }
        }
        __syncthreads();

        kvarn_fold_page<Spec, Geometry>(shared, tid);
#pragma unroll
        for (int r = 0; r < Rows; ++r) { accumulator[r] *= shared.rescale[r]; }

        for (int key = 0; key < G; ++key) {
            const int absolute = page * G + key;
            if (absolute > position) { break; }
            std::int64_t source = 0;
            if (absolute >= first) {
                source = kvarn_fresh_index<Spec, Geometry>(tid, kv_head,
                                                           (absolute - first) + width * row);
            } else {
                source = kvarn_stage_index<Spec, Geometry>(
                    tid, kv_head, kvarn_stage_slot(absolute, G), table_row);
            }
            const float value =
                __bfloat162float(absolute >= first ? v[source] : stage_v[source]);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                accumulator[r] += shared.probability[r * G + key] * value;
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < Rows; ++r) {
        const int q_head = kv_head * Rows + r;
        partial_acc[kvarn_partial_acc_index(q_head, Geometry::QHeads, tid, D, flat, flat_count,
                                            split)] = __float2bfloat16(accumulator[r]);
    }
    if (tid < Rows) {
        const int q_head = kv_head * Rows + tid;
        const std::int64_t index =
            kvarn_partial_stat_index(q_head, Geometry::QHeads, flat, flat_count, split);
        partial_max[index] = shared.running_max[tid];
        partial_sum[index] = shared.running_sum[tid];
    }
}


// ── Q-tiled prefill attention (SM75) ────────────────────────────────────────────────────────
//
// kvarn_gqa_partial_kernel maps one CTA to one query token, which is right for decode and wrong
// for prefill: every CTA re-reads and re-dequantizes the whole record history, so a W-column
// call decodes each record W times. The record region is identical for all of a call's query
// columns (records cover only pages already committed before the call), so a prefill CTA can
// carry a tile of QueryTile columns through one decode of each record.
//
// The tile is sized so QueryTile * Geometry::GroupSize == kKvarnPrefillRows for every registered
// geometry: that fixes the shared-memory footprint under 32 KiB, which is what lets two CTAs of
// 256 threads share a Turing SM's 64 KiB of shared memory.
//
// The call is one split by construction (a prefill call has query columns to spare for
// occupancy), so this kernel normalizes and writes `out` itself instead of going through the
// partial workspace and the reducer.

inline constexpr int kKvarnPrefillRows = 24;

// Query columns per prefill CTA, chosen so the accumulated row count is kKvarnPrefillRows.
template <typename Geometry>
struct KvarnPrefillTile {
    static constexpr int value = kKvarnPrefillRows / Geometry::GroupSize;
    static_assert(value * Geometry::GroupSize == kKvarnPrefillRows,
                  "the prefill row budget must divide into whole query columns");
};

template <typename Spec, typename Geometry, int QueryTile>
struct KvarnPrefillShared {
    static constexpr int D     = Spec::HeadDim;
    static constexpr int G     = Spec::Group;
    static constexpr int Rows  = Geometry::GroupSize;
    static constexpr int R     = QueryTile * Rows;
    static constexpr int Quads = D / G;
    static constexpr int Chunk = D / G; // rows folded per cross-quad reduction round

    static_assert(R % 8 == 0, "the row tile is read eight halves at a time");
    static_assert(R % 4 == 0, "the row tile is read four probabilities at a time");
    static_assert(Chunk * G == D, "one reduction round must cover the block exactly");

    // [channel][row], so a thread reads a whole row tile of one channel as two 128-bit loads.
    __align__(16) __half query[D * R];
    // [key][row], for the same reason on the value pass.
    __align__(16) float probability[G * R];
    // The score buffer and the rotation scratch never overlap in time.
    __align__(16) float work[Quads * Chunk * G > D ? Quads * Chunk * G : D];
    float key_scale[D];
    float key_zero[D];
    float value_channel[D];
    float key_token[G];
    float value_scale[G];
    float value_zero[G];
    float running_max[R];
    float running_sum[R];
    float rescale[R];
    int position[QueryTile];
};

// Eight consecutive halves of one channel's row tile, as floats.
__device__ __forceinline__ void kvarn_load_row_octet(const __half* base, float (&out)[8]) {
    const uint4 packed = *reinterpret_cast<const uint4*>(base);
    const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        out[2 * w + 0] =
            __half2float(__ushort_as_half(static_cast<unsigned short>(words[w] & 0xffffu)));
        out[2 * w + 1] =
            __half2float(__ushort_as_half(static_cast<unsigned short>(words[w] >> 16)));
    }
}

// Folds one page's R x Group scores into the running online-softmax state, leaving the per-key
// weights in shared.probability. One warp owns whole rows, so the fold costs a few shuffles
// instead of a serial scan by a single warp's worth of live threads.
template <typename Spec, typename Geometry, int QueryTile>
__device__ __forceinline__ void kvarn_prefill_fold(
    KvarnPrefillShared<Spec, Geometry, QueryTile>& shared, int warp, int lane) {
    using Shared          = KvarnPrefillShared<Spec, Geometry, QueryTile>;
    constexpr int R       = Shared::R;
    constexpr int G       = Shared::G;
    constexpr int Warps   = Shared::D / kWarpSize;
    constexpr float Log2E = 1.4426950408889634074f;

    for (int r = warp; r < R; r += Warps) {
        float page_max = -CUDART_INF_F;
        for (int key = lane; key < G; key += kWarpSize) {
            page_max = fmaxf(page_max, shared.probability[key * R + r]);
        }
        page_max            = warp_max(page_max);
        const float updated = fmaxf(shared.running_max[r], page_max);

        float page_sum = 0.0f;
        for (int key = lane; key < G; key += kWarpSize) {
            const float score  = shared.probability[key * R + r];
            const float weight = score == -CUDART_INF_F ? 0.0f : exp2f((score - updated) * Log2E);
            shared.probability[key * R + r] = weight;
            page_sum += weight;
        }
        page_sum = warp_sum(page_sum);

        if (lane == 0) {
            const float rescale = shared.running_max[r] == -CUDART_INF_F
                                      ? 0.0f
                                      : exp2f((shared.running_max[r] - updated) * Log2E);
            shared.running_sum[r] = shared.running_sum[r] * rescale + page_sum;
            shared.running_max[r] = updated;
            shared.rescale[r]     = rescale;
        }
    }
    __syncthreads();
}

template <typename Spec, typename Geometry, int QueryTile>
__launch_bounds__(Spec::HeadDim, 2) __global__ void kvarn_gqa_prefill_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ block_tables, std::int32_t table_stride, std::int32_t width,
    std::int32_t query_offset, std::int32_t query_columns,
    const __nv_bfloat16* __restrict__ stage_k, const __nv_bfloat16* __restrict__ stage_v,
    const std::uint8_t* __restrict__ records, float scale, __nv_bfloat16* __restrict__ out) {
    using Shared          = KvarnPrefillShared<Spec, Geometry, QueryTile>;
    constexpr int D       = Shared::D;
    constexpr int G       = Shared::G;
    constexpr int Rows    = Shared::Rows;
    constexpr int R       = Shared::R;
    constexpr int Quads   = Shared::Quads;
    constexpr int Chunk   = Shared::Chunk;
    constexpr int Warps   = D / kWarpSize;

    __shared__ Shared shared;

    const int kv_head = static_cast<int>(blockIdx.x);
    const int tile    = static_cast<int>(blockIdx.y);
    const int row     = static_cast<int>(blockIdx.z);
    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid & (kWarpSize - 1);
    const int warp    = tid >> 5;
    const int slot    = tid % G;
    const int quad    = tid / G;
    const int q_begin = tile * QueryTile;

    const int columns = kvarn_row_columns(valid_columns, row, width);

    // Positions of the tile's query columns; -1 marks a column this row does not own.
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

    const auto write_out = [&](const float* accumulator) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int j     = r / Rows;
            const int query = q_begin + j;
            if (query >= query_columns) { continue; }
            const int q_head = kv_head * Rows + (r - j * Rows);
            const float sum  = accumulator == nullptr ? 0.0f : shared.running_sum[r];
            const float value =
                (shared.position[j] < 0 || !(sum > 0.0f)) ? 0.0f : accumulator[r] / sum;
            out[static_cast<std::int64_t>(tid) +
                static_cast<std::int64_t>(D) *
                    (q_head + Geometry::QHeads * (query + query_columns * row))] =
                __float2bfloat16(value);
        }
    };

    if (columns <= 0 || last_position < 0) {
        write_out(nullptr);
        return;
    }

    const int first     = positions[width * row];
    const int table_row = table_rows[row];
    const std::int32_t* table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;

    const int sink_end        = kKvarnSinkPages * G;
    const int record_page_end = first >= sink_end ? first / G : kKvarnSinkPages;

    // Rotate the tile's queries into the record frame, one row at a time through the scratch.
    for (int r = 0; r < R; ++r) {
        const int j      = r / Rows;
        const int query  = q_begin + j;
        const int q_head = kv_head * Rows + (r - j * Rows);
        shared.work[tid] =
            shared.position[j] < 0
                ? 0.0f
                : __bfloat162float(
                      q[static_cast<std::int64_t>(tid) +
                        static_cast<std::int64_t>(D) *
                            (q_head + Geometry::QHeads * (query + query_columns * row))]);
        kvarn_hadamard_shared<D>(shared.work, tid);
        shared.query[tid * R + r] = __float2half(shared.work[tid]);
    }
    __syncthreads();

    float accumulator[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { accumulator[r] = 0.0f; }

    // ── pass 1: record pages, evaluated in the rotated frame ────────────────────────────────
    for (int page = kKvarnSinkPages; page < record_page_end; ++page) {
        const std::uint8_t* record =
            records + kvarn_record_offset<Spec, Geometry::KVHeads>(table[page], kv_head);

        shared.key_scale[tid]     = kvarn_load_f16(record, Spec::KScaleOff, tid);
        shared.key_zero[tid]      = kvarn_load_f16(record, Spec::KZeroOff, tid);
        shared.value_channel[tid] = kvarn_load_f16(record, Spec::VChannelOff, tid);
        if (tid < G) {
            shared.key_token[tid]   = kvarn_load_f16(record, Spec::KTokenOff, tid);
            shared.value_scale[tid] = kvarn_load_f16(record, Spec::VScaleOff, tid);
            shared.value_zero[tid]  = kvarn_load_f16(record, Spec::VZeroOff, tid);
        }
        __syncthreads();

        const std::uint8_t* key_payload = record + Spec::KPayloadOff;
        const float token_scale         = shared.key_token[slot];
        float partial[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { partial[r] = 0.0f; }
        for (int d = quad * G; d < (quad + 1) * G; ++d) {
            const int code =
                kvarn_unpack<Spec::KeyBits>(key_payload, static_cast<std::int64_t>(d) * G + slot);
            const float key = (code * shared.key_scale[d] + shared.key_zero[d]) * token_scale;
#pragma unroll
            for (int octet = 0; octet < R / 8; ++octet) {
                float tile_q[8];
                kvarn_load_row_octet(shared.query + d * R + octet * 8, tile_q);
#pragma unroll
                for (int i = 0; i < 8; ++i) { partial[octet * 8 + i] += tile_q[i] * key; }
            }
        }

        // Sum the four quads' partial scores, Chunk rows at a time.
#pragma unroll
        for (int base = 0; base < R; base += Chunk) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < Chunk; ++i) {
                shared.work[(quad * Chunk + i) * G + slot] = partial[base + i];
            }
            __syncthreads();
            for (int index = tid; index < Chunk * G; index += D) {
                const int i   = index / G;
                const int key = index - i * G;
                float total   = 0.0f;
#pragma unroll
                for (int qi = 0; qi < Quads; ++qi) {
                    total += shared.work[(qi * Chunk + i) * G + key];
                }
                shared.probability[key * R + base + i] = total * scale;
            }
        }
        __syncthreads();

        kvarn_prefill_fold(shared, warp, lane);
#pragma unroll
        for (int r = 0; r < R; ++r) { accumulator[r] *= shared.rescale[r]; }

        const std::uint8_t* value_payload = record + Spec::VPayloadOff;
        const float channel               = shared.value_channel[tid];
        for (int key = 0; key < G; ++key) {
            const int code = kvarn_unpack<Spec::ValueBits>(
                value_payload, static_cast<std::int64_t>(key) * D + tid);
            const float value =
                (code * shared.value_scale[key] + shared.value_zero[key]) * channel;
            const float4* weights =
                reinterpret_cast<const float4*>(shared.probability + key * R);
#pragma unroll
            for (int quartet = 0; quartet < R / 4; ++quartet) {
                const float4 w = weights[quartet];
                accumulator[quartet * 4 + 0] += w.x * value;
                accumulator[quartet * 4 + 1] += w.y * value;
                accumulator[quartet * 4 + 2] += w.z * value;
                accumulator[quartet * 4 + 3] += w.w * value;
            }
        }
        __syncthreads();
    }

    // Leave the record frame, then refill the query tile in the original frame for pass 2.
    if (record_page_end > kKvarnSinkPages) {
        for (int r = 0; r < R; ++r) {
            shared.work[tid] = accumulator[r];
            kvarn_hadamard_shared<D>(shared.work, tid);
            accumulator[r] = shared.work[tid];
        }
    }
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

    // ── pass 2: sink, tail, and fresh pages, evaluated in the original frame ────────────────
    const int last_page = last_position / G;
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
            std::int64_t source = 0;
            if (absolute >= first) {
                source = kvarn_fresh_index<Spec, Geometry>(0, kv_head,
                                                           (absolute - first) + width * row);
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
            std::int64_t source = 0;
            if (absolute >= first) {
                source = kvarn_fresh_index<Spec, Geometry>(tid, kv_head,
                                                           (absolute - first) + width * row);
            } else {
                source = kvarn_stage_index<Spec, Geometry>(
                    tid, kv_head, kvarn_stage_slot(absolute, G), table_row);
            }
            const float value = __bfloat162float(absolute >= first ? v[source] : stage_v[source]);
            const float4* weights =
                reinterpret_cast<const float4*>(shared.probability + key * R);
#pragma unroll
            for (int quartet = 0; quartet < R / 4; ++quartet) {
                const float4 w = weights[quartet];
                accumulator[quartet * 4 + 0] += w.x * value;
                accumulator[quartet * 4 + 1] += w.y * value;
                accumulator[quartet * 4 + 2] += w.z * value;
                accumulator[quartet * 4 + 3] += w.w * value;
            }
        }
        __syncthreads();
    }

    write_out(accumulator);
}

template <typename Spec, typename Geometry>
__launch_bounds__(Spec::HeadDim) __global__
    void kvarn_gqa_reduce_kernel(const __nv_bfloat16* __restrict__ partial_acc,
                                 const float* __restrict__ partial_max,
                                 const float* __restrict__ partial_sum,
                                 const std::int32_t* __restrict__ valid_columns,
                                 std::int32_t width, std::int32_t query_offset,
                                 std::int32_t query_columns, std::int32_t chunk_columns,
                                 std::int32_t column_begin, std::int32_t splits,
                                 std::int32_t flat_count, __nv_bfloat16* __restrict__ out) {
    constexpr int D       = Spec::HeadDim;
    constexpr float Log2E = 1.4426950408889634074f;

    const int q_head = static_cast<int>(blockIdx.x);
    const int flat   = static_cast<int>(blockIdx.y);
    const int row    = flat / chunk_columns;
    const int local  = flat - row * chunk_columns;
    const int query  = column_begin + local;
    const int column = query_offset + query;
    const int tid    = static_cast<int>(threadIdx.x);

    const bool active = column < kvarn_row_columns(valid_columns, row, width);
    const std::int64_t destination =
        static_cast<std::int64_t>(tid) +
        static_cast<std::int64_t>(D) *
            (q_head + Geometry::QHeads * (query + query_columns * row));

    if (!active) {
        out[destination] = __float2bfloat16(0.0f);
        return;
    }

    float total_max = -CUDART_INF_F;
    for (int split = 0; split < splits; ++split) {
        total_max = fmaxf(total_max, partial_max[kvarn_partial_stat_index(
                                         q_head, Geometry::QHeads, flat, flat_count, split)]);
    }
    if (total_max == -CUDART_INF_F) {
        out[destination] = __float2bfloat16(0.0f);
        return;
    }

    float total_sum = 0.0f;
    float numerator = 0.0f;
    for (int split = 0; split < splits; ++split) {
        const std::int64_t index =
            kvarn_partial_stat_index(q_head, Geometry::QHeads, flat, flat_count, split);
        const float split_sum = partial_sum[index];
        if (split_sum <= 0.0f) { continue; }
        const float weight = exp2f((partial_max[index] - total_max) * Log2E);
        total_sum += split_sum * weight;
        numerator += weight * __bfloat162float(partial_acc[kvarn_partial_acc_index(
                                  q_head, Geometry::QHeads, tid, D, flat, flat_count, split)]);
    }
    out[destination] = __float2bfloat16(total_sum > 0.0f ? numerator / total_sum : 0.0f);
}

} // namespace ninfer::ops
