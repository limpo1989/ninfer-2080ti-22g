// ninfer::ops - KVarN record codec launcher.
#include "ops/launcher/kvarn.h"

#include "ops/kernel/kvarn_gqa.cuh"
#if defined(NINFER_SM75)
#    include "ops/kernel/kvarn_gqa_prefill_sm75.cuh"
#endif

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The record spec is a compile-time property of the kernel, so the launcher resolves the
// registered (format, head geometry) pairs explicitly rather than passing byte offsets.
template <typename Spec, int KVHeads>
void compress_one(const Tensor& k, const Tensor& v, const Tensor& page_ids, std::int32_t tiles,
                  Tensor& records, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(KVHeads), static_cast<unsigned>(tiles));
    kvarn_compress_kernel<Spec, KVHeads><<<grid, Spec::HeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(page_ids.data), static_cast<std::uint8_t*>(records.data),
        kKvarnSinkhornIterations);
}

template <typename Spec, int KVHeads>
void decompress_one(const Tensor& records, const Tensor& page_ids, std::int32_t tiles, Tensor& k,
                    Tensor& v, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(KVHeads), static_cast<unsigned>(tiles));
    kvarn_decompress_kernel<Spec, KVHeads><<<grid, Spec::HeadDim, 0, stream>>>(
        static_cast<const std::uint8_t*>(records.data),
        static_cast<const std::int32_t*>(page_ids.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(v.data));
}

// One CTA covers one (kv_head, split, column). Wide prefill calls already saturate the device
// through their columns, so the split count only has to rescue narrow decode calls; the column
// chunk keeps the split-local partial workspace bounded at large W.
constexpr std::int32_t kKvarnMaxSplits = 32;
// Below this many query columns a call is a decode or a speculative round: the Q-tiled prefill
// kernel would not have enough tiles to fill the device, so the split path serves it instead.
constexpr std::int32_t kKvarnPrefillMinColumns = 128;
constexpr std::int32_t kKvarnChunkColumns      = 64;
constexpr std::int32_t kKvarnBusyColumns       = 64;

#if defined(NINFER_SM75)
template <typename Spec, typename Geometry, int PrefixRows>
void gqa_tc_prefill(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                    const std::int32_t* positions, const std::int32_t* valid_columns,
                    const std::int32_t* table_rows, const std::int32_t* block_tables,
                    std::int32_t table_stride, std::int32_t width, std::int32_t query_offset,
                    std::int32_t query_columns, std::int32_t batch_size,
                    const __nv_bfloat16* stage_k, const __nv_bfloat16* stage_v,
                    const std::uint8_t* records, float scale, const Tensor& prefix_acc,
                    const Tensor& prefix_max, const Tensor& prefix_sum, const Tensor& dense_max,
                    const Tensor& dense_sum, __nv_bfloat16* out, cudaStream_t stream) {
    constexpr int DenseTile = KvarnPrefillTile<Geometry>::value;
    static_assert(sizeof(KvarnTcPrefixShared<Spec, Geometry, PrefixRows>) <= 48 * 1024);
    static_assert(sizeof(KvarnPrefillShared<Spec, Geometry, DenseTile>) <= 48 * 1024);
    auto* prefix          = static_cast<__nv_bfloat16*>(prefix_acc.data);
    auto* prefix_max_data = static_cast<float*>(prefix_max.data);
    auto* prefix_sum_data = static_cast<float*>(prefix_sum.data);
    auto* dense_max_data  = static_cast<float*>(dense_max.data);
    auto* dense_sum_data  = static_cast<float*>(dense_sum.data);

    auto* prefix_kernel        = kvarn_gqa_tc_prefix_kernel<Spec, Geometry, PrefixRows>;
    auto* dense_kernel         = kvarn_gqa_dense_prefill_kernel<Spec, Geometry, DenseTile>;
    static const bool carveout = [prefix_kernel, dense_kernel] {
        cudaFuncSetAttribute(prefix_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(dense_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return true;
    }();
    (void)carveout;

    const dim3 prefix_grid(static_cast<unsigned>((query_columns + PrefixRows - 1) / PrefixRows),
                           static_cast<unsigned>(Geometry::QHeads),
                           static_cast<unsigned>(batch_size));
    prefix_kernel<<<prefix_grid, KvarnTcPrefixShared<Spec, Geometry, PrefixRows>::Threads, 0,
                    stream>>>(q, prefix, positions, valid_columns, table_rows, block_tables,
                              table_stride, width, query_offset, query_columns, records, scale,
                              prefix_max_data, prefix_sum_data);

    const dim3 dense_grid(static_cast<unsigned>(Geometry::KVHeads),
                          static_cast<unsigned>((query_columns + DenseTile - 1) / DenseTile),
                          static_cast<unsigned>(batch_size));
    dense_kernel<<<dense_grid, Spec::HeadDim, 0, stream>>>(
        q, k, v, positions, valid_columns, table_rows, width, query_offset, query_columns, stage_k,
        stage_v, scale, out, dense_max_data, dense_sum_data);

    const dim3 merge_grid(static_cast<unsigned>(Geometry::QHeads),
                          static_cast<unsigned>(query_columns), static_cast<unsigned>(batch_size));
    kvarn_merge_prefill_kernel<Spec, Geometry><<<merge_grid, Spec::HeadDim, 0, stream>>>(
        prefix, prefix_max_data, prefix_sum_data, dense_max_data, dense_sum_data, query_columns,
        out);
}
#endif

template <typename Spec, typename Geometry>
void gqa_one(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
             const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
             const KvarnBatchLayerView& cache, std::int32_t width, std::int32_t query_columns,
             std::int32_t batch_size, const Tensor& partial_acc, const Tensor& partial_max,
             const Tensor& partial_sum, const Tensor& prefix_acc, const Tensor& prefix_max,
             const Tensor& prefix_sum, const Tensor& dense_max, const Tensor& dense_sum,
             Tensor& out, cudaStream_t stream) {
    const auto* positions_data = static_cast<const std::int32_t*>(positions.data);
    const auto* valid_data     = static_cast<const std::int32_t*>(valid_columns.data);
    const auto* rows_data      = static_cast<const std::int32_t*>(kv_table_rows.data);
    const auto* tables_data    = static_cast<const std::int32_t*>(cache.block_tables.data);
    const auto* q_data         = static_cast<const __nv_bfloat16*>(q.data);
    const auto* k_data         = static_cast<const __nv_bfloat16*>(k.data);
    const auto* v_data         = static_cast<const __nv_bfloat16*>(v.data);
    auto* stage_k              = static_cast<__nv_bfloat16*>(cache.stage_k.data);
    auto* stage_v              = static_cast<__nv_bfloat16*>(cache.stage_v.data);
    auto* record_data          = static_cast<std::uint8_t*>(cache.records.data);
    const std::int32_t stride  = cache.block_tables.ne[0];

    // Commit first: a page completing here is built from stage tokens plus this call's fresh
    // K/V, and the attention below must not see it as a record yet.
    const std::int32_t candidates = kvarn_compress_candidates(width);
    const dim3 compress_grid(static_cast<unsigned>(Geometry::KVHeads),
                             static_cast<unsigned>(candidates), static_cast<unsigned>(batch_size));
    kvarn_stage_compress_kernel<Spec, Geometry><<<compress_grid, Spec::HeadDim, 0, stream>>>(
        k_data, v_data, positions_data, valid_data, rows_data, tables_data, stride, width,
        static_cast<const __nv_bfloat16*>(cache.stage_k.data),
        static_cast<const __nv_bfloat16*>(cache.stage_v.data), record_data,
        kKvarnSinkhornIterations);

    const std::int32_t query_offset = width - query_columns;

#if defined(NINFER_SM75)
    if (query_columns >= kKvarnPrefillMinColumns) {
        // The compressed prefix is common to all queries in this call. Evaluate it with a
        // 32-row Tensor Core QK tile, evaluate the small dense region independently, then merge
        // their online-softmax states. The 48-row profile spills on SM75 and is slower.
        gqa_tc_prefill<Spec, Geometry, 32>(
            q_data, k_data, v_data, positions_data, valid_data, rows_data, tables_data, stride,
            width, query_offset, query_columns, batch_size,
            static_cast<const __nv_bfloat16*>(cache.stage_k.data),
            static_cast<const __nv_bfloat16*>(cache.stage_v.data), record_data, scale, prefix_acc,
            prefix_max, prefix_sum, dense_max, dense_sum, static_cast<__nv_bfloat16*>(out.data),
            stream);

        const dim3 append_grid(static_cast<unsigned>(Geometry::KVHeads),
                               static_cast<unsigned>(width), static_cast<unsigned>(batch_size));
        kvarn_stage_append_kernel<Spec, Geometry><<<append_grid, Spec::HeadDim, 0, stream>>>(
            k_data, v_data, positions_data, valid_data, rows_data, width, stage_k, stage_v);
        return;
    }
#else
    (void)prefix_acc;
    (void)prefix_max;
    (void)prefix_sum;
    (void)dense_max;
    (void)dense_sum;
#endif

    // Prefill: one CTA carries a tile of query columns through a single decode of each record,
    // and normalizes its own output because such a call is always one split.
    if (query_columns >= kKvarnPrefillMinColumns) {
        constexpr int Tile         = KvarnPrefillTile<Geometry>::value;
        auto* kernel               = kvarn_gqa_prefill_kernel<Spec, Geometry, Tile>;
        static const bool carveout = [kernel] {
            // Turing splits 64 KiB between L1 and shared per SM; ask for all of it so two CTAs
            // of this kernel's ~27 KiB footprint stay resident.
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            return true;
        }();
        (void)carveout;
        const dim3 grid(static_cast<unsigned>(Geometry::KVHeads),
                        static_cast<unsigned>((query_columns + Tile - 1) / Tile),
                        static_cast<unsigned>(batch_size));
        kernel<<<grid, Spec::HeadDim, 0, stream>>>(
            q_data, k_data, v_data, positions_data, valid_data, rows_data, tables_data, stride,
            width, query_offset, query_columns,
            static_cast<const __nv_bfloat16*>(cache.stage_k.data),
            static_cast<const __nv_bfloat16*>(cache.stage_v.data), record_data, scale,
            static_cast<__nv_bfloat16*>(out.data));

        const dim3 tiled_append_grid(static_cast<unsigned>(Geometry::KVHeads),
                                     static_cast<unsigned>(width),
                                     static_cast<unsigned>(batch_size));
        kvarn_stage_append_kernel<Spec, Geometry><<<tiled_append_grid, Spec::HeadDim, 0, stream>>>(
            k_data, v_data, positions_data, valid_data, rows_data, width, stage_k, stage_v);
        return;
    }

    const std::int32_t chunk = query_columns > 0 ? kvarn_attention_chunk_columns(query_columns) : 1;
    static const bool partial_carveout = [] {
        cudaFuncSetAttribute(kvarn_gqa_partial_kernel<Spec, Geometry>,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return true;
    }();
    (void)partial_carveout;
    // The split count and the partial stride are properties of the whole call, not of one chunk:
    // the wrapper sized the workspace from them, and a short trailing chunk must not re-derive
    // a larger split count and write past it.
    const std::int32_t splits     = kvarn_attention_splits(query_columns, batch_size, chunk);
    const std::int32_t flat_count = chunk * batch_size;
    for (std::int32_t begin = 0; begin < query_columns; begin += chunk) {
        const std::int32_t columns = std::min(chunk, query_columns - begin);
        const dim3 grid(static_cast<unsigned>(Geometry::KVHeads), static_cast<unsigned>(splits),
                        static_cast<unsigned>(columns * batch_size));
        kvarn_gqa_partial_kernel<Spec, Geometry><<<grid, Spec::HeadDim, 0, stream>>>(
            q_data, k_data, v_data, positions_data, valid_data, rows_data, tables_data, stride,
            width, query_offset, query_columns, columns, begin, flat_count,
            static_cast<const __nv_bfloat16*>(cache.stage_k.data),
            static_cast<const __nv_bfloat16*>(cache.stage_v.data), record_data, scale,
            static_cast<__nv_bfloat16*>(partial_acc.data), static_cast<float*>(partial_max.data),
            static_cast<float*>(partial_sum.data));

        const dim3 reduce_grid(static_cast<unsigned>(Geometry::QHeads),
                               static_cast<unsigned>(columns * batch_size));
        kvarn_gqa_reduce_kernel<Spec, Geometry><<<reduce_grid, Spec::HeadDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(partial_acc.data),
            static_cast<const float*>(partial_max.data),
            static_cast<const float*>(partial_sum.data), valid_data, width, query_offset,
            query_columns, columns, begin, splits, flat_count,
            static_cast<__nv_bfloat16*>(out.data));
    }

    // Last: the attention above still reads the tail slots this overwrites.
    const dim3 append_grid(static_cast<unsigned>(Geometry::KVHeads), static_cast<unsigned>(width),
                           static_cast<unsigned>(batch_size));
    kvarn_stage_append_kernel<Spec, Geometry><<<append_grid, Spec::HeadDim, 0, stream>>>(
        k_data, v_data, positions_data, valid_data, rows_data, width, stage_k, stage_v);
}

[[noreturn]] void unsupported() {
    throw std::invalid_argument("kvarn: unsupported (format, kv_heads) combination");
}

} // namespace

void kvarn_compress_launch(const Tensor& k, const Tensor& v, const Tensor& page_ids,
                           KvarnFormat format, std::int32_t kv_heads, std::int32_t tiles,
                           Tensor& records, cudaStream_t stream) {
    if (tiles == 0) { return; }
    switch (format) {
    case KvarnFormat::K4V2G64:
        if (kv_heads == 4) {
            return compress_one<KvarnK4V2, 4>(k, v, page_ids, tiles, records, stream);
        }
        if (kv_heads == 2) {
            return compress_one<KvarnK4V2, 2>(k, v, page_ids, tiles, records, stream);
        }
        if (kv_heads == 1) {
            return compress_one<KvarnK4V2, 1>(k, v, page_ids, tiles, records, stream);
        }
        break;
    case KvarnFormat::K4V4G64:
        if (kv_heads == 4) {
            return compress_one<KvarnK4V4, 4>(k, v, page_ids, tiles, records, stream);
        }
        if (kv_heads == 2) {
            return compress_one<KvarnK4V4, 2>(k, v, page_ids, tiles, records, stream);
        }
        if (kv_heads == 1) {
            return compress_one<KvarnK4V4, 1>(k, v, page_ids, tiles, records, stream);
        }
        break;
    }
    unsupported();
}

void kvarn_decompress_launch(const Tensor& records, const Tensor& page_ids, KvarnFormat format,
                             std::int32_t kv_heads, std::int32_t tiles, Tensor& k, Tensor& v,
                             cudaStream_t stream) {
    if (tiles == 0) { return; }
    switch (format) {
    case KvarnFormat::K4V2G64:
        if (kv_heads == 4) {
            return decompress_one<KvarnK4V2, 4>(records, page_ids, tiles, k, v, stream);
        }
        if (kv_heads == 2) {
            return decompress_one<KvarnK4V2, 2>(records, page_ids, tiles, k, v, stream);
        }
        if (kv_heads == 1) {
            return decompress_one<KvarnK4V2, 1>(records, page_ids, tiles, k, v, stream);
        }
        break;
    case KvarnFormat::K4V4G64:
        if (kv_heads == 4) {
            return decompress_one<KvarnK4V4, 4>(records, page_ids, tiles, k, v, stream);
        }
        if (kv_heads == 2) {
            return decompress_one<KvarnK4V4, 2>(records, page_ids, tiles, k, v, stream);
        }
        if (kv_heads == 1) {
            return decompress_one<KvarnK4V4, 1>(records, page_ids, tiles, k, v, stream);
        }
        break;
    }
    unsupported();
}

std::int32_t kvarn_attention_chunk_columns(std::int32_t width) {
    return width < kKvarnChunkColumns ? width : kKvarnChunkColumns;
}

std::int32_t kvarn_attention_splits(std::int32_t width, std::int32_t batch_size,
                                    std::int32_t chunk_columns) {
    (void)width;
    const std::int32_t columns = chunk_columns * batch_size;
    if (columns >= kKvarnBusyColumns) { return 1; }
    const std::int32_t splits = (kKvarnBusyColumns + columns - 1) / columns;
    return splits < kKvarnMaxSplits ? splits : kKvarnMaxSplits;
}

std::int32_t kvarn_compress_candidates(std::int32_t width) {
    // A call can complete at most one page per Group new tokens, plus the page it started inside.
    return width / kPagedKVPageSize + 2;
}

void kvarn_gqa_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                const Tensor& positions, const Tensor& valid_columns,
                                const Tensor& kv_table_rows, float scale,
                                const KvarnBatchLayerView& cache, std::int32_t q_heads,
                                std::int32_t width, std::int32_t query_columns,
                                std::int32_t batch_size, const Tensor& partial_acc,
                                const Tensor& partial_max, const Tensor& partial_sum,
                                const Tensor& prefix_acc, const Tensor& prefix_max,
                                const Tensor& prefix_sum, const Tensor& dense_max,
                                const Tensor& dense_sum, Tensor& out, cudaStream_t stream) {
    if (cache.kv_heads == 4 && q_heads == 24) {
        if (cache.format == KvarnFormat::K4V2G64) {
            return gqa_one<KvarnK4V2, Gqa27Geometry>(
                q, k, v, positions, valid_columns, kv_table_rows, scale, cache, width,
                query_columns, batch_size, partial_acc, partial_max, partial_sum, prefix_acc,
                prefix_max, prefix_sum, dense_max, dense_sum, out, stream);
        }
        return gqa_one<KvarnK4V4, Gqa27Geometry>(
            q, k, v, positions, valid_columns, kv_table_rows, scale, cache, width, query_columns,
            batch_size, partial_acc, partial_max, partial_sum, prefix_acc, prefix_max, prefix_sum,
            dense_max, dense_sum, out, stream);
    }
    if (cache.kv_heads == 2 && q_heads == 16) {
        if (cache.format == KvarnFormat::K4V2G64) {
            return gqa_one<KvarnK4V2, Gqa35Geometry>(
                q, k, v, positions, valid_columns, kv_table_rows, scale, cache, width,
                query_columns, batch_size, partial_acc, partial_max, partial_sum, prefix_acc,
                prefix_max, prefix_sum, dense_max, dense_sum, out, stream);
        }
        return gqa_one<KvarnK4V4, Gqa35Geometry>(
            q, k, v, positions, valid_columns, kv_table_rows, scale, cache, width, query_columns,
            batch_size, partial_acc, partial_max, partial_sum, prefix_acc, prefix_max, prefix_sum,
            dense_max, dense_sum, out, stream);
    }
    unsupported();
}

} // namespace ninfer::ops::detail
