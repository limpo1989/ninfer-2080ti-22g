// ninfer::ops - KVarN record codec launcher.
#include "ops/launcher/kvarn.h"

#include "ops/kernel/kvarn_attention.cuh"
#include "ops/kernel/kvarn_compress.cuh"

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

// One CTA covers one (kv_head, split, token), so the split count sets occupancy. 32 keeps every
// SM of the smallest supported device busy at kv_heads=2 without splitting below one page.
constexpr std::int32_t kKvarnMaxSplits = 32;

template <typename Spec, typename Geometry>
void attention_one(const Tensor& q, float scale, const Tensor& records, const Tensor& block_table,
                   std::int32_t record_pages, std::int32_t tokens, const Tensor& partial_acc,
                   const Tensor& partial_max, const Tensor& partial_sum, Tensor& out,
                   std::int32_t splits, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(Geometry::KVHeads), static_cast<unsigned>(splits),
                    static_cast<unsigned>(tokens));
    kvarn_attention_partial_kernel<Spec, Geometry><<<grid, Spec::HeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const std::uint8_t*>(records.data),
        static_cast<const std::int32_t*>(block_table.data), record_pages, tokens, scale,
        static_cast<__nv_bfloat16*>(partial_acc.data), static_cast<float*>(partial_max.data),
        static_cast<float*>(partial_sum.data));

    const dim3 reduce_grid(static_cast<unsigned>(Geometry::QHeads),
                           static_cast<unsigned>(tokens));
    kvarn_attention_reduce_kernel<Spec, Geometry><<<reduce_grid, Spec::HeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(partial_acc.data),
        static_cast<const float*>(partial_max.data), static_cast<const float*>(partial_sum.data),
        splits, tokens, static_cast<__nv_bfloat16*>(out.data));
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
        if (kv_heads == 4) { return compress_one<KvarnK4V2, 4>(k, v, page_ids, tiles, records, stream); }
        if (kv_heads == 2) { return compress_one<KvarnK4V2, 2>(k, v, page_ids, tiles, records, stream); }
        if (kv_heads == 1) { return compress_one<KvarnK4V2, 1>(k, v, page_ids, tiles, records, stream); }
        break;
    case KvarnFormat::K4V4G64:
        if (kv_heads == 4) { return compress_one<KvarnK4V4, 4>(k, v, page_ids, tiles, records, stream); }
        if (kv_heads == 2) { return compress_one<KvarnK4V4, 2>(k, v, page_ids, tiles, records, stream); }
        if (kv_heads == 1) { return compress_one<KvarnK4V4, 1>(k, v, page_ids, tiles, records, stream); }
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
        if (kv_heads == 4) { return decompress_one<KvarnK4V2, 4>(records, page_ids, tiles, k, v, stream); }
        if (kv_heads == 2) { return decompress_one<KvarnK4V2, 2>(records, page_ids, tiles, k, v, stream); }
        if (kv_heads == 1) { return decompress_one<KvarnK4V2, 1>(records, page_ids, tiles, k, v, stream); }
        break;
    case KvarnFormat::K4V4G64:
        if (kv_heads == 4) { return decompress_one<KvarnK4V4, 4>(records, page_ids, tiles, k, v, stream); }
        if (kv_heads == 2) { return decompress_one<KvarnK4V4, 2>(records, page_ids, tiles, k, v, stream); }
        if (kv_heads == 1) { return decompress_one<KvarnK4V4, 1>(records, page_ids, tiles, k, v, stream); }
        break;
    }
    unsupported();
}


std::int32_t kvarn_attention_splits(std::int32_t record_pages) {
    if (record_pages <= 0) { return 1; }
    return record_pages < kKvarnMaxSplits ? record_pages : kKvarnMaxSplits;
}

void kvarn_attention_launch(const Tensor& q, float scale, const Tensor& records,
                            const Tensor& block_table, KvarnFormat format, std::int32_t kv_heads,
                            std::int32_t q_heads, std::int32_t record_pages, std::int32_t tokens,
                            const Tensor& partial_acc, const Tensor& partial_max,
                            const Tensor& partial_sum, Tensor& out, cudaStream_t stream) {
    const std::int32_t splits = kvarn_attention_splits(record_pages);
    if (kv_heads == 4 && q_heads == 24) {
        if (format == KvarnFormat::K4V2G64) {
            return attention_one<KvarnK4V2, Gqa27Geometry>(q, scale, records, block_table,
                                                           record_pages, tokens, partial_acc,
                                                           partial_max, partial_sum, out, splits,
                                                           stream);
        }
        return attention_one<KvarnK4V4, Gqa27Geometry>(q, scale, records, block_table,
                                                       record_pages, tokens, partial_acc,
                                                       partial_max, partial_sum, out, splits,
                                                       stream);
    }
    if (kv_heads == 2 && q_heads == 16) {
        if (format == KvarnFormat::K4V2G64) {
            return attention_one<KvarnK4V2, Gqa35Geometry>(q, scale, records, block_table,
                                                           record_pages, tokens, partial_acc,
                                                           partial_max, partial_sum, out, splits,
                                                           stream);
        }
        return attention_one<KvarnK4V4, Gqa35Geometry>(q, scale, records, block_table,
                                                       record_pages, tokens, partial_acc,
                                                       partial_max, partial_sum, out, splits,
                                                       stream);
    }
    unsupported();
}

} // namespace ninfer::ops::detail
