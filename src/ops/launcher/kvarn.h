#pragma once

// ninfer::ops::detail - private launch prototypes for the KVarN record codec.

#include "ninfer/ops/kvarn.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void kvarn_compress_launch(const Tensor& k, const Tensor& v, const Tensor& page_ids,
                           KvarnFormat format, std::int32_t kv_heads, std::int32_t tiles,
                           Tensor& records, cudaStream_t stream);

void kvarn_decompress_launch(const Tensor& records, const Tensor& page_ids, KvarnFormat format,
                             std::int32_t kv_heads, std::int32_t tiles, Tensor& k, Tensor& v,
                             cudaStream_t stream);

std::int32_t kvarn_attention_splits(std::int32_t width, std::int32_t batch_size,
                                    std::int32_t chunk_columns);

std::int32_t kvarn_attention_chunk_columns(std::int32_t width);

std::int32_t kvarn_compress_candidates(std::int32_t width);

void kvarn_gqa_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                const Tensor& positions, const Tensor& valid_columns,
                                const Tensor& kv_table_rows, float scale,
                                const KvarnBatchLayerView& cache, std::int32_t q_heads,
                                std::int32_t width, std::int32_t query_columns,
                                std::int32_t batch_size, const Tensor& partial_acc,
                                const Tensor& partial_max, const Tensor& partial_sum,
                                const Tensor& prefix_acc, const Tensor& prefix_max,
                                const Tensor& prefix_sum, const Tensor& dense_max,
                                const Tensor& dense_sum, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
