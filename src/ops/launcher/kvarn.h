#pragma once

// ninfer::ops::detail - private launch prototypes for the KVarN record codec.

#include "core/kvarn.h"
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

std::int32_t kvarn_attention_splits(std::int32_t record_pages);

void kvarn_attention_launch(const Tensor& q, float scale, const Tensor& records,
                            const Tensor& block_table, KvarnFormat format, std::int32_t kv_heads,
                            std::int32_t q_heads, std::int32_t record_pages, std::int32_t tokens,
                            const Tensor& partial_acc, const Tensor& partial_max,
                            const Tensor& partial_sum, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
