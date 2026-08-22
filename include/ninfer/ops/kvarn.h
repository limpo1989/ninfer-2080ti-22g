#pragma once

#include "core/arena.h"
#include "core/kvarn.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

/**
 * K1: compress whole pages of staged BF16 K/V into KVarN records.
 *
 * `k` and `v` are contiguous BF16 `[head_dim, kv_heads, tiles * group]` in the same
 * channel-major order the GQA cache-append Op consumes. `page_ids` is contiguous I32 `[tiles]`
 * and names the physical record page each tile is written to; `records` is the U8 record plane
 * `[slot_bytes, group, kv_heads, physical_pages]`. Tile `i` consumes source tokens
 * `[i * group, (i + 1) * group)`.
 *
 * For each (tile, kv_head) the Op evaluates, in FP32:
 *
 *   1. the orthonormal symmetric Walsh-Hadamard rotation along head_dim, per token;
 *   2. alternating log-domain column/row standard-deviation normalization of the rotated tile,
 *      keeping the lowest-imbalance scales seen over `kKvarnSinkhornIterations` passes;
 *   3. asymmetric round-to-nearest over each row at the format's key/value bit width;
 *   4. absorption of each row's scale and zero into the matching normalization axis.
 *
 * The reconstruction identity is `k_rot[d,t] = (code * k_s_col[d] + k_zp[d]) * k_s_row[t]` and
 * `v_rot[t,d] = (code * v_s_row[t] + v_zp[t]) * v_s_col[d]`. Every byte of every addressed
 * record is overwritten, including the trailing slot padding. The Op reads no other record and
 * owns no frontier, allocation, or commit authority.
 *
 * The mathematical oracle is ninfer::kvarn_encode_record().
 */
void kvarn_compress(const Tensor& k, const Tensor& v, const Tensor& page_ids, KvarnFormat format,
                    std::int32_t kv_heads, Tensor& records, cudaStream_t stream);

/**
 * K2: expand KVarN records back to BF16 K/V in the ORIGINAL (unrotated) frame.
 *
 * Inverse addressing of K1: `records` and `page_ids` are as above, and `k`/`v` are contiguous
 * BF16 `[head_dim, kv_heads, tiles * group]`. This is the reference expansion used to qualify
 * the codec and to serve cache reads that are not fused into an attention kernel; the fused
 * decode path never materializes it.
 */
void kvarn_decompress(const Tensor& records, const Tensor& page_ids, KvarnFormat format,
                      std::int32_t kv_heads, Tensor& k, Tensor& v, cudaStream_t stream);

/**
 * The record-backed history one attention call reads.
 *
 * `records` is the U8 record plane `[slot_bytes, group, kv_heads, physical_pages]` and
 * `block_table` is contiguous I32 naming the physical page of each logical page. `record_pages`
 * leading logical pages are backed by records; the caller owns everything after them.
 */
struct KvarnAttentionCache {
    Tensor records;
    Tensor block_table;
    KvarnFormat format      = KvarnFormat::K4V2G64;
    std::int32_t kv_heads   = 0;
    std::int32_t record_pages = 0;
};

/** Transient arena capacity kvarn_attention_cached() requires for one call. */
[[nodiscard]] std::size_t kvarn_attention_workspace_capacity_bytes(std::int32_t q_heads,
                                                                   std::int32_t head_dim,
                                                                   std::int32_t tokens,
                                                                   std::int32_t record_pages);

/**
 * K3: grouped-query attention over a record-backed history.
 *
 * For query head h, kvh = floor(h / group), token j, and every key x in
 * `[0, record_pages * group)`:
 *
 *   score[x]     = scale * dot(q[:,h,j], K_cache[:,x,kvh])
 *   out[:,h,j]   = sum_x softmax_x(score)[x] * V_cache[:,x,kvh]
 *
 * where K_cache/V_cache are the records' reconstruction. Every record key is causally visible to
 * every query token, so this Op applies no mask; the caller composes the unquantized remainder of
 * the history separately. q and out are contiguous BF16 `[head_dim, q_heads, tokens]`.
 *
 * The registered geometries are `[256, 24|4]` group 6 and `[256, 16|2]` group 8. The Op mutates
 * no record and owns no frontier.
 */
void kvarn_attention_cached(const Tensor& q, float scale, const KvarnAttentionCache& cache,
                            WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
