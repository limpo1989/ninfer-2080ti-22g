#pragma once

#include "core/arena.h"
#include "core/kvarn.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

// The stage buffer holds each row's unquantized sink pages plus its still-filling tail page.
inline constexpr std::int32_t kKvarnStageTokens = kKvarnSinkTokens + kPagedKVPageSize;

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
 * One layer's KVarN-backed KV storage.
 *
 * `records` is the U8 record plane `[slot_bytes, group, kv_heads, physical_pages]` and
 * `block_tables` is contiguous I32 `[logical_pages, table_rows]`. `stage_k`/`stage_v` are
 * contiguous BF16 `[head_dim, kv_heads, kKvarnStageTokens, table_rows]`: the leading
 * `kKvarnSinkTokens` slots hold each row's permanently unquantized sink, and the remaining
 * `group` slots hold its still-filling tail page.
 */
struct KvarnBatchLayerView {
    Tensor records;
    Tensor block_tables;
    Tensor stage_k;
    Tensor stage_v;
    KvarnFormat format    = KvarnFormat::K4V2G64;
    std::int32_t head_dim = 0;
    std::int32_t kv_heads = 0;
};

/** Transient arena capacity kvarn_gqa_attention() requires at one exact call shape. */
[[nodiscard]] std::size_t kvarn_gqa_attention_workspace_capacity_bytes(std::int32_t q_heads,
                                                                       std::int32_t head_dim,
                                                                       std::int32_t query_columns,
                                                                       std::int32_t batch_size);

/**
 * K3: append K/V for B independent sequences into KVarN storage and compute causal grouped-query
 * attention, the KVarN counterpart of A1.
 *
 * For row b, query head h, kvh = floor(h / group), 0 <= j < Vb, p = positions[j,b], and that
 * row's populated history [0, p]:
 *
 *   score[x]       = scale * dot(q[:,h,j,b], K_cache[b][:,x,kvh]), 0 <= x <= p
 *   out[:,h,j,b]   = sum_x softmax_x(score)[x] * V_cache[b][:,x,kvh]
 *
 * where the history below `positions[0,b]` is the record and stage contents and the rest is this
 * call's own k/v. k/v are contiguous BF16 `[head_dim, kv_heads, W, B]`, positions is contiguous
 * I32 [W,B] whose valid prefix is sequential, kv_table_rows is contiguous I32 [B], and
 * valid_columns is contiguous I32 [B] or empty when every row has W valid columns.
 *
 * q/out are contiguous BF16 `[head_dim, q_heads, Wq, B]` with Wq <= W and address the LAST Wq of
 * the call's columns, so a caller that has to commit W tokens but only needs the final column's
 * attention states that by passing Wq = 1. Invalid columns receive exact BF16 zero. Empty q and
 * out request a commit-only call: the storage effect is unchanged and no attention is computed.
 * Every page is compressed exactly once, so a committed range must not be resubmitted.
 *
 * The Op commits every page the call completes to a record and leaves the remainder in the stage
 * buffer. Its launch shape depends only on W, B, and the head geometry, so it is capturable.
 *
 * The registered geometries are `[256, 24|4]` group 6 and `[256, 16|2]` group 8.
 */
void kvarn_gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                         const Tensor& positions, const Tensor& valid_columns,
                         const Tensor& kv_table_rows, float scale, KvarnBatchLayerView cache,
                         WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
