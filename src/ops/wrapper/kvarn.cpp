// ninfer::ops - KVarN record codec wrapper: public contract validation and dispatch.
#include "ninfer/ops/kvarn.h"

#include "ops/launcher/kvarn.h" // detail::kvarn_compress_launch

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

struct RecordShape {
    KvarnRecordLayout layout;
    std::int32_t head_dim;
    std::int32_t tiles;
};

void require_bf16_staging(const char* op, const Tensor& k, const Tensor& v,
                          const KvarnRecordLayout& layout, std::int32_t kv_heads,
                          std::int32_t tiles) {
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": staged K/V must be BF16");
    }
    if (!k.is_contiguous() || !v.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": staged K/V must be contiguous");
    }
    const std::int32_t tokens = tiles * layout.group;
    for (const Tensor& t : {k, v}) {
        if (t.ne[0] != layout.head_dim || t.ne[1] != kv_heads || t.ne[2] != tokens ||
            t.ne[3] != 1) {
            throw std::invalid_argument(std::string(op) +
                                        ": staged K/V must be [head_dim, kv_heads, tiles * group]");
        }
    }
}

KvarnRecordLayout validate_plane(const char* op, const Tensor& records, KvarnFormat format,
                                 std::int32_t kv_heads, std::int32_t head_dim) {
    if (kv_heads != 1 && kv_heads != 2 && kv_heads != 4) {
        throw std::invalid_argument(std::string(op) + ": kv_heads must be 1, 2, or 4");
    }
    const KvarnRecordLayout layout = kvarn_record_layout(head_dim, format);
    if (records.dtype != DType::U8 || !records.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": record plane must be contiguous U8");
    }
    if (records.ne[0] != static_cast<std::int32_t>(layout.slot_bytes) ||
        records.ne[1] != layout.group || records.ne[2] != kv_heads) {
        throw std::invalid_argument(std::string(op) +
                                    ": record plane must be [slot_bytes, group, kv_heads, pages]");
    }
    return layout;
}

RecordShape validate(const char* op, const Tensor& records, const Tensor& page_ids,
                     KvarnFormat format, std::int32_t kv_heads, std::int32_t head_dim) {
    const KvarnRecordLayout layout = validate_plane(op, records, format, kv_heads, head_dim);
    if (page_ids.dtype != DType::I32 || !page_ids.is_contiguous() || page_ids.ne[0] <= 0 ||
        page_ids.ne[1] != 1 || page_ids.ne[2] != 1 || page_ids.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": page_ids must be contiguous I32 [tiles]");
    }
    return RecordShape{layout, head_dim, page_ids.ne[0]};
}

} // namespace

void kvarn_compress(const Tensor& k, const Tensor& v, const Tensor& page_ids, KvarnFormat format,
                    std::int32_t kv_heads, Tensor& records, cudaStream_t stream) {
    const RecordShape shape =
        validate("kvarn_compress", records, page_ids, format, kv_heads, k.ne[0]);
    require_bf16_staging("kvarn_compress", k, v, shape.layout, kv_heads, shape.tiles);
    detail::kvarn_compress_launch(k, v, page_ids, format, kv_heads, shape.tiles, records, stream);
}

void kvarn_decompress(const Tensor& records, const Tensor& page_ids, KvarnFormat format,
                      std::int32_t kv_heads, Tensor& k, Tensor& v, cudaStream_t stream) {
    const RecordShape shape =
        validate("kvarn_decompress", records, page_ids, format, kv_heads, k.ne[0]);
    require_bf16_staging("kvarn_decompress", k, v, shape.layout, kv_heads, shape.tiles);
    detail::kvarn_decompress_launch(records, page_ids, format, kv_heads, shape.tiles, k, v, stream);
}

std::size_t kvarn_gqa_attention_workspace_capacity_bytes(std::int32_t q_heads,
                                                         std::int32_t head_dim,
                                                         std::int32_t query_columns,
                                                         std::int32_t batch_size) {
    if (q_heads <= 0 || head_dim <= 0 || query_columns <= 0 || batch_size <= 0) {
        throw std::invalid_argument("kvarn_gqa_attention: workspace profile is invalid");
    }
    const std::int32_t chunk  = detail::kvarn_attention_chunk_columns(query_columns);
    const std::int32_t splits = detail::kvarn_attention_splits(query_columns, batch_size, chunk);
    const auto rows =
        static_cast<std::size_t>(q_heads) * static_cast<std::size_t>(chunk) * batch_size * splits;
    constexpr std::size_t kAlign = 256;
    const auto round_up = [](std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; };
    const std::size_t partial =
        round_up(rows * head_dim * sizeof(std::uint16_t)) + 2 * round_up(rows * sizeof(float));
#if defined(NINFER_SM75)
    if (query_columns < 128) { return partial; }
    const auto prefill_rows = static_cast<std::size_t>(q_heads) * query_columns * batch_size;
    const std::size_t tc    = round_up(prefill_rows * head_dim * sizeof(std::uint16_t)) +
                           4 * round_up(prefill_rows * sizeof(float));
    return std::max(partial, tc);
#else
    return partial;
#endif
}

namespace {

void require_i32_vector(const char* op, const Tensor& t, std::int32_t extent, const char* label) {
    if (t.dtype != DType::I32 || !t.is_contiguous() || t.ne[0] != extent || t.ne[1] != 1 ||
        t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": " + label + " must be contiguous I32 [" +
                                    std::to_string(extent) + "]");
    }
}

} // namespace

void kvarn_gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                         const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                         KvarnBatchLayerView cache, WorkspaceArena& workspace, Tensor& out,
                         cudaStream_t stream) {
    constexpr const char* op = "kvarn_gqa_attention";
    if (!std::isfinite(scale)) {
        throw std::invalid_argument(std::string(op) + ": scale must be finite");
    }
    const Tensor* const bf16_tensors[] = {&q, &k, &v, &out};
    const bool commit_only             = q.data == nullptr && out.data == nullptr;
    for (const Tensor* t : bf16_tensors) {
        if (t->data == nullptr && commit_only && (t == &q || t == &out)) { continue; }
        if (t->dtype != DType::BF16 || !t->is_contiguous()) {
            throw std::invalid_argument(std::string(op) + ": q/k/v/out must be contiguous BF16");
        }
    }
    const std::int32_t head_dim   = k.ne[0];
    const std::int32_t width      = k.ne[2];
    const std::int32_t batch_size = k.ne[3];
    // A commit-only call carries no query, so the registered geometry is resolved from the KV
    // head count alone.
    const std::int32_t q_heads       = commit_only ? (cache.kv_heads == 4   ? 24
                                                      : cache.kv_heads == 2 ? 16
                                                                            : 0)
                                                   : q.ne[1];
    const std::int32_t query_columns = commit_only ? 0 : q.ne[2];
    if (head_dim != cache.head_dim || width <= 0 || batch_size <= 0 || q_heads <= 0 ||
        query_columns > width) {
        throw std::invalid_argument(std::string(op) + ": q must be [head_dim, q_heads, Wq<=W, B]");
    }
    if (!commit_only) {
        if (q.ne[0] != head_dim || q.ne[3] != batch_size || query_columns <= 0) {
            throw std::invalid_argument(std::string(op) + ": q must be [head_dim, q_heads, Wq, B]");
        }
        for (int d = 0; d < 4; ++d) {
            if (out.ne[d] != q.ne[d]) {
                throw std::invalid_argument(std::string(op) + ": q/out shapes must match");
            }
        }
    }
    const Tensor* const kv_tensors[] = {&k, &v};
    for (const Tensor* t : kv_tensors) {
        if (t->ne[0] != head_dim || t->ne[1] != cache.kv_heads || t->ne[2] != width ||
            t->ne[3] != batch_size) {
            throw std::invalid_argument(std::string(op) +
                                        ": k/v must be [head_dim, kv_heads, W, B]");
        }
    }
    if (q_heads % cache.kv_heads != 0) {
        throw std::invalid_argument(std::string(op) + ": q_heads must be a multiple of kv_heads");
    }
    if (positions.dtype != DType::I32 || !positions.is_contiguous() || positions.ne[0] != width ||
        positions.ne[1] != batch_size || positions.ne[2] != 1 || positions.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": positions must be contiguous I32 [W, B]");
    }
    require_i32_vector(op, kv_table_rows, batch_size, "kv_table_rows");
    if (valid_columns.data != nullptr) {
        require_i32_vector(op, valid_columns, batch_size, "valid_columns");
    }

    const KvarnRecordLayout layout =
        validate_plane(op, cache.records, cache.format, cache.kv_heads, head_dim);
    if (cache.block_tables.dtype != DType::I32 || !cache.block_tables.is_contiguous() ||
        cache.block_tables.ne[0] <= 0) {
        throw std::invalid_argument(std::string(op) +
                                    ": block_tables must be contiguous I32 [logical_pages, rows]");
    }
    const std::int32_t stage_tokens     = kKvarnStageTokens;
    const Tensor* const stage_tensors[] = {&cache.stage_k, &cache.stage_v};
    for (const Tensor* t : stage_tensors) {
        if (t->dtype != DType::BF16 || !t->is_contiguous() || t->ne[0] != head_dim ||
            t->ne[1] != cache.kv_heads || t->ne[2] != stage_tokens ||
            t->ne[3] != cache.block_tables.ne[1]) {
            throw std::invalid_argument(std::string(op) +
                                        ": stage planes must be BF16 [head_dim, kv_heads, "
                                        "kKvarnStageTokens, table_rows]");
        }
    }
    if (static_cast<std::int32_t>(layout.group) != kPagedKVPageSize) {
        throw std::invalid_argument(std::string(op) + ": record group must equal the KV page size");
    }

    const WorkspaceArena::Scope scope = workspace.scope();
    Tensor partial_acc;
    Tensor partial_max;
    Tensor partial_sum;
    Tensor prefix_acc;
    Tensor prefix_max;
    Tensor prefix_sum;
    Tensor dense_max;
    Tensor dense_sum;
    if (!commit_only) {
#if defined(NINFER_SM75)
        if (query_columns >= 128) {
            prefix_acc =
                workspace.alloc(DType::BF16, {head_dim, q_heads, query_columns, batch_size});
            prefix_max = workspace.alloc(DType::FP32, {q_heads, query_columns, batch_size, 1});
            prefix_sum = workspace.alloc(DType::FP32, {q_heads, query_columns, batch_size, 1});
            dense_max  = workspace.alloc(DType::FP32, {q_heads, query_columns, batch_size, 1});
            dense_sum  = workspace.alloc(DType::FP32, {q_heads, query_columns, batch_size, 1});
        } else
#endif
        {
            const std::int32_t chunk = detail::kvarn_attention_chunk_columns(query_columns);
            const std::int32_t splits =
                detail::kvarn_attention_splits(query_columns, batch_size, chunk);
            partial_acc =
                workspace.alloc(DType::BF16, {head_dim, q_heads, chunk * batch_size, splits});
            partial_max = workspace.alloc(DType::FP32, {q_heads, chunk * batch_size, splits, 1});
            partial_sum = workspace.alloc(DType::FP32, {q_heads, chunk * batch_size, splits, 1});
        }
    }

    detail::kvarn_gqa_attention_launch(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                       cache, q_heads, width, query_columns, batch_size,
                                       partial_acc, partial_max, partial_sum, prefix_acc,
                                       prefix_max, prefix_sum, dense_max, dense_sum, out, stream);
}

} // namespace ninfer::ops
