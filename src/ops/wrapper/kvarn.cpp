// ninfer::ops - KVarN record codec wrapper: public contract validation and dispatch.
#include "ninfer/ops/kvarn.h"

#include "ops/launcher/kvarn.h" // detail::kvarn_compress_launch

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
        throw std::invalid_argument(
            std::string(op) + ": record plane must be [slot_bytes, group, kv_heads, pages]");
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

std::size_t kvarn_attention_workspace_capacity_bytes(std::int32_t q_heads, std::int32_t head_dim,
                                                     std::int32_t tokens,
                                                     std::int32_t record_pages) {
    if (q_heads <= 0 || head_dim <= 0 || tokens <= 0 || record_pages < 0) {
        throw std::invalid_argument("kvarn_attention: workspace profile is invalid");
    }
    const auto rows = static_cast<std::size_t>(q_heads) * tokens *
                      static_cast<std::size_t>(detail::kvarn_attention_splits(record_pages));
    // partial accumulator (BF16), plus the split-local max and sum (FP32), each 256-aligned.
    constexpr std::size_t kAlign = 256;
    const auto round_up = [](std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; };
    return round_up(rows * head_dim * sizeof(std::uint16_t)) + 2 * round_up(rows * sizeof(float));
}

void kvarn_attention_cached(const Tensor& q, float scale, const KvarnAttentionCache& cache,
                            WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "kvarn_attention_cached";
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16 || !q.is_contiguous() ||
        !out.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": q/out must be contiguous BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (q.ne[d] != out.ne[d]) {
            throw std::invalid_argument(std::string(op) + ": q/out shapes must match");
        }
    }
    if (q.ne[3] != 1 || q.ne[2] <= 0 || q.ne[1] <= 0) {
        throw std::invalid_argument(std::string(op) + ": q must be [head_dim, q_heads, tokens]");
    }
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t tokens   = q.ne[2];
    if (!std::isfinite(scale)) {
        throw std::invalid_argument(std::string(op) + ": scale must be finite");
    }
    if (cache.record_pages < 0) {
        throw std::invalid_argument(std::string(op) + ": record_pages must be nonnegative");
    }

    validate_plane(op, cache.records, cache.format, cache.kv_heads, head_dim);
    if (cache.block_table.dtype != DType::I32 || !cache.block_table.is_contiguous() ||
        cache.block_table.ne[0] < cache.record_pages) {
        throw std::invalid_argument(std::string(op) +
                                    ": block_table must be contiguous I32 covering record_pages");
    }
    if (q_heads % cache.kv_heads != 0) {
        throw std::invalid_argument(std::string(op) + ": q_heads must be a multiple of kv_heads");
    }

    const std::int32_t splits         = detail::kvarn_attention_splits(cache.record_pages);
    const WorkspaceArena::Scope scope = workspace.scope();
    Tensor partial_acc = workspace.alloc(DType::BF16, {head_dim, q_heads, tokens, splits});
    Tensor partial_max = workspace.alloc(DType::FP32, {q_heads, tokens, splits, 1});
    Tensor partial_sum = workspace.alloc(DType::FP32, {q_heads, tokens, splits, 1});

    detail::kvarn_attention_launch(q, scale, cache.records, cache.block_table, cache.format,
                                   cache.kv_heads, q_heads, cache.record_pages, tokens,
                                   partial_acc, partial_max, partial_sum, out, stream);
}

} // namespace ninfer::ops
