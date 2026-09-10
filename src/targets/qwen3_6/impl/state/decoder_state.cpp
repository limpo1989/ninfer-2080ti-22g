#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "core/device.h"

#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                              std::int32_t quant_group, std::optional<KvarnFormat> kvarn,
                              std::int32_t table_rows, std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const bool quantized = dtype == DType::I8;
    if (kvarn) {
        if (dtype != DType::U8 || quant_group != 0) {
            throw std::invalid_argument("KVarN KV cache must declare a U8 record plane");
        }
    } else if ((!quantized && (dtype != DType::BF16 || quant_group != 0)) ||
               (quantized && (quant_group != kKvQuantGroup || head_dim % quant_group != 0))) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    PagedKVPoolSpec pool_spec;
    pool_spec.page_group_count      = physical_page_groups;
    pool_spec.logical_page_capacity = logical_pages;
    pool_spec.table_rows            = table_rows;

    PagedKVCacheLayout layout;
    if (kvarn) {
        // One record plane per layer: the per-token slot is the record divided by the page, so a
        // record occupies exactly the page-and-head extent the shared addressing already gives it.
        const KvarnRecordLayout record = kvarn_record_layout(head_dim, *kvarn);
        if (static_cast<std::int32_t>(record.group) != kPagedKVPageSize) {
            throw std::invalid_argument("KVarN record group must equal the KV page size");
        }
        pool_spec.planes.reserve(layers);
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            pool_spec.planes.push_back(
                {DType::U8, static_cast<std::int32_t>(record.slot_bytes), kv_heads, 256});
        }
        layout.stage.reserve(static_cast<std::size_t>(layers) * 2);
        layout.rewrite_checkpoint_stage.reserve(static_cast<std::size_t>(layers) * 2);
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            for (const char* label : {"KVarN stage K", "KVarN stage V"}) {
                layout.stage.push_back(builder.add_tensor(
                    DType::BF16, {head_dim, kv_heads, ops::kKvarnStageTokens, table_rows}, 256,
                    label));
            }
            for (const char* label : {"KVarN rewrite checkpoint tail K",
                                      "KVarN rewrite checkpoint tail V"}) {
                layout.rewrite_checkpoint_stage.push_back(builder.add_tensor(
                    DType::BF16, {head_dim, kv_heads, kPagedKVPageSize, table_rows}, 256, label));
            }
        }
    } else {
        pool_spec.planes.reserve(static_cast<std::size_t>(layers) * (quantized ? 4ULL : 2ULL));
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            pool_spec.planes.push_back({dtype, head_dim, kv_heads, 256});
            pool_spec.planes.push_back({dtype, head_dim, kv_heads, 256});
            if (quantized) {
                pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
                pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
            }
        }
    }

    layout.pool        = plan_paged_kv_pool(builder, pool_spec);
    layout.layers      = layers;
    layout.max_context = capacity;
    layout.kv_heads    = kv_heads;
    layout.head_dim    = head_dim;
    layout.dtype       = dtype;
    layout.quant_group = quant_group;
    layout.kvarn       = kvarn;
    layout.table_rows  = table_rows;
    return layout;
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                spec.kvarn, spec.kv_table_rows, spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   spec.kvarn, spec.kv_table_rows, spec.mtp_physical_page_groups);
    }
    layout.linear_attention = plan_linear_attention_state_pool(builder, spec.linear_attention);
    return layout;
}

std::size_t PagedKVCacheLayout::payload_bytes() const noexcept {
    std::size_t total = pool.payload_bytes();
    for (const TensorRegion& region : stage) { total += region.region.bytes; }
    for (const TensorRegion& region : rewrite_checkpoint_stage) {
        total += region.region.bytes;
    }
    return total;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pool_(backing, layout.pool), layers_(layout.layers), max_context_(layout.max_context),
    kv_heads_(layout.kv_heads), head_dim_(layout.head_dim), dtype_(layout.dtype),
    quant_group_(layout.quant_group), kvarn_(layout.kvarn),
    plane_order_(layout.pool.spec.plane_order), table_rows_(layout.table_rows) {
    stage_.reserve(layout.stage.size());
    for (const TensorRegion& region : layout.stage) { stage_.push_back(region.bind(backing)); }
    rewrite_checkpoint_stage_.reserve(layout.rewrite_checkpoint_stage.size());
    for (const TensorRegion& region : layout.rewrite_checkpoint_stage) {
        rewrite_checkpoint_stage_.push_back(region.bind(backing));
    }
    const std::size_t expected_stages = kvarn_ ? static_cast<std::size_t>(layers_) * 2 : 0;
    if (stage_.size() != expected_stages || rewrite_checkpoint_stage_.size() != expected_stages) {
        throw std::logic_error("KVarN stage layout does not match the cache geometry");
    }
}

std::size_t PagedKVCache::snapshot_bytes(const PagedKVAllocation& allocation) const {
    if (!allocation.belongs_to(pool_)) throw std::invalid_argument("snapshot allocation belongs to another pool");
    std::size_t total = 0;
    for (std::size_t i = 0; i < pool_.plane_count(); ++i)
        total += pool_.plane(i).bytes() / pool_.page_group_count() * allocation.mapped_page_count();
    return total;
}

void PagedKVCache::transfer_snapshot(const PagedKVAllocation& allocation, std::uint8_t* host,
                                     std::size_t bytes, bool restore, cudaStream_t stream) const {
    if (bytes != snapshot_bytes(allocation)) throw std::invalid_argument("invalid KV snapshot extent");
    std::size_t offset = 0;
    const auto pages = allocation.page_ids();
    for (std::size_t i = 0; i < pool_.plane_count(); ++i) {
        const Tensor& plane = pool_.plane(i);
        const auto heads = plane_order_ == PagedKVPlaneOrder::HeadMajor ? plane.ne[3] : 1;
        for (int head = 0; head < heads; ++head) {
            const Tensor slab = plane_order_ == PagedKVPlaneOrder::HeadMajor ? plane.slice(3, head, 1) : plane;
            const int page_axis = plane_order_ == PagedKVPlaneOrder::HeadMajor ? 2 : 3;
            for (std::size_t p = 0; p < pages.size();) {
                std::size_t end = p + 1;
                while (end < pages.size() && pages[end] == pages[end - 1] + 1) ++end;
                const Tensor run = slab.slice(page_axis, pages[p], static_cast<std::int32_t>(end - p));
                if (!run.is_contiguous()) throw std::logic_error("noncontiguous snapshot page run");
                CUDA_CHECK(cudaMemcpyAsync(restore ? run.data : host + offset,
                                           restore ? host + offset : run.data, run.bytes(),
                                           restore ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream));
                offset += run.bytes(); p = end;
            }
        }
    }
}
void PagedKVCache::snapshot_to_host(const PagedKVAllocation& allocation,
                                    std::span<std::uint8_t> destination, cudaStream_t stream) const {
    transfer_snapshot(allocation, destination.data(), destination.size(), false, stream);
}
void PagedKVCache::restore_from_host(PagedKVAllocation& allocation,
                                    std::span<const std::uint8_t> source, cudaStream_t stream) const {
    transfer_snapshot(allocation, const_cast<std::uint8_t*>(source.data()), source.size(), true, stream);
}

ops::KvarnBatchLayerView PagedKVCache::kvarn_batch_layer_view(std::uint32_t layer) const {
    if (!kvarn_ || layer >= layers_ || 2 * layer + 1 >= stage_.size()) {
        throw std::out_of_range("KVarN KV layer is out of range");
    }
    return ops::KvarnBatchLayerView{
        .records      = pool_.plane(layer),
        .block_tables = pool_.block_tables(),
        .stage_k      = stage_[2 * layer],
        .stage_v      = stage_[2 * layer + 1],
        .format       = *kvarn_,
        .head_dim     = head_dim_,
        .kv_heads     = kv_heads_,
    };
}

PagedKVCache::KvarnCheckpointTailView
PagedKVCache::kvarn_rewrite_checkpoint_tail_view(std::uint32_t layer, std::int32_t row) const {
    if (!kvarn_ || layer >= layers_ || row < 0 || row >= table_rows_ ||
        2 * layer + 1 >= rewrite_checkpoint_stage_.size()) {
        throw std::out_of_range("KVarN rewrite checkpoint tail is out of range");
    }
    return KvarnCheckpointTailView{
        .k = rewrite_checkpoint_stage_[2 * layer].slice(3, row, 1),
        .v = rewrite_checkpoint_stage_[2 * layer + 1].slice(3, row, 1),
    };
}

void PagedKVCache::transfer_kvarn_rewrite_checkpoint(std::int32_t row, bool restore,
                                                     cudaStream_t stream) const {
    if (!kvarn_) { return; }
    if (row < 0 || row >= table_rows_) {
        throw std::out_of_range("KVarN rewrite checkpoint row is out of range");
    }
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        const ops::KvarnBatchLayerView current = kvarn_batch_layer_view(layer);
        const KvarnCheckpointTailView checkpoint =
            kvarn_rewrite_checkpoint_tail_view(layer, row);
        const auto transfer = [&](const Tensor& current_stage, const Tensor& checkpoint_stage) {
            const Tensor tail = current_stage.slice(3, row, 1).slice(
                2, kKvarnSinkTokens, kPagedKVPageSize);
            if (!tail.is_contiguous() || !checkpoint_stage.is_contiguous() ||
                tail.bytes() != checkpoint_stage.bytes()) {
                throw std::logic_error("KVarN rewrite checkpoint tail is not contiguous");
            }
            CUDA_CHECK(cudaMemcpyAsync(restore ? tail.data : checkpoint_stage.data,
                                       restore ? checkpoint_stage.data : tail.data, tail.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
        };
        transfer(current.stage_k, checkpoint.k);
        transfer(current.stage_v, checkpoint.v);
    }
}

void PagedKVCache::capture_kvarn_rewrite_checkpoint(std::int32_t row,
                                                    cudaStream_t stream) const {
    transfer_kvarn_rewrite_checkpoint(row, false, stream);
}

void PagedKVCache::restore_kvarn_rewrite_checkpoint(std::int32_t row,
                                                    cudaStream_t stream) const {
    transfer_kvarn_rewrite_checkpoint(row, true, stream);
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const PagedKVAllocation& allocation) const {
    if (!allocation.belongs_to(pool_)) {
        throw std::invalid_argument("Paged KV allocation belongs to another cache pool");
    }
    return PagedKVCacheView(*this, allocation.block_table());
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool quantized     = dtype_ == DType::I8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool quantized     = dtype_ == DType::I8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVBatchLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .block_tables  = pool_.block_tables(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), linear_attention(backing, layout.linear_attention) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
