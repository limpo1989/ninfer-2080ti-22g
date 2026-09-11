// ninfer::ops - sample wrapper: public api validation and dispatch.
#include "ninfer/ops/sampling.h"

#include "ops/common/sampling_workspace.h"
#include "ops/launcher/sampling.h" // detail::sample_batch_launch

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {

void apply_token_bitmask(Tensor& logits, std::int32_t token_domain, const SamplingConfig* configs,
                         std::int32_t columns_per_request, cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || !logits.is_contiguous() || logits.data == nullptr ||
        logits.ne[0] <= 0 || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "apply_token_bitmask: logits must be contiguous BF16 [physical_rows,columns,batch]");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "apply_token_bitmask: token_domain must be in [1, logits.ne[0]]");
    }
    const std::int64_t total_columns = static_cast<std::int64_t>(logits.ne[1]) * logits.ne[2];
    if (columns_per_request <= 0 || total_columns <= 0 ||
        total_columns % columns_per_request != 0) {
        throw std::invalid_argument(
            "apply_token_bitmask: columns do not divide into complete requests");
    }
    if (configs == nullptr) {
        throw std::invalid_argument("apply_token_bitmask: configs must be non-null");
    }
    detail::apply_token_bitmask_launch(logits, token_domain, configs, columns_per_request, stream);
}

std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain, std::int32_t min_lanes,
                                              std::int32_t max_lanes) {
    if (token_domain <= 0 || min_lanes <= 0 || max_lanes < min_lanes) {
        throw std::invalid_argument("sampling workspace: invalid profile or lane interval");
    }
    if (token_domain <= kSamplerTileItems || min_lanes > kSamplerMaxColumns) { return 0; }
    return detail::sampling_workspace_exact_bytes(token_domain,
                                                  std::min(max_lanes, kSamplerMaxColumns));
}

void sample(Tensor& logits, Tensor& out, std::int32_t token_domain, const SamplingConfig* configs,
            const Tensor& logical_positions, std::int32_t purpose, WorkspaceArena& workspace,
            cudaStream_t stream) {
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("sample: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("sample: out must be I32"); }
    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("sample: logits must be rank-2 [physical_rows,B]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("sample: out must be rank-1 [B]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("sample: physical rows must be positive");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("sample: token_domain must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) { throw std::invalid_argument("sample: out shape must be [B]"); }
    if (logical_positions.dtype != DType::I32 || logical_positions.ne[0] != logits.ne[1] ||
        logical_positions.ne[1] != 1 || logical_positions.ne[2] != 1 ||
        logical_positions.ne[3] != 1) {
        throw std::invalid_argument("sample: logical_positions must be I32 [logits.ne[1]]");
    }
    if (logits.ne[1] <= 0) { throw std::invalid_argument("sample: B must be positive"); }
    if (!logits.is_contiguous() || !out.is_contiguous() || !logical_positions.is_contiguous()) {
        throw std::invalid_argument("sample: logits/out/logical_positions must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr || logical_positions.data == nullptr) {
        throw std::invalid_argument("sample: tensor data must be non-null");
    }
    if (configs == nullptr) { throw std::invalid_argument("sample: configs must be non-null"); }

    auto scratch_scope = workspace.scope();
    const std::size_t bytes =
        sampling_workspace_capacity_bytes(token_domain, logits.ne[1], logits.ne[1]);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    apply_token_bitmask(logits, token_domain, configs, 1, stream);
    detail::sample_batch_launch(logits, out, token_domain, configs, logical_positions, purpose,
                                scratch, stream);
}

} // namespace ninfer::ops
