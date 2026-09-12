#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <memory>

namespace ninfer::ops {

// SM75 execution resources for large Q4/Q5 LinearAdd, LinearSwiGLU and input projections.
// The represented inputs, complete mathematical formula, and numerical criteria are those
// of those Ops. FP16 staging and a single BF16 output conversion. Q5 accumulates in FP32;
// Q4 gate/up uses the qualified vendor FP16 compute/output mode. Projection parents use FP32.
// Columns carry a power-of-two scale, restored before the final epilogue, to preserve range.
// One context belongs to one Program. Construction initializes the vendor execution handle;
// every explicit device scratch buffer is provided by the caller's planned workspace.
// Calls on a context are serialized on the Program stream. No weight or request state persists.
class QuantizedPrefillContext {
public:
    QuantizedPrefillContext();
    ~QuantizedPrefillContext();
    QuantizedPrefillContext(const QuantizedPrefillContext&) = delete;
    QuantizedPrefillContext& operator=(const QuantizedPrefillContext&) = delete;
    static bool enabled();
    // A zero capacity means the context route is not selected for this profile/token count.
    static bool admits(QType type, int rows, int k, int tokens);
    static std::size_t workspace_capacity_bytes(QType type, int rows, int k, int max_tokens);
    void run(const Tensor& x, const Weight& weight, Tensor& output,
             WorkspaceArena& workspace, cudaStream_t stream);
    // Same four-projection contracts as GdnInputProj and AttnInputProj; small calls use their
    // existing routes. Large calls share one activation conversion and bounded GEMM scratch.
    static std::size_t gdn_workspace_capacity_bytes(int max_tokens);
    static std::size_t attention_workspace_capacity_bytes(int max_tokens);
    void gdn_input_proj(const Tensor& x, const Weight& qk, const Weight& vz,
        Tensor& qkv, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream);
    void attention_input_proj(const Tensor& x, const Weight& qk, const Weight& gv,
        Tensor& q, Tensor& gate, Tensor& k, Tensor& v, WorkspaceArena& workspace, cudaStream_t stream);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::ops
