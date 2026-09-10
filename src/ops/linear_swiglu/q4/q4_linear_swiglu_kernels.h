#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// One shared weight buffer leaves room for two CTAs even at eight tokens,
// covering the flattened verification batch of two MTP3 lanes.
inline constexpr int kQ4SwiGluLastGemvPair = 8;

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& w,
                                                          Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_small_t_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);

} // namespace ninfer::ops::detail
