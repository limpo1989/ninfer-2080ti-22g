#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Largest token count the fused gate/up decode GEMV is instantiated for. Above it the exact
// small-T MMA core wins: the GEMV's activation tile is T * 2 KiB of shared memory on top of the
// weight staging, and past six tokens that drops the CTA below two per SM and gives back more
// than the single pass over the matrix is worth (measured 683 us against the MMA core's 672 at
// T=8, against 363 versus 651 at T=4).
inline constexpr int kQ4SwiGluLastGemvPair = 6;

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
