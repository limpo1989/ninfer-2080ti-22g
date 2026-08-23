#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_decode_gemv.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

// One warp per output row. Narrow matrices take four warps per CTA so the grid still covers the
// device; the wide ones take eight, which halves the activation re-read per row.
//
// kLastToken is where the exact-small-T MMA core takes the shape back, measured per geometry:
// a warp re-reads the whole activation tile for every row it owns, so the tall-K down projection
// gives the tile up earlier than the others.
template <class Geometry>
struct W8DecodeGemvShape {
    static constexpr int kRowsPerCta = Geometry::kOutputRows >= 16384 ? 8 : 4;
    static constexpr int kLastToken  = Geometry::kInputRows >= 17408 ? 5 : 8;
};

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int kRowsPerCta = W8DecodeGemvShape<Geometry>::kRowsPerCta;
    static_assert((Geometry::kOutputRows % kRowsPerCta) == 0);

    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / kRowsPerCta;
    w8_decode_gemv_kernel<Geometry::kInputRows, ActiveTokens, kRowsPerCta>
        <<<kBlocks, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<W8Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, 1 + static_cast<int>(Offsets)>...};
}

template <class Geometry>
void dispatch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static constexpr auto kLaunchers = make_launchers<Geometry>(
        std::make_index_sequence<W8DecodeGemvShape<Geometry>::kLastToken>{});
    if (weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows ||
        weight.padded_shape[1] != Geometry::kInputRows || x.ne[1] < 1 ||
        x.ne[1] > W8DecodeGemvShape<Geometry>::kLastToken) {
        throw std::invalid_argument("W8 decode GEMV: unsupported exact problem");
    }
    kLaunchers[static_cast<std::size_t>(x.ne[1] - 1)](x, weight, out, stream);
}

} // namespace

void launch_w8_vocabulary_decode_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    dispatch_exact<W8VocabularyProjectionGeometry>(x, weight, out, stream);
}

void launch_w8_mtp_input_decode_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    dispatch_exact<W8MtpInputProjectionGeometry>(x, weight, out, stream);
}

void launch_w8_mtp_attention_decode_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                                         cudaStream_t stream) {
    dispatch_exact<W8MtpAttentionProjectionGeometry>(x, weight, out, stream);
}

void launch_w8_mtp_attention_output_decode_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                                                cudaStream_t stream) {
    dispatch_exact<W8MtpAttentionOutputGeometry>(x, weight, out, stream);
}

void launch_w8_mtp_gate_up_decode_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    dispatch_exact<W8MtpGateUpProjectionGeometry>(x, weight, out, stream);
}

void launch_w8_mtp_down_decode_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream) {
    dispatch_exact<W8MtpDownProjectionGeometry>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
