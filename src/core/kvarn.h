#pragma once

// KVarN structured KV-cache records.
//
// One record holds `group` consecutive tokens of one KV head, in the Hadamard-rotated
// head-dimension frame, as low-bit asymmetric codes with variance-normalized (Sinkhorn)
// per-axis scales:
//
//   K tile is [head_dim, group] (channel-major); its RTN row scale/zero is absorbed into
//   the per-channel Sinkhorn axis, so   k_rot[d,t] = (code * k_s_col[d] + k_zp[d]) * k_s_row[t].
//   V tile is [group, head_dim] (token-major); its RTN row scale/zero is absorbed into
//   the per-token Sinkhorn axis, so     v_rot[t,d] = (code * v_s_row[t] + v_zp[t]) * v_s_col[d].
//
// The Hadamard is orthonormal and symmetric, so scores are unchanged by rotating Q with the
// same matrix, and the attention output is returned to the original frame by one more rotation.
//
// A record spans exactly one paged-KV page, so `group` is kPagedKVPageSize and the per-token
// slot `record_bytes / group` is the KV plane's leading extent.

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer {

enum class KvarnFormat : std::uint8_t {
    K4V2G64 = 0,
    K4V4G64 = 1,
};

struct KvarnPreset {
    KvarnFormat format;
    std::string_view name;
    std::int32_t key_bits;
    std::int32_t value_bits;
    std::int32_t group;
};

// Iterations of the alternating column/row standard-deviation normalization. The reference
// converges by ~4; 8 is indistinguishable from 16 and is the shipped point.
inline constexpr int kKvarnSinkhornIterations = 8;

// Leading tokens of every sequence that stay unquantized. Attention sinks carry extreme
// score mass, and quantizing them is the dominant accuracy loss of low-bit KV.
inline constexpr std::int32_t kKvarnSinkTokens = 128;

[[nodiscard]] std::span<const KvarnPreset> kvarn_presets() noexcept;
[[nodiscard]] const KvarnPreset* kvarn_preset_from_name(std::string_view name) noexcept;
[[nodiscard]] const KvarnPreset& kvarn_preset(KvarnFormat format);

struct KvarnRecordLayout {
    std::int32_t head_dim   = 0;
    std::int32_t group      = 0;
    std::int32_t key_bits   = 0;
    std::int32_t value_bits = 0;

    std::size_t k_payload_off   = 0;
    std::size_t k_payload_bytes = 0;
    std::size_t k_s_col_off     = 0; // [head_dim] FP16, absorbed per-channel scale
    std::size_t k_zp_off        = 0; // [head_dim] FP16, absorbed per-channel zero
    std::size_t k_s_row_off     = 0; // [group]    FP16, per-token Sinkhorn scale

    std::size_t v_payload_off   = 0;
    std::size_t v_payload_bytes = 0;
    std::size_t v_s_col_off     = 0; // [head_dim] FP16, per-channel Sinkhorn scale
    std::size_t v_s_row_off     = 0; // [group]    FP16, absorbed per-token scale
    std::size_t v_zp_off        = 0; // [group]    FP16, absorbed per-token zero

    std::size_t record_bytes = 0; // padded so that record_bytes % group == 0
    std::size_t slot_bytes   = 0; // record_bytes / group
};

// Throws when head_dim is not a supported power-of-two rotation width, or when the bit widths
// do not divide a byte evenly at the record's packing granularity.
[[nodiscard]] KvarnRecordLayout kvarn_record_layout(std::int32_t head_dim, KvarnFormat format);

[[nodiscard]] bool kvarn_head_dim_supported(std::int32_t head_dim) noexcept;

// ── host reference codec ─────────────────────────────────────────────────────────────────
//
// These are the independent mathematical oracle for the device kernels: they evaluate the
// complete KVarN formula in FP32/FP64 from the public BF16 inputs, and they own no staging
// or blocking choice made by any kernel.

// In-place orthonormal symmetric Walsh-Hadamard transform of one `n`-element row.
void kvarn_hadamard(std::span<float> row);

// Alternating log-domain column/row standard-deviation normalization with best-so-far
// selection. `tile` is row-major [rows, cols]; `balanced = tile / s_col / s_row`.
void kvarn_variance_normalize(std::span<const float> tile, std::int32_t rows, std::int32_t cols,
                              int iterations, std::vector<float>& balanced,
                              std::vector<float>& s_col, std::vector<float>& s_row);

// k/v are FP32 [head_dim, group] and [group, head_dim] in the ORIGINAL (unrotated) frame.
void kvarn_encode_record(std::span<const float> k, std::span<const float> v,
                         const KvarnRecordLayout& layout, int sinkhorn_iterations,
                         std::span<std::uint8_t> record);

// Decodes back to the ROTATED frame: k_rot [head_dim, group], v_rot [group, head_dim].
void kvarn_decode_record_rotated(std::span<const std::uint8_t> record,
                                 const KvarnRecordLayout& layout, std::span<float> k_rot,
                                 std::span<float> v_rot);

} // namespace ninfer
