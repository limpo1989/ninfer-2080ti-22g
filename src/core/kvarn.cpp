#include "core/kvarn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::array<KvarnPreset, 2> kPresets{{
    {KvarnFormat::K4V2G64, "kvarn_k4v2_g64", 4, 2, 64},
    {KvarnFormat::K4V4G64, "kvarn_k4v4_g64", 4, 4, 64},
}};

constexpr float kSinkhornStdMin = 1e-3F;
constexpr float kSinkhornStdMax = 1e3F;
constexpr float kSinkhornLogMin = -0.3F;
constexpr float kSinkhornLogMax = 10.0F;
constexpr float kRtnScaleMin    = 1e-10F;

std::uint32_t float_bits(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(std::uint32_t bits) noexcept {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint32_t round_shift_rne(std::uint32_t mantissa, int shift) noexcept {
    const std::uint32_t truncated = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const std::uint32_t half      = 1U << (shift - 1);
    if (remainder > half || (remainder == half && (truncated & 1U) != 0U)) { return truncated + 1U; }
    return truncated;
}

// IEEE binary16 round-to-nearest-even. The record's scale axes are the observable quantization
// boundary, so their rounding is part of the codec contract rather than a kernel staging choice.
std::uint16_t f32_to_f16(float value) noexcept {
    const std::uint32_t x    = float_bits(value);
    const std::uint32_t sign = (x >> 16) & 0x8000U;
    const std::uint32_t ax   = x & 0x7fffffffU;

    if (ax >= 0x7f800000U) {
        const std::uint32_t mantissa = ax & 0x007fffffU;
        return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa != 0U ? 0x0200U : 0U));
    }

    int exponent            = static_cast<int>((ax >> 23) & 0xffU) - 127 + 15;
    std::uint32_t mantissa  = ax & 0x007fffffU;
    if (exponent <= 0) {
        if (exponent < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000U;
        return static_cast<std::uint16_t>(sign | round_shift_rne(mantissa, 14 - exponent));
    }
    if (exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }

    std::uint32_t half_mantissa = round_shift_rne(mantissa, 13);
    if (half_mantissa == 0x0400U) {
        half_mantissa = 0;
        ++exponent;
        if (exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) |
                                      half_mantissa);
}

float f16_to_f32(std::uint16_t half) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(half) & 0x8000U) << 16;
    std::uint32_t exponent   = (static_cast<std::uint32_t>(half) >> 10) & 0x1fU;
    std::uint32_t mantissa   = static_cast<std::uint32_t>(half) & 0x03ffU;

    if (exponent == 0) {
        if (mantissa == 0) { return bits_float(sign); }
        int shifted = -14;
        while ((mantissa & 0x0400U) == 0U) {
            mantissa <<= 1;
            --shifted;
        }
        mantissa &= 0x03ffU;
        return bits_float(sign | (static_cast<std::uint32_t>(shifted + 127) << 23) |
                          (mantissa << 13));
    }
    if (exponent == 31) { return bits_float(sign | 0x7f800000U | (mantissa << 13)); }
    exponent = exponent - 15 + 127;
    return bits_float(sign | (exponent << 23) | (mantissa << 13));
}

void store_f16(std::span<std::uint8_t> record, std::size_t offset, std::int32_t index,
               float value) {
    const std::uint16_t bits = f32_to_f16(value);
    std::memcpy(record.data() + offset + static_cast<std::size_t>(index) * sizeof(bits), &bits,
                sizeof(bits));
}

float load_f16(std::span<const std::uint8_t> record, std::size_t offset, std::int32_t index) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, record.data() + offset + static_cast<std::size_t>(index) * sizeof(bits),
                sizeof(bits));
    return f16_to_f32(bits);
}

std::size_t packed_bytes(std::int64_t values, std::int32_t bits) noexcept {
    return static_cast<std::size_t>((values * bits + 7) / 8);
}

// Values-per-byte packing, low-order value first: value `i` occupies bits
// [(i % pack) * bits, ...) of byte `i / pack`.
void pack_codes(std::span<const std::uint8_t> codes, std::int32_t bits,
                std::span<std::uint8_t> dst) {
    const std::int32_t pack = 8 / bits;
    const std::uint8_t mask = static_cast<std::uint8_t>((1U << bits) - 1U);
    std::fill(dst.begin(), dst.end(), std::uint8_t{0});
    for (std::size_t i = 0; i < codes.size(); ++i) {
        const std::size_t byte  = i / static_cast<std::size_t>(pack);
        const int shift         = static_cast<int>(i % static_cast<std::size_t>(pack)) * bits;
        dst[byte] |= static_cast<std::uint8_t>((codes[i] & mask) << shift);
    }
}

std::uint8_t unpack_code(std::span<const std::uint8_t> src, std::int64_t index,
                         std::int32_t bits) noexcept {
    const std::int32_t pack = 8 / bits;
    const std::uint8_t mask = static_cast<std::uint8_t>((1U << bits) - 1U);
    const int shift         = static_cast<int>(index % pack) * bits;
    return static_cast<std::uint8_t>((src[static_cast<std::size_t>(index / pack)] >> shift) & mask);
}

// Sample standard deviation (Bessel-corrected), accumulated in FP64.
float sample_std(const float* values, std::int32_t count, std::int32_t stride) noexcept {
    double sum    = 0.0;
    double sum_sq = 0.0;
    for (std::int32_t i = 0; i < count; ++i) {
        const double value = values[static_cast<std::size_t>(i) * stride];
        sum += value;
        sum_sq += value * value;
    }
    const double mean     = sum / count;
    const double variance = std::max(0.0, (sum_sq - count * mean * mean) / (count - 1));
    return static_cast<float>(std::sqrt(variance));
}

float imbalance(const std::vector<float>& tile, std::int32_t rows, std::int32_t cols) noexcept {
    float col_min = std::numeric_limits<float>::infinity();
    float col_max = 0.0F;
    float row_min = std::numeric_limits<float>::infinity();
    float row_max = 0.0F;
    for (std::int32_t c = 0; c < cols; ++c) {
        const float value = sample_std(tile.data() + c, rows, cols);
        col_min           = std::min(col_min, value);
        col_max           = std::max(col_max, value);
    }
    for (std::int32_t r = 0; r < rows; ++r) {
        const float value = sample_std(tile.data() + static_cast<std::size_t>(r) * cols, cols, 1);
        row_min           = std::min(row_min, value);
        row_max           = std::max(row_max, value);
    }
    return col_max / std::max(col_min, 1e-8F) + row_max / std::max(row_min, 1e-8F);
}

struct RtnRow {
    float scale;
    float zero;
};

// Asymmetric round-to-nearest over one full row; returns the row's scale and zero point and
// writes the unsigned codes.
RtnRow rtn_row(const float* row, std::int32_t cols, std::int32_t bits, std::uint8_t* codes) {
    const float qmax = static_cast<float>((1U << bits) - 1U);
    const float lo   = *std::min_element(row, row + cols);
    const float hi   = *std::max_element(row, row + cols);
    const float step = std::max((hi - lo) / qmax, kRtnScaleMin);
    for (std::int32_t c = 0; c < cols; ++c) {
        const float code = std::round((row[c] - lo) / step);
        codes[c]         = static_cast<std::uint8_t>(std::clamp(code, 0.0F, qmax));
    }
    return {step, lo};
}

// Quantizes one variance-normalized tile: per-row RTN, the row scale/zero absorbed into the
// row Sinkhorn axis, and the column Sinkhorn axis stored untouched.
void quantize_tile(const std::vector<float>& balanced, const std::vector<float>& s_col,
                   const std::vector<float>& s_row, std::int32_t rows, std::int32_t cols,
                   std::int32_t bits, std::size_t scale_off, std::size_t zero_off,
                   std::size_t other_off, std::size_t payload_off, std::size_t payload_bytes,
                   std::span<std::uint8_t> record) {
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(rows) * cols);
    for (std::int32_t r = 0; r < rows; ++r) {
        const std::size_t base = static_cast<std::size_t>(r) * cols;
        const RtnRow rtn       = rtn_row(balanced.data() + base, cols, bits, codes.data() + base);
        store_f16(record, scale_off, r, s_row[static_cast<std::size_t>(r)] * rtn.scale);
        store_f16(record, zero_off, r, s_row[static_cast<std::size_t>(r)] * rtn.zero);
    }
    for (std::int32_t c = 0; c < cols; ++c) {
        store_f16(record, other_off, c, s_col[static_cast<std::size_t>(c)]);
    }
    pack_codes(codes, bits, record.subspan(payload_off, payload_bytes));
}

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

std::span<const KvarnPreset> kvarn_presets() noexcept { return kPresets; }

const KvarnPreset* kvarn_preset_from_name(std::string_view name) noexcept {
    for (const KvarnPreset& preset : kPresets) {
        if (preset.name == name) { return &preset; }
    }
    return nullptr;
}

const KvarnPreset& kvarn_preset(KvarnFormat format) {
    for (const KvarnPreset& preset : kPresets) {
        if (preset.format == format) { return preset; }
    }
    throw std::invalid_argument("Unknown KVarN format");
}

bool kvarn_head_dim_supported(std::int32_t head_dim) noexcept {
    return head_dim == 128 || head_dim == 256 || head_dim == 512;
}

KvarnRecordLayout kvarn_record_layout(std::int32_t head_dim, KvarnFormat format) {
    if (!kvarn_head_dim_supported(head_dim)) {
        throw std::invalid_argument("KVarN supports head_dim 128, 256, or 512; got " +
                                    std::to_string(head_dim));
    }
    const KvarnPreset& preset = kvarn_preset(format);

    KvarnRecordLayout layout;
    layout.head_dim   = head_dim;
    layout.group      = preset.group;
    layout.key_bits   = preset.key_bits;
    layout.value_bits = preset.value_bits;

    const std::int64_t elements = static_cast<std::int64_t>(head_dim) * preset.group;
    std::size_t offset          = 0;

    layout.k_payload_off   = offset;
    layout.k_payload_bytes = packed_bytes(elements, preset.key_bits);
    offset += layout.k_payload_bytes;
    layout.k_s_col_off = offset;
    offset += static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
    layout.k_zp_off = offset;
    offset += static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
    layout.k_s_row_off = offset;
    offset += static_cast<std::size_t>(preset.group) * sizeof(std::uint16_t);

    layout.v_payload_off   = offset;
    layout.v_payload_bytes = packed_bytes(elements, preset.value_bits);
    offset += layout.v_payload_bytes;
    layout.v_s_col_off = offset;
    offset += static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
    layout.v_s_row_off = offset;
    offset += static_cast<std::size_t>(preset.group) * sizeof(std::uint16_t);
    layout.v_zp_off = offset;
    offset += static_cast<std::size_t>(preset.group) * sizeof(std::uint16_t);

    // The per-token slot is the KV plane's leading extent, so the record must divide evenly by
    // the group. std::lcm(8, group) also keeps every FP16 axis 8-byte addressable.
    layout.record_bytes = align_up(offset, static_cast<std::size_t>(std::lcm(8, preset.group)));
    layout.slot_bytes   = layout.record_bytes / static_cast<std::size_t>(preset.group);
    return layout;
}

void kvarn_hadamard(std::span<float> row) {
    const std::size_t n = row.size();
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("KVarN Hadamard width must be a power of two");
    }
    for (std::size_t stride = 1; stride < n; stride *= 2) {
        for (std::size_t base = 0; base < n; base += 2 * stride) {
            for (std::size_t i = 0; i < stride; ++i) {
                const float a         = row[base + i];
                const float b         = row[base + stride + i];
                row[base + i]         = a + b;
                row[base + stride + i] = a - b;
            }
        }
    }
    const float norm = 1.0F / std::sqrt(static_cast<float>(n));
    for (float& value : row) { value *= norm; }
}

void kvarn_variance_normalize(std::span<const float> tile, std::int32_t rows, std::int32_t cols,
                              int iterations, std::vector<float>& balanced,
                              std::vector<float>& s_col, std::vector<float>& s_row) {
    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    std::vector<float> log_col(static_cast<std::size_t>(cols), 0.0F);
    std::vector<float> log_row(static_cast<std::size_t>(rows), 0.0F);
    std::vector<float> current(tile.begin(), tile.begin() + static_cast<std::ptrdiff_t>(count));

    s_col.assign(static_cast<std::size_t>(cols), 1.0F);
    s_row.assign(static_cast<std::size_t>(rows), 1.0F);
    float best = imbalance(current, rows, cols);

    const auto rebuild = [&]() {
        for (std::int32_t r = 0; r < rows; ++r) {
            const float row_scale = std::exp(log_row[static_cast<std::size_t>(r)]);
            for (std::int32_t c = 0; c < cols; ++c) {
                const std::size_t index = static_cast<std::size_t>(r) * cols + c;
                current[index] =
                    tile[index] / (std::exp(log_col[static_cast<std::size_t>(c)]) * row_scale);
            }
        }
    };

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::int32_t c = 0; c < cols; ++c) {
            const float deviation =
                std::clamp(sample_std(current.data() + c, rows, cols), kSinkhornStdMin,
                           kSinkhornStdMax);
            log_col[static_cast<std::size_t>(c)] =
                std::clamp(log_col[static_cast<std::size_t>(c)] + std::log(deviation),
                           kSinkhornLogMin, kSinkhornLogMax);
        }
        rebuild();

        for (std::int32_t r = 0; r < rows; ++r) {
            const float deviation = std::clamp(
                sample_std(current.data() + static_cast<std::size_t>(r) * cols, cols, 1),
                kSinkhornStdMin, kSinkhornStdMax);
            log_row[static_cast<std::size_t>(r)] =
                std::clamp(log_row[static_cast<std::size_t>(r)] + std::log(deviation),
                           kSinkhornLogMin, kSinkhornLogMax);
        }
        rebuild();

        const float current_imbalance = imbalance(current, rows, cols);
        if (current_imbalance <= best) {
            best = current_imbalance;
            for (std::int32_t c = 0; c < cols; ++c) {
                s_col[static_cast<std::size_t>(c)] = std::exp(log_col[static_cast<std::size_t>(c)]);
            }
            for (std::int32_t r = 0; r < rows; ++r) {
                s_row[static_cast<std::size_t>(r)] = std::exp(log_row[static_cast<std::size_t>(r)]);
            }
        }
    }

    balanced.resize(count);
    for (std::int32_t r = 0; r < rows; ++r) {
        for (std::int32_t c = 0; c < cols; ++c) {
            const std::size_t index = static_cast<std::size_t>(r) * cols + c;
            balanced[index] =
                tile[index] / (s_col[static_cast<std::size_t>(c)] * s_row[static_cast<std::size_t>(r)]);
        }
    }
}

void kvarn_encode_record(std::span<const float> k, std::span<const float> v,
                         const KvarnRecordLayout& layout, int sinkhorn_iterations,
                         std::span<std::uint8_t> record) {
    const std::int32_t d = layout.head_dim;
    const std::int32_t g = layout.group;
    const std::size_t n  = static_cast<std::size_t>(d) * g;
    if (k.size() != n || v.size() != n || record.size() != layout.record_bytes) {
        throw std::invalid_argument("KVarN record encode received mismatched extents");
    }
    std::fill(record.begin(), record.end(), std::uint8_t{0});

    // Rotate into the Hadamard frame: K along its channel axis (one column per token), V along
    // its channel axis (one row per token).
    std::vector<float> k_rot(n);
    std::vector<float> column(static_cast<std::size_t>(d));
    for (std::int32_t t = 0; t < g; ++t) {
        for (std::int32_t e = 0; e < d; ++e) {
            column[static_cast<std::size_t>(e)] = k[static_cast<std::size_t>(e) * g + t];
        }
        kvarn_hadamard(column);
        for (std::int32_t e = 0; e < d; ++e) {
            k_rot[static_cast<std::size_t>(e) * g + t] = column[static_cast<std::size_t>(e)];
        }
    }
    std::vector<float> v_rot(v.begin(), v.end());
    for (std::int32_t t = 0; t < g; ++t) {
        kvarn_hadamard(std::span<float>(v_rot).subspan(static_cast<std::size_t>(t) * d,
                                                       static_cast<std::size_t>(d)));
    }

    std::vector<float> balanced;
    std::vector<float> s_col;
    std::vector<float> s_row;

    kvarn_variance_normalize(k_rot, d, g, sinkhorn_iterations, balanced, s_col, s_row);
    quantize_tile(balanced, s_col, s_row, d, g, layout.key_bits, layout.k_s_col_off,
                  layout.k_zp_off, layout.k_s_row_off, layout.k_payload_off,
                  layout.k_payload_bytes, record);

    kvarn_variance_normalize(v_rot, g, d, sinkhorn_iterations, balanced, s_col, s_row);
    quantize_tile(balanced, s_col, s_row, g, d, layout.value_bits, layout.v_s_row_off,
                  layout.v_zp_off, layout.v_s_col_off, layout.v_payload_off,
                  layout.v_payload_bytes, record);
}

void kvarn_decode_record_rotated(std::span<const std::uint8_t> record,
                                 const KvarnRecordLayout& layout, std::span<float> k_rot,
                                 std::span<float> v_rot) {
    const std::int32_t d = layout.head_dim;
    const std::int32_t g = layout.group;
    const std::size_t n  = static_cast<std::size_t>(d) * g;
    if (record.size() != layout.record_bytes || k_rot.size() != n || v_rot.size() != n) {
        throw std::invalid_argument("KVarN record decode received mismatched extents");
    }

    const std::span<const std::uint8_t> k_payload =
        record.subspan(layout.k_payload_off, layout.k_payload_bytes);
    for (std::int32_t e = 0; e < d; ++e) {
        const float scale = load_f16(record, layout.k_s_col_off, e);
        const float zero  = load_f16(record, layout.k_zp_off, e);
        for (std::int32_t t = 0; t < g; ++t) {
            const std::int64_t index = static_cast<std::int64_t>(e) * g + t;
            const float code         = unpack_code(k_payload, index, layout.key_bits);
            k_rot[static_cast<std::size_t>(index)] =
                (code * scale + zero) * load_f16(record, layout.k_s_row_off, t);
        }
    }

    const std::span<const std::uint8_t> v_payload =
        record.subspan(layout.v_payload_off, layout.v_payload_bytes);
    for (std::int32_t t = 0; t < g; ++t) {
        const float scale = load_f16(record, layout.v_s_row_off, t);
        const float zero  = load_f16(record, layout.v_zp_off, t);
        for (std::int32_t e = 0; e < d; ++e) {
            const std::int64_t index = static_cast<std::int64_t>(t) * d + e;
            const float code         = unpack_code(v_payload, index, layout.value_bits);
            v_rot[static_cast<std::size_t>(index)] =
                (code * scale + zero) * load_f16(record, layout.v_s_col_off, e);
        }
    }
}

} // namespace ninfer
