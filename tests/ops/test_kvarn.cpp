// KVarN record codec conformance.
//
// The oracle is ninfer::kvarn_encode_record()/kvarn_decode_record_rotated(): an independent
// FP32/FP64 evaluation of the complete KVarN formula from the public BF16 staging values. The
// Sinkhorn normalization is an iterative floating-point search with a best-so-far selection, so
// the device kernel is not required to reproduce the oracle's code bytes; it is required to
// reconstruct the staged tile at least as accurately as the oracle does, and its own decode Op
// must invert its own records.

#include "ninfer/ops/kvarn.h"
#include "ops/op_tester.h"

#include <cmath>
#include <limits>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim = 256;

struct Case {
    KvarnFormat format;
    const char* name;
    std::int32_t kv_heads;
    std::int32_t tiles;
    double key_relative_l2;
    double value_relative_l2;
};

// Post-RoPE K carries a few very high-variance channels and V is broadly isotropic; this is the
// structure the rotation and the variance normalization exist to handle.
std::vector<float> staged_keys(std::int32_t kv_heads, std::int32_t tokens, std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(kHeadDim) * kv_heads * tokens);
    fill_uniform(values, seed, -1.0F, 1.0F);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::int32_t channel = static_cast<std::int32_t>(i % kHeadDim);
        values[i] *= channel % 37 == 0 ? 6.0F : 1.0F;
    }
    round_to_bf16(values);
    return values;
}

std::vector<float> staged_values(std::int32_t kv_heads, std::int32_t tokens, std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(kHeadDim) * kv_heads * tokens);
    fill_uniform(values, seed, -1.0F, 1.0F);
    round_to_bf16(values);
    return values;
}

// Extracts one (tile, head) K tile as [head_dim, group] and V tile as [group, head_dim] from the
// staged [head_dim, kv_heads, tokens] layout the Op consumes.
void extract_tile(const std::vector<float>& staged, std::int32_t kv_heads, std::int32_t head,
                  std::int32_t tile, std::int32_t group, bool channel_major,
                  std::vector<float>& out) {
    out.resize(static_cast<std::size_t>(kHeadDim) * group);
    for (std::int32_t t = 0; t < group; ++t) {
        const std::size_t base = static_cast<std::size_t>(kHeadDim) *
                                 (head + static_cast<std::size_t>(kv_heads) * (tile * group + t));
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            const std::size_t index = channel_major
                                          ? static_cast<std::size_t>(d) * group + t
                                          : static_cast<std::size_t>(t) * kHeadDim + d;
            out[index] = staged[base + d];
        }
    }
}

void inverse_rotate_rows(std::vector<float>& tile, std::int32_t rows, std::int32_t width) {
    for (std::int32_t r = 0; r < rows; ++r) {
        kvarn_hadamard(std::span<float>(tile).subspan(static_cast<std::size_t>(r) * width,
                                                      static_cast<std::size_t>(width)));
    }
}

// Rotates a [head_dim, group] channel-major tile back through its column axis.
void inverse_rotate_columns(std::vector<float>& tile, std::int32_t group) {
    std::vector<float> column(static_cast<std::size_t>(kHeadDim));
    for (std::int32_t t = 0; t < group; ++t) {
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            column[static_cast<std::size_t>(d)] = tile[static_cast<std::size_t>(d) * group + t];
        }
        kvarn_hadamard(column);
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            tile[static_cast<std::size_t>(d) * group + t] = column[static_cast<std::size_t>(d)];
        }
    }
}

double relative_l2(const std::vector<float>& got, const std::vector<float>& reference) {
    double error = 0.0;
    double norm  = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double delta = static_cast<double>(got[i]) - reference[i];
        error += delta * delta;
        norm += static_cast<double>(reference[i]) * reference[i];
    }
    return std::sqrt(error / norm);
}

int run_case(const Case& test_case) {
    const KvarnRecordLayout layout = kvarn_record_layout(kHeadDim, test_case.format);
    const std::int32_t group       = layout.group;
    const std::int32_t tokens      = test_case.tiles * group;
    const std::int32_t pages       = test_case.tiles + 3; // records land in non-adjacent pages

    const std::vector<float> keys   = staged_keys(test_case.kv_heads, tokens, 0x51a1);
    const std::vector<float> values = staged_values(test_case.kv_heads, tokens, 0x9e37);

    std::vector<int> page_ids(static_cast<std::size_t>(test_case.tiles));
    for (std::int32_t i = 0; i < test_case.tiles; ++i) { page_ids[static_cast<std::size_t>(i)] = pages - 1 - i; }

    DeviceBuffer device_keys   = to_device_bf16(keys);
    DeviceBuffer device_values = to_device_bf16(values);
    DeviceBuffer device_pages  = to_device_i32(page_ids);
    const std::size_t plane_bytes =
        layout.record_bytes * static_cast<std::size_t>(test_case.kv_heads) * pages;
    GuardedDeviceBuffer device_records(plane_bytes);
    device_records.fill(0x5a);

    Tensor k(device_keys.p, DType::BF16, {kHeadDim, test_case.kv_heads, tokens, 1});
    Tensor v(device_values.p, DType::BF16, {kHeadDim, test_case.kv_heads, tokens, 1});
    Tensor ids(device_pages.p, DType::I32, {test_case.tiles, 1, 1, 1});
    Tensor records(device_records.data(), DType::U8,
                   {static_cast<std::int32_t>(layout.slot_bytes), group, test_case.kv_heads,
                    pages});

    ops::kvarn_compress(k, v, ids, test_case.format, test_case.kv_heads, records, nullptr);
    cuda_check_last_launch("kvarn_compress");
    cuda_synchronize();

    std::vector<std::uint8_t> host_plane(plane_bytes);
    device_records.copy_to_host(host_plane.data(), plane_bytes);

    // Round-trip the device's own records through the device decode Op.
    DeviceBuffer decoded_keys(keys.size() * sizeof(std::uint16_t));
    DeviceBuffer decoded_values(values.size() * sizeof(std::uint16_t));
    Tensor k_out(decoded_keys.p, DType::BF16, {kHeadDim, test_case.kv_heads, tokens, 1});
    Tensor v_out(decoded_values.p, DType::BF16, {kHeadDim, test_case.kv_heads, tokens, 1});
    ops::kvarn_decompress(records, ids, test_case.format, test_case.kv_heads, k_out, v_out,
                          nullptr);
    cuda_check_last_launch("kvarn_decompress");
    cuda_synchronize();

    const std::vector<double> device_keys_back   = from_device_bf16(decoded_keys, keys.size());
    const std::vector<double> device_values_back = from_device_bf16(decoded_values, values.size());

    int failures = 0;
    failures += device_records.verify_guards(test_case.name);

    std::vector<float> tile;
    std::vector<float> oracle_k(static_cast<std::size_t>(kHeadDim) * group);
    std::vector<float> oracle_v(static_cast<std::size_t>(kHeadDim) * group);
    std::vector<std::uint8_t> oracle_record(layout.record_bytes);

    double worst_device_key   = 0.0;
    double worst_device_value = 0.0;
    double worst_oracle_key   = 0.0;
    double worst_oracle_value = 0.0;
    double worst_decode_gap   = 0.0;

    for (std::int32_t tile_index = 0; tile_index < test_case.tiles; ++tile_index) {
        for (std::int32_t head = 0; head < test_case.kv_heads; ++head) {
            std::vector<float> key_tile;
            std::vector<float> value_tile;
            extract_tile(keys, test_case.kv_heads, head, tile_index, group, true, key_tile);
            extract_tile(values, test_case.kv_heads, head, tile_index, group, false, value_tile);

            const std::size_t record_offset =
                layout.record_bytes *
                (static_cast<std::size_t>(head) +
                 static_cast<std::size_t>(test_case.kv_heads) *
                     static_cast<std::size_t>(page_ids[static_cast<std::size_t>(tile_index)]));
            const std::span<const std::uint8_t> device_record(host_plane.data() + record_offset,
                                                              layout.record_bytes);

            // Device record decoded by the host oracle, returned to the original frame.
            std::vector<float> device_k(oracle_k.size());
            std::vector<float> device_v(oracle_v.size());
            kvarn_decode_record_rotated(device_record, layout, device_k, device_v);
            inverse_rotate_columns(device_k, group);
            inverse_rotate_rows(device_v, group, kHeadDim);
            worst_device_key   = std::max(worst_device_key, relative_l2(device_k, key_tile));
            worst_device_value = std::max(worst_device_value, relative_l2(device_v, value_tile));

            // The oracle's own encode, for the accuracy floor the device must reach.
            kvarn_encode_record(key_tile, value_tile, layout, kKvarnSinkhornIterations,
                                oracle_record);
            kvarn_decode_record_rotated(oracle_record, layout, oracle_k, oracle_v);
            inverse_rotate_columns(oracle_k, group);
            inverse_rotate_rows(oracle_v, group, kHeadDim);
            worst_oracle_key   = std::max(worst_oracle_key, relative_l2(oracle_k, key_tile));
            worst_oracle_value = std::max(worst_oracle_value, relative_l2(oracle_v, value_tile));

            // The device decode Op must invert the device records it was given.
            for (std::int32_t t = 0; t < group; ++t) {
                const std::size_t base =
                    static_cast<std::size_t>(kHeadDim) *
                    (head + static_cast<std::size_t>(test_case.kv_heads) *
                                (tile_index * group + t));
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const double key_gap =
                        std::abs(device_keys_back[base + d] -
                                 device_k[static_cast<std::size_t>(d) * group + t]);
                    const double value_gap =
                        std::abs(device_values_back[base + d] -
                                 device_v[static_cast<std::size_t>(t) * kHeadDim + d]);
                    worst_decode_gap = std::max({worst_decode_gap, key_gap, value_gap});
                }
            }
        }
    }

    std::printf("%-16s heads=%d tiles=%d slot=%zuB/token/head (bf16 %dB) "
                "K rel_l2 device=%.4f oracle=%.4f | V rel_l2 device=%.4f oracle=%.4f | "
                "decode gap=%.4f\n",
                test_case.name, test_case.kv_heads, test_case.tiles, layout.slot_bytes,
                kHeadDim * 2 * 2, worst_device_key, worst_oracle_key, worst_device_value,
                worst_oracle_value, worst_decode_gap);

    if (!(worst_device_key <= test_case.key_relative_l2)) {
        std::cerr << test_case.name << ": key reconstruction exceeded " << test_case.key_relative_l2
                  << " (got " << worst_device_key << ")\n";
        ++failures;
    }
    if (!(worst_device_value <= test_case.value_relative_l2)) {
        std::cerr << test_case.name << ": value reconstruction exceeded "
                  << test_case.value_relative_l2 << " (got " << worst_device_value << ")\n";
        ++failures;
    }
    // The kernel must not be materially worse than the oracle's own quantization.
    if (!(worst_device_key <= worst_oracle_key * 1.15 + 1e-3) ||
        !(worst_device_value <= worst_oracle_value * 1.15 + 1e-3)) {
        std::cerr << test_case.name << ": kernel is less accurate than the oracle encode\n";
        ++failures;
    }
    // Both sides decode the same record bytes; only FP16 loads and the rotation order differ.
    if (!(worst_decode_gap <= 0.05)) {
        std::cerr << test_case.name << ": decode Op disagrees with the record contents (gap "
                  << worst_decode_gap << ")\n";
        ++failures;
    }
    return failures;
}

// ── attention over records ──────────────────────────────────────────────────────────────────

struct AttentionCase {
    KvarnFormat format;
    const char* name;
    std::int32_t kv_heads;
    std::int32_t q_heads;
    std::int32_t record_pages;
    std::int32_t tokens;
};

int run_attention_case(const AttentionCase& test_case) {
    const KvarnRecordLayout layout = kvarn_record_layout(kHeadDim, test_case.format);
    const std::int32_t group       = layout.group;
    const std::int32_t keys        = test_case.record_pages * group;
    const std::int32_t pages       = test_case.record_pages + 2;
    const std::int32_t group_size  = test_case.q_heads / test_case.kv_heads;

    const std::vector<float> staged_k = staged_keys(test_case.kv_heads, keys, 0x2f11);
    const std::vector<float> staged_v = staged_values(test_case.kv_heads, keys, 0x77c3);

    // Logical page i lives in a physical page chosen so the table is not the identity.
    std::vector<int> block_table(static_cast<std::size_t>(test_case.record_pages));
    for (std::int32_t i = 0; i < test_case.record_pages; ++i) {
        block_table[static_cast<std::size_t>(i)] = (i * 3 + 1) % pages;
    }

    DeviceBuffer device_k     = to_device_bf16(staged_k);
    DeviceBuffer device_v     = to_device_bf16(staged_v);
    DeviceBuffer device_pages = to_device_i32(block_table);
    const std::size_t plane_bytes =
        layout.record_bytes * static_cast<std::size_t>(test_case.kv_heads) * pages;
    DeviceBuffer device_records(plane_bytes);

    Tensor k(device_k.p, DType::BF16, {kHeadDim, test_case.kv_heads, keys, 1});
    Tensor v(device_v.p, DType::BF16, {kHeadDim, test_case.kv_heads, keys, 1});
    Tensor ids(device_pages.p, DType::I32, {test_case.record_pages, 1, 1, 1});
    Tensor records(device_records.p, DType::U8,
                   {static_cast<std::int32_t>(layout.slot_bytes), group, test_case.kv_heads,
                    pages});
    ops::kvarn_compress(k, v, ids, test_case.format, test_case.kv_heads, records, nullptr);
    cuda_check_last_launch("kvarn_compress");

    std::vector<float> query(static_cast<std::size_t>(kHeadDim) * test_case.q_heads *
                             test_case.tokens);
    fill_uniform(query, 0x1357, -1.0F, 1.0F);
    round_to_bf16(query);
    DeviceBuffer device_query = to_device_bf16(query);
    DeviceBuffer device_out(query.size() * sizeof(std::uint16_t));

    Tensor q(device_query.p, DType::BF16, {kHeadDim, test_case.q_heads, test_case.tokens, 1});
    Tensor out(device_out.p, DType::BF16, {kHeadDim, test_case.q_heads, test_case.tokens, 1});

    ops::KvarnAttentionCache cache{records, ids, test_case.format, test_case.kv_heads,
                                   test_case.record_pages};
    WorkspaceArena workspace(ops::kvarn_attention_workspace_capacity_bytes(
        test_case.q_heads, kHeadDim, test_case.tokens, test_case.record_pages));
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    ops::kvarn_attention_cached(q, scale, cache, workspace, out, nullptr);
    cuda_check_last_launch("kvarn_attention_cached");
    cuda_synchronize();

    const std::vector<double> device_out_host = from_device_bf16(device_out, query.size());

    // Oracle: the records' own logical K/V, returned to the original frame, then FP64 attention.
    std::vector<std::uint8_t> host_plane(plane_bytes);
    cuda_check(cudaMemcpy(host_plane.data(), device_records.p, plane_bytes,
                          cudaMemcpyDeviceToHost),
               "copy records");

    // logical[head][key][channel]
    std::vector<std::vector<double>> logical_k(static_cast<std::size_t>(test_case.kv_heads));
    std::vector<std::vector<double>> logical_v(static_cast<std::size_t>(test_case.kv_heads));
    for (std::int32_t head = 0; head < test_case.kv_heads; ++head) {
        logical_k[static_cast<std::size_t>(head)].resize(static_cast<std::size_t>(keys) * kHeadDim);
        logical_v[static_cast<std::size_t>(head)].resize(static_cast<std::size_t>(keys) * kHeadDim);
        for (std::int32_t page = 0; page < test_case.record_pages; ++page) {
            const std::size_t offset =
                layout.record_bytes *
                (static_cast<std::size_t>(head) +
                 static_cast<std::size_t>(test_case.kv_heads) *
                     static_cast<std::size_t>(block_table[static_cast<std::size_t>(page)]));
            std::vector<float> k_tile(static_cast<std::size_t>(kHeadDim) * group);
            std::vector<float> v_tile(k_tile.size());
            kvarn_decode_record_rotated({host_plane.data() + offset, layout.record_bytes}, layout,
                                        k_tile, v_tile);
            inverse_rotate_columns(k_tile, group);
            inverse_rotate_rows(v_tile, group, kHeadDim);
            for (std::int32_t t = 0; t < group; ++t) {
                const std::size_t key = static_cast<std::size_t>(page) * group + t;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    logical_k[static_cast<std::size_t>(head)][key * kHeadDim + d] =
                        k_tile[static_cast<std::size_t>(d) * group + t];
                    logical_v[static_cast<std::size_t>(head)][key * kHeadDim + d] =
                        v_tile[static_cast<std::size_t>(t) * kHeadDim + d];
                }
            }
        }
    }

    std::vector<double> reference(query.size());
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        for (std::int32_t q_head = 0; q_head < test_case.q_heads; ++q_head) {
            const std::int32_t head = q_head / group_size;
            const std::size_t q_base =
                static_cast<std::size_t>(kHeadDim) * (q_head + test_case.q_heads * token);
            std::vector<double> scores(static_cast<std::size_t>(keys));
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int32_t key = 0; key < keys; ++key) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(query[q_base + d]) *
                           logical_k[static_cast<std::size_t>(head)]
                                    [static_cast<std::size_t>(key) * kHeadDim + d];
                }
                scores[static_cast<std::size_t>(key)] = dot * scale;
                maximum = std::max(maximum, scores[static_cast<std::size_t>(key)]);
            }
            double total = 0.0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                total += score;
            }
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double sum = 0.0;
                for (std::int32_t key = 0; key < keys; ++key) {
                    sum += scores[static_cast<std::size_t>(key)] *
                           logical_v[static_cast<std::size_t>(head)]
                                    [static_cast<std::size_t>(key) * kHeadDim + d];
                }
                reference[q_base + d] = sum / total;
            }
        }
    }

    std::printf("%-42s keys=%d splits<=%d\n", test_case.name, keys,
                test_case.record_pages < 32 ? test_case.record_pages : 32);
    const ReductionCriterion criterion{/*relative_l2*/ 6.0e-3, /*gross_absolute*/ 2.0e-3,
                                       /*gross_relative_to_max_reference*/ 2.0e-2};
    return verify_reduction(test_case.name, device_out_host, reference, criterion);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "kvarn: no CUDA device\n";
        return 77;
    }

    const KvarnRecordLayout k4v2 = kvarn_record_layout(kHeadDim, KvarnFormat::K4V2G64);
    int failures                 = 0;
    if (k4v2.record_bytes % static_cast<std::size_t>(k4v2.group) != 0) {
        std::cerr << "kvarn: record does not divide into per-token slots\n";
        ++failures;
    }

    const Case cases[] = {
        {KvarnFormat::K4V2G64, "kvarn_k4v2_g64", 4, 3, 0.16, 0.62},
        {KvarnFormat::K4V4G64, "kvarn_k4v4_g64", 4, 2, 0.16, 0.17},
        {KvarnFormat::K4V2G64, "kvarn_k4v2_g64", 2, 1, 0.16, 0.62},
    };
    for (const Case& test_case : cases) { failures += run_case(test_case); }

    const AttentionCase attention_cases[] = {
        {KvarnFormat::K4V2G64, "kvarn attention k4v2 pages=5 tokens=2", 4, 24, 5, 2},
        {KvarnFormat::K4V4G64, "kvarn attention k4v4 pages=1 tokens=1", 4, 24, 1, 1},
        {KvarnFormat::K4V2G64, "kvarn attention k4v2 pages=40 tokens=1", 4, 24, 40, 1},
        {KvarnFormat::K4V2G64, "kvarn attention 35b k4v2 pages=3 tokens=1", 2, 16, 3, 1},
    };
    for (const AttentionCase& test_case : attention_cases) {
        failures += run_attention_case(test_case);
    }

    if (failures != 0) {
        std::cerr << "kvarn: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "kvarn: all cases passed\n";
    return 0;
}
