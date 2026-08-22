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

// ── composed attention: sink, records, tail, and fresh K/V ──────────────────────────────────

struct SequenceCase {
    KvarnFormat format;
    const char* name;
    std::int32_t kv_heads;
    std::int32_t q_heads;
    std::int32_t batch;
    std::int32_t prefill;   // first call's width
    std::int32_t steps;     // decode calls after it
    std::int32_t step_width;
};

// Mirrors the Op's own storage rule so the oracle can say where each committed position lives.
std::int32_t stage_slot(std::int32_t position, std::int32_t group) {
    const std::int32_t sink = 2 * group;
    return position < sink ? position : sink + position % group;
}

int run_sequence_case(const SequenceCase& test_case) {
    const KvarnRecordLayout layout = kvarn_record_layout(kHeadDim, test_case.format);
    const std::int32_t group       = layout.group;
    const std::int32_t group_size  = test_case.q_heads / test_case.kv_heads;
    const std::int32_t total = test_case.prefill + test_case.steps * test_case.step_width;
    const std::int32_t pages = total / group + 4;
    const std::int32_t stage_tokens = ops::kKvarnStageTokens;

    // Whole-sequence K/V per row; each call feeds the Op its own slice.
    std::vector<std::vector<float>> keys(static_cast<std::size_t>(test_case.batch));
    std::vector<std::vector<float>> values(static_cast<std::size_t>(test_case.batch));
    for (std::int32_t b = 0; b < test_case.batch; ++b) {
        keys[static_cast<std::size_t>(b)] =
            staged_keys(test_case.kv_heads, total, 0x3300u + static_cast<std::uint32_t>(b));
        values[static_cast<std::size_t>(b)] =
            staged_values(test_case.kv_heads, total, 0x8800u + static_cast<std::uint32_t>(b));
    }

    // Distinct table rows and a non-identity page map, so row and page addressing are exercised.
    std::vector<int> table_rows(static_cast<std::size_t>(test_case.batch));
    for (std::int32_t b = 0; b < test_case.batch; ++b) { table_rows[static_cast<std::size_t>(b)] = b; }
    std::vector<int> block_tables(static_cast<std::size_t>(pages) * test_case.batch);
    for (std::int32_t r = 0; r < test_case.batch; ++r) {
        for (std::int32_t p = 0; p < pages; ++p) {
            block_tables[static_cast<std::size_t>(r) * pages + p] =
                (r * pages + p * 5 + 1) % (pages * test_case.batch);
        }
    }

    const std::size_t plane_bytes = layout.record_bytes *
                                    static_cast<std::size_t>(test_case.kv_heads) * pages *
                                    test_case.batch;
    DeviceBuffer device_records(plane_bytes);
    cuda_check(cudaMemset(device_records.p, 0, plane_bytes), "clear records");
    DeviceBuffer device_stage_k(static_cast<std::size_t>(kHeadDim) * test_case.kv_heads *
                                stage_tokens * test_case.batch * sizeof(std::uint16_t));
    DeviceBuffer device_stage_v(device_stage_k.bytes);
    cuda_check(cudaMemset(device_stage_k.p, 0, device_stage_k.bytes), "clear stage k");
    cuda_check(cudaMemset(device_stage_v.p, 0, device_stage_v.bytes), "clear stage v");
    DeviceBuffer device_tables = to_device_i32(block_tables);
    DeviceBuffer device_rows   = to_device_i32(table_rows);

    ops::KvarnBatchLayerView cache{};
    cache.records = Tensor(device_records.p, DType::U8,
                           {static_cast<std::int32_t>(layout.slot_bytes), group,
                            test_case.kv_heads, pages * test_case.batch});
    cache.block_tables = Tensor(device_tables.p, DType::I32, {pages, test_case.batch, 1, 1});
    cache.stage_k      = Tensor(device_stage_k.p, DType::BF16,
                                {kHeadDim, test_case.kv_heads, stage_tokens, test_case.batch});
    cache.stage_v      = Tensor(device_stage_v.p, DType::BF16,
                                {kHeadDim, test_case.kv_heads, stage_tokens, test_case.batch});
    cache.format       = test_case.format;
    cache.head_dim     = kHeadDim;
    cache.kv_heads     = test_case.kv_heads;

    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    WorkspaceArena workspace(ops::kvarn_gqa_attention_workspace_capacity_bytes(
        test_case.q_heads, kHeadDim,
        std::max(test_case.prefill, test_case.step_width), test_case.batch));

    std::int32_t frontier = 0;
    int failures          = 0;
    std::vector<double> final_out;
    std::vector<float> final_query;
    std::int32_t final_first = 0;
    std::int32_t final_width = 0;

    const std::int32_t calls = 1 + test_case.steps;
    for (std::int32_t call = 0; call < calls; ++call) {
        const std::int32_t width = call == 0 ? test_case.prefill : test_case.step_width;

        std::vector<float> call_k(static_cast<std::size_t>(kHeadDim) * test_case.kv_heads * width *
                                  test_case.batch);
        std::vector<float> call_v(call_k.size());
        std::vector<int> positions(static_cast<std::size_t>(width) * test_case.batch);
        for (std::int32_t b = 0; b < test_case.batch; ++b) {
            for (std::int32_t j = 0; j < width; ++j) {
                positions[static_cast<std::size_t>(b) * width + j] = frontier + j;
                const std::size_t destination =
                    static_cast<std::size_t>(kHeadDim) * test_case.kv_heads * (j + width * b);
                const std::size_t source =
                    static_cast<std::size_t>(kHeadDim) * test_case.kv_heads * (frontier + j);
                std::copy_n(keys[static_cast<std::size_t>(b)].begin() +
                                static_cast<std::ptrdiff_t>(source),
                            static_cast<std::size_t>(kHeadDim) * test_case.kv_heads,
                            call_k.begin() + static_cast<std::ptrdiff_t>(destination));
                std::copy_n(values[static_cast<std::size_t>(b)].begin() +
                                static_cast<std::ptrdiff_t>(source),
                            static_cast<std::size_t>(kHeadDim) * test_case.kv_heads,
                            call_v.begin() + static_cast<std::ptrdiff_t>(destination));
            }
        }

        std::vector<float> query(static_cast<std::size_t>(kHeadDim) * test_case.q_heads * width *
                                 test_case.batch);
        fill_uniform(query, 0x4000u + static_cast<std::uint32_t>(call), -1.0F, 1.0F);
        round_to_bf16(query);

        DeviceBuffer device_k         = to_device_bf16(call_k);
        DeviceBuffer device_v         = to_device_bf16(call_v);
        DeviceBuffer device_query     = to_device_bf16(query);
        DeviceBuffer device_positions = to_device_i32(positions);
        DeviceBuffer device_out(query.size() * sizeof(std::uint16_t));

        Tensor q(device_query.p, DType::BF16,
                 {kHeadDim, test_case.q_heads, width, test_case.batch});
        Tensor k(device_k.p, DType::BF16,
                 {kHeadDim, test_case.kv_heads, width, test_case.batch});
        Tensor v(device_v.p, DType::BF16,
                 {kHeadDim, test_case.kv_heads, width, test_case.batch});
        Tensor pos(device_positions.p, DType::I32, {width, test_case.batch, 1, 1});
        Tensor rows(device_rows.p, DType::I32, {test_case.batch, 1, 1, 1});
        Tensor out(device_out.p, DType::BF16,
                   {kHeadDim, test_case.q_heads, width, test_case.batch});

        ops::kvarn_gqa_attention(q, k, v, pos, Tensor{}, rows, scale, cache, workspace, out,
                                 nullptr);
        cuda_check_last_launch("kvarn_gqa_attention");
        cuda_synchronize();

        if (call + 1 == calls) {
            final_out   = from_device_bf16(device_out, query.size());
            final_query = query;
            final_first = frontier;
            final_width = width;
        }
        frontier += width;
    }

    // Oracle: rebuild each committed position's logical value from the storage the Op left
    // behind, then evaluate ideal FP64 attention for the final call's queries.
    std::vector<std::uint8_t> host_plane(plane_bytes);
    cuda_check(cudaMemcpy(host_plane.data(), device_records.p, plane_bytes, cudaMemcpyDeviceToHost),
               "read records");
    const std::vector<double> host_stage_k =
        from_device_bf16(device_stage_k, device_stage_k.bytes / sizeof(std::uint16_t));
    const std::vector<double> host_stage_v =
        from_device_bf16(device_stage_v, device_stage_v.bytes / sizeof(std::uint16_t));

    const std::int32_t record_page_end = final_first >= 2 * group ? final_first / group : 2;
    std::vector<double> reference(final_query.size());

    for (std::int32_t b = 0; b < test_case.batch; ++b) {
        const std::int32_t table_row = table_rows[static_cast<std::size_t>(b)];
        // logical[head][position][channel] over the committed history.
        std::vector<std::vector<double>> logical_k(static_cast<std::size_t>(test_case.kv_heads));
        std::vector<std::vector<double>> logical_v(static_cast<std::size_t>(test_case.kv_heads));
        for (std::int32_t head = 0; head < test_case.kv_heads; ++head) {
            auto& lk = logical_k[static_cast<std::size_t>(head)];
            auto& lv = logical_v[static_cast<std::size_t>(head)];
            lk.assign(static_cast<std::size_t>(final_first) * kHeadDim, 0.0);
            lv.assign(lk.size(), 0.0);

            for (std::int32_t position = 0; position < final_first; ++position) {
                if (position >= 2 * group && position < record_page_end * group) { continue; }
                const std::int32_t slot = stage_slot(position, group);
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t index =
                        static_cast<std::size_t>(d) +
                        static_cast<std::size_t>(kHeadDim) *
                            (head + static_cast<std::size_t>(test_case.kv_heads) *
                                        (slot + static_cast<std::size_t>(stage_tokens) * table_row));
                    lk[static_cast<std::size_t>(position) * kHeadDim + d] = host_stage_k[index];
                    lv[static_cast<std::size_t>(position) * kHeadDim + d] = host_stage_v[index];
                }
            }
            for (std::int32_t page = 2; page < record_page_end; ++page) {
                const std::size_t offset =
                    layout.record_bytes *
                    (static_cast<std::size_t>(head) +
                     static_cast<std::size_t>(test_case.kv_heads) *
                         static_cast<std::size_t>(
                             block_tables[static_cast<std::size_t>(table_row) * pages + page]));
                std::vector<float> k_tile(static_cast<std::size_t>(kHeadDim) * group);
                std::vector<float> v_tile(k_tile.size());
                kvarn_decode_record_rotated({host_plane.data() + offset, layout.record_bytes},
                                            layout, k_tile, v_tile);
                inverse_rotate_columns(k_tile, group);
                inverse_rotate_rows(v_tile, group, kHeadDim);
                for (std::int32_t t = 0; t < group; ++t) {
                    const std::size_t position = static_cast<std::size_t>(page) * group + t;
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        lk[position * kHeadDim + d] = k_tile[static_cast<std::size_t>(d) * group + t];
                        lv[position * kHeadDim + d] = v_tile[static_cast<std::size_t>(t) * kHeadDim + d];
                    }
                }
            }
        }

        for (std::int32_t j = 0; j < final_width; ++j) {
            const std::int32_t position = final_first + j;
            for (std::int32_t q_head = 0; q_head < test_case.q_heads; ++q_head) {
                const std::int32_t head = q_head / group_size;
                const auto& lk          = logical_k[static_cast<std::size_t>(head)];
                const auto& lv          = logical_v[static_cast<std::size_t>(head)];
                const std::size_t q_base =
                    static_cast<std::size_t>(kHeadDim) *
                    (q_head + static_cast<std::size_t>(test_case.q_heads) * (j + final_width * b));

                std::vector<double> scores(static_cast<std::size_t>(position) + 1);
                double maximum = -std::numeric_limits<double>::infinity();
                for (std::int32_t x = 0; x <= position; ++x) {
                    const double* key = nullptr;
                    std::vector<double> fresh(static_cast<std::size_t>(kHeadDim));
                    if (x < final_first) {
                        key = lk.data() + static_cast<std::size_t>(x) * kHeadDim;
                    } else {
                        const std::size_t base = static_cast<std::size_t>(kHeadDim) *
                                                 (head + static_cast<std::size_t>(test_case.kv_heads) *
                                                             static_cast<std::size_t>(x));
                        for (std::int32_t d = 0; d < kHeadDim; ++d) {
                            fresh[static_cast<std::size_t>(d)] =
                                keys[static_cast<std::size_t>(b)][base + d];
                        }
                        key = fresh.data();
                    }
                    double dot = 0.0;
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        dot += static_cast<double>(final_query[q_base + d]) * key[d];
                    }
                    scores[static_cast<std::size_t>(x)] = dot * scale;
                    maximum = std::max(maximum, scores[static_cast<std::size_t>(x)]);
                }
                double total = 0.0;
                for (double& score : scores) {
                    score = std::exp(score - maximum);
                    total += score;
                }
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    double sum = 0.0;
                    for (std::int32_t x = 0; x <= position; ++x) {
                        double value = 0.0;
                        if (x < final_first) {
                            value = lv[static_cast<std::size_t>(x) * kHeadDim + d];
                        } else {
                            const std::size_t base =
                                static_cast<std::size_t>(kHeadDim) *
                                (head + static_cast<std::size_t>(test_case.kv_heads) *
                                            static_cast<std::size_t>(x));
                            value = values[static_cast<std::size_t>(b)][base + d];
                        }
                        sum += scores[static_cast<std::size_t>(x)] * value;
                    }
                    reference[q_base + d] = sum / total;
                }
            }
        }
    }

    std::printf("%-38s B=%d prefill=%d steps=%d frontier=%d records=[2,%d)\n", test_case.name,
                test_case.batch, test_case.prefill, test_case.steps, final_first + final_width,
                record_page_end);
    const ReductionCriterion criterion{/*relative_l2*/ 8.0e-3, /*gross_absolute*/ 3.0e-3,
                                       /*gross_relative_to_max_reference*/ 3.0e-2};
    failures += verify_reduction(test_case.name, final_out, reference, criterion);
    return failures;
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

    const SequenceCase sequence_cases[] = {
        // Prefill past the sink, then decode across a page boundary.
        {KvarnFormat::K4V2G64, "kvarn gqa k4v2 prefill+decode", 4, 24, 1, 256, 70, 1},
        // Every position still in the sink: no record is ever built.
        {KvarnFormat::K4V2G64, "kvarn gqa k4v2 sink only", 4, 24, 1, 32, 8, 1},
        // Batched decode with independent rows and table rows.
        {KvarnFormat::K4V4G64, "kvarn gqa k4v4 batched decode", 4, 24, 3, 192, 40, 1},
        // Speculative-width decode that crosses a page boundary inside one call.
        {KvarnFormat::K4V2G64, "kvarn gqa k4v2 wide decode", 4, 24, 2, 190, 12, 5},
        // The 35B head geometry.
        {KvarnFormat::K4V2G64, "kvarn gqa 35b k4v2", 2, 16, 1, 320, 20, 1},
        // A prefill width that leaves a short trailing column chunk, verified directly.
        {KvarnFormat::K4V2G64, "kvarn gqa k4v2 ragged prefill", 4, 24, 1, 914, 0, 1},
        {KvarnFormat::K4V2G64, "kvarn gqa k4v2 ragged prefill B=2", 4, 24, 2, 332, 0, 1},
    };
    for (const SequenceCase& test_case : sequence_cases) {
        failures += run_sequence_case(test_case);
    }

    if (failures != 0) {
        std::cerr << "kvarn: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "kvarn: all cases passed\n";
    return 0;
}
