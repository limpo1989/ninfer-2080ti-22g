// Public-Op benchmark for KVarN-backed causal grouped-query attention (K3).
//
// Prefill shape: a history of `--context` committed tokens, then one call that appends and
// attends over `--width` fresh columns. That is exactly the per-layer shape a chunked prefill
// issues, so the reported per-launch time scales to whole-prompt prefill cost.

#include "ninfer/ops/kvarn.h"

#include "core/arena.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim = 256;

struct Options {
    std::vector<std::int32_t> contexts{4096, 10240, 25088};
    std::int32_t width    = 1024;
    std::int32_t kv_heads = 4;
    std::int32_t q_heads  = 24;
    int warmup            = 3;
    int repeat            = 20;
};

std::vector<std::int32_t> parse_list(const char* text) {
    std::vector<std::int32_t> values;
    const std::string source(text);
    std::size_t begin = 0;
    while (begin <= source.size()) {
        const std::size_t comma = source.find(',', begin);
        const std::string item  = source.substr(begin, comma - begin);
        if (!item.empty()) { values.push_back(static_cast<std::int32_t>(std::stol(item))); }
        if (comma == std::string::npos) { break; }
        begin = comma + 1;
    }
    return values;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string flag(argv[index]);
        const auto next = [&]() -> const char* {
            if (index + 1 >= argc) { std::exit(2); }
            return argv[++index];
        };
        if (flag == "--context") { options.contexts = parse_list(next()); }
        else if (flag == "--width") { options.width = static_cast<std::int32_t>(std::stol(next())); }
        else if (flag == "--geometry") {
            const std::string value(next());
            if (value == "35b") { options.kv_heads = 2; options.q_heads = 16; }
        } else if (flag == "--repeat") { options.repeat = std::atoi(next()); }
        else if (flag == "--warmup") { options.warmup = std::atoi(next()); }
        else {
            std::fprintf(stderr, "usage: ninfer_kvarn_attention_bench [--context 4096,25088] "
                                 "[--width 1024] [--geometry 27b|35b] [--repeat N]\n");
            return 2;
        }
    }

    DeviceContext device(0);
    const std::int32_t width    = options.width;
    const std::int32_t kv_heads = options.kv_heads;
    const std::int32_t q_heads  = options.q_heads;
    const float scale           = 0.0625F;

    std::printf("%8s %6s %12s %12s %10s\n", "context", "width", "median_us", "us/token", "GQA");
    for (const std::int32_t context : options.contexts) {
        const std::int32_t total = context + width;
        const std::int32_t pages = total / kPagedKVPageSize + 4;

        const KvarnRecordLayout layout = kvarn_record_layout(kHeadDim, KvarnFormat::K4V2G64);
        DeviceBuffer records(layout.record_bytes * static_cast<std::size_t>(kv_heads) * pages);
        records.fill(0);
        DeviceBuffer stage_k(static_cast<std::size_t>(kHeadDim) * kv_heads *
                             ops::kKvarnStageTokens * sizeof(std::uint16_t));
        DeviceBuffer stage_v(stage_k.bytes);
        stage_k.fill(0);
        stage_v.fill(0);

        std::vector<std::int32_t> table(static_cast<std::size_t>(pages));
        for (std::int32_t p = 0; p < pages; ++p) { table[static_cast<std::size_t>(p)] = p; }
        DeviceBuffer tables(table.size() * sizeof(std::int32_t));
        tables.copy_from_host(table.data(), tables.bytes);
        DeviceBuffer rows(sizeof(std::int32_t));
        rows.fill(0);

        ops::KvarnBatchLayerView cache{};
        cache.records = Tensor(records.p, DType::U8,
                               {static_cast<std::int32_t>(layout.slot_bytes),
                                layout.group, kv_heads, pages});
        cache.block_tables = Tensor(tables.p, DType::I32, {pages, 1, 1, 1});
        cache.stage_k = Tensor(stage_k.p, DType::BF16,
                               {kHeadDim, kv_heads, ops::kKvarnStageTokens, 1});
        cache.stage_v = Tensor(stage_v.p, DType::BF16,
                               {kHeadDim, kv_heads, ops::kKvarnStageTokens, 1});
        cache.format   = KvarnFormat::K4V2G64;
        cache.head_dim = kHeadDim;
        cache.kv_heads = kv_heads;

        DeviceBuffer kv_k = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kv_heads * width);
        DeviceBuffer kv_v = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kv_heads * width);
        DeviceBuffer query = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * q_heads * width);
        DeviceBuffer out(query.bytes);
        out.fill(0);

        WorkspaceArena workspace(
            ops::kvarn_gqa_attention_workspace_capacity_bytes(q_heads, kHeadDim, width, 1));

        std::vector<std::int32_t> positions(static_cast<std::size_t>(width));
        DeviceBuffer device_positions(positions.size() * sizeof(std::int32_t));

        Tensor k(kv_k.p, DType::BF16, {kHeadDim, kv_heads, width, 1});
        Tensor v(kv_v.p, DType::BF16, {kHeadDim, kv_heads, width, 1});
        Tensor pos(device_positions.p, DType::I32, {width, 1, 1, 1});
        Tensor row_ids(rows.p, DType::I32, {1, 1, 1, 1});

        // Build the history with commit-only calls, then time the attending call at `context`.
        for (std::int32_t base = 0; base < context; base += width) {
            for (std::int32_t j = 0; j < width; ++j) {
                positions[static_cast<std::size_t>(j)] = base + j;
            }
            device_positions.copy_from_host(positions.data(), device_positions.bytes);
            Tensor empty;
            ops::kvarn_gqa_attention(empty, k, v, pos, Tensor{}, row_ids, scale, cache, workspace,
                                     empty, device.stream);
        }
        for (std::int32_t j = 0; j < width; ++j) {
            positions[static_cast<std::size_t>(j)] = context + j;
        }
        device_positions.copy_from_host(positions.data(), device_positions.bytes);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));

        Tensor q(query.p, DType::BF16, {kHeadDim, q_heads, width, 1});
        Tensor o(out.p, DType::BF16, {kHeadDim, q_heads, width, 1});

        const bench::ColdTiming timing = bench::measure_launch(
            [&](cudaStream_t stream) {
                ops::kvarn_gqa_attention(q, k, v, pos, Tensor{}, row_ids, scale, cache, workspace,
                                         o, stream);
            },
            device.stream, options.warmup, options.repeat);

        std::printf("%8d %6d %12.1f %12.3f %10s\n", context, width, timing.median_us,
                    timing.median_us / static_cast<double>(width),
                    q_heads == 24 ? "27b" : "35b");
    }
    return 0;
}
