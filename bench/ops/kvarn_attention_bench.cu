// Public-Op benchmark for KVarN-backed causal grouped-query attention (K3).
//
// Prefill shape: a history of `--context` committed tokens, then one call that appends and
// attends over the last `--query-width` columns (all `--width` columns by default).
// --batch and --graph also exercise batched MTP shapes through the same public Op.

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
    std::int32_t query_width = 0;
    std::int32_t batch = 1;
    std::int32_t kv_heads = 4;
    std::int32_t q_heads  = 24;
    int warmup            = 3;
    int repeat            = 20;
    bool graph = false;
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
        else if (flag == "--query-width") { options.query_width = std::stoi(next()); }
        else if (flag == "--batch") { options.batch = std::stoi(next()); }
        else if (flag == "--graph") { options.graph = true; }
        else if (flag == "--geometry") {
            const std::string value(next());
            if (value == "35b") { options.kv_heads = 2; options.q_heads = 16; }
        } else if (flag == "--repeat") { options.repeat = std::atoi(next()); }
        else if (flag == "--warmup") { options.warmup = std::atoi(next()); }
        else {
            std::fprintf(stderr, "usage: ninfer_kvarn_attention_bench [--context 4096,25088] "
                                 "[--width 1024] [--query-width N] [--batch N] [--graph] "
                                 "[--geometry 27b|35b] [--repeat N]\n");
            return 2;
        }
    }

    if (options.query_width == 0) { options.query_width = options.width; }
    if (options.width <= 0 || options.query_width <= 0 || options.query_width > options.width ||
        options.batch < 1 || options.batch > 8 || options.repeat < 1 || options.warmup < 0 ||
        options.contexts.empty() ||
        std::any_of(options.contexts.begin(), options.contexts.end(), [](int n) { return n < 0; })) {
        std::fprintf(stderr, "invalid context, width, query width, batch, or repeat count\n");
        return 2;
    }
    DeviceContext device(0);
    const std::int32_t width    = options.width;
    const std::int32_t queries = options.query_width;
    const std::int32_t batch = options.batch;
    const std::int32_t kv_heads = options.kv_heads;
    const std::int32_t q_heads  = options.q_heads;
    const float scale           = 0.0625F;

    std::printf("%8s %6s %6s %3s %12s %12s %6s %6s\n", "context", "width", "query", "B",
                "median_us", "us/query", "GQA", "launch");
    for (const std::int32_t context : options.contexts) {
        const std::int32_t total = context + width;
        const std::int32_t pages = total / kPagedKVPageSize + 4;

        const KvarnRecordLayout layout = kvarn_record_layout(kHeadDim, KvarnFormat::K4V2G64);
        DeviceBuffer records(layout.record_bytes * static_cast<std::size_t>(kv_heads) * pages * batch);
        records.fill(0);
        DeviceBuffer stage_k(static_cast<std::size_t>(kHeadDim) * kv_heads *
                             ops::kKvarnStageTokens * batch * sizeof(std::uint16_t));
        DeviceBuffer stage_v(stage_k.bytes);
        stage_k.fill(0);
        stage_v.fill(0);

        std::vector<std::int32_t> table(static_cast<std::size_t>(pages) * batch);
        for (std::size_t p = 0; p < table.size(); ++p) { table[p] = static_cast<int>(p); }
        DeviceBuffer tables(table.size() * sizeof(std::int32_t));
        tables.copy_from_host(table.data(), tables.bytes);
        std::vector<int> host_rows(batch);
        for (int b = 0; b < batch; ++b) { host_rows[b] = b; }
        DeviceBuffer rows(batch * sizeof(std::int32_t));
        rows.copy_from_host(host_rows.data(), rows.bytes);

        ops::KvarnBatchLayerView cache{};
        cache.records = Tensor(records.p, DType::U8,
                               {static_cast<std::int32_t>(layout.slot_bytes),
                                layout.group, kv_heads, pages * batch});
        cache.block_tables = Tensor(tables.p, DType::I32, {pages, batch, 1, 1});
        cache.stage_k = Tensor(stage_k.p, DType::BF16,
                               {kHeadDim, kv_heads, ops::kKvarnStageTokens, batch});
        cache.stage_v = Tensor(stage_v.p, DType::BF16,
                               {kHeadDim, kv_heads, ops::kKvarnStageTokens, batch});
        cache.format   = KvarnFormat::K4V2G64;
        cache.head_dim = kHeadDim;
        cache.kv_heads = kv_heads;

        WorkspaceArena workspace(
            ops::kvarn_gqa_attention_workspace_capacity_bytes(q_heads, kHeadDim, queries, batch));
        Tensor row_ids(rows.p, DType::I32, {batch, 1, 1, 1});

        // Bound setup memory and stop exactly at the requested prefix, including ragged lengths.
        // Using the timed width here overshot non-multiple prefixes and was costly for T=1.
        {
            constexpr int setup_chunk = 1024;
            DeviceBuffer history_k = bench::make_bf16(
                static_cast<std::size_t>(kHeadDim) * kv_heads * setup_chunk * batch);
            DeviceBuffer history_v = bench::make_bf16(
                static_cast<std::size_t>(kHeadDim) * kv_heads * setup_chunk * batch);
            DeviceBuffer setup_positions(setup_chunk * batch * sizeof(std::int32_t));
            for (int base = 0; base < context; base += setup_chunk) {
                const int count = std::min(setup_chunk, context - base);
                std::vector<int> positions(count * batch);
                for (int b = 0; b < batch; ++b) {
                    for (int j = 0; j < count; ++j) { positions[b * count + j] = base + j; }
                }
                setup_positions.copy_from_host(positions.data(), positions.size() * sizeof(int));
                Tensor k(history_k.p, DType::BF16, {kHeadDim, kv_heads, count, batch});
                Tensor v(history_v.p, DType::BF16, {kHeadDim, kv_heads, count, batch});
                Tensor pos(setup_positions.p, DType::I32, {count, batch});
                Tensor empty;
                ops::kvarn_gqa_attention(empty, k, v, pos, Tensor{}, row_ids, scale, cache,
                                         workspace, empty, device.stream);
                device.synchronize();
            }
        }

        DeviceBuffer kv_k = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kv_heads * width * batch);
        DeviceBuffer kv_v = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kv_heads * width * batch);
        DeviceBuffer query = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * q_heads * queries * batch);
        DeviceBuffer out(query.bytes);
        out.fill(0);

        std::vector<std::int32_t> positions(static_cast<std::size_t>(width) * batch);
        DeviceBuffer device_positions(positions.size() * sizeof(std::int32_t));

        Tensor k(kv_k.p, DType::BF16, {kHeadDim, kv_heads, width, batch});
        Tensor v(kv_v.p, DType::BF16, {kHeadDim, kv_heads, width, batch});
        Tensor pos(device_positions.p, DType::I32, {width, batch, 1, 1});
        for (int b = 0; b < batch; ++b) {
            for (std::int32_t j = 0; j < width; ++j) {
                positions[static_cast<std::size_t>(b) * width + j] = context + j;
            }
        }
        device_positions.copy_from_host(positions.data(), device_positions.bytes);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));

        Tensor q(query.p, DType::BF16, {kHeadDim, q_heads, queries, batch});
        Tensor o(out.p, DType::BF16, {kHeadDim, q_heads, queries, batch});
        DeviceBuffer saved_k(stage_k.bytes);
        DeviceBuffer saved_v(stage_v.bytes);
        CUDA_CHECK(cudaMemcpyAsync(saved_k.p, stage_k.p, stage_k.bytes,
                                   cudaMemcpyDeviceToDevice, device.stream));
        CUDA_CHECK(cudaMemcpyAsync(saved_v.p, stage_v.p, stage_v.bytes,
                                   cudaMemcpyDeviceToDevice, device.stream));
        const auto launch = [&] {
            ops::kvarn_gqa_attention(q, k, v, pos, Tensor{}, row_ids, scale, cache, workspace,
                                     o, device.stream);
        };
        bench::TimedGraph graph;
        if (options.graph) { graph.capture(device.stream, [&](cudaStream_t) { launch(); }); }
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        std::vector<double> samples;
        for (int i = 0; i < options.warmup + options.repeat; ++i) {
            // Restore the logical input state outside the timed region. Records below the
            // prefix are immutable; records produced by this call are never its own inputs.
            CUDA_CHECK(cudaMemcpyAsync(stage_k.p, saved_k.p, stage_k.bytes,
                                       cudaMemcpyDeviceToDevice, device.stream));
            CUDA_CHECK(cudaMemcpyAsync(stage_v.p, saved_v.p, stage_v.bytes,
                                       cudaMemcpyDeviceToDevice, device.stream));
            CUDA_CHECK(cudaEventRecord(start, device.stream));
            if (options.graph) { graph.launch(device.stream); } else { launch(); }
            CUDA_CHECK(cudaEventRecord(stop, device.stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (i >= options.warmup) { samples.push_back(ms * 1000.0); }
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        const auto timing = bench::summarize_timings(std::move(samples));
        std::printf("%8d %6d %6d %3d %12.1f %12.3f %6s %6s\n", context, width, queries, batch,
                    timing.median_us, timing.median_us / (queries * batch),
                    q_heads == 24 ? "27b" : "35b", options.graph ? "graph" : "direct");
    }
    return 0;
}
