#include "ninfer/engine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

constexpr std::size_t kGiB = 1ULL << 30;

struct TempDirectory {
    explicit TempDirectory(const char* label) {
        std::string pattern =
            (std::filesystem::temp_directory_path() / (std::string(label) + "-XXXXXX")).string();
        const char* created = ::mkdtemp(pattern.data());
        if (created == nullptr) {
            throw std::runtime_error("cannot create state-cache test directory");
        }
        path = created;
    }

    ~TempDirectory() {
        if (remove) {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    }

    std::filesystem::path path;
    bool remove = true;
};

ninfer::EngineOptions engine_options(const char* artifact, const std::filesystem::path& cache,
                                     std::uint32_t idle_ms) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk             = 1024;
    options.kv_cache                  = ninfer::KvCacheStorage::KvarnK4V2;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.state_cache_dir           = cache;
    options.state_cache_max_bytes     = 3 * kGiB;
    options.state_cache_ram_bytes     = 3 * kGiB;
    options.state_cache_idle_ms       = idle_ms;
    return options;
}

ninfer::RequestOptions request_options() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 4;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<ninfer::TokenId> continuation(const std::vector<ninfer::TokenId>& prompt,
                                          const ninfer::GenerationResult& result) {
    std::vector<ninfer::TokenId> tokens = prompt;
    tokens.insert(tokens.end(), result.generated_token_ids.begin(),
                  result.generated_token_ids.end());
    tokens.push_back(198);
    return tokens;
}

std::size_t snapshot_count(const std::filesystem::path& root) {
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".snap") { ++count; }
    }
    return count;
}

bool wait_for_snapshots(const std::filesystem::path& root, std::size_t expected) {
    for (int attempt = 0; attempt < 120; ++attempt) {
        if (snapshot_count(root) >= expected) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

double immediate_queue(const char* artifact) {
    TempDirectory cache("ninfer-state-immediate");
    ninfer::Engine engine(engine_options(artifact, cache.path, 0));
    const std::vector<ninfer::TokenId> prompt(512, 198);
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), request_options());
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare_tokens(continuation(prompt, first)), request_options());
    return second.timings.queue_seconds;
}

double deferred_queue(const char* artifact) {
    TempDirectory cache("ninfer-state-deferred");
    const std::vector<ninfer::TokenId> prompt(512, 198);
    std::vector<ninfer::TokenId> second_prompt;
    ninfer::GenerationResult second;
    {
        ninfer::Engine engine(engine_options(artifact, cache.path, 1000));
        const ninfer::GenerationResult first =
            engine.generate(engine.prepare_tokens(prompt), request_options());
        second_prompt = continuation(prompt, first);
        second        = engine.generate(engine.prepare_tokens(second_prompt), request_options());
        if (!wait_for_snapshots(cache.path, 1) || snapshot_count(cache.path) != 1) {
            throw std::runtime_error("idle coalescing did not persist exactly the latest snapshot");
        }
    }

    ninfer::Engine restored(engine_options(artifact, cache.path, 1000));
    const ninfer::GenerationResult disk = restored.generate(
        restored.prepare_tokens(continuation(second_prompt, second)), request_options());
    if (disk.timings.state_cache_source != 2 || disk.reused_prompt_tokens == 0) {
        throw std::runtime_error("deferred state snapshot did not restore from disk");
    }
    return second.timings.queue_seconds;
}

void forced_replacement_preserves_dirty_state(const char* artifact) {
    TempDirectory cache("ninfer-state-forced-replacement");
    ninfer::Engine engine(engine_options(artifact, cache.path, 60'000));
    const std::vector<ninfer::TokenId> original(512, 198);
    const std::vector<ninfer::TokenId> unrelated(512, 199);
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(original), request_options());

    (void)engine.generate(engine.prepare_tokens(unrelated), request_options());
    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare_tokens(continuation(original, first)), request_options());
    if (restored.timings.state_cache_source != 1 || restored.reused_prompt_tokens == 0) {
        throw std::runtime_error(
            "dirty retained state was lost when an unrelated request replaced its lane");
    }
}

void capacity_eviction_preserves_other_dirty_lane(const char* artifact) {
    TempDirectory cache("ninfer-state-capacity-eviction");
    ninfer::EngineOptions options = engine_options(artifact, cache.path, 60'000);
    options.max_concurrency       = 2;
    ninfer::Engine engine(std::move(options));
    const std::vector<ninfer::TokenId> first_prompt(512, 198);
    const std::vector<ninfer::TokenId> second_prompt(512, 199);
    ninfer::RequestOptions concurrent            = request_options();
    concurrent.execution.requested_output_tokens = 128;
    ninfer::GenerationHandle first_handle =
        engine.submit(engine.prepare_tokens(first_prompt), concurrent);
    ninfer::GenerationHandle second_handle =
        engine.submit(engine.prepare_tokens(second_prompt), concurrent);
    const ninfer::GenerationResult first        = first_handle.wait();
    const ninfer::GenerationResult second       = second_handle.wait();
    const ninfer::RuntimeStats concurrent_stats = engine.runtime_stats();
    if (concurrent_stats.decode_row_rounds <= concurrent_stats.decode_rounds) {
        throw std::runtime_error("capacity-eviction fixture did not populate two active lanes");
    }

    ninfer::RequestOptions oversized            = request_options();
    oversized.execution.requested_output_tokens = 4096;
    ninfer::GenerationHandle eviction           = engine.submit(
        engine.prepare_tokens(continuation(first_prompt, first)), std::move(oversized));
    std::atomic<bool> cancel{false};
    std::atomic<bool> admitted{false};
    std::thread canceller([&] {
        for (int attempt = 0; attempt < 1000; ++attempt) {
            const ninfer::RuntimeStats stats = engine.runtime_stats();
            if (stats.running_requests != 0 || stats.prefilling_requests != 0) {
                admitted.store(true, std::memory_order_release);
                cancel.store(true, std::memory_order_release);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        cancel.store(true, std::memory_order_release);
    });
    ninfer::GenerationResult cancelled;
    try {
        cancelled = eviction.wait(nullptr, ninfer::CancellationView([&] {
                                      return cancel.load(std::memory_order_acquire);
                                  }));
    } catch (...) {
        cancel.store(true, std::memory_order_release);
        canceller.join();
        throw;
    }
    canceller.join();
    if (!admitted.load(std::memory_order_acquire) ||
        cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        throw std::runtime_error("capacity-eviction request did not cancel after admission");
    }

    const ninfer::GenerationResult restored = engine.generate(
        engine.prepare_tokens(continuation(second_prompt, second)), request_options());
    if (restored.timings.state_cache_source != 1 || restored.reused_prompt_tokens == 0) {
        throw std::runtime_error("capacity eviction lost another lane's dirty retained state");
    }
}

int exercise(const char* artifact) {
    const double immediate = immediate_queue(artifact);
    const double deferred  = deferred_queue(artifact);
    forced_replacement_preserves_dirty_state(artifact);
    capacity_eviction_preserves_other_dirty_lane(artifact);
    std::cout << "state-cache queue immediate=" << immediate << "s deferred=" << deferred << "s\n";
    if (!(immediate > 0.1)) {
        std::cerr << "immediate capture did not produce a measurable queue baseline\n";
        return 1;
    }
    if (!(deferred < 0.1 && deferred < immediate * 0.5)) {
        std::cerr << "idle capture did not materially reduce next-request queue time\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }
    try {
        if (const int result = exercise(artifact); result != 0) { return result; }
        std::cout << "state-cache idle coalescing checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
