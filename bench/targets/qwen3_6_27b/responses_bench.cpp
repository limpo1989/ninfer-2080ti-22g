#include "ninfer/engine.h"
#include "serve/responses_schema.h"
#include "serve/translate.h"

#include <cuda_profiler_api.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

struct FirstToken final : ninfer::OutputSink {
    Clock::time_point started = Clock::now();
    double seconds = -1;
    bool profile = false;
    bool profiling = false;
    void publish(ninfer::OutputDelta delta) override {
        if (seconds >= 0 || delta.text.empty()) { return; }
        seconds = std::chrono::duration<double>(Clock::now() - started).count();
        if (profile) {
            if (cudaProfilerStart() != cudaSuccess) { throw std::runtime_error("cudaProfilerStart failed"); }
            profiling = true;
        }
    }
    void stop() {
        if (profiling) {
            cudaProfilerStop();
            profiling = false;
        }
    }
    ~FirstToken() override { stop(); }
};
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ninfer_responses_bench MODEL REQUEST_JSON [output=256] [reps=2] [--profile-decode]\n";
        return 2;
    }
    try {
        const int output = argc > 3 ? std::stoi(argv[3]) : 256;
        const int reps = argc > 4 ? std::stoi(argv[4]) : 2;
        const bool profile = argc > 5 && std::string(argv[5]) == "--profile-decode";
        if (output <= 1 || reps < 1 || (profile && reps != 1)) {
            throw std::invalid_argument("output must exceed 1; profiling requires one repetition");
        }
        std::ifstream file(argv[2]);
        Json body;
        file >> body;
        body["max_output_tokens"] = output;
        auto request = ninfer::serve::parse_responses_request(body, {});
        ninfer::serve::compose_responses_generation_messages(request, {});

        ninfer::EngineOptions engine_options;
        engine_options.artifact_path = argv[1];
        engine_options.max_context = 245760;
        engine_options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(245760);
        engine_options.max_concurrency = 2;
        engine_options.prefill_chunk = 1024;
        engine_options.kv_cache = ninfer::KvCacheStorage::KvarnK4V2;
        engine_options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        engine_options.speculative.draft_tokens = 3;
        engine_options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        ninfer::Engine engine(engine_options);
        ninfer::serve::ServeOptions server;
        server.sampling_overrides.temperature = 0.6F;
        server.sampling_overrides.presence_penalty = 1.0F;
        server.sampling_overrides.seed = 1234;
        server.allow_prefix_reuse = false;
        const auto semantics = ninfer::serve::resolve_prompt_semantics(
            request.generation, server, engine.prompt_capabilities());
        const auto input = ninfer::serve::to_prompt_input(request.generation, semantics,
            [](const auto&) -> ninfer::OwnedMedia {
                throw std::invalid_argument("this benchmark accepts text-only histories");
            });
        const auto options = ninfer::serve::to_request_options(request.generation, server);
        const auto prompt_tokens = engine.count_tokens(input);
        for (int rep = 0; rep < reps; ++rep) {
            FirstToken sink;
            sink.profile = profile;
            auto result = engine.generate(engine.prepare(input), options, &sink);
            sink.stop();
            const auto generated = result.generated_token_ids.size();
            const auto& speculative = result.speculative;
            std::cout << Json{{"rep", rep}, {"prompt", prompt_tokens}, {"output", generated},
                {"ttft_s", sink.seconds}, {"prefill_s", result.timings.prefill_seconds},
                {"prefill_tok_s", prompt_tokens / result.timings.prefill_seconds},
                {"decode_s", result.timings.decode_seconds},
                {"decode_tok_s", generated > 1 ? (generated - 1) / result.timings.decode_seconds : 0},
                {"mtp_acceptance", speculative.drafted_tokens ?
                    double(speculative.accepted_tokens) / speculative.drafted_tokens : 0},
                {"mtp_rounds", speculative.rounds}}.dump() << std::endl;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
