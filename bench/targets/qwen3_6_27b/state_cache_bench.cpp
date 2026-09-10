#include "ninfer/engine.h"
#include "serve/responses_schema.h"
#include "serve/translate.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
Json read_json(const char* path) { std::ifstream f(path); Json j; f >> j; return j; }
void write_json(const char* path, const Json& j) { std::ofstream f(path); f << j.dump(2) << '\n'; if (!f) throw std::runtime_error("cannot write fixture"); }
Json metrics(const ninfer::GenerationResult& r, double wall) {
    return Json{{"input", r.prompt.prompt_tokens}, {"reused", r.reused_prompt_tokens},
        {"output", r.generated_token_ids.size()}, {"tokens", r.generated_token_ids},
        {"ttft", r.timings.first_token_seconds}, {"prefill", r.timings.prefill_seconds},
        {"decode", r.timings.decode_seconds}, {"restore", r.timings.state_restore_seconds},
        {"save", r.timings.state_save_seconds}, {"source", r.timings.state_cache_source}, {"wall", wall}};
}
int main(int argc, char** argv) {
    if (argc != 6) { std::cerr << "Usage: state_cache_bench MODEL REQUEST CACHE_DIR FIXTURE seed|resident|disk|cold|ram\n"; return 2; }
    try {
        const std::string mode = argv[5];
        if (mode != "seed" && mode != "resident" && mode != "disk" && mode != "cold" && mode != "ram") return 2;
        auto request = ninfer::serve::parse_responses_request(read_json(argv[2]), {});
        ninfer::serve::compose_responses_generation_messages(request, {});
        ninfer::EngineOptions options;
        options.artifact_path = argv[1]; options.max_context = 32768;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(32768);
        options.max_concurrency = 1; options.prefill_chunk = 1024;
        options.pending_timeout_ms = 600000;
        options.kv_cache = ninfer::KvCacheStorage::KvarnK4V2;
        options.speculative = {ninfer::SpeculativeBackend::Mtp, 3, ninfer::ProposalHead::Optimized};
        if (mode != "resident" && mode != "cold") {
            options.state_cache_dir = argv[3]; options.state_cache_max_bytes = 8ULL << 30;
            options.state_cache_ram_bytes = 2ULL << 30;
        }
        ninfer::Engine engine(options);
        ninfer::serve::ServeOptions server;
        server.sampling_overrides.temperature = 0.6F; server.sampling_overrides.presence_penalty = 1.0F;
        server.sampling_overrides.seed = 1234;
        const auto semantics = ninfer::serve::resolve_prompt_semantics(request.generation, server, engine.prompt_capabilities());
        auto input = ninfer::serve::to_prompt_input(request.generation, semantics, {});
        auto generation = ninfer::serve::to_request_options(request.generation, server);
        generation.execution.requested_output_tokens = 64;
        Json fixture;
        if (mode == "seed" || mode == "resident" || mode == "ram") {
            const auto started = Clock::now();
            auto first = engine.generate(engine.prepare(input), generation);
            const auto report = metrics(first, std::chrono::duration<double>(Clock::now() - started).count());
            std::cout << Json{{"mode", mode}, {"stage", "first"}, {"metrics", report}}.dump() << std::endl;
            fixture = { {"content", first.content}, {"reasoning", first.reasoning} };
            if (mode == "seed") { write_json(argv[4], fixture); return 0; }
        } else fixture = read_json(argv[4]);
        ninfer::ChatMessage assistant; assistant.role = ninfer::ChatRole::Assistant;
        assistant.reasoning_content = fixture.at("reasoning");
        assistant.parts.push_back({.text = fixture.at("content")}); input.messages.push_back(std::move(assistant));
        ninfer::ChatMessage user; user.role = ninfer::ChatRole::User;
        user.parts.push_back({.text = "请用一句话说明为什么保留已计算的上下文可以减少等待。"});
        input.messages.push_back(std::move(user));
        if (mode == "ram") {
            ninfer::PromptInput filler;
            ninfer::ChatMessage text; text.role = ninfer::ChatRole::User;
            text.parts.push_back({.text = "Say hi."}); filler.messages.push_back(std::move(text));
            auto small = generation; small.execution.requested_output_tokens = 1;
            (void)engine.generate(engine.prepare(filler), small);
        }
        const auto started = Clock::now();
        auto result = engine.generate(engine.prepare(input), generation);
        auto report = metrics(result, std::chrono::duration<double>(Clock::now() - started).count());
        if (mode == "resident") {
            fixture["reference_tokens"] = result.generated_token_ids; write_json(argv[4], fixture);
        } else if (mode == "disk") {
            if (result.timings.state_cache_source != 2 || result.reused_prompt_tokens == 0)
                throw std::runtime_error("no disk restore occurred");
            if (fixture.at("reference_tokens") != report.at("tokens"))
                throw std::runtime_error("disk continuation differs from resident reference");
        } else if (mode == "ram" && (result.timings.state_cache_source != 1 || !result.reused_prompt_tokens))
            throw std::runtime_error("no RAM restore occurred");
        std::cout << Json{{"mode", mode}, {"stage", "continuation"}, {"metrics", report}}.dump() << std::endl;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
