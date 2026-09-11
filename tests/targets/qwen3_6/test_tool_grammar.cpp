#include <ninfer/targets/qwen3_6/tool_grammar.h>

#include "artifact/reader.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr ninfer::TokenId kPlain             = 0;
constexpr ninfer::TokenId kToolPrefix        = 1;
constexpr ninfer::TokenId kMalformedFunction = 2;
constexpr ninfer::TokenId kFunctionOpen      = 3;
constexpr ninfer::TokenId kStop              = 4;
constexpr ninfer::TokenId kParameter         = 5;
constexpr ninfer::TokenId kToolClose         = 6;
constexpr ninfer::TokenId kAtomicFunction    = 7;
constexpr ninfer::TokenId kThinkClose        = 8;
constexpr ninfer::TokenId kUnknownParameter  = 9;
constexpr ninfer::TokenId kCallSeparator     = 10;

int check(bool condition, const char* message) {
    if (condition) return 0;
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

bool allows(std::span<const std::int32_t> mask, ninfer::TokenId token) {
    return (static_cast<std::uint32_t>(mask[static_cast<std::size_t>(token / 32)]) &
            (std::uint32_t{1} << static_cast<unsigned int>(token % 32))) != 0;
}

std::vector<std::string> vocabulary() {
    return {"plain",
            "<tool_call>\n",
            "function=edit>\n",
            "<function=edit>\n",
            "",
            "<parameter=file_path>/a</parameter>",
            "\n</function>\n</tool_call>",
            "<function=edit>\n<parameter=file_path>\n/a\n</parameter>\n"
            "</function>\n</tool_call>",
            "</think>\n\n",
            "<parameter=unknown>/a</parameter>",
            "\n"};
}

std::vector<std::string> tools() {
    return {
        R"({"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]},"strict":false}})"};
}

std::vector<std::string> strict_tools() {
    return {
        R"({"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"],"additionalProperties":false},"strict":true}})"};
}

std::string artifact_resource(const ninfer::artifact::Reader& reader, std::string_view name) {
    const auto payload = reader.payload(name);
    return std::string(reinterpret_cast<const char*>(payload.data.data()), payload.data.size());
}

int test_parallel_draft_masks() {
    ninfer::targets::qwen3_6::ToolGrammarCompiler compiler(vocabulary(), {kStop});
    ninfer::targets::qwen3_6::ToolGrammarState state(compiler.compile(tools(), false));
    const std::size_t words = state.bitmask_words();
    std::vector<std::int32_t> masks(words * 3);
    const std::array<ninfer::TokenId, 2> drafts{kToolPrefix, kMalformedFunction};

    int failures = check(state.fill_draft_masks(masks, 3, drafts),
                         "tool prefix did not activate a grammar mask");
    failures += check(allows(std::span<const std::int32_t>(masks).subspan(0, words), kToolPrefix),
                      "valid tool-call prefix was rejected");
    failures += check(
        !allows(std::span<const std::int32_t>(masks).subspan(words, words), kMalformedFunction),
        "missing '<' function token was accepted");
    failures +=
        check(allows(std::span<const std::int32_t>(masks).subspan(words, words), kFunctionOpen),
              "valid function token was rejected");
    failures +=
        check(allows(std::span<const std::int32_t>(masks).subspan(words, words), kAtomicFunction),
              "valid atomic function token was rejected");
    failures += check(
        allows(std::span<const std::int32_t>(masks).subspan(2 * words, words), kMalformedFunction),
        "unused speculative mask row was not all-allow");

    state.accept(
        std::array<ninfer::TokenId, 4>{kToolPrefix, kFunctionOpen, kParameter, kToolClose});
    failures += check(state.completed(), "complete tool call did not complete the grammar");
    state.accept(std::array<ninfer::TokenId, 2>{kCallSeparator, kToolPrefix});
    std::vector<std::int32_t> second_mask(words);
    failures += check(state.fill_draft_masks(second_mask, 1, {}),
                      "second parallel tool call did not activate a grammar mask");
    failures += check(!allows(second_mask, kMalformedFunction),
                      "second parallel call accepted the missing '<' function token");
    failures += check(allows(second_mask, kFunctionOpen),
                      "second parallel call rejected the valid function token");
    return failures;
}

int test_required_call_blocks_early_stop() {
    ninfer::targets::qwen3_6::ToolGrammarCompiler compiler(vocabulary(), {kStop});
    ninfer::targets::qwen3_6::ToolGrammarState state(compiler.compile(tools(), true));
    const std::size_t words = state.bitmask_words();
    std::vector<std::int32_t> mask(words);

    int failures = check(state.fill_draft_masks(mask, 1, {}),
                         "required tool grammar did not constrain its initial state");
    failures += check(!allows(mask, kStop), "required tool grammar allowed an early stop");
    failures += check(allows(mask, kPlain), "required tool grammar rejected reasoning text");
    state.accept(std::array<ninfer::TokenId, 1>{kPlain});
    failures += check(state.fill_draft_masks(mask, 1, {}),
                      "required grammar stopped constraining after reasoning text");
    failures += check(!allows(mask, kStop), "required grammar allowed stop before a tool call");
    state.accept(
        std::array<ninfer::TokenId, 4>{kToolPrefix, kFunctionOpen, kParameter, kToolClose});
    failures += check(state.completed(), "required tool call did not complete the grammar");
    return failures;
}

int test_thinking_must_close_before_tool_call() {
    ninfer::targets::qwen3_6::ToolGrammarCompiler compiler(vocabulary(), {kStop});
    ninfer::targets::qwen3_6::ToolGrammarState state(compiler.compile(tools(), false, true));
    const std::size_t words = state.bitmask_words();
    std::vector<std::int32_t> mask(words);

    int failures = check(state.fill_draft_masks(mask, 1, {}),
                         "thinking tool grammar did not constrain its initial state");
    failures += check(!allows(mask, kToolPrefix), "tool call was allowed inside reasoning");
    failures += check(!allows(mask, kStop), "EOS was allowed before reasoning closed");
    failures += check(allows(mask, kPlain), "reasoning text was rejected");
    state.accept(std::array<ninfer::TokenId, 2>{kPlain, kThinkClose});
    (void)state.fill_draft_masks(mask, 1, {});
    failures += check(allows(mask, kToolPrefix), "tool call was rejected after reasoning closed");
    failures += check(allows(mask, kStop), "ordinary text response could not stop after reasoning");
    return failures;
}

int test_stop_draft_terminates_preview_and_rolls_back() {
    ninfer::targets::qwen3_6::ToolGrammarCompiler compiler(vocabulary(), {kStop});
    ninfer::targets::qwen3_6::ToolGrammarState state(compiler.compile(tools(), false, true));
    state.accept(std::array<ninfer::TokenId, 2>{kPlain, kThinkClose});
    const std::size_t words = state.bitmask_words();
    std::vector<std::int32_t> masks(words * 3);
    const std::array<ninfer::TokenId, 2> drafts{kStop, kPlain};

    (void)state.fill_draft_masks(masks, 3, drafts);
    int failures = check(allows(std::span<const std::int32_t>(masks).subspan(0, words), kStop),
                         "completed grammar rejected its stop token");
    failures += check(allows(std::span<const std::int32_t>(masks).subspan(words, words), kPlain),
                      "unused row after a stop draft was not all-allow");

    state.accept(std::array<ninfer::TokenId, 1>{kPlain});
    std::vector<std::int32_t> current(words);
    (void)state.fill_draft_masks(current, 1, {});
    failures += check(allows(current, kStop), "rollback no longer permits the stop token");
    return failures;
}

int test_strict_schema_constrains_parameters() {
    ninfer::targets::qwen3_6::ToolGrammarCompiler compiler(vocabulary(), {kStop});
    ninfer::targets::qwen3_6::ToolGrammarState strict(compiler.compile(strict_tools(), false));
    strict.accept(std::array<ninfer::TokenId, 2>{kToolPrefix, kFunctionOpen});
    const std::size_t words = strict.bitmask_words();
    std::vector<std::int32_t> mask(words);
    int failures = check(strict.fill_draft_masks(mask, 1, {}),
                         "strict tool schema did not constrain parameters");
    failures += check(allows(mask, kParameter), "strict schema rejected its declared parameter");
    failures +=
        check(!allows(mask, kUnknownParameter), "strict schema accepted an undeclared parameter");

    ninfer::targets::qwen3_6::ToolGrammarState open(compiler.compile(tools(), false));
    open.accept(std::array<ninfer::TokenId, 2>{kToolPrefix, kFunctionOpen});
    (void)open.fill_draft_masks(mask, 1, {});
    failures += check(allows(mask, kUnknownParameter),
                      "non-strict structural grammar rejected an open parameter");
    return failures;
}

std::vector<std::string> read_tool_workload(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) { throw std::runtime_error("failed to open tool workload: " + path.string()); }
    const nlohmann::json tools = nlohmann::json::parse(stream);
    if (!tools.is_array() || tools.empty()) {
        throw std::invalid_argument("tool workload must be a non-empty JSON array");
    }
    std::vector<std::string> result;
    result.reserve(tools.size());
    for (const auto& tool : tools) { result.push_back(tool.dump()); }
    return result;
}

int test_real_qwen_tokenizer(const std::filesystem::path& artifact_path,
                             const std::filesystem::path* tool_workload_path) {
    ninfer::artifact::Reader reader(artifact_path);
    namespace fi = ninfer::targets::qwen3_6::frontend_internal;
    const fi::Tokenizer tokenizer(
        {.tokenizer_json         = artifact_resource(reader, "frontend/tokenizer.json"),
         .tokenizer_config_json  = artifact_resource(reader, "frontend/tokenizer_config.json"),
         .generation_config_json = artifact_resource(reader, "frontend/generation_config.json")});
    std::vector<std::int32_t> stop_tokens(tokenizer.default_stop_token_ids().begin(),
                                          tokenizer.default_stop_token_ids().end());
    ninfer::targets::qwen3_6::ToolGrammarCompiler compiler(tokenizer.grammar_vocabulary(),
                                                           std::move(stop_tokens));
    const std::vector<std::string> real_tools = {
        R"({"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"patch":{"type":"string"}},"required":["patch"],"additionalProperties":false},"strict":true}})"};
    ninfer::targets::qwen3_6::ToolGrammarState state(compiler.compile(real_tools, true));
    int failures           = check(state.bitmask_words() == (248077U + 31U) / 32U,
                                   "real Qwen tokenizer produced the wrong grammar mask width");
    const std::string call = "<tool_call>\n<function=edit>\n<parameter=patch>\nline 1\nline 2\n"
                             "</parameter>\n</function>\n</tool_call>";
    const std::vector<int> encoded = tokenizer.encode(call);
    state.accept(std::span<const int>(encoded));
    failures +=
        check(state.completed(), "real Qwen tokenizer rejected a multiline strict tool argument");
    if (tool_workload_path != nullptr) {
        const std::vector<std::string> workload = read_tool_workload(*tool_workload_path);
        const auto cold_started                 = std::chrono::steady_clock::now();
        const auto cold_plan                    = compiler.compile(workload, false, true);
        const auto cold_finished                = std::chrono::steady_clock::now();
        const auto cached_plan                  = compiler.compile(workload, false, true);
        const auto cached_finished              = std::chrono::steady_clock::now();
        const double cold_ms =
            std::chrono::duration<double, std::milli>(cold_finished - cold_started).count();
        const double cached_ms =
            std::chrono::duration<double, std::milli>(cached_finished - cold_finished).count();
        failures += check(cold_plan->bitmask_words() == state.bitmask_words() &&
                              cached_plan->bitmask_words() == state.bitmask_words(),
                          "real tool workload produced an invalid mask width");
        std::cout << "real-tool grammar: " << workload.size() << " tools, cold=" << cold_ms
                  << " ms, cached=" << cached_ms << " ms\n";
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    int failures = 0;
    failures += test_parallel_draft_masks();
    failures += test_required_call_blocks_early_stop();
    failures += test_thinking_must_close_before_tool_call();
    failures += test_stop_draft_terminates_preview_and_rolls_back();
    failures += test_strict_schema_constrains_parameters();
    if (argc == 2 || argc == 3) {
        const std::filesystem::path workload = argc == 3 ? argv[2] : std::filesystem::path{};
        failures += test_real_qwen_tokenizer(argv[1], argc == 3 ? &workload : nullptr);
    } else if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [model.ninfer [tools.json]]\n";
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}
