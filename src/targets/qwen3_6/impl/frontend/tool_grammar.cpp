#include <ninfer/targets/qwen3_6/tool_grammar.h>

#include <nlohmann/json.hpp>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ninfer::targets::qwen3_6 {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::string_view kToolOpen = "<tool_call>";

Json function_object(const Json& tool) {
    if (!tool.is_object()) { throw std::invalid_argument("tool definition must be an object"); }
    if (tool.contains("function")) {
        if (!tool.at("function").is_object()) {
            throw std::invalid_argument("tool function definition must be an object");
        }
        return tool.at("function");
    }
    return tool;
}

Json tool_tag(const Json& function) {
    if (!function.contains("name") || !function.at("name").is_string() ||
        function.at("name").get_ref<const std::string&>().empty()) {
        throw std::invalid_argument("tool function name must be a non-empty string");
    }
    const std::string& name = function.at("name").get_ref<const std::string&>();
    if (function.contains("strict") && !function.at("strict").is_boolean()) {
        throw std::invalid_argument("tool function strict must be boolean");
    }
    const bool strict = function.value("strict", false);
    Json schema{{"type", "object"}, {"properties", Json::object()}, {"additionalProperties", true}};
    if (strict) {
        schema = function.value("parameters", Json::object());
        if (!schema.is_object() && !schema.is_boolean()) {
            throw std::invalid_argument("strict tool parameters must be a JSON Schema object");
        }
    } else if (function.contains("parameters") && function.at("parameters").is_object() &&
               function.at("parameters").contains("properties") &&
               function.at("parameters").at("properties").is_object()) {
        for (const auto& [key, value] : function.at("parameters").at("properties").items()) {
            (void)value;
            schema["properties"][key] = Json::object();
        }
    }
    return Json{{"type", "tag"},
                {"begin", "<tool_call>\n<function=" + name + ">\n"},
                {"content", Json{{"type", "json_schema"},
                                 {"json_schema", std::move(schema)},
                                 {"style", "qwen_xml"},
                                 {"any_order", !strict}}},
                {"end", "\n</function>\n</tool_call>"}};
}

std::string structural_tag_json(const std::vector<std::string>& tool_jsons, bool require_tool_call,
                                bool starts_in_reasoning) {
    if (tool_jsons.empty()) {
        throw std::invalid_argument("tool grammar requires at least one tool");
    }

    Json tags = Json::array();
    std::unordered_set<std::string> names;
    for (const std::string& source : tool_jsons) {
        Json tool              = Json::parse(source);
        Json function          = function_object(tool);
        const std::string name = function.value("name", std::string{});
        if (!names.insert(name).second) {
            throw std::invalid_argument("duplicate tool function name: " + name);
        }
        tags.push_back(tool_tag(function));
    }

    Json calls{{"type", "tags_with_separator"},
               {"tags", std::move(tags)},
               {"separator", "\n"},
               {"at_least_one", true}};
    Json prefix{{"type", "any_text"}, {"excludes", Json::array({kToolOpen})}};
    Json with_calls{{"type", "sequence"},
                    {"elements", Json::array({std::move(prefix), std::move(calls)})}};
    Json suffix =
        require_tool_call
            ? std::move(with_calls)
            : Json{{"type", "or"},
                   {"elements",
                    Json::array({Json{{"type", "any_text"}, {"excludes", Json::array({kToolOpen})}},
                                 std::move(with_calls)})}};
    Json format = std::move(suffix);
    if (starts_in_reasoning) {
        format = Json{
            {"type", "sequence"},
            {"elements",
             Json::array(
                 {Json{{"type", "any_text"}, {"excludes", Json::array({"</think>", kToolOpen})}},
                  Json{{"type", "const_string"}, {"value", "</think>\n\n"}}, std::move(format)})}};
    }
    return Json{{"type", "structural_tag"}, {"format", std::move(format)}}.dump();
}

DLTensor cpu_bitmask_tensor(std::int32_t* data, std::int64_t rows, std::int64_t words,
                            std::int64_t* shape) {
    shape[0] = rows;
    shape[1] = words;
    return DLTensor{.data        = data,
                    .device      = DLDevice{kDLCPU, 0},
                    .ndim        = 2,
                    .dtype       = xgrammar::GetBitmaskDLType(),
                    .shape       = shape,
                    .strides     = nullptr,
                    .byte_offset = 0};
}

} // namespace

class ToolGrammarPlan::Impl {
public:
    explicit Impl(xgrammar::CompiledGrammar grammar)
        : compiled(std::move(grammar)), words(static_cast<std::size_t>(xgrammar::GetBitmaskSize(
                                            compiled.GetTokenizerInfo().GetVocabSize()))) {}

    xgrammar::CompiledGrammar compiled;
    std::size_t words = 0;
};

ToolGrammarPlan::ToolGrammarPlan(std::shared_ptr<const Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ToolGrammarPlan::ToolGrammarPlan(const ToolGrammarPlan&)                = default;
ToolGrammarPlan& ToolGrammarPlan::operator=(const ToolGrammarPlan&)     = default;
ToolGrammarPlan::ToolGrammarPlan(ToolGrammarPlan&&) noexcept            = default;
ToolGrammarPlan& ToolGrammarPlan::operator=(ToolGrammarPlan&&) noexcept = default;
ToolGrammarPlan::~ToolGrammarPlan()                                     = default;

std::size_t ToolGrammarPlan::bitmask_words() const noexcept { return impl_ ? impl_->words : 0; }

class ToolGrammarCompiler::Impl {
public:
    Impl(std::vector<std::string> encoded_vocabulary, std::vector<std::int32_t> stop_token_ids)
        : tokenizer(encoded_vocabulary, xgrammar::VocabType::BYTE_LEVEL,
                    static_cast<int>(encoded_vocabulary.size()), std::move(stop_token_ids), false),
          compiler(tokenizer, 8, true, 256LL << 20) {}

    xgrammar::TokenizerInfo tokenizer;
    mutable xgrammar::GrammarCompiler compiler;
    mutable std::mutex mutex;
};

ToolGrammarCompiler::ToolGrammarCompiler(std::vector<std::string> encoded_vocabulary,
                                         std::vector<std::int32_t> stop_token_ids)
    : impl_(std::make_unique<Impl>(std::move(encoded_vocabulary), std::move(stop_token_ids))) {}

ToolGrammarCompiler::ToolGrammarCompiler(ToolGrammarCompiler&&) noexcept            = default;
ToolGrammarCompiler& ToolGrammarCompiler::operator=(ToolGrammarCompiler&&) noexcept = default;
ToolGrammarCompiler::~ToolGrammarCompiler()                                         = default;

std::shared_ptr<const ToolGrammarPlan>
ToolGrammarCompiler::compile(const std::vector<std::string>& tool_jsons, bool require_tool_call,
                             bool starts_in_reasoning) const {
    try {
        const std::string source =
            structural_tag_json(tool_jsons, require_tool_call, starts_in_reasoning);
        std::lock_guard lock(impl_->mutex);
        auto compiled = impl_->compiler.CompileStructuralTag(source);
        return std::shared_ptr<const ToolGrammarPlan>(new ToolGrammarPlan(
            std::make_shared<const ToolGrammarPlan::Impl>(std::move(compiled))));
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(std::string("invalid tool definition for grammar: ") +
                                    error.what());
    } catch (const xgrammar::XGrammarError& error) {
        throw std::invalid_argument(std::string("tool grammar compilation failed: ") +
                                    error.what());
    }
}

class ToolGrammarState::Impl {
public:
    explicit Impl(std::shared_ptr<const ToolGrammarPlan> grammar_plan)
        : plan(std::move(grammar_plan)), matcher(plan->impl_->compiled) {}

    std::shared_ptr<const ToolGrammarPlan> plan;
    xgrammar::GrammarMatcher matcher;
};

ToolGrammarState::ToolGrammarState() noexcept = default;

ToolGrammarState::ToolGrammarState(std::shared_ptr<const ToolGrammarPlan> plan) {
    if (!plan || !plan->impl_) { throw std::invalid_argument("tool grammar plan is empty"); }
    impl_ = std::make_unique<Impl>(std::move(plan));
}

ToolGrammarState::ToolGrammarState(ToolGrammarState&&) noexcept            = default;
ToolGrammarState& ToolGrammarState::operator=(ToolGrammarState&&) noexcept = default;
ToolGrammarState::~ToolGrammarState()                                      = default;

ToolGrammarState::operator bool() const noexcept { return impl_ != nullptr; }

std::size_t ToolGrammarState::bitmask_words() const noexcept {
    return impl_ ? impl_->plan->bitmask_words() : 0;
}

bool ToolGrammarState::fill_draft_masks(std::span<std::int32_t> output, std::uint32_t mask_rows,
                                        std::span<const TokenId> drafts) {
    if (!impl_) { throw std::logic_error("tool grammar state is empty"); }
    const std::size_t words = bitmask_words();
    if (mask_rows == 0 || drafts.size() + 1 > mask_rows ||
        output.size() != words * static_cast<std::size_t>(mask_rows) ||
        words > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument("tool grammar mask storage has an invalid shape");
    }
    std::fill(output.begin(), output.end(), -1);
    std::int64_t shape[2]{};
    DLTensor tensor =
        cpu_bitmask_tensor(output.data(), mask_rows, static_cast<std::int64_t>(words), shape);

    bool constrained     = false;
    std::size_t advanced = 0;
    try {
        for (std::size_t column = 0; column <= drafts.size(); ++column) {
            constrained = impl_->matcher.FillNextTokenBitmask(&tensor, static_cast<int>(column)) ||
                          constrained;
            if (column == drafts.size() || !impl_->matcher.AcceptToken(drafts[column])) { break; }
            ++advanced;
        }
    } catch (...) {
        if (advanced != 0) { impl_->matcher.Rollback(static_cast<int>(advanced)); }
        throw;
    }
    if (advanced != 0) { impl_->matcher.Rollback(static_cast<int>(advanced)); }
    return constrained;
}

void ToolGrammarState::accept(std::span<const TokenId> tokens) {
    if (!impl_) { return; }
    for (const TokenId token : tokens) {
        if (!impl_->matcher.AcceptToken(token)) {
            throw std::logic_error("generated token " + std::to_string(token) +
                                   " violates the active tool grammar");
        }
    }
}

bool ToolGrammarState::completed() const { return impl_ && impl_->matcher.IsCompleted(); }

} // namespace ninfer::targets::qwen3_6
