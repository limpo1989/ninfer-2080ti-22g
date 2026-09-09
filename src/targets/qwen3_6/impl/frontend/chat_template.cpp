#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include "targets/qwen3_6/impl/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr Sha256Digest kThinkingToggleTemplateDigest{
    0xe8, 0x4f, 0x32, 0xa2, 0x3f, 0xdd, 0xa2, 0x76, 0x89, 0xf8, 0x68, 0xaa, 0x4a, 0x1a, 0x56, 0x21,
    0xf4, 0x11, 0x33, 0xe5, 0x1a, 0x48, 0xd7, 0xf3, 0xef, 0xcb, 0xea, 0x28, 0x39, 0x57, 0x42, 0x59,
};

constexpr Sha256Digest kReasoningEffortTemplateDigest{
    0xc3, 0xcf, 0x9e, 0x34, 0xab, 0xf4, 0xf9, 0xe3, 0x6c, 0x2d, 0x72, 0x16, 0x5a, 0xa9, 0xc1, 0x32,
    0xd3, 0xe2, 0xa7, 0x25, 0xb6, 0xc2, 0x58, 0x6a, 0xaa, 0x3a, 0x8a, 0xf9, 0xd7, 0xa8, 0x10, 0x41,
};

constexpr std::string_view kLowReasoningInstructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
    "the conclusion without unnecessary elaboration.";

constexpr std::string_view kXHighReasoningInstructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";

bool is_instruction_role(ChatRole role) noexcept {
    return role == ChatRole::System || role == ChatRole::Developer;
}

void validate_instruction_message(const ChatMessage& message) {
    if (message.has_media()) {
        throw std::invalid_argument(
            "system and developer messages cannot contain images or videos");
    }
    if (!message.reasoning_content.empty() || !message.tool_calls.empty() ||
        !message.tool_call_id.empty()) {
        throw std::invalid_argument("system and developer messages may contain only text content");
    }
}

std::string trim_ascii_whitespace(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return text.substr(begin, end - begin);
}

bool starts_with(const std::string& text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

long last_real_user_query(const std::vector<ChatMessage>& messages) {
    for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i) {
        const ChatMessage& message = messages[static_cast<std::size_t>(i)];
        if (message.role != ChatRole::User) { continue; }
        const std::string content = trim_ascii_whitespace(message.rendered_content());
        if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) {
            return i;
        }
    }
    throw std::invalid_argument("no user query found in chat messages");
}

std::string lstrip_newlines(std::string text) {
    std::size_t begin = 0;
    while (begin < text.size() && text[begin] == '\n') { ++begin; }
    return text.substr(begin);
}

std::string rstrip_newlines(std::string text) {
    std::size_t end = text.size();
    while (end > 0 && text[end - 1] == '\n') { --end; }
    return text.substr(0, end);
}

// Split an assistant turn into (reasoning, content) exactly as the Qwen3.6 jinja
// does when reasoning_content is not provided: reasoning is the text between the
// last <think> and the first </think>; content is everything after the last
// </think>. When there is no </think> the whole thing is content and reasoning is
// empty.
struct ThinkParts {
    std::string reasoning;
    std::string content;
};

ThinkParts derive_think_parts(const std::string& content) {
    ThinkParts parts;
    const std::size_t first_close = content.find("</think>");
    if (first_close == std::string::npos) {
        parts.content = content;
        return parts;
    }
    // reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
    std::string before          = rstrip_newlines(content.substr(0, first_close));
    const std::size_t last_open = before.rfind("<think>");
    std::string reasoning       = (last_open == std::string::npos)
                                      ? before
                                      : before.substr(last_open + std::string("<think>").size());
    parts.reasoning             = lstrip_newlines(std::move(reasoning));
    // content = content.split('</think>')[-1].lstrip('\n')
    const std::size_t last_close = content.rfind("</think>");
    parts.content = lstrip_newlines(content.substr(last_close + std::string("</think>").size()));
    return parts;
}

constexpr std::string_view kToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

std::string tojson_text(const OrderedJson& value) {
    if (value.is_array()) {
        std::string rendered = "[";
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += tojson_text(value[index]);
        }
        rendered += "]";
        return rendered;
    }
    if (value.is_object()) {
        std::string rendered = "{";
        std::size_t index    = 0;
        for (auto it = value.begin(); it != value.end(); ++it, ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += OrderedJson(it.key()).dump();
            rendered += ": ";
            rendered += tojson_text(it.value());
        }
        rendered += "}";
        return rendered;
    }
    return value.dump();
}

std::string parameter_text(const OrderedJson& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    return tojson_text(value);
}

std::string render_tool_call(const ToolCall& call, bool allow_empty_arguments) {
    if (allow_empty_arguments && call.arguments_json.empty()) {
        return "<tool_call>\n<function=" + call.name + ">\n</function>\n</tool_call>";
    }
    OrderedJson args = OrderedJson::parse(call.arguments_json);
    if (!args.is_object()) {
        throw std::invalid_argument("tool call arguments must be a JSON object");
    }

    std::string rendered;
    rendered += "<tool_call>\n<function=";
    rendered += call.name;
    rendered += ">\n";
    for (auto it = args.begin(); it != args.end(); ++it) {
        rendered += "<parameter=";
        rendered += it.key();
        rendered += ">\n";
        rendered += parameter_text(it.value());
        rendered += "\n</parameter>\n";
    }
    rendered += "</function>\n</tool_call>";
    return rendered;
}

std::string render_tools_system_block(const std::vector<std::string>& tool_jsons,
                                      const std::string& leading_instruction,
                                      std::string_view reasoning_instructions) {
    std::string rendered;
    rendered += "<|im_start|>system\n";
    if (!reasoning_instructions.empty()) {
        rendered += reasoning_instructions;
        rendered += "\n\n";
    }
    rendered += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const std::string& tool : tool_jsons) {
        rendered += "\n";
        rendered += tojson_text(OrderedJson::parse(tool));
    }
    rendered += "\n</tools>";
    rendered += std::string(kToolInstructions);
    if (!leading_instruction.empty()) {
        rendered += "\n\n";
        rendered += leading_instruction;
    }
    rendered += "<|im_end|>\n";
    return rendered;
}

// Emit a non-tools system block: <|im_start|>system\n[reasoning][\n\n][leading]<|im_end|>\n
void emit_system_block(std::string& out, std::string_view reasoning_instructions,
                       const std::string& leading_instruction) {
    out += "<|im_start|>system\n";
    if (!reasoning_instructions.empty()) {
        out += reasoning_instructions;
        if (!leading_instruction.empty()) { out += "\n\n"; }
    }
    out += leading_instruction;
    out += "<|im_end|>\n";
}

// Sharp v22.1 overlay: appends a terse steering instruction to the effective
// system content. Applied at the leading-instruction resolution point so it
// covers both the no-tools system block and the tools system block.
static std::string apply_sharp_v22_1_system_overlay(std::string leading_instruction) {
    // Verbatim from Sharp v22.1 chat_template.jinja (_terse block, lines 154-158).
    constexpr std::string_view kSharpTerseInstruction =
        "Answer directly, after thinking. Lead with the answer, then only what it needs to be "
        "correct and usable.\n"
        "Never: open with preamble or pleasantries; restate the question; add filler transitions; "
        "hedge with niceties; or repeat a point you've already made.\n"
        "Always: keep essential steps, caveats, uncertainties, and specifics \u2014 never drop "
        "correctness or a needed warning for brevity. Keep the final answer lean. Use the least "
        "structure that conveys it (plain prose when short; lists or code only when they earn their "
        "place). If genuinely uncertain, say so and explain why \u2014 never omit uncertainty for the "
        "sake of brevity.\n"
        "If a user request is genuinely ambiguous, ask a sharp question, don't guess.";
    if (!leading_instruction.empty()) { leading_instruction += "\n\n"; }
    leading_instruction += std::string(kSharpTerseInstruction);
    return leading_instruction;
}

std::string_view resolve_default_reasoning_instructions(const ChatRenderOptions& options) {
    if (!options.enable_thinking) {
        if (options.reasoning_effort) {
            throw std::invalid_argument(
                "reasoning effort cannot be combined with disabled thinking");
        }
        return {};
    }
    switch (options.reasoning_effort.value_or(ReasoningEffort::XHigh)) {
    case ReasoningEffort::Low:
        return kLowReasoningInstructions;
    case ReasoningEffort::Medium:
        return {};
    case ReasoningEffort::XHigh:
        return kXHighReasoningInstructions;
    }
    throw std::invalid_argument("invalid reasoning effort");
}

std::string_view resolve_sharp_reasoning_instructions(const ChatRenderOptions& options) {
    // Sharp v22.1 silently ignores reasoning effort when thinking is disabled.
    if (!options.enable_thinking) { return {}; }
    switch (options.reasoning_effort.value_or(ReasoningEffort::Medium)) {
    case ReasoningEffort::Medium:
    case ReasoningEffort::None:
        return {};
    case ReasoningEffort::Minimal:
    case ReasoningEffort::Low:
        return kLowReasoningInstructions;
    case ReasoningEffort::High:
    case ReasoningEffort::XHigh:
    case ReasoningEffort::Max:
        return kXHighReasoningInstructions;
    }
    throw std::invalid_argument("invalid reasoning effort");
}


// --- interpreted-template support -----------------------------------------

namespace jj = jinja;

std::string_view reasoning_effort_name(ReasoningEffort effort) {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    throw std::invalid_argument("invalid reasoning effort");
}

std::string_view chat_role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
        return "system";
    case ChatRole::Developer:
        return "developer";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("unsupported chat role value");
}

jj::Value json_to_jinja(const OrderedJson& node) {
    if (node.is_null()) { return jj::Value::none(); }
    if (node.is_boolean()) { return jj::Value::boolean(node.get<bool>()); }
    if (node.is_number_integer() || node.is_number_unsigned()) {
        return jj::Value::integer(node.get<std::int64_t>());
    }
    if (node.is_number_float()) { return jj::Value::number(node.get<double>()); }
    if (node.is_string()) { return jj::Value::string(node.get<std::string>()); }
    if (node.is_array()) {
        jj::Array items;
        items.reserve(node.size());
        for (const OrderedJson& item : node) { items.push_back(json_to_jinja(item)); }
        return jj::Value::array(std::move(items));
    }
    jj::Object entries;
    for (auto it = node.begin(); it != node.end(); ++it) {
        entries.emplace_back(it.key(), json_to_jinja(it.value()));
    }
    return jj::Value::object(std::move(entries));
}

// Message content is a plain string unless the turn carries media, in which
// case it becomes the list-of-parts form that templates branch on.
jj::Value message_content_value(const ChatMessage& message) {
    if (!message.has_media()) { return jj::Value::string(message.rendered_content()); }
    jj::Array parts;
    for (const ChatPart& part : message.parts) {
        jj::Object entry;
        switch (part.kind) {
        case ChatPartKind::Text:
            entry.emplace_back("type", jj::Value::string("text"));
            entry.emplace_back("text", jj::Value::string(part.text));
            break;
        case ChatPartKind::Image:
            entry.emplace_back("type", jj::Value::string("image"));
            break;
        case ChatPartKind::Video:
            entry.emplace_back("type", jj::Value::string("video"));
            break;
        }
        parts.push_back(jj::Value::object(std::move(entry)));
    }
    return jj::Value::array(std::move(parts));
}

jj::Value tool_calls_value(const std::vector<ToolCall>& calls) {
    jj::Array rendered;
    for (const ToolCall& call : calls) {
        jj::Object function;
        function.emplace_back("name", jj::Value::string(call.name));
        // Templates expect a mapping so they can iterate `.items()`; fall back to
        // the raw string when the payload is not a JSON object.
        jj::Value arguments = jj::Value::string(call.arguments_json);
        if (!call.arguments_json.empty()) {
            try {
                const OrderedJson parsed = OrderedJson::parse(call.arguments_json);
                if (parsed.is_object()) { arguments = json_to_jinja(parsed); }
            } catch (const OrderedJson::exception&) {
                // Keep the string form.
            }
        }
        function.emplace_back("arguments", std::move(arguments));

        jj::Object entry;
        if (!call.id.empty()) { entry.emplace_back("id", jj::Value::string(call.id)); }
        entry.emplace_back("type", jj::Value::string("function"));
        entry.emplace_back("function", jj::Value::object(std::move(function)));
        rendered.push_back(jj::Value::object(std::move(entry)));
    }
    return jj::Value::array(std::move(rendered));
}

jj::Value build_render_context(const std::vector<ChatMessage>& messages,
                               const ChatRenderOptions& options) {
    jj::Array rendered_messages;
    for (const ChatMessage& message : messages) {
        jj::Object entry;
        entry.emplace_back("role", jj::Value::string(std::string(chat_role_name(message.role))));
        entry.emplace_back("content", message_content_value(message));
        if (!message.reasoning_content.empty()) {
            entry.emplace_back("reasoning_content", jj::Value::string(message.reasoning_content));
        }
        if (!message.tool_calls.empty()) {
            entry.emplace_back("tool_calls", tool_calls_value(message.tool_calls));
        }
        if (!message.tool_call_id.empty()) {
            entry.emplace_back("tool_call_id", jj::Value::string(message.tool_call_id));
        }
        rendered_messages.push_back(jj::Value::object(std::move(entry)));
    }

    jj::Object context;
    context.emplace_back("messages", jj::Value::array(std::move(rendered_messages)));
    context.emplace_back("add_generation_prompt", jj::Value::boolean(options.add_generation_prompt));
    context.emplace_back("enable_thinking", jj::Value::boolean(options.enable_thinking));
    context.emplace_back("add_vision_id", jj::Value::boolean(options.add_vision_id));
    if (options.reasoning_effort) {
        context.emplace_back(
            "reasoning_effort",
            jj::Value::string(std::string(reasoning_effort_name(*options.reasoning_effort))));
    }
    if (options.preserve_thinking) {
        context.emplace_back("preserve_thinking", jj::Value::boolean(*options.preserve_thinking));
    }
    if (!options.tool_jsons.empty()) {
        jj::Array tools;
        for (const std::string& tool : options.tool_jsons) {
            tools.push_back(json_to_jinja(OrderedJson::parse(tool)));
        }
        context.emplace_back("tools", jj::Value::array(std::move(tools)));
    }
    return jj::Value::object(std::move(context));
}

// A custom template only honours an option if it actually reads the
// corresponding variable, so the advertised capabilities are derived from the
// names the parsed template references rather than assumed.
PromptCapabilities derive_capabilities(const jj::Template& program) {
    const std::vector<std::string>& globals = program.referenced_globals();
    const auto reads = [&](std::string_view name) {
        return std::find(globals.begin(), globals.end(), name) != globals.end();
    };

    PromptCapabilities result;
    result.enable_thinking = reads("enable_thinking");
    if (reads("reasoning_effort")) {
        result.reasoning_effort.low            = true;
        result.reasoning_effort.medium         = true;
        result.reasoning_effort.high           = true;
        result.reasoning_effort.xhigh          = true;
        result.reasoning_effort.default_effort = ReasoningEffort::Medium;
    }
    return result;
}

} // namespace

bool ChatMessage::has_media() const noexcept {
    for (const ChatPart& part : parts) {
        if (part.kind != ChatPartKind::Text) { return true; }
    }
    return false;
}

std::string ChatMessage::rendered_content(bool add_vision_id, int* image_count,
                                          int* video_count) const {
    int local_images = 0;
    int local_videos = 0;
    int& images      = image_count == nullptr ? local_images : *image_count;
    int& videos      = video_count == nullptr ? local_videos : *video_count;
    std::string out;
    for (const ChatPart& part : parts) {
        switch (part.kind) {
        case ChatPartKind::Text:
            out += part.text;
            break;
        case ChatPartKind::Image:
            ++images;
            if (add_vision_id) { out += "Picture " + std::to_string(images) + ": "; }
            out += "<|vision_start|><|image_pad|><|vision_end|>";
            break;
        case ChatPartKind::Video:
            ++videos;
            if (add_vision_id) { out += "Video " + std::to_string(videos) + ": "; }
            out += "<|vision_start|><|video_pad|><|vision_end|>";
            break;
        }
    }
    return out;
}

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source,
                                                    ChatStyle chat_style) {
    const Sha256Digest digest = sha256(source);
    if (digest == kThinkingToggleTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ThinkingToggle, chat_style);
    }
    if (digest == kReasoningEffortTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ReasoningEffort, chat_style);
    }
    // Not one of the transcribed templates: interpret it. This is the path a
    // user-supplied template takes.
    CompiledChatTemplate compiled(ChatTemplateSemantics::Interpreted, chat_style);
    try {
        compiled.program_ = std::make_shared<jinja::Template>(jinja::Template::parse(source));
    } catch (const jinja::Error& error) {
        throw std::invalid_argument("failed to parse chat template (sha256 " + sha256_hex(digest) +
                                    "): " + error.what());
    }
    compiled.interpreted_capabilities_ = derive_capabilities(*compiled.program_);
    return compiled;
}

PromptCapabilities CompiledChatTemplate::capabilities() const noexcept {
    if (semantics_ == ChatTemplateSemantics::Interpreted) { return interpreted_capabilities_; }
    PromptCapabilities result;
    result.enable_thinking = true;
    if (semantics_ == ChatTemplateSemantics::ReasoningEffort) {
        result.reasoning_effort.low            = true;
        result.reasoning_effort.medium         = true;
        result.reasoning_effort.xhigh          = true;
        result.reasoning_effort.default_effort = ReasoningEffort::XHigh;
        if (chat_style_ == ChatStyle::SharpV22_1) {
            result.reasoning_effort.high          = true;
            result.reasoning_effort.default_effort = ReasoningEffort::Medium;
        }
    }
    return result;
}

RenderedChat CompiledChatTemplate::render(const std::vector<ChatMessage>& messages,
                                          ChatRenderOptions options) const {
    if (messages.empty()) { throw std::invalid_argument("chat messages must not be empty"); }

    if (semantics_ == ChatTemplateSemantics::Interpreted) {
        return render_interpreted(messages, options);
    }

    const bool effort_template = semantics_ == ChatTemplateSemantics::ReasoningEffort;
    const std::string_view reasoning_instructions =
        chat_style_ == ChatStyle::SharpV22_1
            ? resolve_sharp_reasoning_instructions(options)
            : (effort_template ? resolve_default_reasoning_instructions(options)
                               : std::string_view{});

    std::size_t message_begin = 0;
    std::string leading_instruction;
    if (is_instruction_role(messages[0].role)) {
        validate_instruction_message(messages[0]);
        leading_instruction = trim_ascii_whitespace(messages[0].rendered_content());
        message_begin       = 1;
    }
    if (chat_style_ == ChatStyle::SharpV22_1) {
        leading_instruction = apply_sharp_v22_1_system_overlay(std::move(leading_instruction));
    }

    std::string rendered;
    const bool has_tools = !options.tool_jsons.empty();
    if (has_tools) {
        rendered += render_tools_system_block(options.tool_jsons, leading_instruction,
                                              reasoning_instructions);
    } else if (!reasoning_instructions.empty() || !leading_instruction.empty()) {
        // May seed a system block with terse overlay or reasoning instructions even when
        // no explicit system/developer message is present (Sharp v22.1 always emits one).
        emit_system_block(rendered, reasoning_instructions, leading_instruction);
    }

    const long last_query_index  = last_real_user_query(messages);
    const bool preserve_thinking = options.preserve_thinking.value_or(effort_template);
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;

    int image_count = 0;
    int video_count = 0;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (i < message_begin) { continue; }
        if (is_instruction_role(message.role)) { validate_instruction_message(message); }
        const std::string content = trim_ascii_whitespace(
            message.rendered_content(options.add_vision_id, &image_count, &video_count));
        if (is_instruction_role(message.role)) {
            rendered += "<|im_start|>system\n";
            rendered += content;
            rendered += "<|im_end|>\n";
            continue;
        }
        if (message.role == ChatRole::User) {
            rendered += "<|im_start|>user\n";
            rendered += content;
            rendered += "<|im_end|>\n";
            continue;
        }
        if (message.role == ChatRole::Tool) {
            const bool opens_group = i > 0 && messages[i - 1].role != ChatRole::Tool;
            const bool closes_group =
                i + 1 == messages.size() || messages[i + 1].role != ChatRole::Tool;
            if (opens_group) { rendered += "<|im_start|>user"; }
            rendered += "\n<tool_response>\n";
            rendered += content;
            rendered += "\n</tool_response>";
            if (closes_group) { rendered += "<|im_end|>\n"; }
            continue;
        }

        if (message.role != ChatRole::Assistant) {
            throw std::invalid_argument("unsupported chat role value");
        }

        // assistant
        std::string reasoning;
        std::string body = content;
        if (!message.reasoning_content.empty()) {
            reasoning = message.reasoning_content;
        } else if (!effort_template) {
            ThinkParts parts = derive_think_parts(content);
            reasoning        = std::move(parts.reasoning);
            body             = std::move(parts.content);
        }
        reasoning = trim_ascii_whitespace(reasoning);

        const bool keep_thinking = preserve_thinking || (static_cast<long>(i) > last_query_index);
        rendered += "<|im_start|>assistant\n";
        if (!preserve_thinking && !rewrite_checkpoint && static_cast<long>(i) > last_query_index) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = rendered.size()};
        }
        // Sharp v22.1 omits the thinking wrapper entirely for history turns that carry no
        // reasoning, matching its `{% if message.reasoning_content or message.reasoning %}` guard.
        // Stock NInfer keeps an empty wrapper; we preserve that for Default and only tighten it
        // under SharpV22_1 so history formatting is byte-faithful to Sharp.
        const bool is_history_turn = static_cast<long>(i) <= last_query_index;
        const bool emit_think = keep_thinking && !(chat_style_ == ChatStyle::SharpV22_1 &&
                                                   is_history_turn && reasoning.empty());
        if (emit_think) {
            rendered += "<think>\n";
            rendered += reasoning;
            rendered += "\n</think>\n\n";
        }
        rendered += body;
        if (!message.tool_calls.empty()) {
            const bool body_has_text = !trim_ascii_whitespace(body).empty();
            for (std::size_t call_index = 0; call_index < message.tool_calls.size(); ++call_index) {
                if (call_index == 0) {
                    if (body_has_text) { rendered += "\n\n"; }
                } else {
                    rendered += "\n";
                }
                rendered += render_tool_call(message.tool_calls[call_index], effort_template);
            }
        }
        rendered += "<|im_end|>\n";
    }

    if (options.add_generation_prompt) {
        rendered += "<|im_start|>assistant\n";
        if (!preserve_thinking && !rewrite_checkpoint) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = rendered.size()};
        }
        if (options.enable_thinking) {
            rendered += "<think>\n";
        } else {
            rendered += "<think>\n\n</think>\n\n";
        }
        if (preserve_thinking) {
            // Response replay retains the deterministic generation prologue. This is the prompt
            // frontier for both thinking modes, so capturing it does not split off a tiny final
            // prefill unit. The complete rendered prefix is tokenized independently below.
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::ResponseReplay, .offset = rendered.size()};
        }
    }
    return RenderedChat{.text = std::move(rendered), .rewrite_checkpoint = rewrite_checkpoint};
}

RenderedChat CompiledChatTemplate::render_interpreted(const std::vector<ChatMessage>& messages,
                                                      const ChatRenderOptions& options) const {
    std::string text;
    try {
        text = program_->render(build_render_context(messages, options));
    } catch (const jinja::Error& error) {
        throw std::invalid_argument(std::string("chat template failed to render: ") +
                                    error.what());
    }

    // The rendered prefix ends exactly where generation begins, so the response
    // replay frontier is the end of the text. This holds for any template and
    // matches what the transcribed renderers emit under preserve_thinking. The
    // TurnClosure checkpoint is deliberately not synthesised: locating the final
    // assistant turn would require assumptions about the template's markup, and
    // omitting it only forgoes a prefix-reuse optimisation.
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    if (options.add_generation_prompt) {
        rewrite_checkpoint = RewriteCheckpointByteSpec{
            .kind = RewriteCheckpointKind::ResponseReplay, .offset = text.size()};
    }
    return RenderedChat{.text = std::move(text), .rewrite_checkpoint = rewrite_checkpoint};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
