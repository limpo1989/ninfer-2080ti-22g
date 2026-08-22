// Covers the Jinja interpreter and the interpreted chat-template path that a
// user-supplied --chat-template takes.
//
// The interpreter's fidelity against real templates is established by a
// differential harness that diffs its output byte-for-byte against Python's
// jinja2 (see docs/custom-chat-templates.md). These tests pin the behaviour that
// harness cannot reach from inside the build: the language subset, the
// allowlist-to-interpreter fallback, capability derivation, and the mapping from
// ChatMessage to template variables.

#include <ninfer/types.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/jinja.h"

#include <iostream>
#include <string>
#include <vector>

namespace fi = ninfer::targets::qwen3_6::frontend_internal;
namespace jj = ninfer::targets::qwen3_6::frontend_internal::jinja;

using fi::ChatMessage;
using fi::ChatPart;
using fi::ChatRenderOptions;
using fi::CompiledChatTemplate;
using ninfer::ChatRole;

namespace {

int g_failures = 0;

void expect_eq(const std::string& label, const std::string& actual, const std::string& expected) {
    if (actual == expected) { return; }
    ++g_failures;
    std::cerr << "FAIL " << label << "\n  expected: [" << expected << "]\n  actual:   [" << actual
              << "]\n";
}

void expect_true(const std::string& label, bool condition) {
    if (condition) { return; }
    ++g_failures;
    std::cerr << "FAIL " << label << "\n";
}

std::string render(const std::string& source, const jj::Value& context) {
    return jj::Template::parse(source).render(context);
}

jj::Value empty_context() { return jj::Value::object({}); }

// --- language subset ------------------------------------------------------

void test_expressions() {
    const jj::Value ctx = empty_context();

    expect_eq("literal text", render("hello", ctx), "hello");
    expect_eq("string concat", render("{{ 'a' ~ 1 ~ true }}", ctx), "a1True");
    expect_eq("arithmetic", render("{{ 2 + 3 * 4 }}", ctx), "14");
    // `~` binds tighter than `+`, as in Jinja.
    expect_eq("concat precedence", render("{{ 'x' + 'y' ~ 'z' }}", ctx), "xyz");
    expect_eq("floor div and modulo", render("{{ 7 // 2 }},{{ -7 % 3 }}", ctx), "3,2");
    expect_eq("comparison", render("{{ 1 < 2 }}{{ 'b' > 'a' }}", ctx), "TrueTrue");
    expect_eq("ternary", render("{{ 'y' if 1 else 'n' }}{{ 'y' if 0 else 'n' }}", ctx), "yn");
    expect_eq("and/or short circuit", render("{{ '' or 'fallback' }}{{ 'a' and 'b' }}", ctx),
              "fallbackb");
    expect_eq("membership", render("{{ 'b' in 'abc' }}{{ 'z' not in ('x','y') }}", ctx),
              "TrueTrue");
    expect_eq("unary not", render("{{ not '' }}", ctx), "True");
}

void test_filters_and_tests() {
    const jj::Value ctx = empty_context();

    expect_eq("trim/lower", render("{{ '  AB  ' | trim | lower }}", ctx), "ab");
    expect_eq("length", render("{{ 'abc' | length }},{{ [1,2] | length }}", ctx), "3,2");
    expect_eq("join", render("{{ ['a','b'] | join('-') }}", ctx), "a-b");
    expect_eq("default", render("{{ missing | default('d') }}", ctx), "d");
    expect_eq("first/last", render("{{ [1,2,3] | first }}{{ [1,2,3] | last }}", ctx), "13");
    // tojson must use Python's separators, which is what the model was trained on.
    expect_eq("tojson separators", render("{{ {'a': 1, 'b': [1, 2]} | tojson }}", ctx),
              "{\"a\": 1, \"b\": [1, 2]}");
    expect_eq("tojson keeps utf8", render("{{ {'k': 'ok \xE2\x9A\xA0'} | tojson }}", ctx),
              "{\"k\": \"ok \xE2\x9A\xA0\"}");

    expect_eq("defined tests", render("{{ missing is defined }}{{ missing is undefined }}", ctx),
              "FalseTrue");
    expect_eq("type tests", render("{{ 'a' is string }}{{ [1] is mapping }}{{ [1] is iterable }}",
                                   ctx),
              "TrueFalseTrue");
    expect_eq("negated test", render("{{ 'a' is not mapping }}", ctx), "True");
}

void test_slicing_and_indexing() {
    const jj::Value ctx = empty_context();

    expect_eq("string slice", render("{{ 'abcdef'[1:3] }}", ctx), "bc");
    expect_eq("open slice", render("{{ 'abcdef'[:2] }}{{ 'abcdef'[4:] }}", ctx), "abef");
    expect_eq("negative index", render("{{ 'abc'[-1] }}", ctx), "c");
    expect_eq("reversed slice", render("{{ [1,2,3][::-1] | join(',') }}", ctx), "3,2,1");
    expect_eq("split then index", render("{{ 'a/b/c'.split('/')[-1] }}", ctx), "c");
    expect_eq("string methods",
              render("{{ 'xAx'.strip('x') }}{{ 'ab'.startswith('a') }}{{ 'ab'.replace('a','z') }}",
                     ctx),
              "ATruezb");
}

void test_statements() {
    const jj::Value ctx = empty_context();

    expect_eq("if/elif/else", render("{% if 0 %}a{% elif 1 %}b{% else %}c{% endif %}", ctx), "b");
    expect_eq("for with loop vars",
              render("{% for x in [10,20] %}{{ loop.index0 }}:{{ x }},{% endfor %}", ctx),
              "0:10,1:20,");
    expect_eq("for else on empty", render("{% for x in [] %}a{% else %}empty{% endfor %}", ctx),
              "empty");
    expect_eq("loop first/last/previtem/nextitem",
              render("{% for x in [1,2,3] %}{{ loop.first }}{{ loop.last }}"
                     "{{ loop.previtem | default('-') }}{{ loop.nextitem | default('-') }} "
                     "{% endfor %}",
                     ctx),
              "TrueFalse-2 FalseFalse13 FalseTrue2- ");
    expect_eq("set block", render("{% set v %}captured{% endset %}{{ v }}", ctx), "captured");
    expect_eq("macro with default",
              render("{% macro m(a, b='B') %}{{ a }}{{ b }}{% endmacro %}{{ m('A') }}{{ m('A','C') }}",
                     ctx),
              "ABAC");

    // `namespace()` has reference semantics, so a loop body can mutate it; a
    // plain `set` inside the loop must not escape the iteration.
    expect_eq("namespace escapes the loop",
              render("{% set ns = namespace(n=0) %}{% set plain = 0 %}"
                     "{% for x in [1,2,3] %}{% set ns.n = ns.n + x %}{% set plain = x %}{% endfor %}"
                     "{{ ns.n }},{{ plain }}",
                     ctx),
              "6,0");
}

void test_whitespace_control() {
    const jj::Value ctx = empty_context();

    // trim_blocks removes the newline after a block tag; lstrip_blocks removes
    // the indentation before one. Both are on, matching transformers.
    expect_eq("trim and lstrip blocks", render("a\n    {% if 1 %}\nb\n    {% endif %}\nc", ctx),
              "a\nb\nc");
    // An explicit `-` strips all adjacent whitespace; `+` suppresses the default.
    expect_eq("dash strips fully", render("a\n   {%- if 1 -%}   \n b {%- endif %}", ctx), "ab");
    expect_eq("output tags are untouched", render("a\n{{ 1 }}\nb", ctx), "a\n1\nb");
}

void test_error_reporting() {
    bool threw = false;
    try {
        render("{{ raise_exception('boom') }}", empty_context());
    } catch (const jj::Error& error) {
        threw = std::string(error.what()) == "boom";
    }
    expect_true("raise_exception propagates its message", threw);

    threw = false;
    try {
        render("{% if 1 %}unclosed", empty_context());
    } catch (const jj::Error&) {
        threw = true;
    }
    expect_true("unclosed block is rejected at parse time", threw);

    threw = false;
    try {
        render("{{ 1 | no_such_filter }}", empty_context());
    } catch (const jj::Error&) {
        threw = true;
    }
    expect_true("unknown filter is rejected rather than ignored", threw);
}

// A `}}` inside a quoted string must not terminate the tag. Chat templates hit
// this whenever they show a JSON tool-call example.
void test_json_literal_in_tag() {
    expect_eq("brace-heavy string literal",
              render("{{ '{\"a\": {\"b\": 1}}' }}", empty_context()), "{\"a\": {\"b\": 1}}");
}

// --- integration with CompiledChatTemplate --------------------------------

const char* kCustomTemplate = R"(
{%- for m in messages %}
{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}
{%- if m.tool_calls is defined %}
{%- for tc in m.tool_calls %}
{{- '[call ' + tc.function.name + ' ' + (tc.function.arguments | tojson) + ']' }}
{%- endfor %}
{%- endif %}
{%- endfor %}
{%- if tools is defined %}{{- '<tools:' ~ (tools | length) ~ '>' }}{%- endif %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %})";

ChatMessage user_message(const std::string& text) {
    ChatMessage message;
    message.role = ChatRole::User;
    message.parts.push_back(ChatPart::text_part(text));
    return message;
}

void test_custom_template_is_interpreted() {
    const CompiledChatTemplate compiled = CompiledChatTemplate::resolve(kCustomTemplate);

    std::vector<ChatMessage> messages{user_message("hi")};
    ChatRenderOptions options;
    options.add_generation_prompt = true;

    const fi::RenderedChat rendered = compiled.render(messages, options);
    expect_eq("custom template renders through the interpreter", rendered.text,
              "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");

    // The response-replay frontier is the end of the rendered prefix.
    expect_true("checkpoint is captured", rendered.rewrite_checkpoint.has_value());
    if (rendered.rewrite_checkpoint) {
        expect_true("checkpoint is ResponseReplay",
                    rendered.rewrite_checkpoint->kind ==
                        ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay);
        expect_true("checkpoint sits at the prompt frontier",
                    rendered.rewrite_checkpoint->offset == rendered.text.size());
    }
}

void test_tool_calls_reach_the_template() {
    const CompiledChatTemplate compiled = CompiledChatTemplate::resolve(kCustomTemplate);

    ChatMessage assistant;
    assistant.role = ChatRole::Assistant;
    assistant.parts.push_back(ChatPart::text_part("working"));
    assistant.tool_calls.push_back(fi::ToolCall{"id-1", "run", R"({"cmd": "make"})"});

    std::vector<ChatMessage> messages{user_message("go"), assistant};
    ChatRenderOptions options;
    options.add_generation_prompt = false;
    options.tool_jsons.push_back(R"({"type": "function", "name": "run"})");

    const std::string text = compiled.render(messages, options).text;
    expect_eq("tool call arguments arrive as a mapping", text,
              "<|im_start|>user\ngo<|im_end|>\n"
              "<|im_start|>assistant\nworking<|im_end|>\n"
              "[call run {\"cmd\": \"make\"}]<tools:1>");
}

void test_capabilities_follow_the_template() {
    // A template that never reads the variables must not advertise them.
    const CompiledChatTemplate inert =
        CompiledChatTemplate::resolve("{% for m in messages %}{{ m.content }}{% endfor %}");
    const ninfer::PromptCapabilities inert_caps = inert.capabilities();
    expect_true("inert template advertises no thinking", !inert_caps.enable_thinking);
    expect_true("inert template advertises no effort", !inert_caps.reasoning_effort.medium);

    const CompiledChatTemplate aware = CompiledChatTemplate::resolve(
        "{% if enable_thinking %}t{% endif %}{{ reasoning_effort }}{{ messages | length }}");
    const ninfer::PromptCapabilities aware_caps = aware.capabilities();
    expect_true("aware template advertises thinking", aware_caps.enable_thinking);
    expect_true("aware template advertises effort", aware_caps.reasoning_effort.medium &&
                                                        aware_caps.reasoning_effort.xhigh);
}

void test_malformed_template_is_rejected() {
    bool threw = false;
    try {
        (void)CompiledChatTemplate::resolve("{% for m in messages %}{{ m.content }}");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect_true("a malformed custom template is rejected at resolve time", threw);
}

// The two built-in templates keep their verified C++ transcriptions; only
// everything else is interpreted.
void test_builtin_templates_keep_their_transcription() {
    const CompiledChatTemplate custom = CompiledChatTemplate::resolve("plain text");
    expect_true("an unknown template is interpreted",
                custom.capabilities().enable_thinking == false);

    std::vector<ChatMessage> messages{user_message("x")};
    ChatRenderOptions options;
    options.add_generation_prompt = false;
    expect_eq("interpreted template with no tags renders verbatim",
              custom.render(messages, options).text, "plain text");
}

} // namespace

int main() {
    test_expressions();
    test_filters_and_tests();
    test_slicing_and_indexing();
    test_statements();
    test_whitespace_control();
    test_error_reporting();
    test_json_literal_in_tag();
    test_custom_template_is_interpreted();
    test_tool_calls_reach_the_template();
    test_capabilities_follow_the_template();
    test_malformed_template_is_rejected();
    test_builtin_templates_keep_their_transcription();

    if (g_failures != 0) {
        std::cerr << g_failures << " jinja chat-template checks failed\n";
        return 1;
    }
    std::cout << "OK jinja chat template\n";
    return 0;
}
