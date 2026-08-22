#pragma once

// A small Jinja2 interpreter, scoped to what chat templates actually use.
//
// This exists because NInfer's chat rendering was previously a hand-written C++
// transcription of two specific templates, gated by a sha256 allowlist. That
// cannot accept a user-supplied template: a template it has not been
// transcribed against either fails the digest check or, worse, renders through
// a transcription written for a different template and silently emits the wrong
// prompt. Interpreting the template removes the transcription step entirely.
//
// Supported subset (anything outside it throws at parse time rather than being
// silently ignored):
//   statements   {% set %} (including {% set x %}..{% endset %}), {% if/elif/else %},
//                {% for .. in .. %} with {% else %}, {% macro %} with default
//                arguments, {% do %}, {# comments #}, and whitespace control via
//                the `-` modifier on any tag.
//   expressions  literals, names, attribute and item access, slicing (with a
//                negative step), calls, filters, tests, conditional expressions,
//                list/tuple literals, and the usual unary/binary operators
//                including `~`, `in`, and `not in`.
//   values       undefined, none, bool, int, double, string, array, object,
//                macro. Arrays and objects have Python reference semantics, so
//                `{% set ns.field = ... %}` on a `namespace()` mutates in place.
//
// Rendering is deterministic and allocation-bounded by the template and inputs;
// there is no I/O, no `include`/`import`, and no access to anything the caller
// did not put in the context.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal::jinja {

class Value;

using Array  = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;

struct MacroDef;

// A Jinja value. Arrays and objects are held by shared_ptr so that assignment
// copies the handle rather than the contents, matching Python semantics: a
// `namespace()` passed into a loop body and mutated there is visible outside it.
class Value {
public:
    enum class Kind : std::uint8_t {
        Undefined,
        None,
        Bool,
        Int,
        Double,
        String,
        Array,
        Object,
        Macro,
    };

    Value() = default;

    static Value undefined() { return Value(); }
    static Value none();
    static Value boolean(bool value);
    static Value integer(std::int64_t value);
    static Value number(double value);
    static Value string(std::string value);
    static Value array(Array value);
    static Value object(Object value);
    static Value macro(std::shared_ptr<const MacroDef> value);

    [[nodiscard]] Kind kind() const noexcept { return kind_; }
    [[nodiscard]] bool is_undefined() const noexcept { return kind_ == Kind::Undefined; }
    [[nodiscard]] bool is_none() const noexcept { return kind_ == Kind::None; }
    [[nodiscard]] bool is_string() const noexcept { return kind_ == Kind::String; }
    [[nodiscard]] bool is_bool() const noexcept { return kind_ == Kind::Bool; }
    [[nodiscard]] bool is_int() const noexcept { return kind_ == Kind::Int; }
    [[nodiscard]] bool is_double() const noexcept { return kind_ == Kind::Double; }
    [[nodiscard]] bool is_number() const noexcept { return is_int() || is_double(); }
    [[nodiscard]] bool is_array() const noexcept { return kind_ == Kind::Array; }
    [[nodiscard]] bool is_object() const noexcept { return kind_ == Kind::Object; }
    [[nodiscard]] bool is_macro() const noexcept { return kind_ == Kind::Macro; }

    // Python truthiness: undefined/none/false/0/""/[]/{} are falsy.
    [[nodiscard]] bool truthy() const noexcept;

    [[nodiscard]] bool as_bool() const noexcept { return bool_; }
    [[nodiscard]] std::int64_t as_int() const noexcept;
    [[nodiscard]] double as_double() const noexcept;
    [[nodiscard]] const std::string& as_string() const;
    [[nodiscard]] const Array& as_array() const;
    [[nodiscard]] Array& as_array();
    [[nodiscard]] const Object& as_object() const;
    [[nodiscard]] Object& as_object();
    [[nodiscard]] const MacroDef& as_macro() const;

    // Object lookup by key; returns undefined when absent.
    [[nodiscard]] const Value* find(std::string_view key) const;
    void set(std::string_view key, Value value);

    // The text `{{ ... }}` would emit for this value.
    [[nodiscard]] std::string to_display_string() const;

    // Python `==` over the supported types.
    [[nodiscard]] bool equals(const Value& other) const;

private:
    Kind kind_ = Kind::Undefined;
    bool bool_ = false;
    std::int64_t int_ = 0;
    double double_    = 0.0;
    std::shared_ptr<std::string> string_;
    std::shared_ptr<Array> array_;
    std::shared_ptr<Object> object_;
    std::shared_ptr<const MacroDef> macro_;
};

// Thrown for both malformed templates (at parse time) and runtime failures,
// including the template calling `raise_exception(...)`.
class Error : public std::exception {
public:
    explicit Error(std::string message) : message_(std::move(message)) {}
    [[nodiscard]] const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};

class Template {
public:
    // Throws jinja::Error if the source is malformed or uses an unsupported
    // construct.
    [[nodiscard]] static Template parse(std::string_view source);

    // `context` must be an object. Throws jinja::Error on a runtime failure.
    [[nodiscard]] std::string render(const Value& context) const;

    // Every name the template reads without having assigned it first — the
    // template's input variables. Used to decide which prompt capabilities a
    // custom template actually honours.
    [[nodiscard]] const std::vector<std::string>& referenced_globals() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal::jinja
