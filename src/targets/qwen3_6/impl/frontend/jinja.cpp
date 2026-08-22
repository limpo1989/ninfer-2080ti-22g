#include "targets/qwen3_6/impl/frontend/jinja.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <functional>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace ninfer::targets::qwen3_6::frontend_internal::jinja {
namespace {

[[noreturn]] void fail(const std::string& message) { throw Error(message); }

std::string to_json_text_impl(const Value& value);

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

bool is_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool is_ident_char(char c) { return is_ident_start(c) || (c >= '0' && c <= '9'); }

std::string strip_left(std::string_view text) {
    std::size_t i = 0;
    while (i < text.size() && is_space(text[i])) { ++i; }
    return std::string(text.substr(i));
}

std::string strip_right(std::string_view text) {
    std::size_t end = text.size();
    while (end > 0 && is_space(text[end - 1])) { --end; }
    return std::string(text.substr(0, end));
}

std::string trim_both(std::string_view text) { return strip_left(strip_right(text)); }

// Python's repr for floats is not reproduced exactly; templates only ever emit
// integers and strings in practice, so a shortest round-trip form that keeps a
// trailing ".0" on integral doubles is sufficient and stable.
std::string format_double(double value) {
    if (std::isfinite(value) && value == std::floor(value) && std::fabs(value) < 1e15) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.1f", value);
        return buffer;
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    // Prefer the shortest representation that round-trips.
    for (int precision = 1; precision < 17; ++precision) {
        char candidate[64];
        std::snprintf(candidate, sizeof(candidate), "%.*g", precision, value);
        if (std::strtod(candidate, nullptr) == value) { return candidate; }
    }
    return buffer;
}

} // namespace

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------

Value Value::none() {
    Value v;
    v.kind_ = Kind::None;
    return v;
}

Value Value::boolean(bool value) {
    Value v;
    v.kind_ = Kind::Bool;
    v.bool_ = value;
    return v;
}

Value Value::integer(std::int64_t value) {
    Value v;
    v.kind_ = Kind::Int;
    v.int_  = value;
    return v;
}

Value Value::number(double value) {
    Value v;
    v.kind_   = Kind::Double;
    v.double_ = value;
    return v;
}

Value Value::string(std::string value) {
    Value v;
    v.kind_   = Kind::String;
    v.string_ = std::make_shared<std::string>(std::move(value));
    return v;
}

Value Value::array(Array value) {
    Value v;
    v.kind_  = Kind::Array;
    v.array_ = std::make_shared<Array>(std::move(value));
    return v;
}

Value Value::object(Object value) {
    Value v;
    v.kind_   = Kind::Object;
    v.object_ = std::make_shared<Object>(std::move(value));
    return v;
}

Value Value::macro(std::shared_ptr<const MacroDef> value) {
    Value v;
    v.kind_  = Kind::Macro;
    v.macro_ = std::move(value);
    return v;
}

bool Value::truthy() const noexcept {
    switch (kind_) {
    case Kind::Undefined:
    case Kind::None:
        return false;
    case Kind::Bool:
        return bool_;
    case Kind::Int:
        return int_ != 0;
    case Kind::Double:
        return double_ != 0.0;
    case Kind::String:
        return !string_->empty();
    case Kind::Array:
        return !array_->empty();
    case Kind::Object:
        return !object_->empty();
    case Kind::Macro:
        return true;
    }
    return false;
}

std::int64_t Value::as_int() const noexcept {
    if (kind_ == Kind::Int) { return int_; }
    if (kind_ == Kind::Double) { return static_cast<std::int64_t>(double_); }
    if (kind_ == Kind::Bool) { return bool_ ? 1 : 0; }
    return 0;
}

double Value::as_double() const noexcept {
    if (kind_ == Kind::Double) { return double_; }
    if (kind_ == Kind::Int) { return static_cast<double>(int_); }
    if (kind_ == Kind::Bool) { return bool_ ? 1.0 : 0.0; }
    return 0.0;
}

const std::string& Value::as_string() const {
    if (kind_ != Kind::String) { fail("value is not a string"); }
    return *string_;
}

const Array& Value::as_array() const {
    if (kind_ != Kind::Array) { fail("value is not a list"); }
    return *array_;
}

Array& Value::as_array() {
    if (kind_ != Kind::Array) { fail("value is not a list"); }
    return *array_;
}

const Object& Value::as_object() const {
    if (kind_ != Kind::Object) { fail("value is not a mapping"); }
    return *object_;
}

Object& Value::as_object() {
    if (kind_ != Kind::Object) { fail("value is not a mapping"); }
    return *object_;
}

const MacroDef& Value::as_macro() const {
    if (kind_ != Kind::Macro) { fail("value is not callable"); }
    return *macro_;
}

const Value* Value::find(std::string_view key) const {
    if (kind_ != Kind::Object) { return nullptr; }
    for (const auto& entry : *object_) {
        if (entry.first == key) { return &entry.second; }
    }
    return nullptr;
}

void Value::set(std::string_view key, Value value) {
    if (kind_ != Kind::Object) { fail("cannot assign a field on a non-mapping value"); }
    for (auto& entry : *object_) {
        if (entry.first == key) {
            entry.second = std::move(value);
            return;
        }
    }
    object_->emplace_back(std::string(key), std::move(value));
}

std::string Value::to_display_string() const {
    switch (kind_) {
    case Kind::Undefined:
    case Kind::None:
        return "";
    case Kind::Bool:
        return bool_ ? "True" : "False";
    case Kind::Int:
        return std::to_string(int_);
    case Kind::Double:
        return format_double(double_);
    case Kind::String:
        return *string_;
    case Kind::Array:
    case Kind::Object:
    case Kind::Macro:
        break;
    }
    // Python's str() of a container; only reachable if a template emits one
    // directly, which chat templates do not do for real output.
    return to_json_text_impl(*this);
}

bool Value::equals(const Value& other) const {
    if (is_number() && other.is_number()) {
        if (is_int() && other.is_int()) { return as_int() == other.as_int(); }
        return as_double() == other.as_double();
    }
    if (kind_ != other.kind_) {
        // Python treats True == 1; keep bool/number cross-comparison consistent.
        if (is_bool() && other.is_number()) { return as_int() == other.as_int(); }
        if (is_number() && other.is_bool()) { return as_int() == other.as_int(); }
        return false;
    }
    switch (kind_) {
    case Kind::Undefined:
    case Kind::None:
        return true;
    case Kind::Bool:
        return bool_ == other.bool_;
    case Kind::String:
        return *string_ == *other.string_;
    case Kind::Array: {
        if (array_->size() != other.array_->size()) { return false; }
        for (std::size_t i = 0; i < array_->size(); ++i) {
            if (!(*array_)[i].equals((*other.array_)[i])) { return false; }
        }
        return true;
    }
    case Kind::Object: {
        if (object_->size() != other.object_->size()) { return false; }
        for (const auto& entry : *object_) {
            const Value* rhs = other.find(entry.first);
            if (rhs == nullptr || !entry.second.equals(*rhs)) { return false; }
        }
        return true;
    }
    default:
        return false;
    }
}

namespace {

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

struct Expr {
    enum class Kind : std::uint8_t {
        Literal,
        Name,
        Attr,
        Item,
        Slice,
        Call,
        Filter,
        Test,
        Unary,
        Binary,
        Ternary,
        ListLit,
        DictLit,
    };

    Kind kind = Kind::Literal;
    Value literal;
    std::string name; // identifier, operator, filter/test name
    std::shared_ptr<Expr> a;
    std::shared_ptr<Expr> b;
    std::shared_ptr<Expr> c;
    std::shared_ptr<Expr> d;
    std::vector<std::shared_ptr<Expr>> args;
    std::vector<std::pair<std::string, std::shared_ptr<Expr>>> kwargs;
    bool negated = false;
};

using ExprPtr = std::shared_ptr<Expr>;

struct Node {
    enum class Kind : std::uint8_t {
        Text,
        Output,
        If,
        For,
        Set,
        SetBlock,
        Macro,
        Do,
    };

    Kind kind = Kind::Text;
    std::string text;                                  // Text body, Set/Macro name
    ExprPtr expr;                                      // Output/Set value, For iterable
    ExprPtr target;                                    // Set target when not a bare name
    std::vector<std::pair<ExprPtr, std::vector<std::shared_ptr<Node>>>> branches; // If
    std::vector<std::shared_ptr<Node>> body;
    std::vector<std::shared_ptr<Node>> else_body;
    std::vector<std::string> loop_targets;
    std::vector<std::pair<std::string, ExprPtr>> params; // Macro
};

using NodePtr = std::shared_ptr<Node>;

} // namespace

// Defined at namespace scope because the header forward-declares it there for
// Value::macro(); its members are the internal AST types above.
struct MacroDef {
    std::string name;
    std::vector<std::pair<std::string, ExprPtr>> params;
    std::vector<NodePtr> body;
};

namespace {

// ---------------------------------------------------------------------------
// Expression tokenizer
// ---------------------------------------------------------------------------

struct Token {
    enum class Kind : std::uint8_t { End, Name, String, Int, Double, Op };
    Kind kind = Kind::End;
    std::string text;
    std::int64_t int_value = 0;
    double double_value    = 0.0;
};

std::string decode_escapes(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] != '\\' || i + 1 >= raw.size()) {
            out.push_back(raw[i]);
            continue;
        }
        const char next = raw[++i];
        switch (next) {
        case 'n':
            out.push_back('\n');
            break;
        case 't':
            out.push_back('\t');
            break;
        case 'r':
            out.push_back('\r');
            break;
        case '0':
            out.push_back('\0');
            break;
        case '\\':
        case '\'':
        case '"':
            out.push_back(next);
            break;
        default:
            out.push_back('\\');
            out.push_back(next);
            break;
        }
    }
    return out;
}

std::vector<Token> tokenize_expression(std::string_view source) {
    static const std::vector<std::string> kMultiCharOps = {"//", "**", "==", "!=", "<=", ">="};
    std::vector<Token> tokens;
    std::size_t i = 0;
    while (i < source.size()) {
        if (is_space(source[i])) {
            ++i;
            continue;
        }
        const char c = source[i];
        if (c == '\'' || c == '"') {
            const char quote = c;
            std::size_t j    = i + 1;
            std::string raw;
            while (j < source.size() && source[j] != quote) {
                if (source[j] == '\\' && j + 1 < source.size()) {
                    raw.push_back(source[j]);
                    raw.push_back(source[j + 1]);
                    j += 2;
                    continue;
                }
                raw.push_back(source[j]);
                ++j;
            }
            if (j >= source.size()) { fail("unterminated string literal in template expression"); }
            Token token;
            token.kind = Token::Kind::String;
            token.text = decode_escapes(raw);
            tokens.push_back(std::move(token));
            i = j + 1;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) != 0) {
            std::size_t j  = i;
            bool is_double = false;
            while (j < source.size() && (std::isdigit(static_cast<unsigned char>(source[j])) != 0 ||
                                         source[j] == '.')) {
                // A '.' only continues the number when a digit follows, so that
                // `1.foo` and slice syntax are not swallowed.
                if (source[j] == '.') {
                    if (j + 1 >= source.size() ||
                        std::isdigit(static_cast<unsigned char>(source[j + 1])) == 0) {
                        break;
                    }
                    is_double = true;
                }
                ++j;
            }
            const std::string text(source.substr(i, j - i));
            Token token;
            if (is_double) {
                token.kind         = Token::Kind::Double;
                token.double_value = std::strtod(text.c_str(), nullptr);
            } else {
                token.kind      = Token::Kind::Int;
                token.int_value = std::strtoll(text.c_str(), nullptr, 10);
            }
            token.text = text;
            tokens.push_back(std::move(token));
            i = j;
            continue;
        }
        if (is_ident_start(c)) {
            std::size_t j = i;
            while (j < source.size() && is_ident_char(source[j])) { ++j; }
            Token token;
            token.kind = Token::Kind::Name;
            token.text = std::string(source.substr(i, j - i));
            tokens.push_back(std::move(token));
            i = j;
            continue;
        }
        bool matched = false;
        for (const std::string& op : kMultiCharOps) {
            if (source.compare(i, op.size(), op) == 0) {
                Token token;
                token.kind = Token::Kind::Op;
                token.text = op;
                tokens.push_back(std::move(token));
                i += op.size();
                matched = true;
                break;
            }
        }
        if (matched) { continue; }
        Token token;
        token.kind = Token::Kind::Op;
        token.text = std::string(1, c);
        tokens.push_back(std::move(token));
        ++i;
    }
    tokens.push_back(Token{});
    return tokens;
}

// ---------------------------------------------------------------------------
// Expression parser
// ---------------------------------------------------------------------------

class ExprParser {
public:
    explicit ExprParser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    ExprPtr parse_full() {
        ExprPtr expr = parse_expression();
        expect_end();
        return expr;
    }

    // Parses a comma-separated tuple without surrounding parentheses, used for
    // `{% for a, b in ... %}` targets and bare tuple expressions.
    ExprPtr parse_expression() { return parse_condexpr(); }

    [[nodiscard]] bool at_end() const { return peek().kind == Token::Kind::End; }

    [[nodiscard]] const Token& peek(std::size_t offset = 0) const {
        const std::size_t index = pos_ + offset;
        return index < tokens_.size() ? tokens_[index] : tokens_.back();
    }

    void expect_end() {
        if (!at_end()) { fail("unexpected trailing input in template expression: " + peek().text); }
    }

    bool accept_op(std::string_view op) {
        if (peek().kind == Token::Kind::Op && peek().text == op) {
            ++pos_;
            return true;
        }
        return false;
    }

    bool accept_name(std::string_view name) {
        if (peek().kind == Token::Kind::Name && peek().text == name) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect_op(std::string_view op) {
        if (!accept_op(op)) {
            fail("expected '" + std::string(op) + "' in template expression, found '" +
                 peek().text + "'");
        }
    }

    std::string expect_name() {
        if (peek().kind != Token::Kind::Name) {
            fail("expected a name in template expression, found '" + peek().text + "'");
        }
        return tokens_[pos_++].text;
    }

private:
    static ExprPtr make(Expr::Kind kind) {
        auto expr  = std::make_shared<Expr>();
        expr->kind = kind;
        return expr;
    }

    static ExprPtr binary(std::string op, ExprPtr a, ExprPtr b) {
        ExprPtr expr = make(Expr::Kind::Binary);
        expr->name   = std::move(op);
        expr->a      = std::move(a);
        expr->b      = std::move(b);
        return expr;
    }

    // condexpr := or ['if' or ['else' condexpr]]
    ExprPtr parse_condexpr() {
        ExprPtr value = parse_or();
        while (peek().kind == Token::Kind::Name && peek().text == "if") {
            ++pos_;
            ExprPtr condition = parse_or();
            ExprPtr otherwise;
            if (accept_name("else")) { otherwise = parse_condexpr(); }
            ExprPtr expr = make(Expr::Kind::Ternary);
            expr->a      = std::move(condition);
            expr->b      = std::move(value);
            expr->c      = std::move(otherwise);
            value        = std::move(expr);
        }
        return value;
    }

    ExprPtr parse_or() {
        ExprPtr left = parse_and();
        while (peek().kind == Token::Kind::Name && peek().text == "or") {
            ++pos_;
            left = binary("or", std::move(left), parse_and());
        }
        return left;
    }

    ExprPtr parse_and() {
        ExprPtr left = parse_not();
        while (peek().kind == Token::Kind::Name && peek().text == "and") {
            ++pos_;
            left = binary("and", std::move(left), parse_not());
        }
        return left;
    }

    ExprPtr parse_not() {
        if (peek().kind == Token::Kind::Name && peek().text == "not") {
            ++pos_;
            ExprPtr expr = make(Expr::Kind::Unary);
            expr->name   = "not";
            expr->a      = parse_not();
            return expr;
        }
        return parse_compare();
    }

    ExprPtr parse_compare() {
        ExprPtr left = parse_math1();
        for (;;) {
            std::string op;
            if (peek().kind == Token::Kind::Op &&
                (peek().text == "==" || peek().text == "!=" || peek().text == "<" ||
                 peek().text == ">" || peek().text == "<=" || peek().text == ">=")) {
                op = tokens_[pos_++].text;
            } else if (peek().kind == Token::Kind::Name && peek().text == "in") {
                ++pos_;
                op = "in";
            } else if (peek().kind == Token::Kind::Name && peek().text == "not" &&
                       peek(1).kind == Token::Kind::Name && peek(1).text == "in") {
                pos_ += 2;
                op = "not in";
            } else {
                break;
            }
            left = binary(std::move(op), std::move(left), parse_math1());
        }
        return left;
    }

    ExprPtr parse_math1() {
        ExprPtr left = parse_concat();
        for (;;) {
            if (peek().kind != Token::Kind::Op) { break; }
            if (peek().text != "+" && peek().text != "-") { break; }
            const std::string op = tokens_[pos_++].text;
            left                 = binary(op, std::move(left), parse_concat());
        }
        return left;
    }

    ExprPtr parse_concat() {
        ExprPtr left = parse_math2();
        while (peek().kind == Token::Kind::Op && peek().text == "~") {
            ++pos_;
            left = binary("~", std::move(left), parse_math2());
        }
        return left;
    }

    ExprPtr parse_math2() {
        ExprPtr left = parse_unary();
        for (;;) {
            if (peek().kind != Token::Kind::Op) { break; }
            const std::string& text = peek().text;
            if (text != "*" && text != "/" && text != "//" && text != "%") { break; }
            const std::string op = tokens_[pos_++].text;
            left                 = binary(op, std::move(left), parse_unary());
        }
        return left;
    }

    ExprPtr parse_unary() {
        if (peek().kind == Token::Kind::Op && (peek().text == "-" || peek().text == "+")) {
            const std::string op = tokens_[pos_++].text;
            ExprPtr expr         = make(Expr::Kind::Unary);
            expr->name           = op;
            expr->a              = parse_unary();
            return parse_postfix(std::move(expr));
        }
        return parse_postfix(parse_primary());
    }

    // Filters and tests bind tighter than any operator, matching Jinja.
    ExprPtr parse_postfix(ExprPtr value) {
        for (;;) {
            if (accept_op(".")) {
                ExprPtr expr = make(Expr::Kind::Attr);
                expr->name   = expect_name();
                expr->a      = std::move(value);
                value        = std::move(expr);
                continue;
            }
            if (peek().kind == Token::Kind::Op && peek().text == "[") {
                ++pos_;
                value = parse_subscript(std::move(value));
                continue;
            }
            if (peek().kind == Token::Kind::Op && peek().text == "(") {
                ++pos_;
                ExprPtr expr = make(Expr::Kind::Call);
                expr->a      = std::move(value);
                parse_call_args(*expr);
                value = std::move(expr);
                continue;
            }
            if (peek().kind == Token::Kind::Op && peek().text == "|") {
                ++pos_;
                ExprPtr expr = make(Expr::Kind::Filter);
                expr->name   = expect_name();
                expr->a      = std::move(value);
                if (peek().kind == Token::Kind::Op && peek().text == "(") {
                    ++pos_;
                    parse_call_args(*expr);
                }
                value = std::move(expr);
                continue;
            }
            if (peek().kind == Token::Kind::Name && peek().text == "is") {
                ++pos_;
                ExprPtr expr  = make(Expr::Kind::Test);
                expr->negated = accept_name("not");
                expr->name    = expect_name();
                expr->a       = std::move(value);
                if (peek().kind == Token::Kind::Op && peek().text == "(") {
                    ++pos_;
                    parse_call_args(*expr);
                }
                value = std::move(expr);
                continue;
            }
            break;
        }
        return value;
    }

    ExprPtr parse_subscript(ExprPtr value) {
        // Distinguish `x[a]` from the slice forms `x[a:b]`, `x[:b]`, `x[::-1]`.
        ExprPtr start;
        if (!(peek().kind == Token::Kind::Op && peek().text == ":")) { start = parse_condexpr(); }
        if (peek().kind == Token::Kind::Op && peek().text == "]") {
            ++pos_;
            ExprPtr expr = make(Expr::Kind::Item);
            expr->a      = std::move(value);
            expr->b      = std::move(start);
            return expr;
        }
        expect_op(":");
        ExprPtr stop;
        if (!(peek().kind == Token::Kind::Op && (peek().text == "]" || peek().text == ":"))) {
            stop = parse_condexpr();
        }
        ExprPtr step;
        if (accept_op(":")) {
            if (!(peek().kind == Token::Kind::Op && peek().text == "]")) { step = parse_condexpr(); }
        }
        expect_op("]");
        ExprPtr expr = make(Expr::Kind::Slice);
        expr->a      = std::move(value);
        expr->b      = std::move(start);
        expr->c      = std::move(stop);
        expr->d      = std::move(step);
        return expr;
    }

    void parse_call_args(Expr& expr) {
        if (accept_op(")")) { return; }
        for (;;) {
            if (peek().kind == Token::Kind::Name && peek(1).kind == Token::Kind::Op &&
                peek(1).text == "=") {
                std::string key = tokens_[pos_].text;
                pos_ += 2;
                expr.kwargs.emplace_back(std::move(key), parse_condexpr());
            } else {
                expr.args.push_back(parse_condexpr());
            }
            if (accept_op(",")) { continue; }
            expect_op(")");
            return;
        }
    }

    ExprPtr parse_primary() {
        const Token& token = peek();
        switch (token.kind) {
        case Token::Kind::String: {
            ExprPtr expr  = make(Expr::Kind::Literal);
            expr->literal = Value::string(token.text);
            ++pos_;
            return expr;
        }
        case Token::Kind::Int: {
            ExprPtr expr  = make(Expr::Kind::Literal);
            expr->literal = Value::integer(token.int_value);
            ++pos_;
            return expr;
        }
        case Token::Kind::Double: {
            ExprPtr expr  = make(Expr::Kind::Literal);
            expr->literal = Value::number(token.double_value);
            ++pos_;
            return expr;
        }
        case Token::Kind::Name: {
            if (token.text == "true" || token.text == "True") {
                ++pos_;
                ExprPtr expr  = make(Expr::Kind::Literal);
                expr->literal = Value::boolean(true);
                return expr;
            }
            if (token.text == "false" || token.text == "False") {
                ++pos_;
                ExprPtr expr  = make(Expr::Kind::Literal);
                expr->literal = Value::boolean(false);
                return expr;
            }
            if (token.text == "none" || token.text == "None") {
                ++pos_;
                ExprPtr expr  = make(Expr::Kind::Literal);
                expr->literal = Value::none();
                return expr;
            }
            ExprPtr expr = make(Expr::Kind::Name);
            expr->name   = token.text;
            ++pos_;
            return expr;
        }
        case Token::Kind::Op:
            if (token.text == "(") {
                ++pos_;
                // A parenthesised group, or a tuple literal such as ('a', 'b').
                if (accept_op(")")) { return make_empty_list(); }
                ExprPtr first = parse_condexpr();
                if (peek().kind == Token::Kind::Op && peek().text == ",") {
                    ExprPtr expr = make(Expr::Kind::ListLit);
                    expr->args.push_back(std::move(first));
                    while (accept_op(",")) {
                        if (peek().kind == Token::Kind::Op && peek().text == ")") { break; }
                        expr->args.push_back(parse_condexpr());
                    }
                    expect_op(")");
                    return expr;
                }
                expect_op(")");
                return first;
            }
            if (token.text == "[") {
                ++pos_;
                ExprPtr expr = make(Expr::Kind::ListLit);
                if (accept_op("]")) { return expr; }
                for (;;) {
                    expr->args.push_back(parse_condexpr());
                    if (accept_op(",")) {
                        if (peek().kind == Token::Kind::Op && peek().text == "]") { break; }
                        continue;
                    }
                    break;
                }
                expect_op("]");
                return expr;
            }
            if (token.text == "{") {
                ++pos_;
                ExprPtr expr = make(Expr::Kind::DictLit);
                if (accept_op("}")) { return expr; }
                for (;;) {
                    ExprPtr key = parse_condexpr();
                    expect_op(":");
                    ExprPtr value = parse_condexpr();
                    expr->args.push_back(std::move(key));
                    expr->args.push_back(std::move(value));
                    if (accept_op(",")) {
                        if (peek().kind == Token::Kind::Op && peek().text == "}") { break; }
                        continue;
                    }
                    break;
                }
                expect_op("}");
                return expr;
            }
            break;
        case Token::Kind::End:
            fail("unexpected end of template expression");
        }
        fail("unexpected token '" + token.text + "' in template expression");
    }

    static ExprPtr make_empty_list() {
        auto expr  = std::make_shared<Expr>();
        expr->kind = Expr::Kind::ListLit;
        return expr;
    }

    std::vector<Token> tokens_;
    std::size_t pos_ = 0;
};

ExprPtr parse_expression_source(std::string_view source) {
    ExprParser parser(tokenize_expression(source));
    return parser.parse_full();
}

// ---------------------------------------------------------------------------
// Template block scanner and statement parser
// ---------------------------------------------------------------------------

struct Block {
    enum class Kind : std::uint8_t { Text, Output, Statement };
    Kind kind = Kind::Text;
    std::string content;
};

// Finds a tag's closing delimiter, skipping over string literals. Chat templates
// emit JSON examples such as `{"arguments": {...}}`, so a literal `}}` can appear
// inside a quoted string and must not be mistaken for the end of the tag.
std::size_t find_tag_end(std::string_view source, std::size_t begin, std::string_view close) {
    char quote = '\0';
    for (std::size_t i = begin; i < source.size(); ++i) {
        const char c = source[i];
        if (quote != '\0') {
            if (c == '\\') {
                ++i;
            } else if (c == quote) {
                quote = '\0';
            }
            continue;
        }
        if (c == '\'' || c == '"') {
            quote = c;
            continue;
        }
        if (source.compare(i, close.size(), close) == 0) { return i; }
    }
    return std::string_view::npos;
}

// Whitespace handling reproduces the environment transformers builds for chat
// templates: trim_blocks=True and lstrip_blocks=True. Those defaults apply to
// block and comment tags only, never to `{{ ... }}`, and an explicit `-` (strip
// all adjacent whitespace) or `+` (suppress the default) overrides them.
enum class WsMode : std::uint8_t { Default, Dash, Plus };

std::vector<Block> scan_blocks(std::string_view source) {
    constexpr bool kTrimBlocks   = true;
    constexpr bool kLstripBlocks = true;

    std::vector<Block> blocks;
    std::string pending_text;
    WsMode pending_right    = WsMode::Plus; // nothing precedes the first text run
    bool pending_right_block = false;

    // Whether the pending text run begins at the start of a source line, which
    // is what lstrip_blocks keys off. A run that starts straight after a tag is
    // mid-line unless the whitespace stripped from its front swallowed a newline.
    bool run_starts_line = true;

    auto take_pending_text = [&](bool& starts_line) {
        std::string text = std::move(pending_text);
        pending_text.clear();
        starts_line = run_starts_line;
        if (pending_right == WsMode::Dash) {
            const std::string stripped = strip_left(text);
            if (text.size() != stripped.size() &&
                text.find('\n') < text.size() - stripped.size()) {
                starts_line = true;
            }
            text = stripped;
        } else if (pending_right == WsMode::Default && pending_right_block && kTrimBlocks) {
            if (!text.empty() && text.front() == '\n') {
                text.erase(text.begin());
                starts_line = true;
            } else if (text.size() >= 2 && text[0] == '\r' && text[1] == '\n') {
                text.erase(text.begin(), text.begin() + 2);
                starts_line = true;
            }
        }
        return text;
    };

    auto flush_text = [&](WsMode left, bool left_is_block) {
        bool starts_line = true;
        std::string text = take_pending_text(starts_line);
        if (left == WsMode::Dash) {
            text = strip_right(text);
        } else if (left == WsMode::Default && left_is_block && kLstripBlocks) {
            // Strip the run of spaces/tabs that reaches back to a newline. When
            // the run is entirely spaces it only counts if the run itself began
            // a line -- otherwise the tag sits mid-line and nothing is stripped.
            std::size_t end = text.size();
            while (end > 0 && (text[end - 1] == ' ' || text[end - 1] == '\t')) { --end; }
            if (end == 0 ? starts_line : text[end - 1] == '\n') { text.resize(end); }
        }
        if (!text.empty()) { blocks.push_back(Block{Block::Kind::Text, std::move(text)}); }
    };

    std::size_t i = 0;
    while (i < source.size()) {
        const std::size_t open = source.find('{', i);
        if (open == std::string_view::npos || open + 1 >= source.size()) {
            pending_text.append(source.substr(i));
            break;
        }
        const char marker = source[open + 1];
        if (marker != '{' && marker != '%' && marker != '#') {
            pending_text.append(source.substr(i, open + 1 - i));
            i = open + 1;
            continue;
        }
        pending_text.append(source.substr(i, open - i));

        const bool is_block     = marker != '{';
        const std::string close = marker == '{' ? "}}" : (marker == '%' ? "%}" : "#}");
        std::size_t body_begin  = open + 2;

        WsMode left = WsMode::Default;
        if (body_begin < source.size() && source[body_begin] == '-') {
            left = WsMode::Dash;
            ++body_begin;
        } else if (body_begin < source.size() && source[body_begin] == '+') {
            left = WsMode::Plus;
            ++body_begin;
        }

        const std::size_t body_end = marker == '#'
                                         ? source.find(close, body_begin)
                                         : find_tag_end(source, body_begin, close);
        if (body_end == std::string_view::npos) {
            fail("unterminated '" + std::string(source.substr(open, 2)) + "' tag in template");
        }

        std::size_t content_end = body_end;
        WsMode right            = WsMode::Default;
        if (content_end > body_begin && source[content_end - 1] == '-') {
            right = WsMode::Dash;
            --content_end;
        } else if (content_end > body_begin && source[content_end - 1] == '+') {
            right = WsMode::Plus;
            --content_end;
        }

        flush_text(left, is_block);
        pending_right       = right;
        pending_right_block = is_block;
        run_starts_line     = false;

        if (marker != '#') {
            Block block;
            block.kind    = marker == '{' ? Block::Kind::Output : Block::Kind::Statement;
            block.content = trim_both(source.substr(body_begin, content_end - body_begin));
            blocks.push_back(std::move(block));
        }
        i = body_end + close.size();
    }
    flush_text(WsMode::Plus, false);
    return blocks;
}

// Splits a statement body into its leading keyword and the remainder.
std::pair<std::string, std::string> split_keyword(const std::string& statement) {
    std::size_t i = 0;
    while (i < statement.size() && is_ident_char(statement[i])) { ++i; }
    return {statement.substr(0, i), trim_both(std::string_view(statement).substr(i))};
}

class StatementParser {
public:
    explicit StatementParser(std::vector<Block> blocks) : blocks_(std::move(blocks)) {}

    std::vector<NodePtr> parse_body(const std::set<std::string>& terminators,
                                    std::string* terminator) {
        std::vector<NodePtr> body;
        for (;;) {
            if (pos_ >= blocks_.size()) {
                if (terminators.empty()) { return body; }
                fail("unexpected end of template; expected one of the closing tags");
            }
            const Block& block = blocks_[pos_];
            if (block.kind == Block::Kind::Text) {
                auto node  = std::make_shared<Node>();
                node->kind = Node::Kind::Text;
                node->text = block.content;
                body.push_back(std::move(node));
                ++pos_;
                continue;
            }
            if (block.kind == Block::Kind::Output) {
                auto node  = std::make_shared<Node>();
                node->kind = Node::Kind::Output;
                node->expr = parse_expression_source(block.content);
                body.push_back(std::move(node));
                ++pos_;
                continue;
            }
            const auto [keyword, rest] = split_keyword(block.content);
            if (terminators.count(keyword) != 0) {
                if (terminator != nullptr) { *terminator = block.content; }
                ++pos_;
                return body;
            }
            body.push_back(parse_statement(keyword, rest));
        }
    }

private:
    NodePtr parse_statement(const std::string& keyword, const std::string& rest) {
        if (keyword == "if") { return parse_if(rest); }
        if (keyword == "for") { return parse_for(rest); }
        if (keyword == "set") { return parse_set(rest); }
        if (keyword == "macro") { return parse_macro(rest); }
        if (keyword == "do") {
            ++pos_;
            auto node  = std::make_shared<Node>();
            node->kind = Node::Kind::Do;
            node->expr = parse_expression_source(rest);
            return node;
        }
        fail("unsupported template statement '" + keyword + "'");
    }

    NodePtr parse_if(const std::string& first_condition) {
        ++pos_;
        auto node  = std::make_shared<Node>();
        node->kind = Node::Kind::If;

        std::string condition = first_condition;
        for (;;) {
            std::string terminator;
            std::vector<NodePtr> branch =
                parse_body({"elif", "else", "endif"}, &terminator);
            node->branches.emplace_back(parse_expression_source(condition), std::move(branch));
            const auto [keyword, rest] = split_keyword(terminator);
            if (keyword == "elif") {
                condition = rest;
                continue;
            }
            if (keyword == "else") {
                node->else_body = parse_body({"endif"}, nullptr);
            }
            return node;
        }
    }

    NodePtr parse_for(const std::string& header) {
        ++pos_;
        auto node  = std::make_shared<Node>();
        node->kind = Node::Kind::For;

        // `a, b in expr` — split on the top-level ` in `.
        ExprParser parser(tokenize_expression(header));
        for (;;) {
            node->loop_targets.push_back(parser.expect_name());
            if (parser.peek().kind == Token::Kind::Op && parser.peek().text == ",") {
                parser.expect_op(",");
                continue;
            }
            break;
        }
        if (!parser.accept_name("in")) { fail("expected 'in' in a for statement"); }
        node->expr = parser.parse_expression();
        // `{% for x in y if cond %}` is not used by chat templates; reject it
        // rather than silently ignoring the filter.
        parser.expect_end();

        std::string terminator;
        node->body = parse_body({"else", "endfor"}, &terminator);
        if (split_keyword(terminator).first == "else") {
            node->else_body = parse_body({"endfor"}, nullptr);
        }
        return node;
    }

    NodePtr parse_set(const std::string& rest) {
        ++pos_;
        auto node = std::make_shared<Node>();

        const std::size_t equals = find_top_level_assign(rest);
        if (equals == std::string::npos) {
            // Block form: {% set name %} ... {% endset %}
            node->kind = Node::Kind::SetBlock;
            node->text = trim_both(rest);
            if (node->text.empty()) { fail("'set' block requires a target name"); }
            node->body = parse_body({"endset"}, nullptr);
            return node;
        }

        node->kind             = Node::Kind::Set;
        const std::string lhs  = trim_both(std::string_view(rest).substr(0, equals));
        const std::string rhs  = trim_both(std::string_view(rest).substr(equals + 1));
        node->expr             = parse_expression_source(rhs);
        ExprPtr target         = parse_expression_source(lhs);
        if (target->kind == Expr::Kind::Name) {
            node->text = target->name;
        } else if (target->kind == Expr::Kind::Attr || target->kind == Expr::Kind::Item) {
            node->target = std::move(target);
        } else {
            fail("unsupported assignment target in a 'set' statement");
        }
        return node;
    }

    NodePtr parse_macro(const std::string& header) {
        ++pos_;
        auto node  = std::make_shared<Node>();
        node->kind = Node::Kind::Macro;

        ExprParser parser(tokenize_expression(header));
        node->text = parser.expect_name();
        parser.expect_op("(");
        if (!parser.accept_op(")")) {
            for (;;) {
                std::string name = parser.expect_name();
                ExprPtr default_value;
                if (parser.accept_op("=")) { default_value = parser.parse_expression(); }
                node->params.emplace_back(std::move(name), std::move(default_value));
                if (parser.accept_op(",")) { continue; }
                parser.expect_op(")");
                break;
            }
        }
        parser.expect_end();
        node->body = parse_body({"endmacro"}, nullptr);
        return node;
    }

    // Finds the '=' that separates a set target from its value, skipping '==',
    // '<=', '>=', '!=' and anything inside brackets, quotes, or a call.
    static std::size_t find_top_level_assign(const std::string& text) {
        int depth = 0;
        char quote = '\0';
        for (std::size_t i = 0; i < text.size(); ++i) {
            const char c = text[i];
            if (quote != '\0') {
                if (c == '\\') {
                    ++i;
                } else if (c == quote) {
                    quote = '\0';
                }
                continue;
            }
            if (c == '\'' || c == '"') {
                quote = c;
                continue;
            }
            if (c == '(' || c == '[' || c == '{') { ++depth; continue; }
            if (c == ')' || c == ']' || c == '}') { --depth; continue; }
            if (depth != 0 || c != '=') { continue; }
            const bool part_of_comparison =
                (i + 1 < text.size() && text[i + 1] == '=') ||
                (i > 0 && (text[i - 1] == '=' || text[i - 1] == '!' || text[i - 1] == '<' ||
                           text[i - 1] == '>'));
            if (!part_of_comparison) { return i; }
            if (i + 1 < text.size() && text[i + 1] == '=') { ++i; }
        }
        return std::string::npos;
    }

    std::vector<Block> blocks_;
    std::size_t pos_ = 0;
};

// ---------------------------------------------------------------------------
// Static analysis: which names does the template read without assigning?
// ---------------------------------------------------------------------------

void collect_globals(const std::vector<NodePtr>& body, std::unordered_set<std::string>& assigned,
                     std::vector<std::string>& referenced,
                     std::unordered_set<std::string>& seen);

void collect_globals_expr(const ExprPtr& expr, const std::unordered_set<std::string>& assigned,
                          std::vector<std::string>& referenced,
                          std::unordered_set<std::string>& seen) {
    if (!expr) { return; }
    if (expr->kind == Expr::Kind::Name) {
        if (assigned.count(expr->name) == 0 && seen.count(expr->name) == 0) {
            seen.insert(expr->name);
            referenced.push_back(expr->name);
        }
        return;
    }
    for (const ExprPtr& child : {expr->a, expr->b, expr->c, expr->d}) {
        collect_globals_expr(child, assigned, referenced, seen);
    }
    for (const ExprPtr& child : expr->args) {
        collect_globals_expr(child, assigned, referenced, seen);
    }
    for (const auto& kwarg : expr->kwargs) {
        collect_globals_expr(kwarg.second, assigned, referenced, seen);
    }
}

void collect_globals(const std::vector<NodePtr>& body, std::unordered_set<std::string>& assigned,
                     std::vector<std::string>& referenced,
                     std::unordered_set<std::string>& seen) {
    for (const NodePtr& node : body) {
        switch (node->kind) {
        case Node::Kind::Text:
            break;
        case Node::Kind::Output:
        case Node::Kind::Do:
            collect_globals_expr(node->expr, assigned, referenced, seen);
            break;
        case Node::Kind::If:
            for (const auto& branch : node->branches) {
                collect_globals_expr(branch.first, assigned, referenced, seen);
                collect_globals(branch.second, assigned, referenced, seen);
            }
            collect_globals(node->else_body, assigned, referenced, seen);
            break;
        case Node::Kind::For:
            collect_globals_expr(node->expr, assigned, referenced, seen);
            for (const std::string& target : node->loop_targets) { assigned.insert(target); }
            assigned.insert("loop");
            collect_globals(node->body, assigned, referenced, seen);
            collect_globals(node->else_body, assigned, referenced, seen);
            break;
        case Node::Kind::Set:
            collect_globals_expr(node->expr, assigned, referenced, seen);
            collect_globals_expr(node->target, assigned, referenced, seen);
            if (!node->text.empty()) { assigned.insert(node->text); }
            break;
        case Node::Kind::SetBlock:
            collect_globals(node->body, assigned, referenced, seen);
            assigned.insert(node->text);
            break;
        case Node::Kind::Macro:
            assigned.insert(node->text);
            for (const auto& param : node->params) {
                collect_globals_expr(param.second, assigned, referenced, seen);
                assigned.insert(param.first);
            }
            collect_globals(node->body, assigned, referenced, seen);
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

std::string json_quote(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 2);
    out.push_back('"');
    for (const unsigned char c : text) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        default:
            if (c < 0x20) {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                out += buffer;
            } else {
                // ensure_ascii=False: pass UTF-8 bytes through unchanged.
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    out.push_back('"');
    return out;
}

// Matches Python's json.dumps(..., ensure_ascii=False) default separators, which
// is what transformers' `tojson` produces and therefore what the model saw
// during training.
std::string to_json_text_impl(const Value& value) {
    switch (value.kind()) {
    case Value::Kind::Undefined:
    case Value::Kind::None:
        return "null";
    case Value::Kind::Bool:
        return value.as_bool() ? "true" : "false";
    case Value::Kind::Int:
        return std::to_string(value.as_int());
    case Value::Kind::Double:
        return format_double(value.as_double());
    case Value::Kind::String:
        return json_quote(value.as_string());
    case Value::Kind::Array: {
        std::string out = "[";
        const Array& items = value.as_array();
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (i != 0) { out += ", "; }
            out += to_json_text_impl(items[i]);
        }
        out += "]";
        return out;
    }
    case Value::Kind::Object: {
        std::string out = "{";
        const Object& entries = value.as_object();
        for (std::size_t i = 0; i < entries.size(); ++i) {
            if (i != 0) { out += ", "; }
            out += json_quote(entries[i].first);
            out += ": ";
            out += to_json_text_impl(entries[i].second);
        }
        out += "}";
        return out;
    }
    case Value::Kind::Macro:
        fail("cannot serialise a macro to JSON");
    }
    return "null";
}

// Python-style index normalisation for sequences.
bool normalise_index(std::int64_t index, std::size_t size, std::size_t& out) {
    if (index < 0) { index += static_cast<std::int64_t>(size); }
    if (index < 0 || index >= static_cast<std::int64_t>(size)) { return false; }
    out = static_cast<std::size_t>(index);
    return true;
}

// Python slice semantics, including a negative step.
template <typename Sequence, typename Emit>
void apply_slice(const Sequence& input, const Value& start, const Value& stop, const Value& step,
                 Emit emit) {
    const auto size    = static_cast<std::int64_t>(input.size());
    std::int64_t stride = step.is_undefined() || step.is_none() ? 1 : step.as_int();
    if (stride == 0) { fail("slice step cannot be zero"); }

    auto clamp = [&](std::int64_t value, std::int64_t low, std::int64_t high) {
        return std::max(low, std::min(value, high));
    };

    std::int64_t begin;
    std::int64_t end;
    if (stride > 0) {
        begin = start.is_undefined() || start.is_none() ? 0 : start.as_int();
        end   = stop.is_undefined() || stop.is_none() ? size : stop.as_int();
        if (begin < 0) { begin += size; }
        if (end < 0) { end += size; }
        begin = clamp(begin, 0, size);
        end   = clamp(end, 0, size);
        for (std::int64_t i = begin; i < end; i += stride) { emit(input[static_cast<std::size_t>(i)]); }
        return;
    }
    begin = start.is_undefined() || start.is_none() ? size - 1 : start.as_int();
    end   = stop.is_undefined() || stop.is_none() ? -1 : stop.as_int();
    if (begin < 0 && !(start.is_undefined() || start.is_none())) { begin += size; }
    if (end < 0 && !(stop.is_undefined() || stop.is_none())) { end += size; }
    begin = clamp(begin, -1, size - 1);
    end   = clamp(end, -1, size - 1);
    for (std::int64_t i = begin; i > end; i += stride) { emit(input[static_cast<std::size_t>(i)]); }
}

class Interpreter {
public:
    explicit Interpreter(const Value& context) { scopes_.push_back({}); seed(context); }

    std::string run(const std::vector<NodePtr>& body) {
        std::string out;
        execute(body, out);
        return out;
    }

private:
    using Frame = std::vector<std::pair<std::string, Value>>;

    void seed(const Value& context) {
        if (!context.is_object()) { fail("template context must be a mapping"); }
        for (const auto& entry : context.as_object()) { scopes_.front().emplace_back(entry); }
    }

    Value* lookup(std::string_view name) {
        for (auto frame = scopes_.rbegin(); frame != scopes_.rend(); ++frame) {
            for (auto entry = frame->rbegin(); entry != frame->rend(); ++entry) {
                if (entry->first == name) { return &entry->second; }
            }
        }
        return nullptr;
    }

    // `set` writes into the innermost frame. Only `for` bodies and macro calls
    // push a frame, so this reproduces Jinja's scoping: an assignment inside a
    // loop does not survive the iteration (which is what `namespace()` is for),
    // while one inside an `if` does.
    void assign(const std::string& name, Value value) {
        Frame& frame = scopes_.back();
        for (auto& entry : frame) {
            if (entry.first == name) {
                entry.second = std::move(value);
                return;
            }
        }
        frame.emplace_back(name, std::move(value));
    }

    void execute(const std::vector<NodePtr>& body, std::string& out) {
        for (const NodePtr& node : body) { execute_node(*node, out); }
    }

    void execute_node(const Node& node, std::string& out) {
        switch (node.kind) {
        case Node::Kind::Text:
            out += node.text;
            return;
        case Node::Kind::Output:
            out += evaluate(node.expr).to_display_string();
            return;
        case Node::Kind::Do:
            (void)evaluate(node.expr);
            return;
        case Node::Kind::If: {
            for (const auto& branch : node.branches) {
                if (evaluate(branch.first).truthy()) {
                    execute(branch.second, out);
                    return;
                }
            }
            execute(node.else_body, out);
            return;
        }
        case Node::Kind::Set: {
            Value value = evaluate(node.expr);
            store(node, std::move(value));
            return;
        }
        case Node::Kind::SetBlock: {
            std::string captured;
            execute(node.body, captured);
            assign(node.text, Value::string(std::move(captured)));
            return;
        }
        case Node::Kind::Macro: {
            auto definition    = std::make_shared<MacroDef>();
            definition->name   = node.text;
            definition->params = node.params;
            definition->body   = node.body;
            assign(node.text, Value::macro(std::move(definition)));
            return;
        }
        case Node::Kind::For:
            execute_for(node, out);
            return;
        }
    }

    void store(const Node& node, Value value) {
        if (node.target == nullptr) {
            assign(node.text, std::move(value));
            return;
        }
        if (node.target->kind == Expr::Kind::Attr) {
            Value owner = evaluate(node.target->a);
            if (!owner.is_object()) {
                fail("cannot assign '" + node.target->name + "' on a non-mapping value");
            }
            owner.set(node.target->name, std::move(value));
            return;
        }
        Value owner = evaluate(node.target->a);
        Value key   = evaluate(node.target->b);
        if (owner.is_object()) {
            owner.set(key.to_display_string(), std::move(value));
            return;
        }
        if (owner.is_array()) {
            std::size_t index = 0;
            if (!normalise_index(key.as_int(), owner.as_array().size(), index)) {
                fail("list assignment index out of range");
            }
            owner.as_array()[index] = std::move(value);
            return;
        }
        fail("unsupported item assignment target");
    }

    void execute_for(const Node& node, std::string& out) {
        const Value iterable = evaluate(node.expr);
        Array items          = iterate(iterable);

        if (items.empty()) {
            execute(node.else_body, out);
            return;
        }

        for (std::size_t index = 0; index < items.size(); ++index) {
            scopes_.push_back({});
            bind_loop_targets(node.loop_targets, items[index]);

            Object loop;
            loop.emplace_back("index0", Value::integer(static_cast<std::int64_t>(index)));
            loop.emplace_back("index", Value::integer(static_cast<std::int64_t>(index) + 1));
            loop.emplace_back("first", Value::boolean(index == 0));
            loop.emplace_back("last", Value::boolean(index + 1 == items.size()));
            loop.emplace_back("length", Value::integer(static_cast<std::int64_t>(items.size())));
            loop.emplace_back(
                "revindex0",
                Value::integer(static_cast<std::int64_t>(items.size() - index - 1)));
            loop.emplace_back("revindex",
                              Value::integer(static_cast<std::int64_t>(items.size() - index)));
            // Undefined on the first/last iteration, as in Jinja.
            if (index > 0) { loop.emplace_back("previtem", items[index - 1]); }
            if (index + 1 < items.size()) { loop.emplace_back("nextitem", items[index + 1]); }
            scopes_.back().emplace_back("loop", Value::object(std::move(loop)));

            execute(node.body, out);
            scopes_.pop_back();
        }
    }

    void bind_loop_targets(const std::vector<std::string>& targets, const Value& item) {
        if (targets.size() == 1) {
            scopes_.back().emplace_back(targets[0], item);
            return;
        }
        const Array unpacked = iterate(item);
        if (unpacked.size() != targets.size()) {
            fail("cannot unpack loop value into " + std::to_string(targets.size()) + " targets");
        }
        for (std::size_t i = 0; i < targets.size(); ++i) {
            scopes_.back().emplace_back(targets[i], unpacked[i]);
        }
    }

    static Array iterate(const Value& value) {
        switch (value.kind()) {
        case Value::Kind::Array:
            return value.as_array();
        case Value::Kind::Object: {
            Array keys;
            for (const auto& entry : value.as_object()) { keys.push_back(Value::string(entry.first)); }
            return keys;
        }
        case Value::Kind::String: {
            Array chars;
            for (const char c : value.as_string()) { chars.push_back(Value::string(std::string(1, c))); }
            return chars;
        }
        case Value::Kind::Undefined:
        case Value::Kind::None:
            return {};
        default:
            fail("value is not iterable");
        }
    }

    // -- expressions --------------------------------------------------------

    Value evaluate(const ExprPtr& expr) {
        if (!expr) { return Value::undefined(); }
        switch (expr->kind) {
        case Expr::Kind::Literal:
            return expr->literal;
        case Expr::Kind::Name: {
            const Value* found = lookup(expr->name);
            return found != nullptr ? *found : Value::undefined();
        }
        case Expr::Kind::Attr:
            return get_attr(evaluate(expr->a), expr->name);
        case Expr::Kind::Item:
            return get_item(evaluate(expr->a), evaluate(expr->b));
        case Expr::Kind::Slice:
            return evaluate_slice(*expr);
        case Expr::Kind::Call:
            return evaluate_call(*expr);
        case Expr::Kind::Filter:
            return apply_filter(*expr);
        case Expr::Kind::Test:
            return apply_test(*expr);
        case Expr::Kind::Unary:
            return evaluate_unary(*expr);
        case Expr::Kind::Binary:
            return evaluate_binary(*expr);
        case Expr::Kind::Ternary:
            return evaluate(expr->a).truthy() ? evaluate(expr->b) : evaluate(expr->c);
        case Expr::Kind::ListLit: {
            Array items;
            items.reserve(expr->args.size());
            for (const ExprPtr& item : expr->args) { items.push_back(evaluate(item)); }
            return Value::array(std::move(items));
        }
        case Expr::Kind::DictLit: {
            Object entries;
            for (std::size_t i = 0; i + 1 < expr->args.size(); i += 2) {
                entries.emplace_back(evaluate(expr->args[i]).to_display_string(),
                                     evaluate(expr->args[i + 1]));
            }
            return Value::object(std::move(entries));
        }
        }
        return Value::undefined();
    }

    static Value get_attr(const Value& owner, const std::string& name) {
        if (owner.is_object()) {
            const Value* found = owner.find(name);
            return found != nullptr ? *found : Value::undefined();
        }
        return Value::undefined();
    }

    static Value get_item(const Value& owner, const Value& key) {
        if (owner.is_object()) {
            const Value* found = owner.find(key.to_display_string());
            return found != nullptr ? *found : Value::undefined();
        }
        if (owner.is_array()) {
            std::size_t index = 0;
            if (!normalise_index(key.as_int(), owner.as_array().size(), index)) {
                fail("list index out of range");
            }
            return owner.as_array()[index];
        }
        if (owner.is_string()) {
            std::size_t index = 0;
            if (!normalise_index(key.as_int(), owner.as_string().size(), index)) {
                fail("string index out of range");
            }
            return Value::string(std::string(1, owner.as_string()[index]));
        }
        return Value::undefined();
    }

    Value evaluate_slice(const Expr& expr) {
        const Value owner = evaluate(expr.a);
        const Value start = evaluate(expr.b);
        const Value stop  = evaluate(expr.c);
        const Value step  = evaluate(expr.d);
        if (owner.is_string()) {
            std::string out;
            apply_slice(owner.as_string(), start, stop, step, [&](char c) { out.push_back(c); });
            return Value::string(std::move(out));
        }
        if (owner.is_array()) {
            Array out;
            apply_slice(owner.as_array(), start, stop, step,
                        [&](const Value& item) { out.push_back(item); });
            return Value::array(std::move(out));
        }
        if (owner.is_undefined() || owner.is_none()) { return Value::array({}); }
        fail("value does not support slicing");
    }

    Value evaluate_unary(const Expr& expr) {
        const Value operand = evaluate(expr.a);
        if (expr.name == "not") { return Value::boolean(!operand.truthy()); }
        if (expr.name == "+") { return operand; }
        if (operand.is_int()) { return Value::integer(-operand.as_int()); }
        return Value::number(-operand.as_double());
    }

    Value evaluate_binary(const Expr& expr) {
        const std::string& op = expr.name;
        if (op == "and") {
            const Value left = evaluate(expr.a);
            return left.truthy() ? evaluate(expr.b) : left;
        }
        if (op == "or") {
            const Value left = evaluate(expr.a);
            return left.truthy() ? left : evaluate(expr.b);
        }

        const Value left  = evaluate(expr.a);
        const Value right = evaluate(expr.b);

        if (op == "~") {
            return Value::string(left.to_display_string() + right.to_display_string());
        }
        if (op == "==") { return Value::boolean(left.equals(right)); }
        if (op == "!=") { return Value::boolean(!left.equals(right)); }
        if (op == "in") { return Value::boolean(contains(right, left)); }
        if (op == "not in") { return Value::boolean(!contains(right, left)); }

        if (op == "<" || op == ">" || op == "<=" || op == ">=") {
            const int order = compare(left, right);
            if (op == "<") { return Value::boolean(order < 0); }
            if (op == ">") { return Value::boolean(order > 0); }
            if (op == "<=") { return Value::boolean(order <= 0); }
            return Value::boolean(order >= 0);
        }

        if (op == "+") {
            if (left.is_string() && right.is_string()) {
                return Value::string(left.as_string() + right.as_string());
            }
            if (left.is_array() && right.is_array()) {
                Array joined = left.as_array();
                const Array& tail = right.as_array();
                joined.insert(joined.end(), tail.begin(), tail.end());
                return Value::array(std::move(joined));
            }
        }

        const bool integral = left.is_int() && right.is_int();
        if (op == "+") {
            return integral ? Value::integer(left.as_int() + right.as_int())
                            : Value::number(left.as_double() + right.as_double());
        }
        if (op == "-") {
            return integral ? Value::integer(left.as_int() - right.as_int())
                            : Value::number(left.as_double() - right.as_double());
        }
        if (op == "*") {
            if (left.is_string() && right.is_int()) {
                std::string repeated;
                for (std::int64_t i = 0; i < right.as_int(); ++i) { repeated += left.as_string(); }
                return Value::string(std::move(repeated));
            }
            return integral ? Value::integer(left.as_int() * right.as_int())
                            : Value::number(left.as_double() * right.as_double());
        }
        if (op == "/") {
            if (right.as_double() == 0.0) { fail("division by zero in template expression"); }
            return Value::number(left.as_double() / right.as_double());
        }
        if (op == "//") {
            if (right.as_int() == 0) { fail("division by zero in template expression"); }
            return Value::integer(static_cast<std::int64_t>(
                std::floor(left.as_double() / right.as_double())));
        }
        if (op == "%") {
            if (right.as_int() == 0) { fail("modulo by zero in template expression"); }
            const std::int64_t a = left.as_int();
            const std::int64_t b = right.as_int();
            // Python's modulo takes the sign of the divisor.
            return Value::integer(((a % b) + b) % b);
        }
        fail("unsupported operator '" + op + "' in template expression");
    }

    static bool contains(const Value& haystack, const Value& needle) {
        if (haystack.is_string()) {
            if (!needle.is_string()) { return false; }
            return haystack.as_string().find(needle.as_string()) != std::string::npos;
        }
        if (haystack.is_array()) {
            for (const Value& item : haystack.as_array()) {
                if (item.equals(needle)) { return true; }
            }
            return false;
        }
        if (haystack.is_object()) { return haystack.find(needle.to_display_string()) != nullptr; }
        return false;
    }

    static int compare(const Value& left, const Value& right) {
        if (left.is_string() && right.is_string()) {
            const int order = left.as_string().compare(right.as_string());
            return order < 0 ? -1 : (order > 0 ? 1 : 0);
        }
        const double a = left.as_double();
        const double b = right.as_double();
        return a < b ? -1 : (a > b ? 1 : 0);
    }

    // -- calls, filters, tests ---------------------------------------------

    std::vector<Value> evaluate_args(const Expr& expr) {
        std::vector<Value> args;
        args.reserve(expr.args.size());
        for (const ExprPtr& arg : expr.args) { args.push_back(evaluate(arg)); }
        return args;
    }

    Value evaluate_call(const Expr& expr) {
        // Method call: the callee is an attribute access.
        if (expr.a && expr.a->kind == Expr::Kind::Attr) {
            const Value owner = evaluate(expr.a->a);
            if (!owner.is_macro()) {
                return call_method(owner, expr.a->name, evaluate_args(expr));
            }
        }
        if (expr.a && expr.a->kind == Expr::Kind::Name) {
            const std::string& name = expr.a->name;
            if (lookup(name) == nullptr) { return call_builtin(name, expr); }
        }
        const Value callee = evaluate(expr.a);
        if (!callee.is_macro()) { fail("attempted to call a value that is not a macro"); }
        return call_macro(callee.as_macro(), evaluate_args(expr), expr);
    }

    Value call_builtin(const std::string& name, const Expr& expr) {
        if (name == "raise_exception") {
            const std::vector<Value> args = evaluate_args(expr);
            fail(args.empty() ? "template raised an exception"
                              : args[0].to_display_string());
        }
        if (name == "namespace") {
            Object entries;
            for (const auto& kwarg : expr.kwargs) {
                entries.emplace_back(kwarg.first, evaluate(kwarg.second));
            }
            return Value::object(std::move(entries));
        }
        if (name == "range") {
            const std::vector<Value> args = evaluate_args(expr);
            std::int64_t begin = 0;
            std::int64_t end   = 0;
            std::int64_t step  = 1;
            if (args.size() == 1) {
                end = args[0].as_int();
            } else if (args.size() >= 2) {
                begin = args[0].as_int();
                end   = args[1].as_int();
                if (args.size() >= 3) { step = args[2].as_int(); }
            }
            if (step == 0) { fail("range() step cannot be zero"); }
            Array items;
            if (step > 0) {
                for (std::int64_t i = begin; i < end; i += step) { items.push_back(Value::integer(i)); }
            } else {
                for (std::int64_t i = begin; i > end; i += step) { items.push_back(Value::integer(i)); }
            }
            return Value::array(std::move(items));
        }
        if (name == "dict") {
            Object entries;
            for (const auto& kwarg : expr.kwargs) {
                entries.emplace_back(kwarg.first, evaluate(kwarg.second));
            }
            return Value::object(std::move(entries));
        }
        fail("unknown function '" + name + "' called by the template");
    }

    Value call_macro(const MacroDef& macro, std::vector<Value> args, const Expr& expr) {
        scopes_.push_back({});
        for (std::size_t i = 0; i < macro.params.size(); ++i) {
            const auto& [name, default_value] = macro.params[i];
            Value bound;
            if (i < args.size()) {
                bound = std::move(args[i]);
            } else {
                bool from_kwarg = false;
                for (const auto& kwarg : expr.kwargs) {
                    if (kwarg.first == name) {
                        bound      = evaluate(kwarg.second);
                        from_kwarg = true;
                        break;
                    }
                }
                if (!from_kwarg) {
                    bound = default_value != nullptr ? evaluate(default_value) : Value::undefined();
                }
            }
            scopes_.back().emplace_back(name, std::move(bound));
        }
        std::string out;
        execute(macro.body, out);
        scopes_.pop_back();
        return Value::string(std::move(out));
    }

    static Value call_method(const Value& owner, const std::string& name,
                             const std::vector<Value>& args) {
        if (owner.is_string()) { return call_string_method(owner.as_string(), name, args); }
        if (owner.is_object()) {
            if (name == "items") {
                Array items;
                for (const auto& entry : owner.as_object()) {
                    items.push_back(Value::array({Value::string(entry.first), entry.second}));
                }
                return Value::array(std::move(items));
            }
            if (name == "keys") {
                Array keys;
                for (const auto& entry : owner.as_object()) {
                    keys.push_back(Value::string(entry.first));
                }
                return Value::array(std::move(keys));
            }
            if (name == "values") {
                Array values;
                for (const auto& entry : owner.as_object()) { values.push_back(entry.second); }
                return Value::array(std::move(values));
            }
            if (name == "get") {
                if (args.empty()) { fail("get() requires a key"); }
                const Value* found = owner.find(args[0].to_display_string());
                if (found != nullptr) { return *found; }
                return args.size() > 1 ? args[1] : Value::none();
            }
        }
        if (owner.is_array() && name == "append") {
            if (args.empty()) { fail("append() requires a value"); }
            // Reference semantics: mutate through the shared handle.
            const_cast<Array&>(owner.as_array()).push_back(args[0]);
            return Value::none();
        }
        if (owner.is_undefined() || owner.is_none()) {
            fail("cannot call '" + name + "' on an undefined value");
        }
        fail("unsupported method '" + name + "' in template");
    }

    static Value call_string_method(const std::string& text, const std::string& name,
                                    const std::vector<Value>& args) {
        auto arg_string = [&](std::size_t index) -> const std::string& {
            if (index >= args.size() || !args[index].is_string()) {
                fail("'" + name + "' expects a string argument");
            }
            return args[index].as_string();
        };

        if (name == "split") {
            Array parts;
            if (args.empty()) {
                // Whitespace split, discarding empty fields, as Python does.
                std::size_t i = 0;
                while (i < text.size()) {
                    while (i < text.size() && is_space(text[i])) { ++i; }
                    const std::size_t begin = i;
                    while (i < text.size() && !is_space(text[i])) { ++i; }
                    if (i > begin) { parts.push_back(Value::string(text.substr(begin, i - begin))); }
                }
                return Value::array(std::move(parts));
            }
            const std::string& sep = arg_string(0);
            if (sep.empty()) { fail("split() separator must not be empty"); }
            std::size_t begin = 0;
            for (;;) {
                const std::size_t hit = text.find(sep, begin);
                if (hit == std::string::npos) {
                    parts.push_back(Value::string(text.substr(begin)));
                    break;
                }
                parts.push_back(Value::string(text.substr(begin, hit - begin)));
                begin = hit + sep.size();
            }
            return Value::array(std::move(parts));
        }
        if (name == "startswith") {
            const std::string& prefix = arg_string(0);
            return Value::boolean(text.size() >= prefix.size() &&
                                  text.compare(0, prefix.size(), prefix) == 0);
        }
        if (name == "endswith") {
            const std::string& suffix = arg_string(0);
            return Value::boolean(text.size() >= suffix.size() &&
                                  text.compare(text.size() - suffix.size(), suffix.size(),
                                               suffix) == 0);
        }
        if (name == "strip" || name == "lstrip" || name == "rstrip") {
            // With an argument, Python strips any character in the set.
            const std::string cutset = args.empty() ? std::string(" \t\r\n\f\v") : arg_string(0);
            std::size_t begin = 0;
            std::size_t end   = text.size();
            if (name != "rstrip") {
                while (begin < end && cutset.find(text[begin]) != std::string::npos) { ++begin; }
            }
            if (name != "lstrip") {
                while (end > begin && cutset.find(text[end - 1]) != std::string::npos) { --end; }
            }
            return Value::string(text.substr(begin, end - begin));
        }
        if (name == "lower") {
            std::string out = text;
            std::transform(out.begin(), out.end(), out.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return Value::string(std::move(out));
        }
        if (name == "upper") {
            std::string out = text;
            std::transform(out.begin(), out.end(), out.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            return Value::string(std::move(out));
        }
        if (name == "replace") {
            const std::string& from = arg_string(0);
            const std::string& to   = arg_string(1);
            if (from.empty()) { return Value::string(text); }
            std::string out;
            std::size_t begin = 0;
            for (;;) {
                const std::size_t hit = text.find(from, begin);
                if (hit == std::string::npos) {
                    out += text.substr(begin);
                    break;
                }
                out += text.substr(begin, hit - begin);
                out += to;
                begin = hit + from.size();
            }
            return Value::string(std::move(out));
        }
        if (name == "join") {
            if (args.empty()) { fail("join() requires a sequence"); }
            std::string out;
            const Array& items = args[0].as_array();
            for (std::size_t i = 0; i < items.size(); ++i) {
                if (i != 0) { out += text; }
                out += items[i].to_display_string();
            }
            return Value::string(std::move(out));
        }
        fail("unsupported string method '" + name + "' in template");
    }

    Value apply_filter(const Expr& expr) {
        const std::string& name = expr.name;
        Value value             = evaluate(expr.a);
        const std::vector<Value> args = evaluate_args(expr);

        if (name == "string") { return Value::string(value.to_display_string()); }
        // Autoescaping is off, so `safe` only marks a value as already-escaped
        // and is an identity here. Chat templates apply it after `tojson` when
        // rendering non-string tool-call arguments.
        if (name == "safe") { return value; }
        if (name == "trim") { return Value::string(trim_both(value.to_display_string())); }
        if (name == "lower" || name == "upper" || name == "replace") {
            return call_string_method(value.to_display_string(), name, args);
        }
        if (name == "capitalize") {
            std::string text = value.to_display_string();
            if (!text.empty()) {
                text[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(text[0])));
            }
            return Value::string(std::move(text));
        }
        if (name == "tojson") { return Value::string(to_json_text_impl(value)); }
        if (name == "length" || name == "count") {
            if (value.is_string()) {
                return Value::integer(static_cast<std::int64_t>(value.as_string().size()));
            }
            if (value.is_array()) {
                return Value::integer(static_cast<std::int64_t>(value.as_array().size()));
            }
            if (value.is_object()) {
                return Value::integer(static_cast<std::int64_t>(value.as_object().size()));
            }
            if (value.is_undefined() || value.is_none()) { return Value::integer(0); }
            fail("length filter applied to a value with no length");
        }
        if (name == "join") {
            const std::string separator = args.empty() ? std::string() : args[0].to_display_string();
            std::string out;
            const Array items = iterate(value);
            for (std::size_t i = 0; i < items.size(); ++i) {
                if (i != 0) { out += separator; }
                out += items[i].to_display_string();
            }
            return Value::string(std::move(out));
        }
        if (name == "items") {
            Array items;
            for (const auto& entry : value.as_object()) {
                items.push_back(Value::array({Value::string(entry.first), entry.second}));
            }
            return Value::array(std::move(items));
        }
        if (name == "list") { return Value::array(iterate(value)); }
        if (name == "first" || name == "last") {
            const Array items = iterate(value);
            if (items.empty()) { return Value::undefined(); }
            return name == "first" ? items.front() : items.back();
        }
        if (name == "reverse") {
            Array items = iterate(value);
            std::reverse(items.begin(), items.end());
            return Value::array(std::move(items));
        }
        if (name == "default" || name == "d") {
            const bool boolean_mode = args.size() > 1 && args[1].truthy();
            const bool missing      = boolean_mode ? !value.truthy() : value.is_undefined();
            if (missing) { return args.empty() ? Value::string("") : args[0]; }
            return value;
        }
        if (name == "int") {
            if (value.is_string()) {
                try {
                    return Value::integer(std::stoll(value.as_string()));
                } catch (const std::exception&) {
                    return args.empty() ? Value::integer(0) : args[0];
                }
            }
            return Value::integer(value.as_int());
        }
        if (name == "float") { return Value::number(value.as_double()); }
        if (name == "abs") {
            if (value.is_int()) { return Value::integer(std::llabs(value.as_int())); }
            return Value::number(std::fabs(value.as_double()));
        }
        fail("unsupported filter '" + name + "' in template");
    }

    Value apply_test(const Expr& expr) {
        const std::string& name = expr.name;
        const Value value       = evaluate(expr.a);
        bool result             = false;

        if (name == "defined") {
            result = !value.is_undefined();
        } else if (name == "undefined") {
            result = value.is_undefined();
        } else if (name == "none" || name == "null") {
            result = value.is_none();
        } else if (name == "string") {
            result = value.is_string();
        } else if (name == "number") {
            result = value.is_number();
        } else if (name == "integer" || name == "int") {
            result = value.is_int();
        } else if (name == "float") {
            result = value.is_double();
        } else if (name == "boolean") {
            result = value.is_bool();
        } else if (name == "mapping") {
            result = value.is_object();
        } else if (name == "sequence") {
            result = value.is_array() || value.is_string() || value.is_object();
        } else if (name == "iterable") {
            result = value.is_array() || value.is_string() || value.is_object();
        } else if (name == "true") {
            result = value.is_bool() && value.as_bool();
        } else if (name == "false") {
            result = value.is_bool() && !value.as_bool();
        } else if (name == "equalto" || name == "eq" || name == "sameas") {
            const std::vector<Value> args = evaluate_args(expr);
            if (args.empty()) { fail("'" + name + "' test requires an argument"); }
            result = value.equals(args[0]);
        } else if (name == "in") {
            const std::vector<Value> args = evaluate_args(expr);
            if (args.empty()) { fail("'in' test requires an argument"); }
            result = contains(args[0], value);
        } else if (name == "odd") {
            result = (value.as_int() % 2) != 0;
        } else if (name == "even") {
            result = (value.as_int() % 2) == 0;
        } else {
            fail("unsupported test '" + name + "' in template");
        }
        return Value::boolean(expr.negated ? !result : result);
    }

    std::vector<Frame> scopes_;
};

} // namespace

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

struct Template::Impl {
    std::vector<NodePtr> body;
    std::vector<std::string> referenced_globals;
};

Template Template::parse(std::string_view source) {
    auto impl = std::make_shared<Impl>();
    StatementParser parser(scan_blocks(source));
    impl->body = parser.parse_body({}, nullptr);

    std::unordered_set<std::string> assigned;
    std::unordered_set<std::string> seen;
    collect_globals(impl->body, assigned, impl->referenced_globals, seen);

    Template compiled;
    compiled.impl_ = std::move(impl);
    return compiled;
}

std::string Template::render(const Value& context) const {
    if (!impl_) { fail("template has not been parsed"); }
    Interpreter interpreter(context);
    return interpreter.run(impl_->body);
}

const std::vector<std::string>& Template::referenced_globals() const {
    static const std::vector<std::string> kEmpty;
    return impl_ ? impl_->referenced_globals : kEmpty;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal::jinja
