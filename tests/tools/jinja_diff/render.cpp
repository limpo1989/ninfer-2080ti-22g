// Differential harness: render a jinja template with our interpreter so the
// output can be diffed against Python's jinja2 on the same JSON context.
#include "targets/qwen3_6/impl/frontend/jinja.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

namespace jj = ninfer::targets::qwen3_6::frontend_internal::jinja;
using Json   = nlohmann::ordered_json;

jj::Value from_json(const Json& node) {
    if (node.is_null()) { return jj::Value::none(); }
    if (node.is_boolean()) { return jj::Value::boolean(node.get<bool>()); }
    if (node.is_number_integer() || node.is_number_unsigned()) {
        return jj::Value::integer(node.get<std::int64_t>());
    }
    if (node.is_number_float()) { return jj::Value::number(node.get<double>()); }
    if (node.is_string()) { return jj::Value::string(node.get<std::string>()); }
    if (node.is_array()) {
        jj::Array items;
        for (const auto& item : node) { items.push_back(from_json(item)); }
        return jj::Value::array(std::move(items));
    }
    jj::Object entries;
    for (auto it = node.begin(); it != node.end(); ++it) {
        entries.emplace_back(it.key(), from_json(it.value()));
    }
    return jj::Value::object(std::move(entries));
}

std::string slurp(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: render <template.jinja> <context.json>\n";
        return 2;
    }
    try {
        const jj::Template compiled = jj::Template::parse(slurp(argv[1]));
        if (argc > 3 && std::string(argv[3]) == "--globals") {
            for (const std::string& name : compiled.referenced_globals()) {
                std::cout << name << "\n";
            }
            return 0;
        }
        std::cout << compiled.render(from_json(Json::parse(slurp(argv[2]))));
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
