// CPU-only inspection of token IDs from NINFER_TRACE_PREFIX diagnostics.
#include "artifact/reader.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include <nlohmann/json.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ninfer_token_decode MODEL.ninfer TOKEN_ID...\n";
        return 2;
    }
    try {
        ninfer::artifact::Reader reader(argv[1]);
        const auto resource = [&](const char* name) {
            const auto data = reader.payload(name).data;
            return std::string(reinterpret_cast<const char*>(data.data()), data.size());
        };
        const auto vocabulary = resource("frontend/tokenizer.json");
        const auto config = resource("frontend/tokenizer_config.json");
        const auto generation = resource("frontend/generation_config.json");
        ninfer::targets::qwen3_6::frontend_internal::Tokenizer tokenizer({vocabulary, config, generation});
        for (int i = 2; i < argc; ++i) {
            const int id = std::stoi(argv[i]);
            const auto bytes = tokenizer.decode_token_bytes(id);
            std::ostringstream hex;
            for (unsigned char c : bytes) { hex << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); }
            std::cout << nlohmann::json{{"id", id}, {"text", bytes}, {"hex", hex.str()}}
                .dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
