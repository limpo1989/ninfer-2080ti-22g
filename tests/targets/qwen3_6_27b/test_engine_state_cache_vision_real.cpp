#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

constexpr std::size_t kGiB = 1ULL << 30;

ninfer::EngineOptions engine_options(const char* artifact, const std::filesystem::path& cache) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk             = 1024;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = true;
    options.state_cache_dir           = cache;
    options.state_cache_max_bytes     = 3 * kGiB;
    options.state_cache_ram_bytes     = 3 * kGiB;
    options.state_cache_idle_ms       = 0;
    return options;
}

std::vector<std::uint8_t> gradient_ppm(int width = 1024, int height = 1024) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < width * height; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::MessagePart image_part(const std::vector<std::uint8_t>& bytes) {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = bytes;
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    return image;
}

ninfer::ChatMessage text_message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return message;
}

ninfer::ChatMessage assistant_message(const ninfer::GenerationResult& result) {
    ninfer::ChatMessage message = text_message(ninfer::ChatRole::Assistant, result.content);
    message.reasoning_content   = result.reasoning;
    return message;
}

ninfer::PromptInput conversation(const std::vector<std::uint8_t>& image,
                                 const std::vector<ninfer::GenerationResult>& responses) {
    ninfer::ChatMessage initial;
    initial.role = ninfer::ChatRole::User;
    initial.parts.push_back(image_part(image));
    initial.parts.push_back(
        ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                            .text  = "Describe the dominant visual pattern briefly.",
                            .media = {}});

    ninfer::PromptInput input;
    input.messages.push_back(std::move(initial));
    for (std::size_t index = 0; index < responses.size(); ++index) {
        input.messages.push_back(assistant_message(responses[index]));
        input.messages.push_back(text_message(
            ninfer::ChatRole::User, index == 0 ? "Give one additional visual detail."
                                               : "Summarize both observations in one sentence."));
    }
    input.options.enable_thinking = false;
    return input;
}

ninfer::RequestOptions request_options(bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 4;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

void evict_resident_lane(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> unrelated{248045, 846, 198, 5834, 248046, 198};
    ninfer::RequestOptions options            = request_options(false);
    options.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(unrelated), options);
    if (result.generated_token_ids.size() != 1) {
        throw std::runtime_error("resident eviction request did not complete");
    }
}

int exercise(const char* artifact) {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "ninfer-vision-state-XXXXXX").string();
    const char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
        std::cerr << "cannot create Vision state-cache test directory\n";
        return 1;
    }
    const std::filesystem::path cache(created);
    const std::vector<std::uint8_t> image = gradient_ppm();
    std::vector<ninfer::GenerationResult> history;
    try {
        {
            ninfer::Engine engine(engine_options(artifact, cache));
            history.push_back(
                engine.generate(engine.prepare(conversation(image, {})), request_options(true)));
            if (!history.back().prompt.has_media ||
                history.back().generated_token_ids.size() != 4) {
                throw std::runtime_error("Vision snapshot source request did not complete");
            }
            evict_resident_lane(engine);
            history.push_back(engine.generate(engine.prepare(conversation(image, history)),
                                              request_options(true)));
            const auto& restored = history.back();
            if (restored.timings.state_cache_source != 1 || restored.reused_prompt_tokens == 0 ||
                restored.timings.vision_seconds != 0.0) {
                throw std::runtime_error("Vision continuation did not restore from RAM");
            }
        }

        {
            ninfer::Engine engine(engine_options(artifact, cache));
            history.push_back(engine.generate(engine.prepare(conversation(image, history)),
                                              request_options(true)));
            const auto& restored = history.back();
            if (restored.timings.state_cache_source != 2 || restored.reused_prompt_tokens == 0 ||
                restored.timings.vision_seconds != 0.0) {
                throw std::runtime_error("Vision continuation did not restore from disk");
            }
        }

        {
            ninfer::Engine engine(engine_options(artifact, cache));
            const ninfer::GenerationResult restored = engine.generate(
                engine.prepare(conversation(image, history)), request_options(true));
            if (restored.timings.state_cache_source != 2 || restored.reused_prompt_tokens == 0 ||
                restored.timings.vision_seconds != 0.0) {
                throw std::runtime_error("chained Vision continuation did not restore from disk");
            }
        }

        std::vector<std::uint8_t> changed = image;
        changed.back() ^= 0x5aU;
        {
            ninfer::Engine engine(engine_options(artifact, cache));
            const ninfer::GenerationResult miss = engine.generate(
                engine.prepare(conversation(changed, history)), request_options(true));
            if (miss.timings.state_cache_source != 0 || !(miss.timings.vision_seconds > 0.0)) {
                throw std::runtime_error("changed image incorrectly restored a Vision snapshot");
            }
        }
        std::filesystem::remove_all(cache);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << " (cache: " << cache << ")\n";
        return 1;
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }
    if (const int result = exercise(artifact); result != 0) { return result; }
    std::cout << "Vision RAM/disk state-cache restore checks passed\n";
    return 0;
}
