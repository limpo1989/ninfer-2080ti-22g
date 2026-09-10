#include "targets/qwen3_6/impl/runtime/program.h"
#include <nlohmann/json.hpp>
#include <bit>
#include <cstdio>
#include <sstream>
#include <iomanip>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {
using SnapshotJson = nlohmann::json;
std::string snapshot_alias(std::uint32_t n, std::uint64_t hash) {
    std::ostringstream out; out << n << '-' << std::hex << hash; return out.str();
}
std::vector<std::string> snapshot_aliases(const std::vector<TokenId>& tokens,
                                          std::uint32_t minimum_frontier = 0) {
    std::vector<std::string> aliases;
    if (tokens.size() > minimum_frontier) { aliases.reserve(tokens.size() - minimum_frontier); }
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        hash ^= static_cast<std::uint32_t>(tokens[i]); hash *= 1099511628211ULL;
        if (i + 1 > minimum_frontier) {
            aliases.push_back(snapshot_alias(static_cast<std::uint32_t>(i + 1), hash));
        }
    }
    return aliases;
}
std::string snapshot_image_key(const std::vector<TokenId>& tokens) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const TokenId token : tokens) {
        hash ^= static_cast<std::uint32_t>(token); hash *= 1099511628211ULL;
    }
    return runtime::state_hash_key(snapshot_alias(static_cast<std::uint32_t>(tokens.size()), hash));
}
std::vector<Tensor> snapshot_tensors(ProgramImplCore& p, std::uint32_t lane, bool checkpoint) {
    std::vector<Tensor> result;
    result.push_back(p.sequences[lane].tail_hidden);
    if (checkpoint) result.push_back(p.sequences[lane].rewrite_checkpoint_hidden);
    for (const auto slot : {LinearStateSlots::current_state_slot(lane, p.max_concurrency),
                            LinearStateSlots::rewrite_checkpoint_state_slot(lane, p.max_concurrency)}) {
        if (!checkpoint && slot >= static_cast<int>(p.max_concurrency)) continue;
        for (std::uint32_t layer = 0; layer < p.decoder->linear_attention.layer_count(); ++layer) {
            result.push_back(p.decoder->linear_attention.conv_slot(layer, slot));
            result.push_back(p.decoder->linear_attention.recurrent_slot(layer, slot));
        }
    }
    for (auto* cache : {&p.decoder->text_kv, p.decoder->mtp_cache()}) {
        if (!cache || !cache->kvarn()) continue;
        for (std::uint32_t layer = 0; layer < cache->layers(); ++layer) {
            const auto view = cache->kvarn_batch_layer_view(layer);
            result.push_back(view.stage_k.slice(3, lane, 1));
            result.push_back(view.stage_v.slice(3, lane, 1));
            if (checkpoint) {
                const auto checkpoint_tail =
                    cache->kvarn_rewrite_checkpoint_tail_view(layer, lane);
                result.push_back(checkpoint_tail.k);
                result.push_back(checkpoint_tail.v);
            }
        }
    }
    return result;
}
std::size_t cache_page_bytes(const qwen3_6::PagedKVCache& cache, std::uint32_t pages) {
    std::size_t size = 0;
    for (std::size_t i = 0; i < cache.pool().plane_count(); ++i)
        size += cache.pool().plane(i).bytes() / cache.pool().page_group_count() * pages;
    return size;
}
std::shared_ptr<runtime::StateSnapshotImage> capture_image(ProgramImplCore& p, std::uint32_t lane,
                                                           std::string key) {
    const auto& s = p.sequences[lane];
    auto image = std::make_shared<runtime::StateSnapshotImage>();
    auto keys = snapshot_aliases(s.ledger);
    image->key = std::move(key);
    if (s.execution_frontier) image->aliases.push_back(keys.at(s.execution_frontier - 1));
    if (s.rewrite_checkpoint.valid) image->aliases.push_back(keys.at(s.rewrite_checkpoint.frontier - 1));
    const auto main_pages = s.kv->text.mapped_page_count();
    const auto mtp_pages = s.kv->backend ? s.kv->backend->mapped_page_count() : 0U;
    SnapshotJson j{{"version", 3}, {"execution", s.execution_frontier}, {"ledger_frontier", s.ledger_frontier},
        {"text_valid", s.text_kv_valid}, {"mtp_valid", s.mtp_kv_valid}, {"rope_delta", s.rope_delta},
        {"checkpoint", s.rewrite_checkpoint.valid}, {"checkpoint_kind", static_cast<int>(s.rewrite_checkpoint.kind)},
        {"checkpoint_frontier", s.rewrite_checkpoint.frontier}, {"tail_valid", s.tail_hidden_valid},
        {"ledger", s.ledger}, {"identity", SnapshotJson::binary(s.prefix_identity.serialize())},
        {"main_pages", main_pages}, {"mtp_pages", mtp_pages}};
    const auto tensors = snapshot_tensors(p, lane, s.rewrite_checkpoint.valid);
    auto bytes = cache_page_bytes(p.decoder->text_kv, main_pages);
    if (s.kv->backend) bytes += cache_page_bytes(*p.decoder->mtp_cache(), mtp_pages);
    for (const auto& tensor : tensors) bytes += tensor.bytes();
    image->metadata = SnapshotJson::to_cbor(j);
    if (!p.state_cache->can_store(bytes, image->metadata.size())) return nullptr;
    image->payload.resize(bytes);
    std::size_t offset = 0;
    for (const auto& tensor : tensors) {
        if (!tensor.is_contiguous()) throw std::logic_error("snapshot state is not contiguous");
        CUDA_CHECK(cudaMemcpyAsync(image->payload.data() + offset, tensor.data, tensor.bytes(),
                                   cudaMemcpyDeviceToHost, p.device.stream));
        offset += tensor.bytes();
    }
    const auto main_bytes = p.decoder->text_kv.snapshot_bytes(s.kv->text);
    p.decoder->text_kv.snapshot_to_host(s.kv->text,
        std::span<std::uint8_t>(image->payload).subspan(offset, main_bytes), p.device.stream);
    offset += main_bytes;
    if (s.kv->backend) p.decoder->mtp_cache()->snapshot_to_host(*s.kv->backend,
        std::span<std::uint8_t>(image->payload).subspan(offset), p.device.stream);
    p.device.synchronize(); // all source pages remain owned until their host copy is complete
    return image;
}
}

void ProgramImplCore::configure_state_cache(const EngineOptions& options) {
    if (options.state_cache_dir.empty() || options.state_cache_max_bytes == 0) return;
    if (options.enable_vision || speculative_backend == SpeculativeBackend::DFlash)
        throw std::invalid_argument("state-cache phase 1 supports text with ordinary decode or MTP");
    static_assert(std::endian::native == std::endian::little);
    // Conservative cache namespace: an executable rebuild or artifact replacement invalidates
    // existing images. No raw pointer, physical page ID, or CUDA Graph is persisted.
    SnapshotJson identity{{"state_format", 3}, {"artifact", runtime::state_file_identity(options.artifact_path)},
        {"executable", runtime::state_file_identity("/proc/self/exe")},
        {"sm", device.sm()}, {"capacity", capacity}, {"kv_storage", static_cast<int>(kv_storage)},
        {"spec", static_cast<int>(speculative_backend)}, {"draft", draft_window},
        {"proposal", static_cast<int>(proposal_head)}, {"prefill_chunk", prefill_chunk},
        {"style", static_cast<int>(options.chat_style)}, {"template", options.chat_template_override}};
    state_cache = std::make_unique<runtime::StateSnapshotCache>(options.state_cache_dir,
        options.state_cache_max_bytes, options.state_cache_ram_bytes, identity.dump());
}
runtime::StateSnapshotLoad ProgramImplCore::lookup_state(const PreparedPromptData& prompt,
                                                         std::uint32_t minimum_frontier) {
    if (!state_cache || !prompt.identity.reusable || prompt.has_media()) return {};
    auto aliases = snapshot_aliases(prompt.token_ids, minimum_frontier);
    std::reverse(aliases.begin(), aliases.end());
    return state_cache->lookup(aliases);
}
void ProgramImplCore::save_state(std::uint32_t lane) noexcept {
    if (!state_cache || !has_retained_lane(lane) || !sequences[lane].kv ||
        sequences[lane].ledger.size() < 256) return; // avoid fixed GDN snapshot cost for tiny/warmup requests
    try {
        const std::string key = snapshot_image_key(sequences[lane].ledger);
        if (state_cache->contains(key)) return;
        const auto started = Clock::now();
        auto image = capture_image(*this, lane, key);
        if (image) {
            const auto size = image->payload.size();
            const bool kept = state_cache->put(std::move(image));
            std::fprintf(stderr, "[state-cache] capture lane=%u bytes=%zu kept=%d seconds=%.4f\n",
                lane, size, kept, std::chrono::duration<double>(Clock::now() - started).count());
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[state-cache] capture skipped: %s\n", error.what());
    }
}

bool ProgramImplCore::restore_state(std::uint32_t lane, const PreparedPromptData& prompt,
                                   const runtime::StateSnapshotImage& image) noexcept {
    if (!state_cache || lane >= max_concurrency || requests[lane].lifecycle == Lifecycle::Active ||
        requests[lane].lifecycle == Lifecycle::Pending || requests[lane].lifecycle == Lifecycle::Prefilling) return false;
    bool mutated = false;
    try {
        const auto j = SnapshotJson::from_cbor(image.metadata);
        if (j.at("version") != 3) return false;
        auto ledger = j.at("ledger").get<std::vector<TokenId>>();
        const auto execution = j.at("execution").get<std::uint32_t>();
        const auto text_valid = j.at("text_valid").get<std::uint32_t>();
        const auto mtp_valid = j.at("mtp_valid").get<std::uint32_t>();
        const auto checkpoint = j.at("checkpoint").get<bool>();
        const auto checkpoint_frontier = j.at("checkpoint_frontier").get<std::uint32_t>();
        const auto kind = j.at("checkpoint_kind").get<int>();
        const auto main_pages = j.at("main_pages").get<std::uint32_t>();
        const auto mtp_pages = j.at("mtp_pages").get<std::uint32_t>();
        if (ledger.empty() || ledger.size() > capacity + 1ULL || ledger.size() != j.at("ledger_frontier") ||
            execution > text_valid || text_valid > capacity || mtp_valid > text_valid ||
            execution > ledger.size() || main_pages != pages_for_tokens(text_valid) ||
            mtp_pages != pages_for_tokens(mtp_valid) || main_pages == 0 ||
            (speculative_backend == SpeculativeBackend::Mtp) != (mtp_pages != 0) ||
            (checkpoint && (checkpoint_frontier == 0 || checkpoint_frontier > text_valid || kind < 0 || kind > 1))) return false;
        qwen3_6::detail::ResidentPrefixIdentity identity;
        identity.deserialize(j.at("identity").get_binary());
        if (identity.size() != ledger.size()) return false;
        const bool append = execution && qwen3_6::detail::prefix_matches(prompt, ledger, identity, execution);
        const bool rewrite = checkpoint && qwen3_6::detail::prefix_matches(prompt, ledger, identity, checkpoint_frontier);
        if (!append && !rewrite) return false;
        const auto tensors = snapshot_tensors(*this, lane, checkpoint);
        std::size_t expected = cache_page_bytes(decoder->text_kv, main_pages);
        if (mtp_pages) expected += cache_page_bytes(*decoder->mtp_cache(), mtp_pages);
        for (const auto& tensor : tensors) expected += tensor.bytes();
        if (expected != image.payload.size()) return false;

        auto& s = sequences[lane];
        evict_retained_lane(lane); mutated = true;
        reserve_sequence_kv(s, main_pages, mtp_pages);
        s.kv->text.materialize_pages(main_pages, device.stream);
        if (mtp_pages) s.kv->backend->materialize_pages(mtp_pages, device.stream);
        std::size_t offset = 0;
        for (const auto& tensor : tensors) {
            CUDA_CHECK(cudaMemcpyAsync(tensor.data, image.payload.data() + offset, tensor.bytes(),
                                       cudaMemcpyHostToDevice, device.stream));
            offset += tensor.bytes();
        }
        const auto main_bytes = cache_page_bytes(decoder->text_kv, main_pages);
        decoder->text_kv.restore_from_host(s.kv->text,
            std::span<const std::uint8_t>(image.payload).subspan(offset, main_bytes), device.stream);
        offset += main_bytes;
        if (mtp_pages) decoder->mtp_cache()->restore_from_host(*s.kv->backend,
            std::span<const std::uint8_t>(image.payload).subspan(offset), device.stream);
        device.synchronize();
        s.ledger = std::move(ledger); s.prefix_identity = std::move(identity);
        s.execution_frontier = execution; s.ledger_frontier = s.ledger.size();
        s.text_kv_valid = text_valid; s.mtp_kv_valid = mtp_valid;
        s.rope_delta = j.at("rope_delta").get<std::int32_t>();
        s.rewrite_checkpoint = {checkpoint, static_cast<RewriteCheckpointKind>(kind), checkpoint_frontier};
        s.tail_hidden_valid = j.at("tail_valid").get<bool>(); s.mtp_draft_count = 0; s.retained = true;
        requests[lane].lifecycle = Lifecycle::Complete;
        if (std::getenv("NINFER_VERIFY_STATE_RESTORE")) {
            const auto actual = capture_image(*this, lane, snapshot_image_key(sequences[lane].ledger));
            if (!actual || actual->payload != image.payload || actual->metadata != image.metadata)
                throw std::runtime_error("restored state differs from saved bytes");
            std::fprintf(stderr, "[state-cache] verified bytes=%zu\n", image.payload.size());
        }
        return true;
    } catch (const std::exception& error) {
        if (mutated) { device.synchronize(); clear_lane(sequences[lane], requests[lane]); }
        std::fprintf(stderr, "[state-cache] restore ignored: %s\n", error.what());
        return false;
    }
}
} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
