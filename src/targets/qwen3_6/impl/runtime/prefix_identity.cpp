#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace ninfer::targets::qwen3_6::detail {
namespace {

bool same_grid(const VisionGrid& left, const VisionGrid& right) {
    return left.temporal == right.temporal && left.height == right.height &&
           left.width == right.width;
}

bool same_spans(const std::vector<TokenSpan>& left, const std::vector<TokenSpan>& right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
                                                     [](const TokenSpan& a, const TokenSpan& b) {
                                                         return a.begin == b.begin &&
                                                                a.count == b.count;
                                                     });
}

bool same_item(const VisionItem& left, const VisionItem& right) {
    return left.modality == right.modality && same_grid(left.grid, right.grid) &&
           left.patch_begin == right.patch_begin && left.patch_count == right.patch_count &&
           left.content_digest == right.content_digest && left.timestamps == right.timestamps &&
           same_spans(left.token_spans, right.token_spans);
}

bool valid_item(const VisionItem& item, std::size_t tokens, std::size_t previous_end,
                std::size_t* item_end) {
    const auto modality = static_cast<std::uint8_t>(item.modality);
    if ((modality != static_cast<std::uint8_t>(PromptModality::Image) &&
         modality != static_cast<std::uint8_t>(PromptModality::Video)) ||
        item.grid.temporal <= 0 || item.grid.height <= 0 || item.grid.width <= 0 ||
        item.patch_count == 0 ||
        item.patch_begin > std::numeric_limits<std::size_t>::max() - item.patch_count ||
        item.timestamps.size() > std::numeric_limits<std::uint32_t>::max() ||
        item.token_spans.empty() ||
        item.token_spans.size() > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    std::size_t span_end = 0;
    for (std::size_t index = 0; index < item.token_spans.size(); ++index) {
        const TokenSpan span = item.token_spans[index];
        if (span.count == 0 || span.begin > tokens || span.count > tokens - span.begin ||
            (index != 0 && span.begin < span_end)) {
            return false;
        }
        span_end = span.begin + span.count;
    }
    if (item.token_spans.front().begin < previous_end) { return false; }
    for (const double timestamp : item.timestamps) {
        if (!std::isfinite(timestamp)) { return false; }
    }
    *item_end = span_end;
    return true;
}

bool valid_identity(const std::vector<std::uint8_t>& token_types,
                    const std::array<std::vector<std::int32_t>, 3>& positions,
                    const std::vector<VisionItem>& items) {
    for (const auto& axis : positions) {
        if (axis.size() != token_types.size()) { return false; }
    }
    for (const std::uint8_t type : token_types) {
        if (type != 0 && type != static_cast<std::uint8_t>(PromptModality::Image) &&
            type != static_cast<std::uint8_t>(PromptModality::Video)) {
            return false;
        }
    }
    std::vector<bool> covered(token_types.size(), false);
    std::size_t previous_end = 0;
    for (const VisionItem& item : items) {
        std::size_t item_end = 0;
        if (!valid_item(item, token_types.size(), previous_end, &item_end)) { return false; }
        const std::uint8_t modality = static_cast<std::uint8_t>(item.modality);
        for (const TokenSpan span : item.token_spans) {
            for (std::size_t index = span.begin; index < span.begin + span.count; ++index) {
                if (covered[index] || token_types[index] != modality) { return false; }
                covered[index] = true;
            }
        }
        previous_end = item_end;
    }
    for (std::size_t index = 0; index < token_types.size(); ++index) {
        if ((token_types[index] != 0) != covered[index]) { return false; }
    }
    return true;
}

template <typename UInt>
void append_unsigned(std::vector<std::uint8_t>& out, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    for (std::size_t byte = 0; byte < sizeof(UInt); ++byte) {
        out.push_back(static_cast<std::uint8_t>(value >> (8U * byte)));
    }
}

void append_item(std::vector<std::uint8_t>& out, const VisionItem& item) {
    append_unsigned(out, static_cast<std::uint8_t>(item.modality));
    append_unsigned(out, std::bit_cast<std::uint32_t>(item.grid.temporal));
    append_unsigned(out, std::bit_cast<std::uint32_t>(item.grid.height));
    append_unsigned(out, std::bit_cast<std::uint32_t>(item.grid.width));
    append_unsigned(out, static_cast<std::uint64_t>(item.patch_begin));
    append_unsigned(out, static_cast<std::uint64_t>(item.patch_count));
    out.insert(out.end(), item.content_digest.begin(), item.content_digest.end());
    append_unsigned(out, static_cast<std::uint32_t>(item.timestamps.size()));
    for (const double timestamp : item.timestamps) {
        append_unsigned(out, std::bit_cast<std::uint64_t>(timestamp));
    }
    append_unsigned(out, static_cast<std::uint32_t>(item.token_spans.size()));
    for (const TokenSpan span : item.token_spans) {
        append_unsigned(out, static_cast<std::uint64_t>(span.begin));
        append_unsigned(out, static_cast<std::uint64_t>(span.count));
    }
}

template <typename UInt>
UInt read_unsigned(std::span<const std::uint8_t> bytes, std::size_t* offset) {
    static_assert(std::is_unsigned_v<UInt>);
    if (*offset > bytes.size() || sizeof(UInt) > bytes.size() - *offset) {
        throw std::invalid_argument("truncated prefix identity");
    }
    UInt value = 0;
    for (std::size_t byte = 0; byte < sizeof(UInt); ++byte) {
        value |= static_cast<UInt>(bytes[*offset + byte]) << (8U * byte);
    }
    *offset += sizeof(UInt);
    return value;
}

std::size_t read_size(std::span<const std::uint8_t> bytes, std::size_t* offset) {
    const std::uint64_t value = read_unsigned<std::uint64_t>(bytes, offset);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("prefix identity size exceeds size_t");
    }
    return static_cast<std::size_t>(value);
}

void hash_byte(std::uint64_t* hash, std::uint8_t value) {
    *hash ^= value;
    *hash *= 1099511628211ULL;
}

template <typename UInt>
void hash_unsigned(std::uint64_t* hash, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    for (std::size_t byte = 0; byte < sizeof(UInt); ++byte) {
        hash_byte(hash, static_cast<std::uint8_t>(value >> (8U * byte)));
    }
}

void hash_item(std::uint64_t* hash, const VisionItem& item) {
    hash_byte(hash, 0x02U);
    hash_unsigned(hash, static_cast<std::uint8_t>(item.modality));
    hash_unsigned(hash, std::bit_cast<std::uint32_t>(item.grid.temporal));
    hash_unsigned(hash, std::bit_cast<std::uint32_t>(item.grid.height));
    hash_unsigned(hash, std::bit_cast<std::uint32_t>(item.grid.width));
    hash_unsigned(hash, static_cast<std::uint64_t>(item.patch_begin));
    hash_unsigned(hash, static_cast<std::uint64_t>(item.patch_count));
    for (const std::uint8_t byte : item.content_digest) { hash_byte(hash, byte); }
    hash_unsigned(hash, static_cast<std::uint32_t>(item.timestamps.size()));
    for (const double timestamp : item.timestamps) {
        hash_unsigned(hash, std::bit_cast<std::uint64_t>(timestamp));
    }
    hash_unsigned(hash, static_cast<std::uint32_t>(item.token_spans.size()));
    for (const TokenSpan span : item.token_spans) {
        hash_unsigned(hash, static_cast<std::uint64_t>(span.begin));
        hash_unsigned(hash, static_cast<std::uint64_t>(span.count));
    }
}

bool prefix_item_count(const std::vector<VisionItem>& items, std::size_t tokens,
                       std::size_t* count) {
    *count          = 0;
    bool saw_suffix = false;
    for (const VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        const TokenSpan& first = item.token_spans.front();
        const TokenSpan& last  = item.token_spans.back();
        if (first.count == 0 || last.count == 0 ||
            last.begin > std::numeric_limits<std::size_t>::max() - last.count) {
            return false;
        }
        const std::size_t end = last.begin + last.count;
        if (end <= tokens) {
            if (saw_suffix) { return false; }
            ++*count;
        } else if (first.begin >= tokens) {
            saw_suffix = true;
        } else {
            // A reusable frontier may not divide the consumers of one Vision item.
            return false;
        }
    }
    return true;
}

} // namespace

void ResidentPrefixIdentity::reserve(std::size_t tokens) {
    token_types_.reserve(tokens);
    for (auto& axis : positions_) { axis.reserve(tokens); }
}

void ResidentPrefixIdentity::clear() noexcept {
    token_types_.clear();
    for (auto& axis : positions_) { axis.clear(); }
    vision_items_.clear();
}

void ResidentPrefixIdentity::assign(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("prepared prompt identity metadata has an invalid shape");
    }
    token_types_ = prompt.token_types;
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin = prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * tokens);
        positions_[axis].assign(begin, begin + static_cast<std::ptrdiff_t>(tokens));
    }
    vision_items_ = prompt.vision_items;
}

void ResidentPrefixIdentity::append_generated(std::size_t count, std::int32_t rope_delta) {
    const std::size_t begin = size();
    if (count > std::numeric_limits<std::size_t>::max() - begin) {
        throw std::overflow_error("generated prefix identity length overflows size_t");
    }
    for (std::size_t offset = 0; offset < count; ++offset) {
        const std::size_t index = begin + offset;
        if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("generated prefix position exceeds int32");
        }
        const std::int64_t position = static_cast<std::int64_t>(index) + rope_delta;
        if (position < std::numeric_limits<std::int32_t>::min() ||
            position > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("generated MRoPE position exceeds int32");
        }
        token_types_.push_back(0);
        for (auto& axis : positions_) { axis.push_back(static_cast<std::int32_t>(position)); }
    }
}

void ResidentPrefixIdentity::truncate(std::size_t tokens) {
    if (tokens > size()) {
        throw std::out_of_range("cannot extend resident prefix identity by truncation");
    }
    std::size_t retained_items = 0;
    if (!prefix_item_count(vision_items_, tokens, &retained_items)) {
        throw std::logic_error("resident prefix truncation divides a Vision item");
    }
    token_types_.resize(tokens);
    for (auto& axis : positions_) { axis.resize(tokens); }
    vision_items_.resize(retained_items);
}

bool ResidentPrefixIdentity::matches(const PreparedPromptData& prompt, std::size_t count) const {
    const std::size_t prompt_tokens = prompt.token_ids.size();
    if (count > prompt_tokens || count > size() || prompt.token_types.size() != prompt_tokens ||
        prompt.positions.size() != 3 * prompt_tokens) {
        return false;
    }
    if (!std::equal(prompt.token_types.begin(),
                    prompt.token_types.begin() + static_cast<std::ptrdiff_t>(count),
                    token_types_.begin())) {
        return false;
    }
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * prompt_tokens);
        if (!std::equal(begin, begin + static_cast<std::ptrdiff_t>(count),
                        positions_[axis].begin())) {
            return false;
        }
    }

    std::size_t incoming_items = 0;
    std::size_t resident_items = 0;
    if (!prefix_item_count(prompt.vision_items, count, &incoming_items) ||
        !prefix_item_count(vision_items_, count, &resident_items) ||
        incoming_items != resident_items) {
        return false;
    }
    for (std::size_t i = 0; i < incoming_items; ++i) {
        if (!same_item(prompt.vision_items[i], vision_items_[i])) { return false; }
    }
    return true;
}

bool prefix_matches(const PreparedPromptData& prompt, const std::vector<TokenId>& resident_tokens,
                    const ResidentPrefixIdentity& resident_identity, std::size_t count) {
    if (count > prompt.token_ids.size() || count > resident_tokens.size()) { return false; }
    const auto end = prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(count);
    const auto mismatch = std::mismatch(prompt.token_ids.begin(), end, resident_tokens.begin());
    if (mismatch.first != end) {
        static const bool trace = std::getenv("NINFER_TRACE_PREFIX") != nullptr;
        if (trace) {
            std::fprintf(
                stderr,
                "[prefix-mismatch] frontier=%zu prompt=%zu common=%zu incoming=%d resident=%d\n",
                         count, prompt.token_ids.size(),
                         static_cast<std::size_t>(mismatch.first - prompt.token_ids.begin()),
                         *mismatch.first, *mismatch.second);
            if (count + 1 == resident_tokens.size()) {
                const auto position =
                    static_cast<std::size_t>(mismatch.first - prompt.token_ids.begin());
                const auto begin = position > 3 ? position - 3 : 0;
                std::fprintf(stderr, "[prefix-window] start=%zu incoming=", begin);
                for (auto i = begin; i < std::min(prompt.token_ids.size(), position + 6); ++i) {
                    std::fprintf(stderr, "%d,", prompt.token_ids[i]);
                }
                std::fprintf(stderr, " resident=");
                for (auto i = begin; i < std::min(resident_tokens.size(), position + 6); ++i) {
                    std::fprintf(stderr, "%d,", resident_tokens[i]);
                }
                std::fprintf(stderr, "\n");
            }
        }
        return false;
    }
    return resident_identity.matches(prompt, count);
}

std::vector<PersistentPrefixFingerprint>
ResidentPrefixIdentity::persistent_fingerprints(std::span<const TokenId> tokens,
                                                std::uint32_t minimum_frontier) const {
    if (tokens.size() != token_types_.size() ||
        tokens.size() > std::numeric_limits<std::uint32_t>::max() ||
        !valid_identity(token_types_, positions_, vision_items_)) {
        throw std::invalid_argument("persistent prefix identity is invalid");
    }
    std::vector<PersistentPrefixFingerprint> result;
    if (tokens.size() > minimum_frontier) { result.reserve(tokens.size() - minimum_frontier); }
    std::uint64_t hash                = 14695981039346656037ULL;
    constexpr std::string_view domain = "ninfer-multimodal-state-prefix-v4";
    for (const char byte : domain) { hash_byte(&hash, static_cast<std::uint8_t>(byte)); }

    std::size_t item_index = 0;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        hash_byte(&hash, 0x01U);
        hash_unsigned(&hash, static_cast<std::uint32_t>(tokens[index]));
        hash_unsigned(&hash, token_types_[index]);
        for (const auto& axis : positions_) {
            hash_unsigned(&hash, std::bit_cast<std::uint32_t>(axis[index]));
        }

        const std::uint32_t frontier = static_cast<std::uint32_t>(index + 1);
        while (item_index < vision_items_.size()) {
            const VisionItem& item = vision_items_[item_index];
            const TokenSpan& last  = item.token_spans.back();
            if (last.begin + last.count != frontier) { break; }
            hash_item(&hash, item);
            ++item_index;
        }
        const bool divides_item = item_index < vision_items_.size() &&
                                  vision_items_[item_index].token_spans.front().begin < frontier &&
                                  frontier < vision_items_[item_index].token_spans.back().begin +
                                                 vision_items_[item_index].token_spans.back().count;
        if (frontier > minimum_frontier && !divides_item) {
            result.push_back(PersistentPrefixFingerprint{frontier, hash});
        }
    }
    if (item_index != vision_items_.size()) {
        throw std::invalid_argument("persistent prefix identity has unfinished Vision items");
    }
    return result;
}

std::vector<std::uint8_t> ResidentPrefixIdentity::serialize() const {
    if (token_types_.size() > std::numeric_limits<std::uint32_t>::max() ||
        vision_items_.size() > std::numeric_limits<std::uint32_t>::max() ||
        !valid_identity(token_types_, positions_, vision_items_)) {
        throw std::invalid_argument("prefix identity is invalid");
    }
    std::vector<std::uint8_t> out;
    const std::uint32_t count = static_cast<std::uint32_t>(token_types_.size());
    append_unsigned(out, count);
    out.insert(out.end(), token_types_.begin(), token_types_.end());
    for (const auto& axis : positions_) {
        for (const std::int32_t position : axis) {
            append_unsigned(out, std::bit_cast<std::uint32_t>(position));
        }
    }
    const std::uint32_t item_count = static_cast<std::uint32_t>(vision_items_.size());
    append_unsigned(out, item_count);
    for (const VisionItem& item : vision_items_) { append_item(out, item); }
    return out;
}

void ResidentPrefixIdentity::deserialize(std::span<const std::uint8_t> bytes) {
    std::size_t offset = 0;
    const std::uint32_t count = read_unsigned<std::uint32_t>(bytes, &offset);
    if (count > (bytes.size() - offset) / 13) {
        throw std::invalid_argument("invalid prefix token count");
    }
    std::vector<std::uint8_t> token_types(count);
    if (token_types.size() > bytes.size() - offset) {
        throw std::invalid_argument("truncated prefix identity");
    }
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(offset), token_types.size(),
                token_types.begin());
    offset += token_types.size();
    std::array<std::vector<std::int32_t>, 3> positions;
    for (auto& axis : positions) {
        axis.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            axis.push_back(
                std::bit_cast<std::int32_t>(read_unsigned<std::uint32_t>(bytes, &offset)));
        }
    }
    const std::uint32_t item_count          = read_unsigned<std::uint32_t>(bytes, &offset);
    constexpr std::size_t kMinimumItemBytes = 85;
    if (item_count > (bytes.size() - offset) / kMinimumItemBytes) {
        throw std::invalid_argument("invalid Vision item count");
    }
    std::vector<VisionItem> items;
    items.reserve(item_count);
    for (std::uint32_t index = 0; index < item_count; ++index) {
        VisionItem item;
        item.modality = static_cast<PromptModality>(read_unsigned<std::uint8_t>(bytes, &offset));
        item.grid.temporal =
            std::bit_cast<std::int32_t>(read_unsigned<std::uint32_t>(bytes, &offset));
        item.grid.height =
            std::bit_cast<std::int32_t>(read_unsigned<std::uint32_t>(bytes, &offset));
        item.grid.width = std::bit_cast<std::int32_t>(read_unsigned<std::uint32_t>(bytes, &offset));
        item.patch_begin = read_size(bytes, &offset);
        item.patch_count = read_size(bytes, &offset);
        if (item.content_digest.size() > bytes.size() - offset) {
            throw std::invalid_argument("truncated prefix identity");
        }
        std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(offset), item.content_digest.size(),
                    item.content_digest.begin());
        offset += item.content_digest.size();
        const std::uint32_t timestamp_count = read_unsigned<std::uint32_t>(bytes, &offset);
        if (timestamp_count > (bytes.size() - offset) / sizeof(std::uint64_t)) {
            throw std::invalid_argument("invalid Vision timestamp count");
        }
        item.timestamps.reserve(timestamp_count);
        for (std::uint32_t timestamp = 0; timestamp < timestamp_count; ++timestamp) {
            item.timestamps.push_back(
                std::bit_cast<double>(read_unsigned<std::uint64_t>(bytes, &offset)));
        }
        const std::uint32_t span_count = read_unsigned<std::uint32_t>(bytes, &offset);
        if (span_count > (bytes.size() - offset) / (2 * sizeof(std::uint64_t))) {
            throw std::invalid_argument("invalid Vision span count");
        }
        item.token_spans.reserve(span_count);
        for (std::uint32_t span = 0; span < span_count; ++span) {
            item.token_spans.push_back(
                TokenSpan{read_size(bytes, &offset), read_size(bytes, &offset)});
        }
        items.push_back(std::move(item));
    }
    if (offset != bytes.size()) {
        throw std::invalid_argument("prefix identity has trailing bytes");
    }
    if (!valid_identity(token_types, positions, items)) {
        throw std::invalid_argument("persisted prefix identity is invalid");
    }
    token_types_  = std::move(token_types);
    positions_    = std::move(positions);
    vision_items_ = std::move(items);
}

} // namespace ninfer::targets::qwen3_6::detail
