#pragma once

// Mapping from the public KV storage selection to the storage facts the target planner and the
// Ops need. Keeping it in one place stops the CLI, the server, and the planner from each
// re-deriving a partial view of the enum.

#include "core/dtype.h"
#include "core/kvarn.h"

#include <ninfer/types.h>

#include <optional>
#include <stdexcept>
#include <string_view>

namespace ninfer {

[[nodiscard]] constexpr bool kv_storage_is_kvarn(KvCacheStorage storage) noexcept {
    return storage == KvCacheStorage::KvarnK4V2 || storage == KvCacheStorage::KvarnK4V4;
}

[[nodiscard]] constexpr std::optional<KvarnFormat>
kv_storage_kvarn_format(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::KvarnK4V2: return KvarnFormat::K4V2G64;
    case KvCacheStorage::KvarnK4V4: return KvarnFormat::K4V4G64;
    default:                        return std::nullopt;
    }
}

[[nodiscard]] constexpr std::string_view kv_storage_name(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16:    return "bf16";
    case KvCacheStorage::Int8Group64: return "int8-group64";
    case KvCacheStorage::KvarnK4V2:   return "kvarn-k4v2-g64";
    case KvCacheStorage::KvarnK4V4:   return "kvarn-k4v4-g64";
    }
    return "unknown";
}

// Accepts the flag spellings the CLI and the server share.
[[nodiscard]] constexpr std::optional<KvCacheStorage>
kv_storage_from_name(std::string_view text) noexcept {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "kvarn" || text == "kvarn-k4v2") { return KvCacheStorage::KvarnK4V2; }
    if (text == "kvarn-k4v4") { return KvCacheStorage::KvarnK4V4; }
    return std::nullopt;
}

} // namespace ninfer
