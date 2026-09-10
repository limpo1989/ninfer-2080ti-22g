#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::runtime {
struct StateSnapshotImage {
    std::string key;
    std::vector<std::string> aliases;
    std::vector<std::uint8_t> metadata;
    std::vector<std::uint8_t> payload;
};
struct StateSnapshotLoad {
    std::shared_future<std::shared_ptr<const StateSnapshotImage>> result;
    std::string source;
    std::uint32_t frontier = 0;
    [[nodiscard]] bool ready() const {
        return !result.valid() || result.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
};
// Immutable images. Queued/in-flight writes count against RAM. Payload I/O and checksums
// execute on a dedicated worker; lookup consults only an in-memory alias index.
class StateSnapshotCache {
public:
    StateSnapshotCache(std::filesystem::path directory, std::size_t disk_bytes,
                       std::size_t ram_bytes, std::string compatibility);
    ~StateSnapshotCache();
    StateSnapshotCache(const StateSnapshotCache&) = delete;
    StateSnapshotCache& operator=(const StateSnapshotCache&) = delete;
    [[nodiscard]] bool contains(const std::string& key) const;
    [[nodiscard]] bool can_store(std::size_t payload_bytes,
                                 std::size_t metadata_bytes) const noexcept;
    [[nodiscard]] bool put(std::shared_ptr<const StateSnapshotImage> image);
    [[nodiscard]] StateSnapshotLoad lookup(const std::vector<std::string>& aliases);
    void flush();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
[[nodiscard]] std::string state_file_identity(const std::filesystem::path& path);
[[nodiscard]] std::string state_hash_key(const std::string& value);
} // namespace ninfer::runtime
