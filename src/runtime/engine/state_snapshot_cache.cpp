#include "runtime/engine/state_snapshot_cache.h"
#include <nlohmann/json.hpp>
#include <zlib.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <list>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace ninfer::runtime {
namespace {
using Json = nlohmann::json;
constexpr std::array<char,8> kMagic{'N','I','S','N','A','P','2','\0'};
struct Header {
    std::array<char,8> magic = kMagic;
    std::uint64_t metadata_bytes = 0, payload_bytes = 0;
    std::uint32_t metadata_crc = 0, payload_crc = 0;
};
static_assert(sizeof(Header) == 32);
std::uint32_t crc(const std::vector<std::uint8_t>& bytes) {
    return static_cast<std::uint32_t>(crc32_z(0, bytes.data(), bytes.size()));
}
bool valid_key(const std::string& key) {
    return !key.empty() && key.size() <= 128 && key.find_first_not_of("0123456789abcdef-") == std::string::npos;
}
void write_all(int fd, const void* data, std::size_t size) {
    auto p = static_cast<const char*>(data);
    while (size) {
        const auto n = ::write(fd, p, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("state cache write failed");
        p += n; size -= static_cast<std::size_t>(n);
    }
}
struct Descriptor {
    std::string key;
    std::vector<std::string> aliases;
    std::vector<std::uint8_t> metadata;
    Header header;
    std::size_t file_bytes() const { return sizeof(Header) + header.metadata_bytes + header.payload_bytes; }
};
Descriptor read_descriptor(const std::filesystem::path& path, std::size_t limit,
                           const std::string& compatibility) {
    std::ifstream input(path, std::ios::binary);
    Header h;
    input.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!input || h.magic != kMagic || h.metadata_bytes > (32ULL << 20) ||
        h.payload_bytes > limit || h.metadata_bytes > limit - h.payload_bytes ||
        std::filesystem::file_size(path) != sizeof(h) + h.metadata_bytes + h.payload_bytes)
        throw std::runtime_error("invalid state cache header");
    std::vector<std::uint8_t> raw(h.metadata_bytes);
    input.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (!input || crc(raw) != h.metadata_crc) throw std::runtime_error("state metadata checksum mismatch");
    const auto j = Json::from_cbor(raw);
    if (j.at("compatibility") != compatibility) throw std::runtime_error("incompatible state cache");
    Descriptor d{j.at("key").get<std::string>(), j.at("aliases").get<std::vector<std::string>>(),
                 j.at("metadata").get_binary(), h};
    if (!valid_key(d.key) || path.stem() != d.key) throw std::runtime_error("invalid state cache key");
    return d;
}
}
std::string state_hash_key(const std::string& value) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char c : value) { hash ^= c; hash *= 1099511628211ULL; }
    std::ostringstream out; out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}
std::string state_file_identity(const std::filesystem::path& path) {
    struct stat s{};
    if (::stat(path.c_str(), &s)) throw std::runtime_error("cannot identify state cache artifact/executable");
    std::ostringstream out;
    out << std::filesystem::canonical(path).string() << ':' << s.st_dev << ':' << s.st_ino << ':' << s.st_size;
#ifdef __APPLE__
    out << ':' << s.st_mtimespec.tv_sec << ':' << s.st_mtimespec.tv_nsec << ':' << s.st_ctimespec.tv_sec << ':' << s.st_ctimespec.tv_nsec;
#else
    out << ':' << s.st_mtim.tv_sec << ':' << s.st_mtim.tv_nsec << ':' << s.st_ctim.tv_sec << ':' << s.st_ctim.tv_nsec;
#endif
    return out.str();
}
struct StateSnapshotCache::Impl {
    struct Entry {
        Descriptor descriptor;
        std::shared_ptr<const StateSnapshotImage> image;
        StateSnapshotLoad loading;
        bool dirty = false, disk = false;
        std::list<std::string>::iterator lru;
    };
    std::filesystem::path directory;
    std::string compatibility;
    std::size_t disk_limit, ram_limit, disk_bytes = 0, ram_bytes = 0;
    int lock_fd = -1;
    std::mutex mutex;
    std::condition_variable changed;
    std::unordered_map<std::string, std::shared_ptr<Entry>> entries;
    std::unordered_map<std::string, std::string> aliases;
    std::list<std::string> lru;
    std::deque<std::function<void()>> jobs;
    std::size_t active = 0;
    bool stopping = false;
    std::thread worker;
    Impl(std::filesystem::path root, std::size_t disk, std::size_t ram, std::string identity)
        : directory(std::move(root) / state_hash_key(identity)), compatibility(std::move(identity)), disk_limit(disk), ram_limit(ram) {
        if (!disk || !ram) throw std::invalid_argument("state cache requires positive RAM and disk budgets");
        std::filesystem::create_directories(directory); ::chmod(directory.c_str(), 0700);
        lock_fd = ::open((directory / ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lock_fd < 0 || ::flock(lock_fd, LOCK_EX | LOCK_NB)) {
            if (lock_fd >= 0) ::close(lock_fd);
            throw std::runtime_error("state cache namespace is already in use or inaccessible");
        }
        try {
            std::vector<std::filesystem::directory_entry> files;
            for (const auto& f : std::filesystem::directory_iterator(directory)) {
                if (f.path().extension() == ".snap") files.push_back(f);
                else if (f.path().filename().string().starts_with(".pending-")) {
                    std::error_code ec; std::filesystem::remove(f.path(), ec);
                }
            }
            std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.last_write_time() < b.last_write_time(); });
            for (const auto& f : files) {
                try {
                    auto e = std::make_shared<Entry>();
                    e->descriptor = read_descriptor(f.path(), disk_limit, compatibility); e->disk = true;
                    add_entry(e); disk_bytes += e->descriptor.file_bytes();
                } catch (const std::exception&) { /* Invalid cache files are never candidates. */ }
            }
            trim_disk(); worker = std::thread([this] { run(); });
        } catch (...) { ::close(lock_fd); throw; }
    }
    ~Impl() {
        { std::lock_guard lock(mutex); stopping = true; }
        changed.notify_all(); worker.join(); ::close(lock_fd);
    }
    void add_entry(const std::shared_ptr<Entry>& e) {
        const auto& key = e->descriptor.key;
        lru.push_front(key); e->lru = lru.begin(); entries[key] = e;
        for (const auto& a : e->descriptor.aliases) aliases[a] = key;
    }
    void trim_disk() {
        for (auto it = lru.rbegin(); it != lru.rend() && disk_bytes > disk_limit; ++it) {
            auto& e = entries.at(*it);
            if (!e->disk || e->dirty) continue;
            std::error_code ec;
            if (std::filesystem::remove(directory / (e->descriptor.key + ".snap"), ec)) {
                disk_bytes -= e->descriptor.file_bytes(); e->disk = false;
            }
        }
        for (auto it = entries.begin(); it != entries.end();) {
            auto e = it->second;
            if (e->disk || e->image || e->dirty) { ++it; continue; }
            for (const auto& a : e->descriptor.aliases) {
                const auto found = aliases.find(a);
                if (found != aliases.end() && found->second == it->first) aliases.erase(found);
            }
            lru.erase(e->lru); it = entries.erase(it);
        }
    }
    bool reserve_ram(std::size_t bytes) {
        if (bytes > ram_limit) return false;
        for (auto it = lru.rbegin(); it != lru.rend() && ram_bytes > ram_limit - bytes; ++it) {
            auto& e = entries.at(*it);
            if (!e->dirty && e->image && e->image.use_count() == 1) {
                ram_bytes -= e->image->payload.size() + e->image->metadata.size(); e->image.reset();
            }
        }
        return ram_bytes <= ram_limit - bytes;
    }
    void write_image(const std::shared_ptr<Entry>& e) {
        const auto image = e->image;
        const auto metadata = Json::to_cbor(Json{{"compatibility", compatibility}, {"key", image->key},
                                     {"aliases", image->aliases}, {"metadata", Json::binary(image->metadata)}});
        Header h{kMagic, metadata.size(), image->payload.size(), crc(metadata), crc(image->payload)};
        std::string temporary = (directory / ".pending-XXXXXX").string();
        int fd = ::mkstemp(temporary.data());
        if (fd < 0) throw std::runtime_error("cannot create state cache temporary file");
        try {
            write_all(fd, &h, sizeof(h)); write_all(fd, metadata.data(), metadata.size());
            write_all(fd, image->payload.data(), image->payload.size());
            if (::fsync(fd)) throw std::runtime_error("state cache fsync failed");
            ::close(fd); fd = -1;
            std::filesystem::rename(temporary, directory / (image->key + ".snap"));
            const int dirfd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
            if (dirfd < 0) throw std::runtime_error("cannot sync state cache directory");
            const auto status = ::fsync(dirfd); ::close(dirfd);
            if (status) throw std::runtime_error("state cache directory sync failed");
        } catch (...) {
            if (fd >= 0) ::close(fd);
            std::error_code ec; std::filesystem::remove(temporary, ec); throw;
        }
        std::lock_guard lock(mutex);
        e->descriptor.header = h; e->disk = true; e->dirty = false;
        disk_bytes += e->descriptor.file_bytes(); trim_disk();
    }
    void run() noexcept {
        for (;;) {
            std::function<void()> job;
            { std::unique_lock lock(mutex);
              changed.wait(lock, [&] { return stopping || !jobs.empty(); });
              if (jobs.empty() && stopping) return;
              job = std::move(jobs.front()); jobs.pop_front(); ++active; }
            job();
            { std::lock_guard lock(mutex); --active; }
            changed.notify_all();
        }
    }
};
StateSnapshotCache::StateSnapshotCache(std::filesystem::path directory, std::size_t disk_bytes,
                                     std::size_t ram_bytes, std::string compatibility)
    : impl_(std::make_unique<Impl>(std::move(directory), disk_bytes, ram_bytes, std::move(compatibility))) {}
StateSnapshotCache::~StateSnapshotCache() = default;
bool StateSnapshotCache::contains(const std::string& key) const {
    std::lock_guard lock(impl_->mutex);
    return impl_->entries.contains(key);
}
bool StateSnapshotCache::can_store(std::size_t payload_bytes,
                                   std::size_t metadata_bytes) const noexcept {
    if (payload_bytes > impl_->ram_limit || metadata_bytes > impl_->ram_limit - payload_bytes) {
        return false;
    }
    const std::size_t image_bytes = payload_bytes + metadata_bytes;
    return sizeof(Header) <= impl_->disk_limit && image_bytes <= impl_->disk_limit - sizeof(Header);
}
bool StateSnapshotCache::put(std::shared_ptr<const StateSnapshotImage> image) {
    if (!image || !valid_key(image->key)) return false;
    std::lock_guard lock(impl_->mutex);
    if (impl_->entries.contains(image->key)) return true;
    const auto size = image->payload.size() + image->metadata.size();
    if (!can_store(image->payload.size(), image->metadata.size()) || !impl_->reserve_ram(size)) return false;
    auto e = std::make_shared<Impl::Entry>();
    e->descriptor = {image->key, image->aliases, image->metadata, {}};
    e->image = std::move(image); e->dirty = true;
    impl_->add_entry(e); impl_->ram_bytes += size;
    impl_->jobs.push_back([this, e] {
        try { impl_->write_image(e); }
        catch (const std::exception& error) {
            std::fprintf(stderr, "[state-cache] write failed: %s\n", error.what());
            std::lock_guard lock(impl_->mutex); e->dirty = false;
        }
    });
    impl_->changed.notify_one(); return true;
}
StateSnapshotLoad StateSnapshotCache::lookup(const std::vector<std::string>& aliases) {
    std::lock_guard lock(impl_->mutex);
    for (const auto& alias : aliases) {
        const auto found = impl_->aliases.find(alias);
        if (found == impl_->aliases.end()) continue;
        const auto entry = impl_->entries.find(found->second);
        if (entry == impl_->entries.end()) continue;
        auto e = entry->second;
        impl_->lru.splice(impl_->lru.begin(), impl_->lru, e->lru);
        const auto frontier =
            static_cast<std::uint32_t>(std::stoul(alias.substr(0, alias.find('-'))));
        if (e->loading.result.valid() && !e->loading.ready()) {
            StateSnapshotLoad ticket = e->loading;
            ticket.frontier = frontier;
            return ticket;
        }
        auto promise = std::make_shared<std::promise<std::shared_ptr<const StateSnapshotImage>>>();
        StateSnapshotLoad ticket{promise->get_future().share(), e->image ? "ram" : "disk",
                                 frontier};
        if (e->image) { promise->set_value(e->image); return ticket; }
        if (!e->disk) continue;
        const auto size = e->descriptor.header.payload_bytes + e->descriptor.metadata.size();
        if (!impl_->reserve_ram(size)) return {};
        impl_->ram_bytes += size; e->dirty = true; e->loading = ticket;
        impl_->jobs.push_front([this, e, promise, size] {
            try {
                const auto path = impl_->directory / (e->descriptor.key + ".snap");
                const auto d = read_descriptor(path, impl_->disk_limit, impl_->compatibility);
                auto image = std::make_shared<StateSnapshotImage>();
                image->key = d.key; image->aliases = d.aliases; image->metadata = d.metadata;
                image->payload.resize(d.header.payload_bytes);
                std::ifstream input(path, std::ios::binary);
                input.seekg(sizeof(Header) + d.header.metadata_bytes);
                input.read(reinterpret_cast<char*>(image->payload.data()), image->payload.size());
                if (!input || crc(image->payload) != d.header.payload_crc) throw std::runtime_error("state payload checksum mismatch");
                { std::lock_guard lock(impl_->mutex); e->image = image; e->dirty = false; e->loading = {}; }
                promise->set_value(std::move(image));
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[state-cache] load ignored: %s\n", error.what());
                { std::lock_guard lock(impl_->mutex); impl_->ram_bytes -= size; e->dirty = false; e->loading = {};
                  for (const auto& a : e->descriptor.aliases) {
                      const auto it = impl_->aliases.find(a);
                      if (it != impl_->aliases.end() && it->second == e->descriptor.key) impl_->aliases.erase(it);
                  } }
                promise->set_value(nullptr);
            }
        });
        impl_->changed.notify_one(); return ticket;
    }
    return {};
}
void StateSnapshotCache::flush() {
    std::unique_lock lock(impl_->mutex);
    impl_->changed.wait(lock, [&] { return impl_->jobs.empty() && impl_->active == 0; });
}
} // namespace ninfer::runtime
