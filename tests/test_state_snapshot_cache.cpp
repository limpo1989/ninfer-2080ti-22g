#include "runtime/engine/state_snapshot_cache.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace ninfer::runtime;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
std::shared_ptr<StateSnapshotImage> image(std::string key, std::size_t bytes, unsigned char value) {
    auto out = std::make_shared<StateSnapshotImage>();
    out->key = key; out->aliases = {"100-" + key}; out->metadata = {1,2,3}; out->payload.assign(bytes, value); return out;
}
int main() {
    std::string pattern = (std::filesystem::temp_directory_path() / "ninfer-state-test-XXXXXX").string();
    const char* path = ::mkdtemp(pattern.data());
    if (!path) return 1;
    const std::filesystem::path root(path);
    try {
        {
            StateSnapshotCache cache(root, 10000, 1500, "model-a");
            check(cache.can_store(1000, 3), "valid image rejected by capacity preflight");
            check(!cache.can_store(1500, 1), "oversize RAM image passed capacity preflight");
            check(!cache.contains("abc"), "missing key reported present");
            auto abc = image("abc", 1000, 0x5a);
            abc->aliases.push_back("200-abc");
            check(cache.put(std::move(abc)), "store failed"); cache.flush();
            check(cache.contains("abc"), "stored key reported missing");
            check(cache.put(image("abc", 1000, 0xa5)), "duplicate store failed");
            check(cache.lookup({"100-abc"}).result.get()->payload[10] == 0x5a,
                  "duplicate store replaced immutable image");
            auto hit = cache.lookup({"100-abc"});
            check(hit.source == "ram" && hit.ready() && hit.result.get()->payload[10] == 0x5a, "RAM hit failed");
            hit = {};
            check(cache.put(image("def", 1000, 0xa5)), "RAM eviction failed"); cache.flush();
            auto disk = cache.lookup({"100-abc"});
            auto same = cache.lookup({"200-abc"});
            check(disk.source == "disk" && disk.result.get()->payload == image("abc",1000,0x5a)->payload,
                  "disk reload after RAM eviction failed");
            check(same.result.get() == disk.result.get(), "concurrent aliases did not share one immutable load");
            check(disk.frontier == 100 && same.frontier == 200,
                  "shared load did not preserve the matched alias frontier");
            check(!cache.put(image("bad", 2000, 1)), "oversize RAM accepted");
        }
        {
            StateSnapshotCache cache(root, 10000, 1500, "model-a");
            auto disk = cache.lookup({"100-def"});
            check(disk.source == "disk" && disk.result.get()->payload[10] == 0xa5, "restart disk hit failed");
        }
        {
            StateSnapshotCache other(root, 10000, 1500, "model-b");
            check(!other.lookup({"100-abc"}).result.valid(), "model namespace isolation failed");
        }
        const auto file = root / state_hash_key("model-a") / "abc.snap";
        { std::fstream out(file, std::ios::binary|std::ios::in|std::ios::out); out.seekp(-1, std::ios::end); out.put('\0'); }
        {
            StateSnapshotCache cache(root, 10000, 1500, "model-a");
            const auto bad = cache.lookup({"100-abc"});
            check(bad.result.valid() && !bad.result.get(), "corrupt payload was accepted");
        }
        {
            StateSnapshotCache cache(root / "budget", 2500, 1500, "model-a");
            check(cache.put(image("aaa",1000,1)), "put1"); cache.flush();
            check(cache.put(image("bbb",1000,2)), "put2"); cache.flush();
            check(cache.put(image("ccc",1000,3)), "put3"); cache.flush();
        }
        {
            StateSnapshotCache cache(root / "budget", 2500, 1500, "model-a");
            check(!cache.lookup({"100-aaa"}).result.valid(), "disk oldest entry survived pressure");
            check(cache.lookup({"100-ccc"}).result.get()->payload[0] == 3, "newest disk entry lost");
        }
        std::filesystem::remove_all(root);
        std::cout << "state snapshot RAM/disk/restart/budget/integrity checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << " (artifacts: " << root << ")\n"; return 1;
    }
}
