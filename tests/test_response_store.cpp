#include "serve/response_store.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

ChatTurn text_turn(ninfer::ChatRole role, std::string text) {
    ChatTurn turn;
    turn.role = role;
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "input_text";
    part.text     = std::move(text);
    turn.content.push_back(std::move(part));
    return turn;
}

StoredResponse record(std::string id, ResponseContext context) {
    StoredResponse value;
    value.id = std::move(id);
    value.response =
        nlohmann::json{{"id", value.id}, {"object", "response"}, {"status", "completed"}};
    value.input_items.push_back(nlohmann::json{{"id", "msg_" + value.id}, {"type", "message"}});
    value.context = std::move(context);
    return value;
}

int test_context_dag() {
    const ResponseContext first =
        append_response_context({}, {text_turn(ninfer::ChatRole::User, "one"),
                                     text_turn(ninfer::ChatRole::Assistant, "a")});
    const ResponseContext second =
        append_response_context(first, {text_turn(ninfer::ChatRole::User, "two"),
                                        text_turn(ninfer::ChatRole::Assistant, "b")});
    const std::vector<ChatTurn> flattened = flatten_response_context(second);
    int failures                          = 0;
    failures += check(flattened.size() == 4, "context chain flattened all turns");
    failures += check(flattened[0].content[0].text == "one" && flattened[3].content[0].text == "b",
                      "context chain preserves chronological order");
    failures += check(second->parent.get() == first.get(), "context nodes share their parent");
    return failures;
}

int test_lru_and_delete() {
    ResponseStore store(2, 1ULL << 20, 1ULL << 20);
    const ResponseContext root =
        append_response_context({}, {text_turn(ninfer::ChatRole::User, "root")});
    store.put(record("resp_1", root));
    const ResponseContext child =
        append_response_context(root, {text_turn(ninfer::ChatRole::Assistant, "child")});
    store.put(record("resp_2", child));
    (void)store.get("resp_1"); // resp_2 becomes the least-recently used entry.
    store.put(record("resp_3",
                     append_response_context(root, {text_turn(ninfer::ChatRole::User, "fork")})));

    int failures = 0;
    failures += check(store.get("resp_1") != nullptr, "get refreshes LRU recency");
    failures += check(store.get("resp_2") == nullptr, "least-recent response evicted");
    failures += check(store.get("resp_3") != nullptr, "new response retained");
    failures += check(store.size() == 2 && store.bytes() != 0, "store reports bounded usage");
    failures += check(store.erase("resp_1"), "stored response deleted");
    failures += check(!store.erase("resp_1") && store.get("resp_1") == nullptr,
                      "deleted response is no longer addressable");
    // The child/fork context owns a shared parent even when the parent's public
    // response entry is deleted.
    const std::shared_ptr<const StoredResponse> fork = store.get("resp_3");
    failures += check(fork && flatten_response_context(fork->context).size() == 2,
                      "descendant context survives parent response deletion");
    return failures;
}

int test_oversized_record() {
    ResponseStore store(4, 256, 1ULL << 20);
    StoredResponse large = record(
        "resp_large",
        append_response_context({}, {text_turn(ninfer::ChatRole::User, std::string(1024, 'x'))}));
    std::string code;
    try {
        store.put(std::move(large));
    } catch (const ApiException& exception) { code = exception.error().code; }
    int failures = 0;
    failures += check(code == "response_store_capacity_exceeded",
                      "oversized response fails deterministically");
    failures += check(store.size() == 0, "oversized insertion does not mutate store");
    return failures;
}

int test_generated_tool_replay() {
    ResponseStore store(4, 1ULL << 20, 1ULL << 20);
    ChatTurn saved = text_turn(ninfer::ChatRole::Assistant, "Done");
    saved.reasoning_content = "Write the requested file.\n";
    saved.tool_calls.push_back({"call_write", "write", R"json({"path":"demo.py","content":"print(1)"})json"});
    saved.replay_content = "Done\n<tool_call>\n<function=write>\n"
        "<parameter=path>\"demo.py\"</parameter>\n"
        "<parameter=content>print(1)</parameter>\n</function>\n</tool_call>";
    store.remember_tool_output(saved);
    ChatTurn incoming = saved;
    incoming.replay_content.reset();
    incoming.tool_calls.front().arguments_json = R"json({"content":"print(1)","path":"demo.py"})json";
    GenerationRequest request;
    request.messages = {incoming};
    store.restore_tool_outputs(request.messages);
    const auto restored = to_prompt_input(request, {}, {});
    int failures = check(restored.messages.front().parts.front().text == *saved.replay_content &&
                             restored.messages.front().tool_calls.empty(),
                         "unchanged tool replay retains original quotes and separators exactly once");
    failures += check(store.size() == 0, "format replay does not publish store:false responses");

    for (int edit = 0; edit < 4; ++edit) {
        request.messages = {incoming};
        auto& changed = request.messages.front();
        if (edit == 0) changed.tool_calls.front().arguments_json = R"json({"path":"other.py","content":"print(1)"})json";
        if (edit == 1) changed.tool_calls.front().id = "call_other";
        if (edit == 2) changed.reasoning_content = "Changed thinking";
        if (edit == 3) changed.content.front().text = "Changed answer";
        store.restore_tool_outputs(request.messages);
        const auto translated = to_prompt_input(request, {}, {});
        failures += check(translated.messages.front().tool_calls.size() == 1 &&
                              translated.messages.front().parts.front().text == changed.content.front().text,
                          "edited history uses the supplied fields instead of stale original markup");
    }
    return failures;
}

ChatTurn replay_turn(std::string id, std::size_t padding = 0) {
    ChatTurn turn = text_turn(ninfer::ChatRole::Assistant, "Done");
    turn.tool_calls.push_back({std::move(id), "read", R"({"path":"demo.py"})"});
    turn.replay_content = "Done\n<tool_call>\n<function=read>\n"
        "<parameter=path>demo.py</parameter>\n</function>\n</tool_call>" + std::string(padding, '\n');
    return turn;
}

bool replay_matches(ResponseStore& store, const ChatTurn& saved) {
    std::vector<ChatTurn> incoming{saved};
    incoming.front().replay_content.reset();
    store.restore_tool_outputs(incoming);
    return incoming.front().replay_content == saved.replay_content;
}

int test_replay_memory_budget() {
    ResponseStore store(2, 1ULL << 20, 8192);
    const auto first = replay_turn("first", 5000);
    const auto second = replay_turn("second", 5000);
    store.remember_tool_output(first);
    int failures = check(replay_matches(store, first), "entry fitting the replay budget is retained");
    store.remember_tool_output(second);
    failures += check(!replay_matches(store, first) && replay_matches(store, second),
                      "byte pressure evicts the oldest replay and updates its lookup index");
    const auto oversized = replay_turn("oversized", 16384);
    store.remember_tool_output(oversized);
    failures += check(!replay_matches(store, oversized) && replay_matches(store, second),
                      "oversized replay does not displace usable cached entries");
    ResponseStore disabled(2, 1ULL << 20, 0);
    disabled.remember_tool_output(first);
    failures += check(!replay_matches(disabled, first), "zero budget disables replay caching");
    return failures;
}

int test_replay_has_no_count_limit() {
    ResponseStore store(2, 1ULL << 20, 8ULL << 20);
    for (int i = 0; i < 3000; ++i) {
        store.remember_tool_output(replay_turn("call_" + std::to_string(i)));
    }
    return check(replay_matches(store, replay_turn("call_0")) &&
                     replay_matches(store, replay_turn("call_2999")),
                 "more than 2048 replays remain available while the byte budget permits");
}

int test_replay_replaces_duplicate_ids() {
    ResponseStore store(2, 1ULL << 20, 8192);
    const auto other = replay_turn("other");
    store.remember_tool_output(other);
    ChatTurn newest;
    for (int i = 0; i < 20; ++i) {
        newest = replay_turn("same_id", 1000 + i);
        store.remember_tool_output(newest);
    }
    return check(replay_matches(store, newest) && replay_matches(store, other),
                 "replacement retains the newest original text without accumulating old copies");
}

} // namespace

int main() {
    int failures = 0;
    failures += test_context_dag();
    failures += test_lru_and_delete();
    failures += test_oversized_record();
    failures += test_generated_tool_replay();
    failures += test_replay_memory_budget();
    failures += test_replay_has_no_count_limit();
    failures += test_replay_replaces_duplicate_ids();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
