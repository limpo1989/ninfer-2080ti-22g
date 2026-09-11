#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6 {

class ToolGrammarPlan {
public:
    ToolGrammarPlan(const ToolGrammarPlan&);
    ToolGrammarPlan& operator=(const ToolGrammarPlan&);
    ToolGrammarPlan(ToolGrammarPlan&&) noexcept;
    ToolGrammarPlan& operator=(ToolGrammarPlan&&) noexcept;
    ~ToolGrammarPlan();

    [[nodiscard]] std::size_t bitmask_words() const noexcept;

private:
    class Impl;
    explicit ToolGrammarPlan(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class ToolGrammarCompiler;
    friend class ToolGrammarState;
};

class ToolGrammarCompiler {
public:
    ToolGrammarCompiler(std::vector<std::string> encoded_vocabulary,
                        std::vector<std::int32_t> stop_token_ids);
    ToolGrammarCompiler(ToolGrammarCompiler&&) noexcept;
    ToolGrammarCompiler& operator=(ToolGrammarCompiler&&) noexcept;
    ~ToolGrammarCompiler();

    ToolGrammarCompiler(const ToolGrammarCompiler&)            = delete;
    ToolGrammarCompiler& operator=(const ToolGrammarCompiler&) = delete;

    [[nodiscard]] std::shared_ptr<const ToolGrammarPlan>
    compile(const std::vector<std::string>& tool_jsons, bool require_tool_call,
            bool starts_in_reasoning = false) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class ToolGrammarState {
public:
    ToolGrammarState() noexcept;
    explicit ToolGrammarState(std::shared_ptr<const ToolGrammarPlan> plan);
    ToolGrammarState(ToolGrammarState&&) noexcept;
    ToolGrammarState& operator=(ToolGrammarState&&) noexcept;
    ~ToolGrammarState();

    ToolGrammarState(const ToolGrammarState&)            = delete;
    ToolGrammarState& operator=(const ToolGrammarState&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::size_t bitmask_words() const noexcept;

    // Writes one mask for the current state and one after each legal draft token. Remaining rows
    // stay all-allow after the first illegal draft because target verification stops there.
    [[nodiscard]] bool fill_draft_masks(std::span<std::int32_t> output, std::uint32_t mask_rows,
                                        std::span<const TokenId> drafts);
    void accept(std::span<const TokenId> tokens);
    [[nodiscard]] bool completed() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen3_6
