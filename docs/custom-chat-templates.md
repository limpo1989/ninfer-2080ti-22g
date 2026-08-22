# Custom chat templates

NInfer can render a chat template supplied at run time instead of the one baked
into the `.ninfer` artifact:

```
ninfer-serve model.ninfer --chat-template path/to/chat_template.jinja
```

The file is read once at startup. If it cannot be opened, is empty, or is not
valid Jinja, startup fails with an error rather than silently falling back.

## Why this needed an interpreter

Chat rendering used to be a hand-written C++ transcription of two specific
templates, selected by a sha256 allowlist. That design cannot accept a
user-supplied template. A template outside the allowlist either fails the digest
check, or — worse — gets rendered by a transcription written for a *different*
template, silently producing a prompt the model was never trained on.

That risk is concrete. `sharp_qwen_chat_template.jinja`
(`qwen3.8-froggeric-v22.3.1`) looks superficially like the template the
`--chat-style sharp-v22.1` transcription was written against, but differs in the
tool-call instruction block, adds a JSON/XML `tool_call_format` switch,
`<|think_*|>` control tokens, consecutive-tool-failure warnings, and tool
response truncation. Rendering it through the existing transcription would have
produced wrong prompts with no error.

So templates outside the allowlist are now *interpreted*. The two built-in
templates keep their verified transcriptions, so behaviour for shipped artifacts
is unchanged.

## Supported Jinja subset

The interpreter targets what chat templates actually use. Anything outside the
subset raises an error at parse time rather than being ignored:

- **Statements** — `set` (including `{% set x %}…{% endset %}`), `if`/`elif`/
  `else`, `for` with `else` and the `loop` variable (`index`, `index0`, `first`,
  `last`, `length`, `revindex`, `revindex0`, `previtem`, `nextitem`), `macro`
  with default arguments, `do`, comments, and whitespace control via `-`/`+`.
- **Expressions** — literals, attribute and item access, slicing (including a
  negative step), calls, filters, tests, conditional expressions, list/tuple/dict
  literals, and the usual operators including `~`, `in`, and `not in`.
- **Filters** — `string`, `lower`, `upper`, `capitalize`, `trim`, `replace`,
  `tojson`, `length`/`count`, `join`, `list`, `items`, `first`, `last`,
  `reverse`, `default`/`d`, `int`, `float`, `abs`.
- **Tests** — `defined`, `undefined`, `none`, `string`, `number`, `integer`,
  `float`, `boolean`, `mapping`, `sequence`, `iterable`, `true`, `false`,
  `equalto`/`eq`, `in`, `odd`, `even`.
- **Methods** — strings: `split`, `startswith`, `endswith`, `strip`, `lstrip`,
  `rstrip`, `lower`, `upper`, `replace`, `join`; mappings: `items`, `keys`,
  `values`, `get`; lists: `append`.
- **Globals** — `namespace()`, `range()`, `dict()`, `raise_exception()`.

There is no `include`, `import`, or file access, and a template can only reach
the variables listed below.

Two behaviours are matched deliberately because the model's training data
depends on them:

- `trim_blocks` and `lstrip_blocks` are **on**, matching the environment
  `transformers` builds for chat templates.
- `tojson` uses Python's `json.dumps(..., ensure_ascii=False)` separators —
  `", "` and `": "`, not the compact form.

## Template variables

| Variable | Type | Notes |
| --- | --- | --- |
| `messages` | list | Each has `role`, `content`, and optionally `reasoning_content`, `tool_calls`, `tool_call_id`. |
| `tools` | list | Present only when the request supplies tools. |
| `add_generation_prompt` | bool | |
| `enable_thinking` | bool | |
| `reasoning_effort` | string | One of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Present only when the request sets it. |
| `preserve_thinking` | bool | Present only when the request sets it. |
| `add_vision_id` | bool | |

`content` is a plain string unless the turn carries media, in which case it is
the list-of-parts form (`{"type": "text"|"image"|"video", …}`). A tool call's
`arguments` is a mapping when the payload parses as a JSON object, otherwise the
raw string.

## Advertised capabilities

The server rejects request options the template does not honour. For an
interpreted template those capabilities are derived from the variable names the
parsed template actually reads: `enable_thinking` is advertised only if the
template references `enable_thinking`, and reasoning-effort support only if it
references `reasoning_effort`. A template that ignores a variable will not have
requests silently accepted against it.

## Prefix-reuse checkpoints

Interpreted templates emit a `ResponseReplay` rewrite checkpoint at the end of
the rendered prefix when `add_generation_prompt` is set — that offset is the
prompt frontier for any template. The `TurnClosure` checkpoint is deliberately
not synthesised, since locating the final assistant turn would require assuming
a particular markup. Omitting it costs a prefix-reuse optimisation; it does not
affect correctness.

## Verifying fidelity

The interpreter is checked against Python's `jinja2` by rendering the same
template and context through both and requiring byte-identical output:

```
tests/tools/jinja_diff/run.sh <build-dir> <template.jinja> [more templates...]
```

For example, over the shipped fixtures and the Sharp template:

```
mkdir -p /tmp/jdiff
tests/tools/jinja_diff/run.sh /tmp/jdiff \
  sharp_qwen_chat_template.jinja \
  tests/fixtures/frontend/reasoning_effort_chat_template.jinja \
  tests/fixtures/frontend/thinking_toggle_chat_template.jinja
```

This runs every context in `tests/tools/jinja_diff/contexts/` (tool sessions,
vision turns, thinking toggles, `<|think_*|>` control tokens, truncation limits,
multi-system heads, unusual roles) against each template. It requires `jinja2`
installed. Add a context file to extend coverage.

`ninfer_qwen3_6_jinja_chat_template_test` covers what the differential cannot
reach from inside the build: the language subset, the allowlist-to-interpreter
fallback, capability derivation, and the `ChatMessage`-to-variable mapping.
