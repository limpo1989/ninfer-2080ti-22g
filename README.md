# NInfer (RTX 2080 Ti 22GB / Turing SM75 Port)

> Selected checkpoints. Maximum single-GPU inference performance.

This repository is a specialized port of [NInfer](https://github.com/Neroued/ninfer) (originally developed by [@Neroued](https://github.com/Neroued)) optimized for NVIDIA Turing architecture (`sm_75`, tuned specifically for the **RTX 2080 Ti 22GB** modded card), while retaining compatibility with Ampere (`sm_86`) and Blackwell (`sm_120a`). It executes text and multimodal (image/video) prompts through a fast local CLI or OpenAI/Anthropic-compatible HTTP servers.

**Current measured default: 568.11 tok/s prefill at 8,083 input tokens, falling to
266.85 tok/s at 62,960 input tokens; MTP3 decode falls from 25.79 to 17.06 tok/s.**
Every point is the mean of two complete 256-token generations on one RTX 2080 Ti 22GB at
250 W with Turbo disabled. Both runs are retained below; see the
[measurement conditions and remaining bounds](docs/performance.md).

**Cross-process disk restore reused 8,039 / 8,043 tokens (99.95%)** and returned the same
deterministic request result. The [persistent state cache](#persistent-state-cache) retains model
state in RAM and on disk, preserving expensive prefill work across compatible rebuilds.

---

## Persistent State Cache

Long agent histories can take tens of seconds to process again after eviction or a server restart.
NInfer saves completed conversation state in **VRAM → RAM → DISK** tiers. When the same token
prefix returns, it restores the saved state and computes the new suffix. Completed disk snapshots
survive process restarts with compatible model files, state ABI, and configuration.

### Verified continuation restore

An 8,043-token continuation was restored after a service process restart while keeping the state
ABI, artifact and execution configuration unchanged:

| State source | TTFB | Engine prefill | Restore | Reused input |
|---|---:|---:|---:|---:|
| **Disk snapshot after process restart** | **1.09s** | 0.08s | 0.09s | **8,039 / 8,043 (99.95%)** |

The response status, cached-token count and deterministic arithmetic result were verified after a
service restart. This test establishes disk restore behavior for the current namespace.

### Enable and inspect

The deployment launcher enables **8 GiB RAM + 50 GiB disk** by default, allocating memory on demand:

```bash
# Configure larger budgets; values are in MiB. Choose a disk with sufficient free space.
./deploy/restart-ninfer.sh restart \
  --state-cache-dir /path/to/ssd/ninfer-state \
  --state-cache-ram-mib 8192 --state-cache-max-mib 32768

# Inspect RAM/disk restore source and GPU upload time; capture timing is in the service log.
./deploy/restart-ninfer.sh watch --details
./deploy/restart-ninfer.sh logs
```

The same cache flags are available on `ninfer-serve`; direct launches require an explicit directory
and positive disk budget. Set `--state-cache-max-mib 0` to disable state caching. For an existing
service, pass the same overrides when opening watch so its configuration header reflects them.

Disk reads, checksums, and durable writes run on a dedicated I/O worker. The restart check restored
the disk snapshot in 0.09s; capture and upload briefly occupy the inference worker.

This first version supports **text with ordinary decode or MTP**. Clients resend conversation
history after restart; public `previous_response_id` records and tool-format replay metadata remain
process-local. Rebuilding preserves the cache namespace while the explicit state ABI, model file
identity and execution configuration remain unchanged.
Budgets apply to the current namespace, while old namespaces remain available for manual cleanup.
See the [serving options](docs/serving.md#server-options) and the deployment details below for the
complete storage behavior and benchmark modes.

---

## Supported Models & Artifacts

NInfer uses standalone `.ninfer` container artifacts embedding packed weights and tokenizer resources:

| Model | Weights | NInfer artifact | Artifact size | RTX 2080 Ti status |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 16.29 GiB | Supported |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 16.96 GiB | Current measured deployment |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 21.22 GiB | Supported |

*Note: `nvfp4` is unsupported on Turing (`sm_75`) and Ampere (`sm_86`). Use `groupwise-int`
(W8A16) artifacts.*

---

## VRAM & Context Sizing (22GB Target)

On an RTX 2080 Ti 22GB (~22,528 MiB addressable), available device memory is allocated between model weights, speculative draft structures, CUDA runtime workspaces, and the paged KV cache pool.

### 1. KV Cache Quantization: `--kv-dtype kvarn` (Long-context default)
- **BF16 (`--kv-dtype bf16`)**: Consumes **64.0 KiB per token** (64 MiB per 1,000 context tokens on 27B).
- **INT8 Group-64 (`--kv-dtype int8`)**: Consumes **33.0 KiB per token**, halving KV memory footprint relative to BF16.
- **KVarN (`--kv-dtype kvarn`, 4-bit key / 2-bit value)**: Consumes **13.9 KiB per token**, ~4.6x smaller than BF16. `--kv-dtype kvarn-k4v4` (4-bit value) costs 17.9 KiB per token.

KVarN keeps each sequence's first 128 positions and its still-filling tail page unquantized in BF16
and compresses every complete 64-token page into one structured record. One additional BF16 tail
page per lane preserves exact rewrite-checkpoint state. The current SM75 route uses Tensor Cores for
compressed-prefix QK/PV, stable LSE merging, and shared transformed queries for 2..127-query
verification. `--spec dflash` is not supported under KVarN; `--spec mtp` is.

### 2. Current Measured Deployment

| Model/profile | KV | Context / shared KV | Concurrency | Loaded weights | Runtime reservation | Planner slack |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-27B `groupwise-int` text** | KVarN | 245,760 / 245,760 | 2 | 16.67 GiB | 4.55 GiB | 106.37 MiB |

These values come from the current bounded-prefill build at startup. The engine also reported
265.31 MiB free after startup; `nvidia-smi` showed approximately 21,740 MiB used. Artifact file
size and loaded weight bytes are different quantities. Capacity is not extrapolated to the other
supported artifacts: use their startup planner result when selecting a deployment profile.

---

## Performance (RTX 2080 Ti 22GB)

Current end-to-end measurements use Qwen3.8-27B `groupwise-int`, KVarN, MTP3, 1,024-token
prefill chunks, concurrency two, seed 1234, a 256-token output, a 250 W GPU limit and Turbo
disabled. Prefix reuse is disabled so every input token is computed. The first three requests use
`preserve_thinking=false`; the 62,960-token request preserves the recorded reasoning history.

| Input tokens | Prefill run 1 | Prefill run 2 | Mean |
|---:|---:|---:|---:|
| 8,083 | 577.90 tok/s | 558.32 tok/s | **568.11 tok/s** |
| 18,783 | 463.17 tok/s | 445.53 tok/s | **454.35 tok/s** |
| 21,160 | 442.42 tok/s | 425.44 tok/s | **433.93 tok/s** |
| 62,960 | 270.44 tok/s | 263.26 tok/s | **266.85 tok/s** |

| Input tokens | Decode run 1 | Decode run 2 | Mean | MTP acceptance |
|---:|---:|---:|---:|---:|
| 8,083 | 26.27 tok/s | 25.31 tok/s | **25.79 tok/s** | 45.69% |
| 18,783 | 22.33 tok/s | 21.46 tok/s | **21.89 tok/s** | 42.51% |
| 21,160 | 22.08 tok/s | 21.28 tok/s | **21.68 tok/s** | 44.20% |
| 62,960 | 17.10 tok/s | 17.02 tok/s | **17.06 tok/s** | 42.77% |

Every run generated all 256 requested tokens. During the measured regions, average GPU
utilization was 99.1%–99.9%, peak temperature was 76 C, and host CPU idle never fell below 96%.
Only the benchmark process used the GPU. The second run at each point reflects the thermally stable
state and is 2.7%–3.9% slower than the first, so both values are shown instead of reporting a peak.
Decode is an end-to-end workload result: prompt content and MTP acceptance also affect it, so the
rows show observed throughput at each length rather than a length-only scaling law.

Replaying the identical 8,043-token cache-check prompt reuses 8,039 input tokens and reaches the first generated
token in approximately 1.09s after a process restart. Reused tokens are not counted as raw prefill
throughput. Cold prefill values above are full-request averages; live rolling throughput falls as
each new chunk attends to a longer prefix.

The selected SM75 implementation uses bounded Q4/Q5 cuBLAS prefill for chunks of at least 256
tokens, Tensor Core KVarN QK/PV, stable LSE merging and shared transformed queries for 2..127-query
verification. Final outputs and persistent state retain their required formats, and the changed Ops
pass their independent complete-formula numerical oracles. Diagnostic controls are documented in
[performance](docs/performance.md) and [KVarN records](docs/maintainer/kvarn-records.md).

An ordinary decode step streams approximately 14.66 GiB of projection weights. At the measured
approximately 555 GB/s streaming bandwidth, weight reads alone impose a loose 35.3 tok/s ceiling
before KV scans, recurrent state, unpacking, synchronization and sampling. The measured long-context
rate remains materially below that ceiling; see the
[current bound](docs/performance.md#interpreting-the-remaining-upper-bound).

`ninfer_responses_bench` accepts a recorded Responses request and uses the deployment's context,
concurrency, KVarN, MTP3, temperature 0.6, and presence penalty 1.0 through the same request
translation as HTTP serving. Its output limit applies to the benchmark only, and it never executes
returned tool calls. `--profile-decode` brackets GPU profiling from the first visible token onward:

```bash
./build/bench/ninfer_responses_bench /path/to/model.ninfer /path/to/request.json 512 2
```

`tools/bench/replay_harness_session.mjs` reconstructs text requests from a DeepSeek Harness v0
session log and checks their token counts. `tools/bench/run_pi_ai_cache.mjs` tests the pi-ai
Responses serialization path. Both need a separately installed `@earendil-works/pi-ai` package
whose directory is supplied through `NINFER_PI_AI_ROOT`; they do not modify the client or its settings.

### Structured tool decoding

Qwen function calls use XGrammar structural decoding on the ordinary and MTP paths. Automatic tool
choice does not force a call or constrain normal reasoning/text, but once the model starts the
`<tool_call>` marker, per-token masks require a declared function name and a complete Qwen XML call.
MTP applies the grammar independently at every draft and bonus position. `strict:true` functions
also enforce their parameter JSON Schema; non-strict functions retain open parameters while keeping
the envelope structurally valid. Malformed tool-shaped output fails closed instead of appearing as
assistant text. DFlash retains parser-only automatic tool handling and rejects `strict:true`,
required, or named tool contracts rather than silently ignoring their constraints.

### Responses cache reuse

The builtin template replays the actual `<think>` / `</think>` tokens. Responses replay prefers
raw reasoning `content` over its `summary`, so clients that return both do not duplicate it.
Assistant text and immediately following function calls replay as one assistant turn, and tool
parameters keep their generation order through JSON parsing.
String parameters lose only the single framing newline on each side of the XML value; indentation,
tabs, and intentional blank lines remain part of the argument. Unchanged tool responses can replay
their original generated markup after the server verifies IDs, names, argument values, reasoning,
and assistant text. This prevents formatting normalization from invalidating long prefixes.
These fixes allow exact `append_frontier` reuse during tool loops with `preserve_thinking` off.
The tool-format replay cache is limited only by estimated record/index memory, with a default
budget of 1 GiB and no entry-count limit. Set `--tool-replay-cache-mib N` on `ninfer-serve` to
change the budget in MiB, or use zero to disable this cache. It allocates on demand, replaces
duplicate tool-call IDs, and evicts the oldest records under memory pressure. Tool-call IDs are
hash-indexed so lookup does not scan the full cache as it grows.
The current cross-executable state-restore result above is the README's cache measurement. Tool-loop
reuse is checked behaviorally because its exact hit percentage depends on the appended tool output,
edited history and whether a new user turn strips prior reasoning.

`ninfer_token_decode MODEL.ninfer TOKEN_ID...` decodes trace IDs directly from the artifact's
embedded tokenizer on the CPU. It can distinguish an actual token mismatch from SSE text chunking.

To reproduce against a running service:

```bash
python3 tools/bench/run_responses_cache.py --base-url http://127.0.0.1:8321/v1 --mode streamed --min-hit 0.98
python3 tools/bench/run_responses_cache.py --base-url http://127.0.0.1:8321/v1 --mode stored --min-hit 0.98
```

`deploy/codex-ninfer.config.toml` is an optional Codex profile. Copy it to
`~/.codex/ninfer.config.toml` and select it with `codex --profile ninfer`. Its two-hour stream idle
limit covers long queue and prefill phases at the full 245K context; NInfer continues to send
transport-only SSE comment pings every five seconds.

---

## Fork Features & Customizations

- **Custom W8 GEMM & Split-K Kernels**: Tailored for Turing SM75 thread-block limits and register allocation.
- **Decode-shaped GEMV routes**: One-warp-per-row W8 cores for the vocabulary head and the MTP projections, and a fused Q4 gate/up core that carries a whole speculative verify step through one pass over the matrix — the exact-small-T MMA cores run at a third of Turing's streaming read rate at these token counts. Details in [`docs/maintainer/turing-decode-gemv.md`](docs/maintainer/turing-decode-gemv.md).
- **GDN Routing Optimization**: Routes GDN gating projections to `MmaUnsplit` for token counts $T \ge 9$, resolving cooperative launch limits on Turing.
- **22GB VRAM Memory Tuning**: Startup sizing headroom and paged INT8/BF16 KV allocation profiles calibrated for 22GB capacity.
- **Reasoning Effort Control**: Configurable thinking depth via `--reasoning-effort none|minimal|low|medium|high|xhigh` (`none` disables thinking; `minimal`/`low` concise reasoning; `medium`/`high`/`xhigh` comprehensive reasoning).

---

## Requirements

- **OS**: 64-bit Linux (or WSL2).
- **GPU**: NVIDIA GPU with Turing `sm_75` (RTX 2080 Ti 22GB), Ampere `sm_86`, or Blackwell `sm_120a`.
- **CUDA**: CUDA Toolkit >= 12.8 and compatible NVIDIA driver.
- **Build Tools**: CMake >= 3.28, Ninja, C++20 compiler (GCC >= 11 or Clang >= 14), `pkg-config`.
- **System Libraries**:
  - FFmpeg development libraries (`libavformat >= 60`, `libavcodec >= 60`, `libavutil >= 58`, `libswscale >= 7`)
  - zlib development headers/library (`zlib1g-dev` on Debian/Ubuntu), used for state-cache checksums
  - `libcurl >= 7.85`

---

## Build

```bash
git clone https://github.com/mr-september/ninfer-2080ti-22g.git
cd ninfer-2080ti-22g

# Build for Turing sm_75 (default)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --parallel
```

*(For Ampere or Blackwell, set `-DCMAKE_CUDA_ARCHITECTURES=86` or `-DCMAKE_CUDA_ARCHITECTURES=120a`.)*

Targets:
- `build/apps/ninfer`: CLI inference runner.
- `build/apps/ninfer-serve`: OpenAI & Anthropic HTTP server.

---

## Model Download

Download registered `groupwise-int` `.ninfer` artifacts via the Hugging Face CLI:

```bash
pip install huggingface-hub

# Qwen3.6-27B (groupwise-int)
hf download neroued/Qwen3.6-27B-NInfer qwen3_6_27b.ninfer --local-dir models

# Qwen3.8-27B (groupwise-int)
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --local-dir models

# Qwen3.6-35B-A3B (groupwise-int)
hf download neroued/Qwen3.6-35B-A3B-NInfer qwen3_6_35b_a3b.ninfer --local-dir models
```

---

## CLI Usage

### Text Generation
```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain virtual memory in three sentences." \
  --max-context 8192 \
  --max-new 256 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

### Multimodal Input (Vision)
```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 256 \
  --vision \
  --kv-dtype int8
```

---

## Server Usage

Start the HTTP server:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 \
  --port 8080 \
  --max-context 16384 \
  --kv-dtype int8 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

For the RTX 2080 Ti 22GB deployment bundle, `deploy/restart-ninfer.sh` provides an idempotent
restart command. It resolves the adjacent `ninfer-src/` and `models/` directories, waits for old GPU
allocations to drain, starts the public HTTP server, and waits for `/v1/models` to become healthy:

```bash
./deploy/restart-ninfer.sh restart
./deploy/restart-ninfer.sh restart-daemon
./deploy/restart-ninfer.sh status
./deploy/restart-ninfer.sh watch
./deploy/restart-ninfer.sh watch --details
./deploy/restart-ninfer.sh logs
./deploy/restart-ninfer.sh stop
```

`restart` attaches a fixed-width live table after startup with Queue, Prefill, TTFB, prefill and
decode rates, generated tokens, prompt-cache hit rate, MTP acceptance, and wall time. `Ctrl+C`
exits only the view. Use `restart-daemon` (or `--daemon`) when the restart command must return
immediately.
The duration columns are displayed in seconds after the request completes: `Queue` measures Engine
submission to scheduler admission; `Prefill` is the Engine's recorded prefill duration; `TTFB`
retains the full server-side time to first token, including queueing, prompt preparation, and work
up to that token. Thinking tokens count. Queue plus Prefill need not equal TTFB because preparation
and other scheduling overhead also contribute. `Prefill/s` and `Decode/s` are token rates, while
`Wall` is the full request duration. Missing durations in older logs show `-`; requests with no
generated token show `-` for TTFB. Restart the rebuilt server to emit Queue and Prefill durations.

The watch header shows the GPU model and refreshes every three seconds with running/queued requests,
inference-process CPU utilization, GPU utilization, VRAM, fan percentage, temperature, and power.
CPU is measured from process-wide CPU-time deltas over the sampling interval: 100% means one fully
busy core, and 150% means 1.5 cores. The first sample or an unavailable process shows `-`.
It keeps the script's launch configuration visible above the live
status: URL, nonempty API key, model, context/KV and output limits, KVarN, MTP, concurrency, and
prefill chunk size. Scheduler counts use the server's five-second throughput snapshots;
unknown counts show `-`. `Input/New` is total prompt tokens / tokens requiring prefill. `Finish`
distinguishes completion, tool calls, output limits, cancellation, rejection, and queue timeouts;
errors include their messages even without `--details`. `--details` adds request IDs, thinking
token counts, and prefix reuse paths. Narrow terminals wrap each request into a compact block.
Terminal colors highlight throughput in cyan, successful completion in green, queued/cancelled
or output-limited requests in yellow, and errors in red. Borders and secondary details are muted.
Color is automatic for terminals; redirected output stays plain. Use `--color never` or `NO_COLOR=1`
to disable colors, or `--color always` to retain them in a snapshot.
The view uses a Python 3 standard-library helper and never stops the server. `watch --once` prints
a snapshot without entering the live view. Watch also opens when the service is stopped, showing
its state and recent logs; it follows the PID file when the service starts again. The script starts
the server in a separate session so Ctrl+C in the restart/watch terminal cannot stop inference.
Reopen watch to use the new layout; the new thinking
counter and idle-transition snapshots require restarting the rebuilt server.

The deployment script enables API-key authentication using the fixed shared `API_KEY` near its
top. Configure clients with that same key using `Authorization: Bearer <key>` or `x-api-key: <key>`.
Startup and status probes include the key. The Ready line shows the configured key after the URL
when it is nonempty. Script changes take effect on the next server restart;
direct `ninfer-serve` launches still require an explicit `--api-key` to enable authentication.

Its tested defaults are KVarN, MTP3, concurrency 2, a 245,760-token maximum context and shared KV
capacity, and `default-max-tokens=32768`. The current bounded-prefill build uses approximately
21,740 MiB according to `nvidia-smi`; the engine reports 106.37 MiB of planner slack. Paths and
sizing remain overridable through the `NINFER_MODEL`, `NINFER_BIN`, `NINFER_MAX_CONTEXT`,
`NINFER_KV_CAPACITY`, and `NINFER_DEFAULT_MAX_TOKENS` environment variables.

### Vision cost on the RTX 2080 Ti 22GB

Vision adds its encoder weights, fixed workspace and request-transient buffers. Its context capacity
has not been requalified after the bounded-prefill workspace change, so this README does not publish
a numerical Vision limit. Requalify an explicit context against the current startup planner before
using Vision on the 22GB card. Image/video expansion and text share the configured context, while
merged Vision tokens also consume the model's media envelope. Vision remains opt-in; the default
launcher uses the measured 245,760-token text profile and its persistent state cache.

The 32,768-token default leaves more shared KV capacity available for concurrent work. Clients can
still request up to the remaining context capacity explicitly: 4K–16K for ordinary tool work,
16K–32K for typical long answers, and 128K only when that much output is genuinely required.
Admission accounts for the declared output budget as well as the prompt, so an unnecessarily large
value can keep an otherwise compatible second request queued. This changes capacity planning, not
model quality, unless generation actually reaches the requested limit.
`NINFER_PREFILL_CHUNK` and `NINFER_DRAFT_TOKENS` override the prefill chunk and MTP window for
controlled experiments. The script leaves `preserve_thinking` off by default.
The launcher also accepts `--tool-replay-cache-mib N`, for example
`./deploy/restart-ninfer.sh restart --tool-replay-cache-mib 2048`, or the
`NINFER_TOOL_REPLAY_CACHE_MIB` environment variable (default `1024`). The command-line value takes
precedence, and the selected budget appears in the watch header. The Responses object/context store
has its own separate limits. The launcher enables retained-state caching with 8 GiB RAM
and 50 GiB disk under `/data/ninfer-state-cache` when that data volume is available, falling back
to `$BUNDLE_ROOT/state-cache` otherwise. Override with `NINFER_STATE_CACHE_DIR`,
`NINFER_STATE_CACHE_RAM_MIB`, and `NINFER_STATE_CACHE_MAX_MIB`, or the corresponding
`--state-cache-dir`, `--state-cache-ram-mib`, and `--state-cache-max-mib` flags. State capture waits
for 1,000 ms of complete Engine idleness by default. Override it with
`NINFER_STATE_CACHE_IDLE_MS` or `--state-cache-idle-ms`; zero restores immediate capture. A zero
disk budget disables this feature. Direct `ninfer-serve` launches leave it disabled unless a
directory and positive disk budget are supplied.

Completed text or multimodal requests with at least 256 retained tokens can capture immutable
continuation images (Main/MTP KV, current and checkpoint KVarN stages, current/checkpoint GDN
state, hidden state and prefix identity).
The final result is published before capture begins, so snapshot copying does not extend that
request's reported Wall time. With the default idle delay, a new pending or active request takes
priority; repeated completions reset the deadline and coalesce each lane to its latest state. Once
the Engine remains idle, it captures one dirty lane, then checks the queue again before another.
Graceful shutdown captures any still-valid dirty lanes. Compatible resident continuations keep
coalescing without a copy. A full reset, external snapshot restore, or capacity eviction that would
discard a dirty retained lane captures it first; that exceptional copy is included in the incoming
request's Queue time. An identical state already present in RAM or on disk is detected before any
GPU-to-host copy.

Capture and upload synchronize on the compute stream and briefly occupy the GPU worker. A request
arriving after copying has already started can still wait for that copy to finish. Payload
reads, checksums, writes, fsync and atomic publication run on a separate I/O worker. RAM includes
pending writes; pressure evicts clean images or skips new captures instead of blocking for disk.
Images are persisted in the background after completion, even if their GPU state remains resident,
so a later restart can reuse them. An interrupted write is never a valid cache hit. Saved state
uses existing numeric formats without recompression. Ordinary and MTP execution support both text
and multimodal snapshots; DFlash configurations reject enabling state caching. A multimodal key
includes token IDs and types, three-axis MRoPE positions, media SHA-256, modality, grid, patch
layout, timestamps, and consumer spans. No alias is published at a frontier that divides one Vision
item, and restore repeats the complete identity comparison before mutating GPU state.

Snapshots retain post-Vision model state rather than original media bytes. After a restart the
client must submit the same image or video again so NInfer can acquire it, derive its immutable
identity, and locate the snapshot. A media-preprocessing cache hit avoids repeated decode/resize
work; a state-cache hit then skips Vision GPU encoding and model prefill for the matched prefix.

Cache directories are separated by artifact identity, explicit state ABI, storage configuration
and template. A rebuild preserves cache compatibility when that ABI and configuration are unchanged.
State layouts, codecs and persistent-state semantics require an ABI bump. The switch from executable
identity to ABI identity creates one new namespace; older namespaces remain available for manual
removal. Budgets apply to the current namespace.
Startup reads descriptors only. Once a request has an admissible free lane, it compares the exact
GPU-resident prefix first. It starts an asynchronous RAM/disk load only when a stored prefix is
longer, avoiding disk materialization for an equal or better GPU hit.
Recoverable cache corruption falls back to normal prefill. New physical pages and mappings are
allocated on restore, leaving CUDA Graph addresses stable. The disk cache does not persist public
Responses IDs: after restart clients must resend their history. Tool-format replay metadata is
also process-local, so normalized tool history may still reduce the reusable suffix.

The idle policy favors interactive latency over retaining every intermediate tool turn: a crash or
branch before the latest idle capture may reuse an older snapshot and recompute the suffix. It does
not change model output, resident prefix reuse, or prefill/decode kernels.

`watch --details` shows `State` and GPU upload time (`Restore`). Disk-read waiting is included in
Queue; upload occurs after admission and is included in TTFB. Wall ends when the final result is
published. The following capture is reported separately as a `[state-cache] capture` service-log
line; it can still pause other active inference while copying from the GPU. These are not
decode-throughput improvements. The opt-in
`ninfer_state_cache_bench MODEL REQUEST_JSON CACHE_DIR FIXTURE_JSON MODE` exercises the actual
Responses translation and public Engine with `seed`, `resident`, `ram`, `disk`, and `cold` modes.
`disk` checks the generated tokens against the `resident` reference. Set
`NINFER_VERIFY_STATE_RESTORE=1` for byte-for-byte upload verification during correctness testing;
leave it unset for performance measurement.
`NINFER_SEED` optionally fixes the server's sampling seed for repeatable measurements; leaving it
unset retains fresh request seeds. The script identifies processes by their executable path so
wrapper command lines are not mistaken for the inference service.

CUDA stream synchronization defaults to blocking the host thread until GPU work completes,
avoiding the default driver's busy wait without a fixed sleep between decode rounds. Set
`NINFER_CUDA_WAIT=spin` before starting the process to compare with busy waiting;
`blocking` (or an unset value) selects the default. This changes host waiting only, leaving the
GPU computation and request cancellation boundaries unchanged.

Turbo mode is disabled by default. Add `--turbo` to a starting action to request the card's tested
280 W maximum and ensure the aggressive `nvidia-fan-curve.service` is active:

```bash
./deploy/restart-ninfer.sh restart-daemon --turbo
# Or restart and enter the live dashboard:
./deploy/restart-ninfer.sh --turbo
```

Applying the power limit and starting the fan service require sudo. `stop` dynamically reads and
restores the driver's default power limit (250 W on the tested card); every `restart` therefore
restores the default before selecting the next mode. A failed turbo startup also rolls back to the
default limit. `NINFER_GPU_POWER_LIMIT_W` remains available as an advanced manual startup override,
but it does not replace `--turbo`'s fan-service check. Current default deployment and the final
performance measurements use 250 W.

Driver 595.99.02 exposes manual fan control through NVML. The tested fan curve keeps the driver's
84 C target (so temperature control does not lower clocks), runs at 45% through 40 C, and ramps to
100% at 70 C. Install its system service with:

```bash
sudo install -m 0755 deploy/nvidia-fan-curve.py /usr/local/sbin/nvidia-fan-curve.py
sudo install -m 0644 deploy/nvidia-fan-curve.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-fan-curve.service
```

### Request Example
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "You are a precise technical assistant."},
      {"role": "user", "content": "What is the difference between paging and segmentation?"}
    ],
    "max_tokens": 256,
    "temperature": 0.6
  }'
```

---

## Capabilities & Architecture

- **Batched Decode**: Small-scale concurrent request scheduling with round-boundary compaction and CUDA Graph replay.
- **Speculative Decoding**: Multi-Token Prediction (MTP) with draft windows (1–5 tokens) and optimized draft heads; DFlash support on 35B-A3B.
- **Structural Tool Decoding**: XGrammar C++ constraints for Qwen XML calls, including native
  per-position MTP verification and strict JSON Schema enforcement.
- **Memory Management**: Paged INT8 (group-64) and BF16 KV cache with automatic VRAM capacity detection and prefix reuse.
- **Native Vision**: Image and video token encoding with frozen request-transient allocations.
- **Compiled Chat Frontend**: In-engine chat template rendering avoiding Python/Jinja runtime overhead.

---

## Documentation

- [CLI Usage Guide](docs/cli.md)
- [HTTP Serving Protocol](docs/serving.md)
- [Paged KV Cache Architecture](docs/maintainer/paged-kv-cache.md)
- [Concurrent Inference Engine](docs/maintainer/concurrent-inference-architecture.md)
- [CLI Input Examples](examples/cli/)

---

## Acknowledgements & Upstream Project

- **Original Project:** [NInfer](https://github.com/Neroued/ninfer) by [@Neroued](https://github.com/Neroued).
- **Original Checkpoint Artifacts:** [neroued on Hugging Face](https://huggingface.co/neroued).

---

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).
Model weights are subject to their respective upstream licenses ([Qwen License](https://huggingface.co/Qwen)).
