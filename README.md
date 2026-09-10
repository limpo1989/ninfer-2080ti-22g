# NInfer (RTX 2080 Ti 22GB / Turing SM75 Port)

> Selected checkpoints. Maximum single-GPU inference performance.

This repository is a specialized port of [NInfer](https://github.com/Neroued/ninfer) (originally developed by [@Neroued](https://github.com/Neroued)) optimized for NVIDIA Turing architecture (`sm_75`, tuned specifically for the **RTX 2080 Ti 22GB** modded card), while retaining compatibility with Ampere (`sm_86`) and Blackwell (`sm_120a`). It executes text and multimodal (image/video) prompts through a fast local CLI or OpenAI/Anthropic-compatible HTTP servers.

---

## Supported Models & Artifacts

NInfer uses standalone `.ninfer` container artifacts embedding packed weights and tokenizer resources:

| Model | Weights | NInfer Artifact | Size | 22GB VRAM Residency |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 16.29 GiB | Supported (~5.5 GiB KV headroom) |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 16.96 GiB | Supported (~5.0 GiB KV headroom) |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 21.22 GiB | Supported (~0.8 GiB KV headroom) |

*Note: For Turing (`sm_75`) and Ampere (`sm_86`) do not support `nvfp4`. Use `groupwise-int` (W8A16) artifacts.*

---

## VRAM & Context Sizing (22GB Target)

On an RTX 2080 Ti 22GB (~22,528 MiB addressable), available device memory is allocated between model weights, speculative draft structures, CUDA runtime workspaces, and the paged KV cache pool.

### 1. KV Cache Quantization: `--kv-dtype int8` (Recommended)
- **BF16 (`--kv-dtype bf16`)**: Consumes **64.0 KiB per token** (64 MiB per 1,000 context tokens on 27B).
- **INT8 Group-64 (`--kv-dtype int8`)**: Consumes **33.0 KiB per token**, halving KV memory footprint relative to BF16.
- **KVarN (`--kv-dtype kvarn`, 4-bit key / 2-bit value)**: Consumes **13.9 KiB per token**, ~4.6x smaller than BF16. `--kv-dtype kvarn-k4v4` (4-bit value) costs 17.9 KiB per token.

KVarN keeps each sequence's first 128 positions and its still-filling tail page unquantized in BF16 and compresses every complete 64-token page into one structured record, so the extra fixed cost is independent of `--max-context` and its advantage grows with context length. It is a **capacity** format first: decode is marginally faster than BF16, and prefill is slower, though the Q-tiled prefill kernel closed most of that gap — on a 31K-token prompt KVarN prefill went from 0.59x BF16 to 0.85x (2.9x on the KVarN attention Op itself). `--spec dflash` is not supported under KVarN; `--spec mtp` is.

### 2. Context Limits & Concurrency

| Model | Weight Footprint | KV Pool Headroom | Max Context (`--kv-dtype int8`) | Concurrency (`--max-concurrency`) |
|---|:---:|:---:|:---:|:---:|
| **Qwen3.6-27B** | ~16.29 GiB | ~5.0 – 5.5 GiB | Up to 131,072 (128K) | 1 – 4 active requests |
| **Qwen3.8-27B** | ~16.96 GiB | ~4.5 – 5.0 GiB | Up to 131,072 (128K) | 1 – 4 active requests |
| **Qwen3.6-35B-A3B** | ~21.22 GiB | ~0.7 – 0.9 GiB | 4,096 – 8,192 (4K–8K) | 1 active request |

### 3. Execution Configuration Notes
- **27B Deployments**: Standard configuration uses `--kv-dtype int8` with `--kv-capacity auto` (or `--max-context 32768` / `65536`). Speculative decoding (`--spec mtp --draft-tokens 3 --lm-head-draft`) allocates ~0.8 GiB for draft parameters and CUDA Graph state.
- **35B-A3B Deployments**: Requires `--kv-dtype int8`, `--max-context 4096` (or `8192`), and `--max-concurrency 1` to stay within the 22GB ceiling.

---

## Performance (RTX 2080 Ti 22GB)

Earlier reference measurements on NVIDIA GeForce RTX 2080 Ti (`TU102` / `sm_75`, 22 GB VRAM mod) with **Qwen3.8-27B
Dense** (`groupwise-int`, greedy generation, $T_{\text{new}} = 256$ tokens, `--max-context 4096`,
one fixed prompt per row). Current Linux measurements and defaults appear under the SM75 fast path below.

### Committed decode throughput

| KV cache | Speculation | Decode throughput | Tokens / round |
|---|---|:---:|:---:|
| BF16 | none (autoregressive) | **24.62 tok/s** | 1.00 |
| BF16 | MTP, draft window 2 | **44.18 tok/s** | 2.34 |
| BF16 | MTP, draft window 3 | **44.81 tok/s** | 2.73 |
| INT8 group-64 | none (autoregressive) | **25.50 tok/s** | 1.00 |
| INT8 group-64 | MTP, draft window 2 | **43.90 tok/s** | 2.28 |
| INT8 group-64 | MTP, draft window 3 | **41.57 tok/s** | 2.54 |

A decode step reads 15.9 GiB of weights, and this card sustains ≈ 555 GB/s on a streaming read,
so ≈ 31 ms (≈ 32 tok/s) is the autoregressive floor; speculation is what carries the committed
rate past it. See [`docs/maintainer/turing-decode-gemv.md`](docs/maintainer/turing-decode-gemv.md)
for the decode-path kernel routes and their measured before/after.

### Long context

At 34,342 prompt tokens with `--kv-dtype kvarn` and MTP draft window 2: prefill **168 tok/s**,
decode **34.40 tok/s**, 3.00 tokens per round, planted needle retrieved verbatim.

### Prefill throughput
- **Short Prompt ($T = 22$ tokens):** ~71 – 78 tok/s
- **Medium Prompt ($T = 62$ tokens):** ~134 – 135 tok/s
- Current long and short prefill routes are described below; the earlier prompt measurements above used a different host and route set.

### SM75 groupwise-int prefill fast path

Turing has neither `cp.async` nor native BF16 Tensor Core instructions. The SM75 long-prefill
path therefore converts the already-loaded BF16 fragments to FP16, executes native FP16 HMMA with
FP32 accumulation, and overlaps the next quantized-weight and activation loads through registers.
This is enabled by default for the Qwen3.8 Q4 SwiGLU and Q5 residual-add C128 routes. SM75 grouped
Q4/Q5 attention/GDN input projections also use FP16 HMMA. Both BF16 panels are scaled by 256
before conversion, then partial sums are scaled back by 1/65536. A warp-wide check permits only
zeros and magnitudes in [2^-22, 255], making conversion exact and keeping nonzero FP16 operands
normal; other panels fall back to FP32. The GDN chunk output uses native HMMA
only when the BF16 operands are exactly representable in FP16. Accumulation and persistent state
retain their original dtypes; FP32 accumulation order can differ and bitwise output identity is
not promised. The registered operators retain their independent FP32/FP64 oracle thresholds.

Measured on RTX 2080 Ti 22GB, driver 595.99.02, CUDA 12.8.93, with the 7,680-token NIAH prompt,
INT8 KV, MTP3, and the same aggressive fan curve for every run:

| Configuration | Prefill | Decode | Result |
|---|---:|---:|---|
| Original Q4/Q5 paths | 77.51 tok/s | 36.01 tok/s | `ORCHID=4938` |
| Q4 FP16 HMMA only | 98.62 tok/s | 36.73 tok/s | `ORCHID=4938` |
| Q5 FP16 HMMA only | 112.82 tok/s | 36.60 tok/s | `ORCHID=4938` |
| Default Q4 + Q5 fast paths | **201.34 tok/s** | **38.94 tok/s** | `ORCHID=4938` |

The focused Q4 and Q5 independent-oracle tests pass with the default fast paths. Override the
routes only for diagnosis: `NINFER_SWIGLU_KMODE=0` or `NINFER_Q5ADD_KMODE=0` selects the original
path, `=2` selects FP16 HMMA without register prefetch, and `=3` selects the default full fast path.
The experimental W8 `KMODE=3` is not enabled by default and is not part of this optimization.

Short prefills use the fused Q4 C128 route from 17 tokens and the Q5 C128 route from 49 tokens.
Q4 decode/verify uses the paired GEMV through eight columns. Its next weight tile is held in
registers, so one shared staging buffer suffices; this also keeps the eight-column batch resident
at two CTAs per SM. The measured Q4 64-token latency fell
from 14.48 ms to 3.49 ms; Q5 down-projection at 80 tokens fell from 6.99 ms to 2.94 ms. Set
`NINFER_SWIGLU_FUSED_PREFILL=0`, `NINFER_Q5ADD_SMALL_HMMA=0`, `NINFER_GROUPED_HMMA=0`, or
`NINFER_GDN_OUTPUT_HMMA=0` to disable the corresponding optimization for comparison.
Detailed per-layer profiling is opt-in with `NINFER_PROFILE_PREFILL=1`; ordinary serving metrics
remain available without per-layer CUDA timing events.

The KVarN decode kernel limits register allocation to support two resident CTAs and requests
the corresponding shared-memory carveout. On SM75 this reduced its register count from about
247 to 128 per thread without spills. At a 27,885-token history and four query columns, the
isolated attention latency decreased from 1.677 ms to 1.425 ms. Q4 paired gate/up latency changed
from 368 to 348 microseconds at four columns and from 1,261 to 543 microseconds at eight columns.
These changes retain the existing numerical formats and oracle tolerances.

On the Linux dual E5-2690 v3 host (125 GiB RAM), the public Engine benchmark with the same
Qwen3.8-27B artifact, KVarN, 8,192 input tokens, 256 decode tokens, MTP3, and two repetitions
measured:

This is a forced-length greedy token-stream benchmark. Its MTP acceptance was approximately 98%;
agent conversations with longer histories and stochastic sampling require their own measurements.

| Configuration | Prefill | Decode |
|---|---:|---:|
| Earlier Q4/Q5 fast paths, 250 W GPU limit | 185.10 tok/s | 40.07 tok/s |
| Final defaults with exact operand scaling, default 250 W | 196.88 tok/s | 39.93 tok/s |

At the same default power limit, prefill improved by 6.4%. Decode was effectively unchanged
(-0.3%, within run-to-run variation); this measurement does not establish a decode speedup.

The benchmark context capacity was 32,768; serving retains 245,760 and the 131,072 default
output allowance. This host retains the card's default 250 W power limit. The restart script
does not change it unless explicitly requested; the fan-control systemd unit does not set power.
The new kernels are enabled directly in source on SM75, so their effect does not depend on
setting environment variables in the restart script. The delivered path avoids conversion
rounding in the new input-projection kernels. FP32 accumulation order may still differ.

```bash
LD_LIBRARY_PATH="$HOME/.local/ninfer-deps/lib:/usr/local/cuda-12.8/lib64" \
  ./build/bench/ninfer_bench --weights /path/to/qwen3_8_27b.ninfer \
  -pg 8192,256 -r 2 --warmup 0 --max-ctx 32768 --kv-dtype kvarn \
  --mtp-draft-tokens 3 --lm-head-draft -o json
```

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
At the default 250 W limit, Codex CLI 0.153.4 with three sequential shell calls produced warm
continuation hits of 98.65%, 99.09%, and 99.09% with 0.834–0.956 s time to first token;
a controlled five-request tool loop with an initial 6,583-token prompt reached 99.55–99.56%
after the cold start.
Cold starts, large new tool outputs, edited history, and stripping reasoning at a new user turn
are excluded from this warm-continuation claim.

A recorded DeepSeek Harness request with 25,063 input tokens reproduced a later tool-continuation
hit of 70.34% and 55.80 s to first token. Preserving parameter whitespace and original tool markup
raised that continuation to 99.76% and reduced the wait to 1.21 s. A further continuation retained
99.89%. The first response's client-observed decode rate changed from 28.40 to 29.09 tok/s;
the request, seed, output count (916), and MTP acceptance (84.4%) matched. Subsequent output lengths
changed after the corrected history, so their total wall times are not used as a decode-speed claim.

`ninfer_token_decode MODEL.ninfer TOKEN_ID...` decodes trace IDs directly from the artifact's
embedded tokenizer on the CPU. It can distinguish an actual token mismatch from SSE text chunking.

To reproduce against a running service:

```bash
python3 tools/bench/run_responses_cache.py --base-url http://127.0.0.1:8321/v1 --mode streamed --min-hit 0.98
python3 tools/bench/run_responses_cache.py --base-url http://127.0.0.1:8321/v1 --mode stored --min-hit 0.98
```

`deploy/codex-ninfer.config.toml` is an optional Codex profile. Copy it to
`~/.codex/ninfer.config.toml` and select it with `codex --profile ninfer`.

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
capacity, and `default-max-tokens=131072`. On the 22,528 MiB card this uses about 21,610 MiB after
startup while retaining about 220 MiB of planner slack. Paths and sizing remain overridable through
the `NINFER_MODEL`, `NINFER_BIN`, `NINFER_MAX_CONTEXT`, `NINFER_KV_CAPACITY`, and
`NINFER_DEFAULT_MAX_TOKENS` environment variables.
`NINFER_PREFILL_CHUNK` and `NINFER_DRAFT_TOKENS` override the prefill chunk and MTP window for
controlled experiments. The script leaves `preserve_thinking` off by default.
The launcher also accepts `--tool-replay-cache-mib N`, for example
`./deploy/restart-ninfer.sh restart --tool-replay-cache-mib 2048`, or the
`NINFER_TOOL_REPLAY_CACHE_MIB` environment variable (default `1024`). The command-line value takes
precedence, and the selected budget appears in the watch header. The Responses object/context store
has its own separate limits.
`NINFER_SEED` optionally fixes the server's sampling seed for repeatable measurements; leaving it
unset retains fresh request seeds. The script identifies processes by their executable path so
wrapper command lines are not mistaken for the inference service.

CUDA stream synchronization defaults to blocking the host thread until GPU work completes,
avoiding the default driver's busy wait without a fixed sleep between decode rounds. Set
`NINFER_CUDA_WAIT=spin` before starting the process to compare with busy waiting;
`blocking` (or an unset value) selects the default. This changes host waiting only, leaving the
GPU computation and request cancellation boundaries unchanged.
On the dual E5-2690 v3 / RTX 2080 Ti host with driver 595.99.02, a controlled 20 ms
asynchronous stream-completion test (40 interleaved samples per mode) reduced the waiting thread's
CPU use from 100.0% to 0.35%. Median completion-to-return latency was 104 microseconds with
blocking synchronization, versus 4 microseconds with spin and 952 microseconds with 1 ms polling.
This measures host waiting overhead, not end-to-end model throughput.

The optional GPU power override is disabled by default. To explicitly request 280 W at startup,
use `NINFER_GPU_POWER_LIMIT_W=280 ./deploy/restart-ninfer.sh restart-daemon`; applying it requires
sudo. An unset or empty variable leaves the current driver limit unchanged. To restore this
host's default after an opt-in run, use `sudo nvidia-smi -i 0 -pl 250`. Current deployment and the
final performance measurements use 250 W.

Driver 595.99.02 exposes manual fan control through NVML. The tested fan curve keeps the driver's
84 C target (so temperature control does not lower clocks), runs at 45% through 40 C, and ramps to
100% at 67 C. Install its system service with:

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
