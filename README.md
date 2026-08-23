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

Measured on NVIDIA GeForce RTX 2080 Ti (`TU102` / `sm_75`, 22 GB VRAM mod) with **Qwen3.8-27B
Dense** (`groupwise-int`, greedy generation, $T_{\text{new}} = 256$ tokens, `--max-context 4096`,
one fixed prompt per row).

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
- **Long Prefill ($T \ge 1024$ – $2048$ tokens):** Routed via GDN `MmaUnsplit` to operate within Turing SM75 shared-memory and cooperative CTA launch limits (1 CTA / SM, 40 KiB smem).

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
