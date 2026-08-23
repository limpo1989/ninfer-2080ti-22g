# Turing decode GEMV routes

## Status and scope

This document is the current authority for the decode-shaped (small-`T`) routes added for
Turing SM75 in the W8 Linear and Q4 LinearSwiGLU families, and for why they exist. Route
selection itself stays in `src/ops/linear/w8/w8_dispatch.cpp` and
`src/ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp`; this file records the measured reasoning
behind the cut-offs those tables encode.

## The problem

Single-stream decode on a 27B `groupwise-int` artifact is a weight-streaming workload: one step
reads 15.9 GiB of quantized weights, so its floor is set by DRAM. An RTX 2080 Ti sustains
**≈ 555 GB/s** on a pure streaming read (measured; 616 GB/s spec), which puts the floor for one
autoregressive step at ≈ 31 ms.

Two families of exact small-`T` kernel were sitting far away from that floor on SM75:

- **The exact-small-T MMA cores** (`w8_small_t_mma_kernel`, `q4_small_t_mma_kernel`) own a 16-row
  output tile per CTA and refill warp-shared staging through two block barriers at every K group.
  At `T = 1..8` almost every output column of that tile is padding, and Turing has no async copy,
  so the `cp.async` pipeline degenerates to a load/store pair that stalls the warp at the shared
  store. The 248,320-row vocabulary head ran at **172 GB/s**, the MTP module's 34,816-row gate/up
  projection at **98 GB/s**.
- **The fused Q4 gate/up GEMV** existed only for `T = 1`. Every `T > 1` call — that is, every
  speculative verify step — fell back to the MMA core and paid a second full read of the matrix
  worth of stall: 569 µs at `T = 4` against 228 µs at `T = 1`, for four times the tokens on a
  matrix that is read once either way.

Because a speculative round is one verify step plus `k` draft steps, both effects land on the
same critical path, and MTP was buying much less than its acceptance length implied.

## The routes

### W8 decode GEMV (`src/ops/linear/w8/w8_decode_gemv.cuh`)

One warp owns one output row and streams a private 16-byte code vector per K phase; a lane's 16
codes always sit inside one 32-wide quantization group, so a single half-warp scale load plus a
broadcast covers the phase. There is no shared memory and no barrier, which leaves the loop a
dependency-free load stream, and the activations are re-read from L1 (a few KiB, shared by every
warp on the SM). Products are formed in FP32 from exactly-represented int8 codes and bf16
activations, which is at least as accurate as the MMA route it replaces.

`W8DecodeGemvShape` carries the two per-geometry knobs: eight warps per CTA for matrices of
16,384 rows or more (four for the narrow ones, so the grid still covers the device), and the
measured token cut-off — 5 for the tall-K down projection, 8 elsewhere. A warp re-reads the whole
activation vector for every row it owns, which is what the MMA core's 16-row tile amortizes, so
the tall-K shape gives that trade up earlier.

### Q4 fused gate/up decode GEMV (`src/ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu`)

The `T = 1` kernel is now templated on `ActiveTokens` and covers `T = 1..8`, so a verify step
reads the matrix once and carries all of its tokens through it — the token count costs arithmetic
instead of DRAM. Two Turing-specific details carry the bandwidth:

- the next tile's codes and scales are prefetched into **registers** and spilled to shared only
  after the current tile has been consumed, so a warp's global loads stay in flight across the
  whole unpack rather than stalling at the shared store;
- the activations are staged **one K tile at a time**, which makes the CTA's shared footprint
  independent of `T` (whole-vector staging would be `T · 10 KiB` and would cost occupancy at
  `T = 4`) and leaves the bf16x2 reads bank-conflict-free — lane `L` reads word `L` of its group.

## Measured

RTX 2080 Ti 22GB, `qwen3_8_27b` (`groupwise-int`), cold-cache Op benchmarks
(`bench/ops/linear_bench.cu`, `bench/ops/q4_linear_swiglu_bench.cu`):

W8 Linear at `T = 1`:

| Shape | before | after |
|---|---:|---:|
| 248320 × 5120 (vocabulary head) | 7,430 µs (182 GB/s) | **2,409 µs (561 GB/s)** |
| 34816 × 5120 (MTP gate/up) | 1,909 µs (99 GB/s) | **350 µs (542 GB/s)** |
| 5120 × 17408 (MTP down) | 606 µs (156 GB/s) | **188 µs (504 GB/s)** |
| 14336 × 5120 (MTP attention) | 360 µs (217 GB/s) | **162 µs (483 GB/s)** |
| 5120 × 10240 (MTP input) | 354 µs (158 GB/s) | **113 µs (493 GB/s)** |
| 5120 × 6144 (MTP attention out) | 223 µs (150 GB/s) | **77 µs (436 GB/s)** |

Q4 fused gate/up, 34816 × 5120, across the token range:

| T | before | after |
|---:|---:|---:|
| 1 | 262 µs (361 GB/s) | **213 µs (444 GB/s)** |
| 2 | 657 µs | **248 µs** |
| 3 | 653 µs | **313 µs** |
| 4 | 651 µs | **363 µs** |
| 6 | 668 µs | **445 µs** |
| 7 | 686 µs | 686 µs (unchanged; back on the MMA core) |

End to end, same prompt and 256 greedy tokens each time, `--max-context 4096`:

| KV | speculation | before | after | delta |
|---|---|---:|---:|---:|
| bf16 | none | 21.76 tok/s | **24.62** | 1.13× |
| bf16 | MTP, 2 drafts | 30.27 tok/s | **44.18** | 1.46× |
| bf16 | MTP, 3 drafts | 28.80 tok/s | **44.81** | 1.56× |
| int8 g64 | none | 21.88 tok/s | **25.50** | 1.17× |
| int8 g64 | MTP, 2 drafts | 30.55 tok/s | **43.90** | 1.44× |
| int8 g64 | MTP, 3 drafts | 32.47 tok/s | **41.57** | 1.28× |

At 34,342 prompt tokens with KVarN records the same configuration decodes at **34.40 tok/s** and
retrieves a planted needle verbatim.

## What was tried and rejected

The same `ActiveTokens` generalization was written for the Q5 row-split GEMV (MLP down
projection, GDN input and output projections) and **measured worse than the tiled SIMT cores it
would have replaced** — 295 µs against 199 µs at `T = 4` on the 5120 × 17408 residual shape. Q5's
split nibble/high-bit unpack costs roughly ten instructions per two weights, so those kernels are
issue-bound rather than latency-bound at small `T`, and adding a token only adds work. The Turing
register prefetch was also tried on that kernel on its own and cost 17% at `T = 1` (74.6 µs
against 63.5 µs on 6144 × 5120), presumably to register pressure. Both were reverted; the Q5
routes are unchanged.

The consequence is that the Q5 shapes now dominate what is left of a decode step, and the next
real lever there is FP16 tensor cores on the dequantized product rather than more SIMT tuning.
