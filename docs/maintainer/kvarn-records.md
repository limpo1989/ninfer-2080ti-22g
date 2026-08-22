# KVarN 结构化 KV 记录

本文定义 KVarN KV 压缩格式在 NInfer 中的 record 布局、数学契约、Op 边界与 storage 契约。它是
codec、attention 与 KVarN cache family 的 authority；page ownership、capacity 与 frontier 语义仍由
[Paged KV Context Store](paged-kv-cache.md) 规定。

---

## 1. 格式

KVarN 把一个 KV head 的连续 `group` 个 token 压缩成一条 record。压缩在 **Hadamard 旋转域**中进行：

1. 沿 head_dim 做正交对称 Walsh-Hadamard 旋转（每个 token 一次）；
2. 对旋转后的 tile 做 log 域交替 column/row 标准差归一化（Sinkhorn-like），保留全过程中
   imbalance 最低的一组 scale，而不是最后一次迭代的；
3. 对每一行做非对称 RTN 量化到 `key_bits` / `value_bits`；
4. 把该行的 RTN scale 与 zero 吸收进对应的归一化轴。

因为旋转是正交的，把 Q 用同一个矩阵旋转后得到的 score 与原始域完全一致；attention 输出再旋转
一次即回到原始域。任何路径都不需要把去量化后的 K/V tile 落到显存。

重构恒等式：

```
k_rot[d,t] = (code * k_s_col[d] + k_zp[d]) * k_s_row[t]      K tile 为 [head_dim, group]
v_rot[t,d] = (code * v_s_row[t] + v_zp[t]) * v_s_col[d]      V tile 为 [group, head_dim]
```

`group` 恒等于 paged KV 的 `P=64`：一个 record 精确覆盖一个 page，于是 per-token slot
`record_bytes / group` 就是该 plane 的 leading extent，record plane 与其他 KV plane 使用同一套
`[leading, P, Hkv, Nphysical]` 寻址。

### 1.1 已注册 preset

| preset | key bits | value bits | group | record bytes (D=256) | B/token/head |
|---|---|---|---|---|---|
| `kvarn_k4v2_g64` | 4 | 2 | 64 | 14208 | 222 |
| `kvarn_k4v4_g64` | 4 | 4 | 64 | 18304 | 286 |

同几何下 BF16 为 1024 B/token/head，INT8-G64 为 528 B/token/head。Key 的误差经 softmax 指数放大，
value 的误差被 softmax 权重平均掉，所以 preset 把预算优先给 key。

### 1.2 Record 字段顺序

`k_payload`、`k_s_col[D]`、`k_zp[D]`、`k_s_row[G]`、`v_payload`、`v_s_col[D]`、`v_s_row[G]`、
`v_zp[G]`，全部 scale 轴为 FP16。payload 低位优先打包：第 `i` 个 code 位于字节 `i / (8/bits)` 的
`(i % (8/bits)) * bits` 位。record 末尾补齐到 `lcm(8, group)`，使 per-token slot 为整数。

`ninfer::kvarn_record_layout()` 是唯一的 offset authority；device 侧 `KvarnRecordSpec` 是它的
compile-time 镜像，两者不一致时 wrapper 拒绝 launch。

---

## 2. 未量化前缀

前 `kKvarnSinkTokens = 128` 个 token 不进入 record。attention sink 承载了极端的 score 质量，量化
它们是低比特 KV 的主要精度损失来源。未满 `group` 的尾部 page 同样不能压缩，因为 tile 的 scale 是
整块共享的。这两段以 BF16 保存在 **stage buffer** 中，并与 record 段组合成完整历史。

每一行（table row）的 stage 是 `kKvarnStageTokens = kKvarnSinkTokens + P = 192` 个 token：前 128 个
slot 是永久不量化的 sink，后 `P` 个 slot 是仍在填充的尾页。一个 position `p` 落在哪一段由 `p` 与
sink 边界、以及行的 frontier 所在页决定，不需要额外的元数据：

```text
p <  kKvarnSinkTokens                  -> stage slot p                （sink，永久）
floor(p / P) == floor(frontier / P)    -> stage slot 128 + (p mod P)  （尾页，未压缩）
otherwise                              -> record page block_table[floor(p / P)]
```

sink 区间同时也在 record plane 中占据 page（前两个 logical page），但这些 record 永不被读取；这样
logical page index 与 position 的关系对 KVarN 与 BF16 完全一致，capacity accounting 不需要特例。

---

## 3. Op 边界

`include/ninfer/ops/kvarn.h` 定义三个 Op：

- **K1 `kvarn_compress`**：把整页的 BF16 K/V staging 编码成 record，纯函数，无 frontier、allocation
  或 commit 权限；
- **K2 `kvarn_decompress`**：把 record 还原为原始域 BF16 K/V，用于 codec 资格认定与非融合读取；
- **K3 `kvarn_gqa_attention`**：KVarN 版的 A1。把一次调用的 K/V append 进 KVarN storage，并对
  `sink + record + tail + 本次 K/V` 做一次 causal grouped-query attention。

K3 是 storage-mutating 的：它把本次调用**填满**的每一个 page 压缩成 record，其余留在 stage 中。因此
每个 page 只被压缩一次，已 commit 的范围不得重复提交。它接受 `Wq <= W`，`q`/`out` 对齐到本次调用的
最后 `Wq` 列，从而支持「必须 commit W 个 token 但只需要最后一列 attention」的 MTP/speculative 调用；
`q` 与 `out` 同时为空即 commit-only 调用。

K3 对 record 段不施加 mask：record 段整体位于 `positions[0]` 之下，每个 record key 对每次调用的
所有 query token 都是因果可见的。sink、tail 与本次 K/V 段各自产生 split-local
`(max, sum, accumulator)`，全部并入同一次 online-softmax 归约，因此组合不需要重新展开历史。

K3 的 launch shape 只依赖 `W`、`B` 与 head geometry，不依赖 frontier，因此 decode 仍可被 CUDA graph
捕获。已注册几何为 `[256, 24|4]`（group 6）与 `[256, 16|2]`（group 8）。

K3 的实现 profile（非语义要求）：一个 CTA 负责一个 `(kv_head, split, query token)`，`HeadDim` 个
线程。score 阶段把线程映射到 (page 内 token, 通道四分之一)，value 阶段映射到通道，两者都按 payload
的存储顺序读取。record 段的 score 在旋转域中求值；split-local accumulator 在写出前旋转回原始域——
旋转是线性的，与归约的加权求和可交换，所以它与 BF16 段的 partial 可以直接相加。

split 数量由 wrapper 一次性按 `W`、`B` 定出并据此给 partials 定容；launcher 不得按 column chunk
重新推导 split 数量。

---

## 4. Storage 契约

`ops::KvarnBatchLayerView` 是一层 KVarN KV 的完整存储视图：

| 成员 | 形状 | 说明 |
|---|---|---|
| `records` | U8 `[slot_bytes, P, Hkv, Nphysical]` | 每层一个 record plane |
| `block_tables` | I32 `[logical_pages, table_rows]` | 与 BF16/INT8 共用的 block table |
| `stage_k` / `stage_v` | BF16 `[D, Hkv, kKvarnStageTokens, table_rows]` | 每行的 sink + 尾页 |

因为 `slot_bytes = record_bytes / P` 是整数，record plane 与其他 KV plane 使用完全相同的
`[leading, P, Hkv, Nphysical]` 寻址，pool 与 allocator 不需要任何 KVarN-specific 分支：KVarN cache
family 只是「一个 U8 plane / 层、`leading = slot_bytes`」的普通 homogeneous pool，外加一块与 pool
无关、按 `table_rows` 定容的 stage 张量。

stage 是固定开销（`2 * L * D * Hkv * 192 * table_rows * 2 B`），与 `max_context` 无关，所以 KVarN 的
capacity 优势随上下文长度单调变好。

DFlash 在 KVarN 下被拒绝：它的 context append 需要一个 per-token BF16 plane 可写，而 KVarN 只对
超出 sink 的整页保留 record。MTP 则正常工作，其 KV pool 与 Main Text 各自独立地采用同一 KVarN family。

---

## 5. 正确性资格

数学 oracle 是 `ninfer::kvarn_encode_record()` / `kvarn_decode_record_rotated()`：从公开 BF16 输入
出发、以 FP32/FP64 完整求值 KVarN 公式的独立实现。

Sinkhorn 是带 best-so-far 选择的迭代浮点搜索，device kernel **不要求**复现 oracle 的 code 字节；
要求是：

- 对 staged tile 的重构精度不劣于 oracle 自身的量化精度；
- K2 能反演 K1 写出的 record；
- K3 的输出对「record 逻辑值上的 FP64 理想 attention」满足 reduction 判据。

`tests/ops/test_kvarn.cpp` 分两组：codec case 覆盖两个 preset、两种 head geometry 与多页 / 单页；
sequence case 在 K3 上跑完整的 prefill + decode 序列，覆盖「全部位置仍在 sink 内」「decode 跨页」
「批量 B=2/3 且 table row 各异」「speculative 宽度在一次调用内跨页」「35B head geometry」以及
「留下短尾 column chunk 的 prefill 宽度」——最后一项是 split-count 回归的直接守卫。

---

## 6. 已测性能

RTX 2080 Ti 22GB、`qwen3_8_27b`（groupwise-int 权重）、greedy、单请求、无 speculative backend，
needle-in-a-haystack 提示：

| prompt tokens | KV dtype | prefill tok/s | decode tok/s |
|---:|---|---:|---:|
| 10,019 | `bf16` | 230.1 | 18.0 |
| 10,019 | `kvarn` | 184.6 | 19.8 |
| 24,920 | `bf16` | 212.8 | 15.2 |
| 24,920 | `kvarn` | 136.2 | 17.9 |

两种 dtype 在这两个长度上都能从提示中间取回 needle（`kvarn` 在 24,920 token 上逐字引用了该条
记录）。

Decode 略快于 BF16：KV traffic 小了 4.6 倍，抵消掉了解码 record 的额外算术。

Prefill 则随上下文变慢，且差距在拉大（10k 为 BF16 的 0.80x，25k 为 0.64x）。原因是 K3 的 CTA 划分
是 decode 形状的：每个 query token 一个 CTA，各自重新扫描并解码整段 record 历史，没有像 flash
attention 那样把 Q 分块、让一个 K/V tile 被整块 Q 复用。对 `W = prefill_chunk` 的调用，record 解码
工作量因此被放大约 `W` 倍。要在长上下文 prefill 上追平，需要给 K3 增加 Q-tiling 的 prefill profile，
让同一个 record 在 shared memory 中解码一次、服务整个 query tile。这是纯性能工作，不改变本文的任何
数学或 storage 契约。

Capacity 侧的实测：在同一份约 2.66 GiB 的 runtime 预算下，`bf16` 的 `--max-context` 上限在 32,768
附近，`kvarn` 在 131,072 成功、172,032 失败——与 bytes/token 的 4.6x 比值一致。
