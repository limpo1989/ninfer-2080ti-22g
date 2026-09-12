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

K3 有两套实现 profile（均非语义要求），由 query column 数选择：

**decode profile**（`Wq < 128`）：一个 CTA 负责一个 `(kv_head, split, query token)`，`HeadDim` 个
线程。score 阶段把线程映射到 (page 内 token, 通道四分之一)，value 阶段映射到通道，两者都按 payload
的存储顺序读取。split-local partial 写入 workspace，由 reduce kernel 归并。split 数量由 wrapper
一次性按 `W`、`B` 定出并据此给 partials 定容；launcher 不得按 column chunk 重新推导 split 数量。

**SM75 prefill profile**（`Wq >= 128`）把历史拆成两个数学上独立的 attention state：

- compressed prefix：一个 CTA 负责 `(q_head, 32 query rows, row)`。Q 和解码后的 K 以 FP16 输入
  `mma.sync.m16n8k8` 计算 score；PV 保持 FP32 标量累加，避免 256 通道 Tensor Core PV accumulator
  造成寄存器 spill；
- dense current：sink、未满 tail 与本次 K/V 沿用 24 行 Q-tiled kernel，一个 CTA 负责
  `(kv_head, query tile, row)`；
- merge：两侧分别发布 `(max, sum, normalized output)`，最后用共同最大值重标定权重并做稳定 LSE
  合并。

record 段对调用内所有 query 都相同，32 行 tile 把 record 解码和元数据读取摊薄；dense 侧仍保留原始
域的因果 mask。没有 compressed record 的首个 chunk 会让 prefix CTA 直接发布空 state，merge 保留
dense 输出，因此不会为无 prefix 的 Q 做无意义旋转。该实现使用约 36.1 KiB shared memory、无 local
stack spill；实测 48 行 profile 会产生 192 B/thread stack spill，故不进入生产 dispatch。

非 SM75 构建仍使用原来的 24 行 fused Q-tiled prefill kernel。两种实现的语义相同：record score 在
旋转域中求值，prefix accumulator 在合并前旋回原始域；旋转是线性的，与加权求和可交换。

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
「留下短尾 column chunk 的 prefill 宽度」——倒数第二项是 split-count 回归的直接守卫。

另有四个 chunked prefill case：最后一次调用既宽（走 prefill profile）又坐在一段 record 历史之上，
这是唯一能覆盖 prefill profile record pass 的形状（其余 case 的末次调用都是 `records=[2,2)`）。
其中三个的末次宽度不是 query tile 的整数倍，覆盖尾部残 tile；一个是 B=2。

这些 case 的末次 `first` 都对齐到页边界，因为 oracle 只能表达这一种起点：它按调用**前**的
record 边界从 stage 缓冲重建 `[record_end, first)` 段，而该调用自身的 append 已经把那些 tail slot
让给了新的尾页。起点落在页中间时 oracle 读到的是错的 K/V——这是 oracle 的限制，不是 Op 的。

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

上表中的两行 `kvarn` prefill 是 **Q-tiling 之前**的数字，只反映 decode profile 的形状代价，已被
下面的 prefill profile 取代。

### Q-tiled prefill profile

`ninfer_kvarn_attention_bench`（27B 几何、K4V2、`W = Wq = 1024`、`B = 1`，每次 launch 的中位耗时；
即一层一次 prefill chunk 的全部 K3 成本）：

| 已提交上下文 | decode profile | prefill profile | 提速 |
|---:|---:|---:|---:|
| 2,048 | 45.95 ms | 18.54 ms | 2.48x |
| 8,192 | 138.82 ms | 48.77 ms | 2.85x |
| 16,384 | 260.44 ms | 91.28 ms | 2.85x |
| 24,576 | 383.07 ms | 133.79 ms | 2.86x |

同一条 31.2k token 的 needle 提示上端到端跑完整模型（`qwen3_8_27b`、greedy、单请求）：

| KV dtype / K3 profile | prefill 墙钟 | prefill tok/s | 相对 bf16 | decode tok/s | needle |
|---|---:|---:|---:|---:|---|
| `bf16` | 149.3 s | 208.93 | 1.00x | 15.42 | 取回 |
| `kvarn`，decode profile（Q-tiling 之前） | 251.4 s | 124.25 | 0.59x | 17.53 | 取回 |
| `kvarn`，Q-tiled prefill profile | 175.1 s | 178.17 | 0.85x | 17.52 | 取回 |

端到端 1.43x 小于 kernel 的 2.86x，因为 attention 只占 prefill 的一部分：按这两次 `kvarn` 测量
反推，K3 从约 116 s 降到约 40 s，而同一条提示上的非 attention prefill 工作约 135 s 不变。也就是说
K3 现在只占 prefill 的约 23%，即使把它做到零，这条提示的 prefill 上限也只有 1.86x——余下与 BF16
的 0.85x 差距里，只有一小部分还能靠 K3 拿回来。

decode 不受影响（`Wq < 128` 仍走 decode profile），三次运行都逐字取回了 needle。

Capacity 侧的实测：在同一份约 2.66 GiB 的 runtime 预算下，`bf16` 的 `--max-context` 上限在 32,768
附近，`kvarn` 在 131,072 成功、172,032 失败——与 bytes/token 的 4.6x 比值一致。

### SM75 compressed-prefix Tensor Core profile

同一 public Op benchmark、RTX 2080 Ti 22GB、K4V2、27B 几何、`W = Wq = 1024`、`B = 1`；表中是
完整 K3 调用中位耗时，不是单独的 MMA kernel：

| 已提交上下文 | 24-row scalar | 32-row TC QK + LSE merge | 提速 |
|---:|---:|---:|---:|
| 0 | 6.867 ms | 6.733 ms | 2.0% |
| 2,048 | 18.553 ms | 16.077 ms | 13.3% |
| 8,192 | 49.812 ms | 44.296 ms | 11.1% |
| 24,576 | 136.355 ms | 121.387 ms | 11.0% |
| 65,536 | 355.252 ms | 315.848 ms | 11.1% |
| 127,488 | 699.483 ms | 617.704 ms | 11.7% |

在 24,576 / 127,488 两个 anchor 上，`Wq=512` 分别快 9.1% / 10.6%，`Wq=2048` 分别快
8.0% / 14.1%。35B 几何的 `Wq=1024` 在 8,192 / 127,488 上分别快 9.2% / 12.7%。这些数据同时
包含 prefix、dense、merge、stage commit/append 与 workspace traffic；decode（`Wq < 128`）不变。

### SM75 small-query reuse profile

SM75 默认启用 `NINFER_KVARN_TC_QUERY_REUSE=1`；设置为 `0` 可进行同二进制 A/B。
该路径已通过独立 oracle 和重复真实请求验证，适用于 2..127 个 query columns。
该路径把同一 KV head 下的 query heads 与相邻 query tokens 放入一个 MMA tile：`Wq=2` 使用
16-row tile，`3<=Wq<128` 使用 32-row tile、每组最多四个 token。`Wq=1` 保留更快的 scalar 路径。
PV 只累加逻辑 query rows，不为 MMA 补齐的空行执行无效乘加。历史 record 按 prefix 分段，
每段独立计算 QK / FP32 PV / softmax，再将逆旋转后的结果与 dense sink/tail/current 区域进行稳定
LSE 合并。split 数只依赖公开 shape，保持 CUDA Graph 拓扑和 workspace 地址稳定。workspace
容量覆盖小 query 分段数变化产生的内部峰值，供 family 按最大 width 规划时使用。

该路径已通过 CUDA 12.8.93 / GCC 12.3 编译和 GPU 数值验证，沿用 FP64 attention oracle 的原有准则。
16-row tile 限制为每线程最多 128 个寄存器以允许两 CTA 驻留；K4V2 编译结果有 16 bytes 栈空间，
但完整 Op 的实测仍明显快于基线。以下为 RTX 2080 Ti 22GB、250 W、Turbo 关闭、27B 几何、
K4V2、CUDA Graph 的完整 K3 算子结果。它们不是整模型吞吐，也不代表 `Wq>=128` 的长 prefill 收益。

| Prefix / 每请求 | Wq | B | Scalar | Query reuse | 算子加速比 |
|---:|---:|---:|---:|---:|---:|
| 27,885 | 4 | 1 | 1.665 ms | 1.007 ms | 1.65x |
| 127,488 | 4 | 1 | 7.261 ms | 3.590 ms | 2.02x |
| 127,488 | 2 | 1 | 3.861 ms | 2.239 ms | 1.72x |
| 127,488 | 8 | 1 | 14.599 ms | 6.615 ms | 2.21x |
| 127,488 | 40 | 1 | 71.285 ms | 38.440 ms | 1.85x |
| 65,536 | 4 | 2 | 7.772 ms | 3.648 ms | 2.13x |

Wq=4、B=1 使用 A/B/B/A 交错顺序，每次 200 次预热、30 次测量；表中为两次中位数的平均值。
其余形状每次 30 次测量，Wq=2/8/40 的预热次数分别为 500/150/50，B=2 为 150 次。单 query 的
候选在 128K 上从 1.964 ms 退化到 2.153 ms，因此已从候选调度中排除；不将这个回退留给 MTP draft。

同一个 public Op benchmark 可通过进程环境切换两条路径，示例：

```bash
NINFER_KVARN_TC_QUERY_REUSE=0 ./build/bench/ninfer_kvarn_attention_bench \
  --context 27885,127488 --width 4 --graph --warmup 200 --repeat 30
NINFER_KVARN_TC_QUERY_REUSE=1 ./build/bench/ninfer_kvarn_attention_bench \
  --context 27885,127488 --width 4 --graph --warmup 200 --repeat 30
```

`--batch 2` 覆盖两请求的验证几何；`--query-width 4 --width 16` 覆盖 `Wq<W`。历史通过最多
1024-token 的 commit-only 调用精确构建，避免窄 width 设置造成大量初始化调用或越过指定 prefix。
每次计时前恢复 stage 输入，恢复操作不计入 Op 延迟；被测调用新生成的 records 不作为该调用的
历史输入。输出同时报告 width、query width、batch 和 direct/graph 模式。

`ninfer_kvarn_test` 的新增 oracle 场景覆盖 128K prefix、非对齐页边界、ragged query tile、不同有效
长度与空行、`Wq<W`、两种 head geometry 和 CUDA Graph。oracle 从提交的 BF16 输入取未量化
sink/tail，从独立 codec 解码已存储的 records，再完整计算 FP64 attention；它不读取已被本次 append
覆盖的旧 tail slots 来推断输入。另对可见 sink/tail 的持久化结果做精确检查。
可用 `--case name-substring` 选择受修改影响的场景；2-query 分支另通过了混合有效长度的 Graph
场景与 128K prefix 场景，避免仅用 4-query 的另一种 MMA tile 证明它。

#### Real-request MTP result

同一 RTX 2080 Ti 22GB、250 W、Turbo 关闭，使用 recorded Responses 请求、MTP3、chunk 1024、
启动 concurrency 2、temperature 0.6、presence penalty 1.0。测试时停止服务，确认没有其他 GPU
计算进程或编译负载，并记录每两秒的 GPU 频率、温度和功耗。

在旧基准的 `preserve_thinking=false` 配置下，输入为 27,849 tokens。seed 1234/1235/1236 各重复
两次，以 A/B/B/A 顺序比较同一新 prefill 版本的 query reuse 开关。按提交 token 总数除以 decode
总时间统计，吞吐 **19.77 → 21.69 tok/s（+9.7%）**，三个 seed 都有净收益。

显式开启客户端选项 `preserve_thinking=true` 后，同一记录包含 **62,920 tokens**，后续重放完整复用
62,920 tokens。在相同 prefill 实现上，三个 seed 的 MTP3 decode 合计吞吐为：

| Query reuse | Decode |
|---|---:|
| 关闭 | 15.12 tok/s |
| 开启 | 19.23 tok/s |

这两组配置的输入和缓存行为不同，不混合统计。生成轨迹随数值路径和 seed 改变；直接使用提交量
与时间，不要求概率输出逐 token 相同，也不以接受率单独判断性能。旧版本全部优化关闭时，同一
62K reasoning-preserving 配置为 15.90 tok/s；完整优化的版本比较详见 [performance](../performance.md)。

### SM75 Tensor Core PV

长 prefill（Wq>=128）默认启用 `NINFER_KVARN_TC_PV=1`，设置为 `0` 可测量 scalar PV 对照。
同一 key tile 的 QK 只算一次。八个 warp 分成两个 query 组与四个输出维度组，各自累加 16x64
输出，保留 FP32 softmax/LSE，P/V 用 FP16 进入 FP32 MMA。最终继续与 dense sink/tail/current
区域做稳定 LSE 合并。

该 32-row 实现使用 **178 registers/thread、39,040 shared bytes，stack/local 均为零**。
独立 FP64 oracle 覆盖 chunked prefill、ragged/B=2、27B/35B 两种 head geometry。128K 场景
执行实际 128 queries，并对 query 0/31/64/127 的全部 heads/dimensions 计算完整 prefix 公式；
这不是将另一条 GPU 路径的输出作为 oracle。

完整 Op 的代表性 Graph 测量（Wq=W=1024）：

| Prefix | Scalar PV | Tensor Core PV |
|---|---:|---:|
| 24,576 | 123.22 ms | 95.95 ms |
| 127,488 | 624.14 ms | 478.63 ms |

这些是 attention Op 数据。整模型 prefill 还包含投影、GDN、FFN 和其他工作，不能把算子比例
直接当作整模型增幅。

另一个方案将 QK 从两个 warp 分到八个 warp，通过 shared memory 交换 score 后沿用原 softmax。
它通过了相同 oracle，降到 173 registers 且没有 spill，但 A/B/B/A 完整 Op 差异不足 1%，
没有稳定收益，因此已移除。更多活跃 warp 本身不构成性能结论。
