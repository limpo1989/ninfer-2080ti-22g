# KVarN 结构化 KV 记录

本文定义 KVarN KV 压缩格式在 NInfer 中的 record 布局、数学契约和 Op 边界。它是 codec 与
attention 的 authority；page ownership、capacity 与 frontier 语义仍由
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
整块共享的。这两段由调用方以 BF16 保存，并与 record 段组合成完整历史。

---

## 3. Op 边界

`include/ninfer/ops/kvarn.h` 定义三个 Op，均不拥有 frontier、allocation 或 commit 权限：

- **K1 `kvarn_compress`**：把整页的 BF16 K/V staging 编码成 record；
- **K2 `kvarn_decompress`**：把 record 还原为原始域 BF16 K/V，用于 codec 资格认定与非融合读取；
- **K3 `kvarn_attention_cached`**：直接在 record 上做 grouped-query attention，输出 BF16。

K3 对 record 段不施加 mask：契约要求每个 record key 对每个 query token 都是因果可见的。sink 与
tail 段由调用方单独计算；两段的 split-local `(max, sum, accumulator)` 可以并入同一次 online-softmax
归约，因此组合不需要重新展开历史。

K3 的实现 profile（非语义要求）：一个 CTA 负责一个 `(kv_head, split, query token)`，`HeadDim` 个
线程。score 阶段把线程映射到 (page 内 token, 通道四分之一)，value 阶段映射到通道，两者都按 payload
的存储顺序读取。split-local accumulator 在写出前旋转回原始域——旋转是线性的，与归约的加权求和可交换。

---

## 4. 正确性资格

数学 oracle 是 `ninfer::kvarn_encode_record()` / `kvarn_decode_record_rotated()`：从公开 BF16 输入
出发、以 FP32/FP64 完整求值 KVarN 公式的独立实现。

Sinkhorn 是带 best-so-far 选择的迭代浮点搜索，device kernel **不要求**复现 oracle 的 code 字节；
要求是：

- 对 staged tile 的重构精度不劣于 oracle 自身的量化精度；
- K2 能反演 K1 写出的 record；
- K3 的输出对「record 逻辑值上的 FP64 理想 attention」满足 reduction 判据。

`tests/ops/test_kvarn.cpp` 覆盖两个 preset、两种 head geometry、多页 / 单页 / 40 页 split-K 以及
多 query token。
