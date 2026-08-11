# 11 · attention kernel 与算子融合

前面各章处理的是「调度什么时候算」。本章处理「算的时候怎么快」，即算子层。

这一层的内容对不写 CUDA 的岗位同样重要：它决定了引擎能支持多长的上下文、为什么某些配置会退化、以及 profile 结果该怎么读。

---

## 1. 问题：attention 为什么需要专门的 kernel

矩阵乘可以直接调用 cuBLAS，效率接近硬件上限。attention 不能，原因有三：

1. **中间矩阵太大**。注意力分数矩阵是 `[N, N]`，与序列长度的平方成正比。N = 32768 时这个矩阵是 2 GB，N = 131072 时是 34 GB。单是它就装不下。
2. **中间矩阵的访存是浪费的**。它写回显存只是为了再读出来做 softmax，之后就丢弃。
3. **prefill 与 decode 的形状完全不同**。prefill 是 `[N, d] × [d, N]`，decode 是 `[1, d] × [d, N]`。前者是标准矩阵乘，后者是矩阵向量乘，最优实现不同。

---

## 2. FlashAttention

### 2.1 核心思想

不构造完整的 `[N, N]` 矩阵，而是分块计算，用增量方式维护 softmax 的归一化。

```
对每个 query 块 i：
    维护三个运行状态：当前最大值 m、指数和 l、累加输出 o
    对每个 key/value 块 j：
        s = q_i · k_j^T / sqrt(d)            ← 唯一的中间矩阵，尺寸 [bq, bkv]
        m_new = max(m, rowmax(s))
        alpha = exp(m - m_new)               ← 已累计结果的重缩放系数
        p = exp(s - m_new)
        l = alpha * l + rowsum(p)
        o = alpha * o + p · v_j
        m = m_new
    输出 o / l
```

### 2.1.1 为什么需要重缩放

softmax 的标准做法是先减去最大值再取指数，避免 `exp` 溢出：

```
softmax(s)_i = exp(s_i - m) / Σ exp(s_j - m)，其中 m = max(s)
```

分块计算的困难在于：处理第一块时还不知道全局最大值。若后面某块出现更大的值，前面已经用旧的 m 算出来的结果就全错了。

用一个 4 个元素的例子说明。真实分数是 `s = [1, 2, 8, 3]`，分成两块 `[1, 2]` 与 `[8, 3]`。

**处理第一块**（此时 m = 2）：

```
p = exp([1,2] - 2) = [0.368, 1.0]
l = 0.368 + 1.0 = 1.368
o = 0.368·v0 + 1.0·v1
```

**处理第二块**，发现新的最大值 8：

```
m_new = 8
```

现在问题出现了：`l` 与 `o` 里的每一项都是按 `exp(s - 2)` 算的，而正确的应当是 `exp(s - 8)`。两者相差一个固定倍数：

```
exp(s - 8) = exp(s - 2) × exp(2 - 8) = exp(s - 2) × exp(-6)
```

这个 `exp(2 - 8)` 就是代码里的 `alpha = exp(m - m_new)`。**它对第一块里的每一项都是同一个数**，因此只要把已累计的 `l` 与 `o` 各乘一次 alpha，就等价于当初就用 m_new 算过一遍：

```
alpha = exp(2 - 8) = 0.00248
l = 0.00248 × 1.368 + rowsum(exp([8,3] - 8))
  = 0.00339 + (1.0 + 0.00674) = 1.01013
```

直接一次性计算做对照：

```
exp([1,2,8,3] - 8) = [0.00091, 0.00248, 1.0, 0.00674]
求和 = 1.01013
```

两者一致。

**这一步是整个算法成立的关键**：alpha 让「用旧的最大值算出的部分结果」可以被无损地修正为「用新的最大值算出的结果」，因此不需要预先知道全局最大值，也不需要保留任何中间数据。第 8 节代码的建议实验之一就是去掉 alpha，观察结果如何出错。

### 2.2 不是近似算法

代码的验证结果：

```
N=  128 d= 64  最大绝对误差 8.882e-16  一致
N=  512 d= 64  最大绝对误差 6.661e-16  一致
N= 1024 d=128  最大绝对误差 6.106e-16  一致
```

误差在双精度浮点的舍入范围内。**FlashAttention 省的是显存与访存，不是精度**。这一点在面试中经常被问，回答「它用了近似」是错的。

### 2.3 收益

中间矩阵的显存占用：

| N | 朴素实现的中间矩阵 | 分块实现的中间矩阵 | 比值 |
|---|---|---|---|
| 2048 | 8.4 MB | 8.2 KB | 1024x |
| 8192 | 134.2 MB | 8.2 KB | 16384x |
| 32768 | 2147.5 MB | 8.2 KB | 262144x |
| 131072 | 34359.7 MB | 8.2 KB | 4194304x |

分块实现的中间矩阵大小固定（由 SRAM 容量决定），与序列长度无关。

访存量的对比（单头，d = 128）：

| N | 朴素访存 | 分块访存 | 比值 |
|---|---|---|---|
| 512 | 2.0 MB | 0.5 MB | 3.8x |
| 2048 | 26.7 MB | 2.1 MB | 12.8x |
| 8192 | 408.9 MB | 12.6 MB | 32.5x |
| 32768 | 6467.6 MB | 151.0 MB | 42.8x |

**分块实现的计算量没有减少，甚至因为重复读 KV 略有增加**。收益全部来自访存。这是一个典型的用计算换访存的设计，符合第 02 章的结论：现代 GPU 的算力增长快于带宽，因此这个交换是划算的。

### 2.4 版本差异

| 版本 | 主要改动 |
|---|---|
| FlashAttention-1 | 提出分块与 online softmax |
| FlashAttention-2 | 调整并行划分（按序列维度并行而不只是按 batch 与头），减少非矩阵乘操作 |
| FlashAttention-3 | 针对 Hopper 的异步与 fp8 优化 |
| FlashDecoding | 针对 decode 场景：batch 与头数不足以填满 GPU 时，把 KV 序列维度也切分并行 |

FlashDecoding 值得单独说明：decode 时 query 只有 1 个 token，天然的并行度只有「batch × 头数」。当 batch 较小时这个数字远小于 SM 数量，GPU 大部分空闲。解决办法是把 KV 的序列维度也切开，多个线程块各算一段，最后再合并（合并时需要用各段的 m 与 l 做加权，与 online softmax 的合并逻辑相同）。

---

## 3. PagedAttention 的 kernel

分页之后 KV 不再连续，kernel 需要改造。代码中的实现：

```python
for pos in range(seq_len):
    logical = pos // block_size
    offset  = pos % block_size
    phys    = block_table[logical]      # 间接寻址：查表
    k = kv_cache[phys, offset, 0]
    v = kv_cache[phys, offset, 1]
    ...
```

验证结果：分页实现与连续实现的最大绝对误差 8.327e-17，完全一致。**分页只改变数据的物理位置与寻址方式，不改变计算结果**。

真实 kernel 中的处理比这个循环高效得多：

- block table 一次性读入共享内存，不是每个 token 查一次
- 以 block 为单位处理，块内是连续访存，只有跨块时才有跳跃
- 与 FlashAttention 的分块结合，block_size 通常就是 kernel 的 KV 分块大小

开销来源：一次间接访问，以及每 block_size 个 token 一次地址跳跃。论文报告的开销在 20%-26%，换来的是显存利用率从 20%-40% 提升到接近 100%。

---

## 4. 算子融合

### 4.1 收益来源

逐元素算子（归一化、激活、残差）的计算量可以忽略，但访存量不可忽略：每个算子都要把中间张量写回显存再读出来。第 03 章的统计显示，prefill 场景下这部分占总访存量的 56%。

代码给出的估算（32 层，hidden 4096，8192 token）：

```
未融合时这些算子的访存量: 44.0 GB
在 2 TB/s 带宽上对应      : 22.0 ms
融合后                    : 接近 0
```

融合的做法是把相邻算子写进同一个 kernel，中间结果保留在寄存器或共享内存中。常见的融合组合：

| 融合 | 说明 |
|---|---|
| RMSNorm + 后续的矩阵乘 | 归一化的输出直接进入矩阵乘的输入阶段 |
| SiLU + 逐元素乘 | SwiGLU 的两个分支 |
| 残差相加 + 下一个 RMSNorm | 两个逐元素操作合并 |
| 矩阵乘 + 偏置 + 激活 | epilogue 融合，cuBLAS 与 CUTLASS 都支持 |
| RoPE + KV cache 写入 | 位置编码后直接写入目标槽位 |

### 4.2 融合的边界

不是所有算子都能融合。限制条件：

- 相邻算子之间不能有需要全局同步的操作（例如跨 token 的归约）
- 融合后的 kernel 占用的寄存器与共享内存不能超限，否则占用率下降反而变慢
- 融合增加了 kernel 的特化程度，模型结构变化时需要重写

因此编译式方案（TensorRT-LLM、torch.compile）在融合上有优势：它们在图层面自动寻找可融合的子图，而不是手写。

---

## 5. CUDA Graph

### 5.1 问题

每次 kernel 启动有 5-10 微秒的固定开销。一步 decode 的 kernel 数量：

| 每层 kernel 数 | 总启动次数 | 启动开销 | 占 decode 步（19.6 ms）的比例 |
|---|---|---|---|
| 5 | 160 | 1.12 ms | 5.7% |
| 10 | 320 | 2.24 ms | 11.4% |
| 20 | 640 | 4.48 ms | 22.9% |

当 batch 较小、模型较小时，这个比例更高。

### 5.2 机制与代价

CUDA Graph 把一段固定的 kernel 序列录制为一张图，之后一次提交即可执行全部 kernel，启动开销降到接近 0。

代价是**图内的张量形状必须固定**。而推理引擎的 batch 大小每步都在变化。解决办法是按 batch 大小分档录制多张图（例如 1, 2, 4, 8, 16, ..., 256），运行时向上取整到最近的档位，多出的位置用 padding 填充。

这带来三个实际影响：

1. 启动时间变长（要录制多张图）与显存占用增加（每张图有自己的固定缓冲区）
2. 实际 batch 略小于档位时会浪费部分计算
3. `max_num_seqs` 应当设置为档位上的值，否则最大 batch 会被向上取整到更大的档位

prefill 阶段的 token 数变化范围太大，通常不使用 CUDA Graph；启用 chunked prefill 后 chunk 大小固定，则有可能覆盖。

---

## 6. kernel backend 的选择

引擎通常支持多个 attention 后端，运行时按硬件与配置选择：

| 后端 | 特点 |
|---|---|
| FlashAttention | prefill 场景的主力，成熟度最高 |
| FlashInfer | 对 decode 与分页场景优化，支持更多变体（MLA、投机解码的树形注意力） |
| Triton 实现 | 可移植性好，便于修改与研究；性能通常略低于手写 CUDA |
| xFormers | 早期方案，覆盖面广 |
| 厂商专用 | TensorRT-LLM 的内置 kernel，CPU/NPU 的对应实现 |

选择的依据包括：GPU 架构、head_dim 是否被支持、是否需要滑动窗口或 ALiBi、是否启用了 fp8 KV、是否是 MLA 结构。引擎中通常有一个 backend 选择函数，读源码时可以从它入手，能一次看清所有约束条件。

**这也是部署时的一个常见问题来源**：某个配置组合（例如 head_dim 不是 64 的倍数 + fp8 KV + 滑动窗口）没有对应的 kernel，引擎会回退到较慢的实现或直接报错。

---

## 7. 引擎实现对照

| 引擎 | attention 层的组织 |
|---|---|
| vLLM | `vllm/attention/` 下的 backend 抽象，`AttentionImpl` 接口，V1 中统一了 prefill 与 decode 的调用路径 |
| SGLang | `python/sglang/srt/layers/attention/` 下多个 backend，可用 `--attention-backend` 指定 |
| TensorRT-LLM | 编译期选择并生成 plugin |

vLLM V1 的一个重要改动是统一 prefill 与 decode 的 kernel 调用：一步之内既有多 token 的 prefill 序列又有单 token 的 decode 序列，用变长（varlen）接口一次处理。这要求 kernel 支持每个序列不同的 query 长度，是 chunked prefill 成为默认路径的前提。

---

## 8. 代码

`code/ch11_flash_attention.py` 用 numpy 实现：

1. 朴素 attention（构造完整分数矩阵）
2. 分块 attention（online softmax），含因果掩码与整块跳过
3. 两者的数值一致性断言，以及显存与访存的对比表
4. PagedAttention 的 decode 路径，含与连续实现的一致性断言
5. 算子融合与 CUDA Graph 的收益估算

运行：

```bash
python3 code/ch11_flash_attention.py
```

建议的实验：

- 修改 `flash_attention` 的 `block_q` 与 `block_kv`，验证结果不变。真实 kernel 中这两个值由 SRAM 容量与寄存器数量决定。
- 去掉 `alpha` 的重缩放（即不处理最大值的更新），观察结果如何出错。这能说明 online softmax 中那一步的必要性。
- 把因果掩码去掉，比较耗时。因果掩码让计算量减半，但需要 kernel 能跳过整块被掩码的部分——代码中的 `if np.all(np.isneginf(s)): continue` 就是这个优化。

本章讲的是算法。这些算法在 GPU 上具体怎么执行——线程块与 warp 的划分、共享内存的使用、tensor core 的 MMA 指令、为什么 occupancy 会限制融合的收益——见附录 A。
