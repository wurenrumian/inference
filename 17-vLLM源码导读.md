# 17 · vLLM 源码导读

前面各章的机制在 vLLM 中都有对应实现。本章给出阅读路径：从哪里入手、按什么顺序、每个模块对应前面哪一章。

**版本说明**：vLLM 的目录结构与接口变动较快，本章基于 V1 架构（2025 年起的默认引擎）。具体路径以你所读版本的代码为准，模块的职责划分与调用顺序相对稳定。

---

## 1. 进程结构

V1 把工作拆到多个进程，避免 Python GIL 让 CPU 侧工作阻塞 GPU 执行循环。

```
API 进程（前端）
  ├─ HTTP 服务、参数校验、聊天模板渲染
  ├─ tokenize
  └─ detokenize 与流式返回
        │
        │ ZMQ（进程间消息）
        ▼
EngineCore 进程（引擎核心）
  ├─ Scheduler          调度决策
  ├─ KVCacheManager     block 分配与前缀缓存
  └─ ModelExecutor
        │
        ▼
Worker 进程（每张卡一个，TP > 1 时多个）
  ├─ ModelRunner        构造输入张量、执行前向
  ├─ 模型定义
  └─ AttentionBackend   attention kernel
```

对应关系：

| 进程/模块 | 对应章节 |
|---|---|
| API 进程 | 01（前处理与后处理） |
| Scheduler | 08、09 |
| KVCacheManager | 06、07 |
| ModelRunner | 01（第 7 步）、11 |
| AttentionBackend | 11 |
| Sampler | 12 |

---

## 2. 目录地图

主要目录及其内容：

| 路径 | 内容 |
|---|---|
| `vllm/entrypoints/openai/` | OpenAI 兼容的 HTTP 接口 |
| `vllm/v1/engine/` | 引擎核心：`core.py`（EngineCore 主循环）、`processor.py`（前处理）、`output_processor.py`（后处理） |
| `vllm/v1/core/` | 调度与显存：`sched/scheduler.py`、`kv_cache_manager.py`、`block_pool.py`、`kv_cache_utils.py` |
| `vllm/v1/worker/` | `gpu_model_runner.py`（输入构造与执行）、`gpu_worker.py` |
| `vllm/v1/attention/backends/` | attention backend 的实现与选择 |
| `vllm/v1/sample/` | 采样器与 logits 处理器 |
| `vllm/model_executor/models/` | 各模型的结构定义 |
| `vllm/model_executor/layers/` | 通用层：线性层（含并行切分）、归一化、旋转位置编码、量化方法 |
| `vllm/distributed/` | 并行组的建立与集合通信 |
| `vllm/config/` | 全部配置项的定义，是查参数含义的地方 |

---

## 3. 一次请求的调用路径

对照第 01 章的生命周期图：

```
1. entrypoints/openai/serving_chat.py
     解析请求，渲染聊天模板

2. v1/engine/processor.py
     tokenize，构造 EngineCoreRequest

3. （ZMQ 发送到 EngineCore 进程）

4. v1/engine/core.py 的主循环
     add_request → scheduler.add_request → waiting 队列

5. scheduler.schedule()
     决定本步的 batch，返回 SchedulerOutput

6. v1/worker/gpu_model_runner.py 的 execute_model
     构造输入张量 → 前向 → 采样

7. scheduler.update_from_output()
     更新序列状态，判断结束

8. （ZMQ 返回到 API 进程）

9. v1/engine/output_processor.py
     detokenize，检查 stop 字符串，流式返回
```

---

## 4. 一步迭代的内部

### 4.1 调度：`scheduler.schedule()`

这是全书第 08 章内容的实现。核心逻辑：

```
1. 遍历 running 队列，为每个序列申请下一步需要的 block
   - 申请失败 → 抢占 running 队列尾部的请求（第 09 章）
2. 在 token 预算内，从 waiting 队列取请求做 prefill
   - 先查前缀缓存，命中的部分不需要计算（第 07 章）
   - 按 chunk 切分（第 10 章）
3. 产出 SchedulerOutput，其中关键字段是
   num_scheduled_tokens: {请求 id → 本步要处理的 token 数}
```

**V1 的关键简化**：调度结果不区分请求处于 prefill 还是 decode 阶段，统一表示为「本步处理多少 token」。prefill 请求的值是一个 chunk，decode 请求的值是 1。这使得混合执行成为唯一的代码路径。

阅读时重点关注：token 预算与序列数预算如何共同作用、抢占的触发条件、前缀命中如何影响需要计算的 token 数。

### 4.2 显存：`kv_cache_manager.py` 与 `block_pool.py`

这是第 06、07 章内容的实现。

| 方法 | 作用 |
|---|---|
| `get_computed_blocks` | 查前缀缓存，返回可复用的块与命中长度 |
| `allocate_slots` | 为本步新增的 token 分配 block |
| `free` | 请求结束时释放，块进入可复用池 |
| `cache_full_blocks` | 把写满的块登记到哈希表，供后续命中 |

`BlockPool` 内部维护三样东西：空闲块的链表、块哈希到块的映射、块的引用计数。这与第 06 章代码中的 `BlockAllocator` 结构一致。

阅读时重点关注：块的三种状态如何转换、LRU 淘汰在哪里触发、块哈希如何构造（是否包含 LoRA id 与多模态内容的哈希）。

### 4.3 执行：`gpu_model_runner.py`

这是第 01 章第 7 步与第 11 章内容的实现。主要工作：

```
1. 按 SchedulerOutput 构造输入张量
   input_ids、positions、slot_mapping、block_tables、seq_lens
2. 选择是否走 CUDA Graph（batch 是否命中已录制的档位）
3. 执行前向
4. 调用 Sampler
5. 返回采样结果
```

这个文件通常是整个引擎中最长、最复杂的一个，因为它要处理所有的特殊情况（多模态输入、LoRA、投机解码、编码器-解码器结构）。第一次阅读时应当只跟主路径，跳过分支。

阅读时重点关注：`slot_mapping` 的构造（对应第 06 章的槽位寻址）、输入张量的增量更新（对应第 08 章的调度开销优化）、CUDA Graph 的分档逻辑（第 11 章）。

---

## 5. 阅读顺序建议

不要从 `main` 开始逐行读。按以下顺序，每一步都能独立形成理解：

| 顺序 | 目标 | 入口 |
|---|---|---|
| 1 | 理解数据结构 | `v1/core/sched/output.py`、`v1/request.py`，看清 Request 与 SchedulerOutput 有哪些字段 |
| 2 | 理解调度 | `scheduler.py` 的 `schedule()`，只看主路径 |
| 3 | 理解显存 | `kv_cache_manager.py` 的四个主要方法 |
| 4 | 理解执行 | `gpu_model_runner.py` 的 `execute_model` |
| 5 | 理解模型结构 | `model_executor/models/llama.py`，对照第 03 章 |
| 6 | 理解并行 | `model_executor/layers/linear.py` 中的列切分与行切分，对照第 15 章 |
| 7 | 理解 attention | `v1/attention/backends/` 的 backend 选择函数，看清所有约束条件 |

第 7 步的 backend 选择函数值得特别关注：它把「什么配置组合有可用的 kernel」这一信息集中在一处，是部署踩坑时最有用的一段代码。

---

## 6. 动手方法

### 6.1 打日志比读代码有效

在 `schedule()` 的末尾打印每步的决策：

```python
logger.info("step: prefill=%s decode=%s free_blocks=%d waiting=%d",
            ...)
```

跑一个小模型，发几个不同长度的请求，观察：

- prefill 与 decode 如何混合
- chunk 如何切分
- 前缀命中时 `num_computed_tokens` 的初值
- block 何时分配、何时释放
- 抢占何时触发

这比静读代码快得多，因为调度逻辑的分支很多，静读难以判断哪条是主路径。

### 6.2 无 GPU 时的做法

本机没有 GPU 时仍然可以：

| 做法 | 说明 |
|---|---|
| 读源码并画调用图 | 不需要运行 |
| 跑单元测试 | vLLM 的部分测试（调度器、block manager 的逻辑测试）不需要 GPU |
| 用 CPU 后端跑小模型 | 速度很慢，但能验证流程 |
| 对照本教材的模拟器 | 第 06、07、08 章的代码实现了相同的逻辑，可以对比理解 |

第 4 项是本教材代码的用途之一：把 vLLM 中被工程细节包裹的核心逻辑单独实现一遍，理解之后再回去读真实代码会容易很多。

### 6.3 找到可以提交的改动

从容易到难：

| 类型 | 说明 |
|---|---|
| 文档 | 补充参数说明、修正过时描述 |
| 测试 | 为已有逻辑补充边界条件的测试 |
| good first issue | 项目标记的入门任务 |
| 小功能 | 新增一个采样参数、支持一个新的模型结构 |
| 性能 | 需要 GPU 环境与 profile 数据 |

一个被合入的 PR 的说服力高于任何自述。在没有 GPU 的条件下，前三类是可行的。

---

## 7. V0 与 V1 的差异

面试中可能被问到，也影响你读到的代码属于哪一代。

| 项 | V0 | V1 |
|---|---|---|
| 调度粒度 | 一步要么全 prefill 要么全 decode | 混合，统一为「每请求本步处理多少 token」 |
| chunked prefill | 可选 | 默认 |
| 前缀缓存 | 可选 | 默认 |
| 抢占 | swap 与 recompute | 只有 recompute |
| 队列 | waiting / running / swapped | waiting / running |
| 进程结构 | 单进程为主 | API 与 EngineCore 分离 |
| 调度与执行 | 同步 | 支持异步重叠 |

变化的共同方向是**减少代码路径的分支**：把可选特性变为默认，把两种模式统一为一种。这降低了维护成本，也让性能优化（异步、CUDA Graph）更容易实施。

---

## 8. 与本教材各章的对照表

| 本教材章节 | vLLM 中的位置 |
|---|---|
| 01 请求生命周期 | `entrypoints/` → `v1/engine/` → `v1/core/` → `v1/worker/` |
| 03 计算结构 | `model_executor/models/llama.py` |
| 05 KV cache | `v1/core/kv_cache_utils.py` 中的容量计算 |
| 06 PagedAttention | `v1/core/block_pool.py` |
| 07 前缀共享 | `kv_cache_manager.get_computed_blocks` |
| 08 连续批处理 | `v1/core/sched/scheduler.py` |
| 09 抢占 | `scheduler.py` 中的抢占分支 |
| 10 chunked prefill | `scheduler.py` 中的 chunk 切分 |
| 11 kernel | `v1/attention/backends/` |
| 12 采样 | `v1/sample/` |
| 13 量化 | `model_executor/layers/quantization/` |
| 14 投机解码 | `v1/spec_decode/` |
| 15 分布式 | `distributed/`、`model_executor/layers/linear.py` |

本章没有配套代码。前面各章的模拟器就是这些模块的简化版本，建议对照阅读。
