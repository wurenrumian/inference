#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 11 章 · FlashAttention 与 PagedAttention 的算法实现

四部分：
  1. 朴素 attention：实体化 N×N 注意力矩阵
  2. 分块 attention（FlashAttention 的核心）：online softmax，不实体化
  3. 两者的数值一致性与显存占用对比
  4. PagedAttention 的 decode kernel：按 block table 间接寻址

只用 numpy，目的是把算法讲清楚，不涉及 CUDA 实现。
"""

import numpy as np

np.random.seed(0)
# numpy 2.0 在 macOS 的 BLAS 后端上对 matmul 会发出与本例无关的浮点警告，
# 计算结果不受影响（第 1、4 节有数值一致性断言）。
np.seterr(all="ignore")


def sep(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


# ---------------------------------------------------------------- 1


def naive_attention(q, k, v, causal=True):
    """朴素实现：显式构造 [N, N] 的注意力矩阵。

    q: [N, d]   k: [M, d]   v: [M, d]
    """
    n, d = q.shape
    m = k.shape[0]
    scores = q @ k.T / np.sqrt(d)              # [N, M] ← 峰值显存在这里
    if causal:
        mask = np.triu(np.ones((n, m), dtype=bool), k=m - n + 1)
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=-1, keepdims=True)
    return p @ v, scores.nbytes


# ---------------------------------------------------------------- 2


def flash_attention(q, k, v, block_q=64, block_kv=64, causal=True):
    """分块实现：online softmax，从不构造完整的 [N, M] 矩阵。

    对每个 query 块，遍历 kv 块，用运行中的最大值与归一化因子增量更新
    输出。峰值额外显存只有一个 [block_q, block_kv] 的小块。
    """
    n, d = q.shape
    m = k.shape[0]
    out = np.zeros((n, d), dtype=np.float64)
    peak = 0

    for i0 in range(0, n, block_q):
        i1 = min(i0 + block_q, n)
        qi = q[i0:i1]                            # [bq, d]
        # 运行状态：当前最大值 mi、指数和 li、累加输出 oi
        mi = np.full((i1 - i0, 1), -np.inf)
        li = np.zeros((i1 - i0, 1))
        oi = np.zeros((i1 - i0, d))

        for j0 in range(0, m, block_kv):
            j1 = min(j0 + block_kv, m)
            kj, vj = k[j0:j1], v[j0:j1]
            s = qi @ kj.T / np.sqrt(d)           # [bq, bkv] ← 唯一的中间矩阵
            peak = max(peak, s.nbytes)
            if causal:
                # query 的绝对位置 i0+a 只能看到 kv 位置 <= i0+a+(m-n)
                rows = np.arange(i0, i1)[:, None] + (m - n)
                cols = np.arange(j0, j1)[None, :]
                s = np.where(cols > rows, -np.inf, s)
                if np.all(np.isneginf(s)):
                    continue                     # 整块被掩码，可直接跳过

            m_new = np.maximum(mi, s.max(axis=-1, keepdims=True))
            # 用新的最大值重新缩放已累计的结果
            alpha = np.exp(mi - m_new)
            p = np.exp(s - m_new)
            li = alpha * li + p.sum(axis=-1, keepdims=True)
            oi = alpha * oi + p @ vj
            mi = m_new

        out[i0:i1] = oi / li
    return out, peak


# ---------------------------------------------------------------- 3


def part1_and_2():
    sep("1. 数值一致性：分块实现与朴素实现的结果相同")
    for n, d in ((128, 64), (512, 64), (1024, 128)):
        q = np.random.randn(n, d)
        k = np.random.randn(n, d)
        v = np.random.randn(n, d)
        a, mem_a = naive_attention(q, k, v)
        b, mem_b = flash_attention(q, k, v)
        err = np.abs(a - b).max()
        print("  N=%5d d=%3d  最大绝对误差 %.3e  %s"
              % (n, d, err, "一致" if err < 1e-10 else "不一致"))
        assert err < 1e-10, "分块实现与朴素实现结果不一致"
    print("\n分块计算的结果与一次性计算完全相同（误差在浮点精度内）。")
    print("FlashAttention 不是近似算法，它省的是显存与访存，不是精度。")

    sep("2. 中间矩阵的显存占用")
    print("%10s %10s %18s %18s %10s"
          % ("N", "d", "朴素中间矩阵", "分块中间矩阵", "比值"))
    for n in (512, 2048, 8192, 32768, 131072):
        d = 128
        naive_bytes = n * n * 2                  # fp16 的 [N, N]
        flash_bytes = 64 * 64 * 2                # 一个 [64, 64] 小块
        print("%10d %10d %15.1f MB %15.1f KB %9.0fx"
              % (n, d, naive_bytes / 1e6, flash_bytes / 1e3,
                 naive_bytes / float(flash_bytes)))
    print("\n朴素实现的中间矩阵与序列长度的平方成正比：32K 序列需要 2 GB，")
    print("128K 序列需要 34 GB，单是这一个中间张量就装不下。")
    print("分块实现的中间矩阵大小固定，与序列长度无关。")
    print("这是长上下文推理必须用 FlashAttention 的原因。")

    sep("3. 访存量的减少")
    print("朴素实现要把 [N, N] 矩阵写回显存再读出来做 softmax，")
    print("分块实现的中间块留在片上（SRAM），不落显存。\n")
    print("%10s %20s %20s %10s"
          % ("N", "朴素访存(MB)", "分块访存(MB)", "比值"))
    d, nh = 128, 32
    for n in (512, 2048, 8192, 32768):
        # 朴素：QKV 各读一遍 + 分数矩阵写一遍读两遍（softmax 前后）
        naive = (3 * n * d + 3 * n * n) * 2 / 1e6
        # 分块：QKV 各读一遍（KV 可能被多个 q 块重复读，按 N/block_q 次算）
        flash = (n * d + 2 * n * d * max(1, n // 4096) + n * d) * 2 / 1e6
        print("%10d %20.1f %20.1f %9.1fx" % (n, naive, flash, naive / flash))
    print("\n访存量的差距随 N 增大而扩大，这是 FlashAttention 提速的来源。")
    print("注意分块实现的计算量并没有减少，甚至因为重复读 KV 略有增加。")


# ---------------------------------------------------------------- 4


def paged_decode_attention(q, kv_cache, block_table, seq_len, block_size):
    """PagedAttention 的 decode 路径：单个 query 对全部历史 KV。

    q          : [d]            当前 token 的 query
    kv_cache   : [num_blocks, block_size, 2, d]   全局 KV 池
    block_table: 逻辑块 → 物理块
    seq_len    : 该序列当前的长度
    """
    d = q.shape[0]
    scores = np.empty(seq_len)
    vals = np.empty((seq_len, d))
    for pos in range(seq_len):
        logical = pos // block_size
        offset = pos % block_size
        phys = block_table[logical]              # 间接寻址：查表
        kk = kv_cache[phys, offset, 0]
        vv = kv_cache[phys, offset, 1]
        scores[pos] = q @ kk / np.sqrt(d)
        vals[pos] = vv
    scores -= scores.max()
    p = np.exp(scores)
    p /= p.sum()
    return p @ vals


def part4():
    sep("4. PagedAttention 的 decode kernel")
    d, block_size, num_blocks = 64, 16, 32
    seq_len = 100

    # 构造一个物理上分散的 KV cache
    kv_cache = np.zeros((num_blocks, block_size, 2, d))
    rng = np.random.RandomState(3)
    n_blk = (seq_len + block_size - 1) // block_size
    block_table = [int(x) for x in rng.permutation(num_blocks)[:n_blk]]
    print("序列长度 %d，block_size %d，需要 %d 个块"
          % (seq_len, block_size, len(block_table)))
    print("block table（逻辑块 → 物理块）: %s" % block_table)
    print("物理块在池中不连续。\n")

    # 同时构造一份逻辑上连续的 KV，用于对照
    k_flat = rng.randn(seq_len, d)
    v_flat = rng.randn(seq_len, d)
    for pos in range(seq_len):
        phys = block_table[pos // block_size]
        kv_cache[phys, pos % block_size, 0] = k_flat[pos]
        kv_cache[phys, pos % block_size, 1] = v_flat[pos]

    q = rng.randn(d)
    paged = paged_decode_attention(q, kv_cache, block_table, seq_len, block_size)

    # 对照：连续布局的标准实现
    s = k_flat @ q / np.sqrt(d)
    s -= s.max()
    p = np.exp(s)
    p /= p.sum()
    ref = p @ v_flat

    err = np.abs(paged - ref).max()
    print("分页实现与连续实现的最大绝对误差: %.3e  %s"
          % (err, "一致" if err < 1e-12 else "不一致"))
    assert err < 1e-12, "分页寻址实现有误"
    print("\n分页只改变数据的物理位置与寻址方式，不改变计算结果。")
    print("kernel 的改动是：把「按步长计算地址」换成「查 block table 得到")
    print("物理块号，再算块内偏移」。多出的开销是一次间接访问，")
    print("以及每 %d 个 token 一次地址跳跃带来的访存连续性下降。" % block_size)


# ---------------------------------------------------------------- 5


def part5():
    sep("5. 算子融合与 CUDA Graph 的收益估算")
    print("融合：省掉中间张量的写回与读出。以第 03 章的统计为例，")
    print("prefill 场景下逐元素算子占访存量的 56%，融合后可消掉大部分。\n")
    layers, hidden, tokens = 32, 4096, 8192
    unfused = 0
    # RMSNorm ×2、SiLU 与乘、残差 ×2 的中间张量读写
    unfused += 2 * 2 * tokens * hidden * 2 * layers      # RMSNorm
    unfused += 3 * tokens * 14336 * 2 * layers           # SiLU 与乘
    unfused += 3 * 2 * tokens * hidden * 2 * layers      # 残差
    print("  未融合时这些算子的访存量: %.1f GB" % (unfused / 1e9))
    print("  融合后（中间结果留在寄存器与共享内存）: 接近 0")
    print("  在 2 TB/s 带宽上对应 %.1f ms" % (unfused / 2.0e12 * 1e3))

    print("\nCUDA Graph：省掉 kernel 启动开销。")
    print("%12s %14s %16s %14s"
          % ("每层kernel数", "总启动次数", "启动开销(ms)", "占decode步比例"))
    step_ms = 19.6
    for per_layer in (5, 10, 20):
        n = per_layer * layers
        overhead = n * 7e-3        # 每次启动约 7 微秒
        print("%12d %14d %16.2f %13.1f%%"
              % (per_layer, n, overhead, 100 * overhead / step_ms))
    print("\nCUDA Graph 把整个序列录制为一个图，一次提交，启动开销降到接近 0。")
    print("代价是图内的张量形状必须固定，因此引擎要按 batch 大小分档录制")
    print("多张图（例如 1, 2, 4, 8, ..., 256），运行时向上取整到最近的档位，")
    print("多出的部分用 padding 填充。这也是 max_num_seqs 需要是常见档位的原因。")


def main():
    part1_and_2()
    part4()
    part5()
    print("\n观察建议")
    print("  1. 修改 flash_attention 的 block_q 与 block_kv，验证结果不变。")
    print("     真实 kernel 中这两个值由 SRAM 容量决定。")
    print("  2. 第 2 节的表说明：不用 FlashAttention 时，32K 以上的上下文")
    print("     在单卡上无法实现，与 KV cache 的容量无关。")


if __name__ == "__main__":
    main()
