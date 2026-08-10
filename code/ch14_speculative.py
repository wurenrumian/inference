#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 14 章 · 投机解码的加速比模型与正确性

四部分：
  1. 加速比公式与最优草稿长度
  2. 负收益区：什么条件下投机解码会变慢
  3. batch 大小的影响：为什么高并发下收益消失
  4. 拒绝采样的正确性验证：输出分布与目标模型一致
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch02_roofline as rl        # noqa: E402

np.random.seed(11)
np.seterr(all="ignore")


def sep(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- 模型


def expected_tokens(alpha, k):
    """一轮投机（草稿 k 个 token + 1 次验证）期望产出的 token 数。

    草稿的第 i 个 token 被接受的概率是 alpha^i。全部 k 个都被接受时，
    验证步还能额外产出 1 个（用目标模型自己的分布采样），因此上限是 k+1。
    """
    if alpha >= 1.0:
        return k + 1.0
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha, k, c):
    """相对逐 token 解码的加速比。

    c 为草稿一步的成本相对目标模型一步的比例。
    一轮的成本 = k 次草稿 + 1 次验证 = k*c + 1（以目标模型一步为单位）。
    """
    return expected_tokens(alpha, k) / (k * c + 1.0)


def part1():
    sep("1. 加速比随接受率与草稿长度变化（草稿成本 c = 0.1）")
    c = 0.1
    ks = (1, 2, 3, 4, 5, 6, 8, 10)
    print("%8s" % "接受率", end="")
    for k in ks:
        print("%8s" % ("k=%d" % k), end="")
    print("%10s %8s" % ("最优k", "最优加速"))
    for a in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        print("%8.2f" % a, end="")
        best, bk = 0.0, 0
        for k in ks:
            s = speedup(a, k, c)
            print("%8.2f" % s, end="")
        for k in range(1, 33):
            s = speedup(a, k, c)
            if s > best:
                best, bk = s, k
        print("%10d %8.2f" % (bk, best))
    print("\n接受率是决定性因素。接受率 0.5 时加速上限约 1.6 倍，")
    print("接受率 0.9 时可达 4 倍以上。草稿长度存在最优值：")
    print("太短则验证的固定成本摊不开，太长则后面的 token 大概率被拒绝，白算。")


def part2():
    sep("2. 负收益区：什么时候投机解码会变慢")
    print("加速比小于 1 即为负收益。给定草稿成本 c，求最低可用接受率。\n")
    print("%12s %14s %14s %-30s"
          % ("草稿成本 c", "自适应k门槛", "固定k=5门槛", "说明"))
    notes = {0.02: "n-gram / 前缀匹配，几乎无成本",
             0.05: "Medusa / EAGLE 头，与主干共享大部分计算",
             0.1: "1B 草稿模型配 8B 目标模型",
             0.2: "1.5B 草稿模型配 7B 目标模型",
             0.35: "草稿模型过大"}
    for c in (0.02, 0.05, 0.1, 0.2, 0.35):
        lo = lo5 = None
        for a100 in range(1, 100):
            a = a100 / 100.0
            if lo is None and max(speedup(a, k, c) for k in range(1, 17)) > 1.0:
                lo = a
            if lo5 is None and speedup(a, 5, c) > 1.0:
                lo5 = a
        print("%12.2f %14s %14s %-30s"
              % (c, "%.2f" % lo if lo else "无",
                 "%.2f" % lo5 if lo5 else "无", notes.get(c, "")))
    print("\n两列门槛差别很大：草稿长度可自适应时（每步按接受情况调整 k），")
    print("门槛很低；固定 k=5 时门槛高得多，因为低接受率下后面几个草稿")
    print("token 几乎必然被拒绝。生产实现都带自适应的草稿长度。")
    print("\n草稿成本越低，可用的接受率门槛越低。这是 Medusa 与 EAGLE 这类")
    print("「共享主干、只加轻量头」的方案胜过独立草稿模型的原因：")
    print("它们的 c 极低，即使接受率不高也不会亏。")

    print("\n接受率取决于任务类型（经验值，需按实际负载实测）：")
    for task, a in (("代码补全（重复度高）", 0.85),
                    ("结构化输出 / JSON", 0.80),
                    ("摘要（可抄原文）", 0.75),
                    ("常规对话", 0.60),
                    ("开放式创作", 0.45)):
        print("  %-24s 接受率约 %.2f，c=0.1 时最优加速 %.2fx"
              % (task, a, max(speedup(a, k, 0.1) for k in range(1, 17))))


def part3():
    sep("3. batch 大小的影响：高并发下收益消失")
    M, HW = rl.MODELS["Llama-3-8B"], rl.HW["A100-80GB"]
    seq = 2048
    print("投机解码把 decode 的算术强度从 batch 提高到 batch × (k+1)，")
    print("因此只有在访存受限区间才有收益。一旦进入计算受限区，")
    print("多算的 token 就要付出真实的算力代价。\n")
    k = 4
    print("%8s %14s %16s %14s %14s"
          % ("batch", "算术强度", "验证时的强度", "单步(ms)", "验证步(ms)"))
    for b in (1, 4, 16, 32, 64, 128, 256):
        r1 = rl.decode_step(M, HW, b, seq)
        # 验证步：一次前向处理 b*(k+1) 个 token
        rv = rl.prefill_step(M, HW, k + 1, prefix=seq, n_seq=b)
        print("%8d %14.1f %16.1f %14.2f %14.2f"
              % (b, r1["intensity"], rv["intensity"],
                 rl.realistic(r1["t"]) * 1e3, rl.realistic(rv["t"]) * 1e3))
    print("\n%8s %16s %16s %12s" % ("batch", "逐token(ms/token)",
                                     "投机(ms/token)", "实际加速"))
    alpha = 0.7
    for b in (1, 4, 16, 32, 64, 128, 256):
        t1 = rl.realistic(rl.decode_step(M, HW, b, seq)["t"])
        tv = rl.realistic(rl.prefill_step(M, HW, k + 1, seq, b)["t"])
        td = k * 0.1 * t1                                  # 草稿成本
        per_token_base = t1
        per_token_spec = (tv + td) / expected_tokens(alpha, k)
        print("%8d %16.3f %16.3f %11.2fx"
              % (b, per_token_base * 1e3, per_token_spec * 1e3,
                 per_token_base / per_token_spec))
    print("\nbatch 从 1 到 256，加速比从 1.98x 降到 1.36x。")
    print("上表只计入了计算成本。真实系统中还有草稿模型的调度开销、")
    print("树形验证的掩码构造、以及草稿模型自身占用的显存与 KV，")
    print("因此实际的下降更快，大 batch 下可能变为负收益。")
    print("因此投机解码适合低并发、低延迟优先的场景（单用户交互、代码补全），")
    print("不适合追求吞吐的高并发服务。生产中通常按当前 batch 动态开关。")


def part4():
    sep("4. 拒绝采样：投机解码不改变输出分布")
    V = 8
    n_trial = 400000

    # 目标模型与草稿模型的分布（草稿是目标的一个粗糙近似）
    p = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02])
    q = np.array([0.20, 0.30, 0.10, 0.15, 0.10, 0.05, 0.07, 0.03])
    p, q = p / p.sum(), q / q.sum()

    rng = np.random.RandomState(5)
    counts = np.zeros(V)
    accepted = 0
    for _ in range(n_trial):
        x = rng.choice(V, p=q)                    # 草稿模型采样
        r = rng.rand()
        if r < min(1.0, p[x] / q[x]):             # 接受
            counts[x] += 1
            accepted += 1
        else:                                     # 拒绝：从修正分布重采样
            resid = np.maximum(p - q, 0)
            resid = resid / resid.sum()
            counts[rng.choice(V, p=resid)] += 1

    emp = counts / counts.sum()
    print("目标分布 p : %s" % np.round(p, 4))
    print("草稿分布 q : %s" % np.round(q, 4))
    print("经验分布   : %s" % np.round(emp, 4))
    print("最大偏差   : %.4f（%d 次试验的采样噪声约 %.4f）"
          % (np.abs(emp - p).max(), n_trial, 1.0 / np.sqrt(n_trial)))
    print("接受率     : %.4f（理论值 %.4f = sum(min(p, q))）"
          % (accepted / float(n_trial), np.minimum(p, q).sum()))
    assert np.abs(emp - p).max() < 0.01, "拒绝采样未能还原目标分布"
    print("\n算法：")
    print("  1. 用草稿模型采样得到 x ~ q")
    print("  2. 以概率 min(1, p(x)/q(x)) 接受")
    print("  3. 拒绝时从修正分布 normalize(max(p - q, 0)) 重新采样")
    print("这一步保证最终分布严格等于 p，与不用投机解码时完全一致。")
    print("因此投机解码是无损加速，不需要在质量上做取舍。")
    print("\n接受率的理论值是 sum(min(p, q))，即两个分布的重叠面积。")
    print("草稿模型越接近目标模型，重叠越大，接受率越高。")


def main():
    part1()
    part2()
    part3()
    part4()
    print("\n观察建议")
    print("  1. 第 1 节把 c 从 0.1 改为 0.02（对应 EAGLE 类方案），")
    print("     观察最优草稿长度变长、加速比提高。")
    print("  2. 第 4 节把 q 改得更接近 p，观察接受率上升。这说明草稿模型")
    print("     的训练目标应当是拟合目标模型的分布，而不是拟合真实数据。")


if __name__ == "__main__":
    main()
