#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 09 章 · 抢占：换出与重算的代价对比

三部分：
  1. swap（换出到主机内存）与 recompute（丢弃 KV 重新 prefill）的代价曲线
  2. 抢占对象的选择策略对比
  3. 抢占的连锁效应：一次抢占引发多少额外工作
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch02_roofline as rl        # noqa: E402
import ch05_kvcache as kv         # noqa: E402

M = rl.MODELS["Llama-3-8B"]
HW = rl.HW["A100-80GB"]

PCIE_BW = 64e9          # PCIe Gen4 x16 单向约 64 GB/s
NVLINK_BW = 600e9       # 对照：NVLink


def kv_bytes(seq_len):
    return kv.kv_per_token(M["layers"], M["kv_heads"], M["head_dim"]) * seq_len


def swap_cost(seq_len, bw=PCIE_BW):
    """换出 + 换入，各传一次。"""
    return 2.0 * kv_bytes(seq_len) / bw


def recompute_cost(seq_len, cached_prefix=0):
    """丢弃 KV 后重新 prefill。命中前缀缓存的部分不需要重算。"""
    todo = max(0, seq_len - cached_prefix)
    if todo == 0:
        return 0.0
    return rl.realistic(rl.prefill_step(M, HW, todo, cached_prefix)["t"])


def sep(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def part1():
    sep("1. 换出与重算的代价对比（Llama-3-8B，A100 + PCIe Gen4）")
    print("%10s %12s %12s %12s %12s %10s"
          % ("序列长度", "KV(MB)", "换出+入(ms)", "重算(ms)", "重算/换出", "更优"))
    for s in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        sw = swap_cost(s)
        rc = recompute_cost(s)
        print("%10d %12.1f %12.1f %12.1f %12.2f %10s"
              % (s, kv_bytes(s) / 1e6, sw * 1e3, rc * 1e3, rc / sw,
                 "换出" if sw < rc else "重算"))
    print("\n两者的代价都与序列长度近似成正比，因此比值基本不随长度变化。")
    print("在本配置下换出更快，因为 PCIe 传输 KV 的速度高于重新计算的速度。")
    print("但换出还有两项隐藏代价，见第 2 节。")

    print("\n换个角度：换出的代价与链路带宽直接相关")
    print("%-16s %14s %14s %12s" % ("链路", "带宽(GB/s)", "8K序列(ms)", "对比重算"))
    rc8 = recompute_cost(8192)
    for name, bw in (("PCIe Gen3", 16e9), ("PCIe Gen4", 64e9),
                     ("PCIe Gen5", 128e9), ("NVLink", NVLINK_BW)):
        sw = swap_cost(8192, bw)
        print("%-16s %14.0f %14.1f %12s"
              % (name, bw / 1e9, sw * 1e3, "换出更优" if sw < rc8 else "重算更优"))


def part2():
    sep("2. 前缀缓存改变了结论")
    print("重算时若命中前缀缓存，只需重算未命中的部分。")
    print("%10s %14s %14s %14s %10s"
          % ("序列长度", "命中前缀", "重算(ms)", "换出+入(ms)", "更优"))
    for s, hit in ((8192, 0), (8192, 4096), (8192, 7168), (8192, 8192),
                   (32768, 0), (32768, 16384), (32768, 30720)):
        rc = recompute_cost(s, hit)
        sw = swap_cost(s)
        print("%10d %14d %14.1f %14.1f %10s"
              % (s, hit, rc * 1e3, sw * 1e3, "换出" if sw < rc else "重算"))
    print("\n单看传输与计算的账，换出在多数情况下更快（比值 19-27 倍）。")
    print("但实际系统中重算的代价通常远低于表中数值，原因有两点：")
    print("  1. 被抢占的请求释放 block 后，这些 block 进入可复用池而不是")
    print("     立即清零。若在显存压力缓解前未被覆盖，重新调度时前缀")
    print("     100%% 命中，重算代价为 0（表中最后一种情况）。")
    print("  2. 重算与其他请求的 prefill 合并在同一步，权重只读一遍，")
    print("     边际代价低于表中按独占计算的数值。")
    print("这是 vLLM V1 取消 swap 路径的原因：前缀缓存默认开启后，")
    print("swap 的收益不足以抵消它带来的实现复杂度与主机内存占用。")

    print("\n换出的两项隐藏代价：")
    print("  1. 占用主机内存。换出 100 个 8K 序列需要 %.1f GB 主机内存。"
          % (100 * kv_bytes(8192) / 1e9))
    print("  2. 占用 PCIe 带宽，与权重加载、日志、指标上报争抢；")
    print("     且换入必须在该请求恢复计算之前完成，是同步阻塞的。")


def part3():
    sep("3. 抢占对象的选择")
    rng = random.Random(9)
    seqs = []
    for i in range(40):
        seqs.append(dict(sid=i,
                         cur=rng.randint(200, 8000),
                         remain=rng.randint(10, 800),
                         arrival=rng.random() * 100))
    need = 60      # 需要腾出的 block 数
    bs = 16

    def simulate(name, key, reverse=False):
        pool = sorted(seqs, key=key, reverse=reverse)
        freed, victims = 0, []
        for s in pool:
            if freed >= need:
                break
            freed += s["cur"] // bs
            victims.append(s)
        lost = sum(s["cur"] for s in victims)          # 需要重算的 token
        cost = sum(recompute_cost(s["cur"]) for s in victims)
        return name, len(victims), freed, lost, cost

    print("需要腾出 %d 个 block（%d 个 token 的空间），40 个候选序列\n"
          % (need, need * bs))
    print("%-26s %8s %10s %14s %14s"
          % ("策略", "牺牲数", "腾出block", "需重算token", "重算代价(ms)"))
    rows = [
        simulate("最晚加入优先（LIFO）", lambda s: -s["arrival"]),
        simulate("最长序列优先", lambda s: -s["cur"]),
        simulate("最短序列优先", lambda s: s["cur"]),
        simulate("剩余输出最多优先", lambda s: -s["remain"]),
        simulate("剩余输出最少优先", lambda s: s["remain"]),
    ]
    for name, n, freed, lost, cost in rows:
        print("%-26s %8d %10d %14d %14.1f" % (name, n, freed, lost, cost * 1e3))

    print("\n三种考量互相冲突：")
    print("  牺牲长序列：一次就能腾出足够空间，牺牲数少，但重算代价最高。")
    print("  牺牲短序列：重算便宜，但要牺牲很多个请求，影响面大。")
    print("  牺牲剩余输出最多的：它还要占用显存很久，收益持续时间长，")
    print("    但输出长度不可预知，只能用已生成长度做近似。")
    print("vLLM 采用最晚加入优先，理由是保证先到的请求先完成，避免饥饿，")
    print("且实现简单——running 队列本身就是按加入顺序排列的。")


def part4():
    sep("4. 抢占的连锁效应")
    print("一次抢占的直接代价是重算，间接代价是它会推迟所有其他请求。\n")
    lens = [1000, 4000, 8000, 16000]
    print("%12s %14s %16s %18s"
          % ("被抢占序列", "重算代价(ms)", "等价于几步decode", "batch32时浪费的token"))
    dstep = rl.realistic(rl.decode_step(M, HW, 32, 4096)["t"])
    for s in lens:
        rc = recompute_cost(s)
        steps = rc / dstep
        print("%12d %14.1f %16.1f %18.0f"
              % (s, rc * 1e3, steps, steps * 32))
    print("\n最后一列是这次抢占让整个系统少产出的 token 数（其他 31 个请求")
    print("在这段时间内本来可以继续生成）。抢占一个 16000 token 的序列，")
    print("代价相当于系统少产出上千个 token。")
    print("\n因此抢占应当被视为异常而非常态。第 08 章的准入控制把抢占次数")
    print("从 50 降到 0，goodput 从 40 升到 68，就是这个原因。")
    print("监控中应当把抢占次数作为一级指标，持续大于 0 说明容量不足。")


def main():
    part1()
    part2()
    part3()
    part4()
    print("\n观察建议")
    print("  1. 把 PCIE_BW 改为 NVLINK_BW，看换出的代价如何变化。")
    print("     这对应把 KV 换到同机的另一张卡而不是主机内存。")
    print("  2. 第 2 节说明前缀缓存与抢占策略是耦合的：命中率越高，")
    print("     重算越便宜，换出的价值越低。")


if __name__ == "__main__":
    main()
