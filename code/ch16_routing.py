#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 16 章 · 多实例路由策略的对比

在同一负载上比较六种路由策略，指标为：
  - 前缀缓存命中率（决定 prefill 计算量）
  - 实例间负载的不均衡度
  - 估算的平均 TTFT

负载来自第 07 章的生成器（多 system prompt + 多轮对话）。
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch02_roofline as rl        # noqa: E402
import ch07_prefix_cache as pc    # noqa: E402

M = rl.MODELS["Llama-3-8B"]
HW = rl.HW["A100-80GB"]

N_INSTANCE = 8
CACHE_TOKENS_PER_INSTANCE = 60000
BLOCK = 16


def sep(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


class Instance(object):
    def __init__(self, idx):
        self.idx = idx
        self.cache = pc.HashedBlockCache(CACHE_TOKENS_PER_INSTANCE // BLOCK, BLOCK)
        self.inflight = 0
        self.served = 0
        self.prefill_tokens = 0      # 实际需要计算的 token 数
        self.total_tokens = 0

    def handle(self, prompt, full):
        hit = self.cache.lookup_and_insert(prompt, full)
        self.served += 1
        self.total_tokens += len(prompt)
        self.prefill_tokens += len(prompt) - hit
        return hit


# ---------------------------------------------------------------- 策略


def route_round_robin(state, prompt, insts):
    i = state["rr"] % len(insts)
    state["rr"] += 1
    return insts[i]


def route_random(state, prompt, insts):
    return state["rng"].choice(insts)


def route_least_loaded(state, prompt, insts):
    return min(insts, key=lambda x: x.served)


def route_hash_prefix(state, prompt, insts):
    """按 prompt 的前若干 token 做一致性哈希，保证相同前缀落到同一实例。"""
    key = tuple(prompt[:64])
    return insts[hash(key) % len(insts)]


def route_prefix_aware(state, prompt, insts):
    """查询每个实例的缓存，选命中最长的；命中相同则选负载低的。"""
    best, best_hit = None, -1
    for x in insts:
        h = 0
        prev = 0
        for i in range(0, len(prompt) - BLOCK + 1, BLOCK):
            prev = hash((prev, tuple(prompt[i:i + BLOCK])))
            if prev in x.cache.pool:
                h += BLOCK
            else:
                break
        if h > best_hit or (h == best_hit and x.served < best.served):
            best, best_hit = x, h
    return best


def route_prefix_aware_balanced(state, prompt, insts):
    """前缀感知 + 负载上限：命中最长的实例若已过载则退回负载最低的。"""
    cand = route_prefix_aware(state, prompt, insts)
    avg = sum(x.served for x in insts) / float(len(insts))
    if cand.served > avg * 1.2:
        return min(insts, key=lambda x: x.served)
    return cand


POLICIES = [
    ("轮询", route_round_robin),
    ("随机", route_random),
    ("最少请求数", route_least_loaded),
    ("前缀哈希（前 64 token）", route_hash_prefix),
    ("前缀感知（查缓存）", route_prefix_aware),
    ("前缀感知 + 负载上限", route_prefix_aware_balanced),
]


def run(policy_fn, reqs):
    insts = [Instance(i) for i in range(N_INSTANCE)]
    state = {"rr": 0, "rng": random.Random(0)}
    for prompt, full in reqs:
        inst = policy_fn(state, prompt, insts)
        inst.handle(prompt, full)
    return insts


def summarize(insts):
    total = sum(x.total_tokens for x in insts)
    need = sum(x.prefill_tokens for x in insts)
    served = [x.served for x in insts]
    avg = sum(served) / float(len(served))
    imbalance = (max(served) - min(served)) / avg if avg else 0
    return dict(hit=1.0 - need / float(total), need=need,
                imbalance=imbalance, served=served)


def main():
    global N_INSTANCE
    reqs, sys_prompts, kinds = pc.make_workload(n_req=1200)
    total = sum(len(p) for p, _ in reqs)
    print("负载: %d 个请求，%d 个 prompt token，%d 个实例"
          % (len(reqs), total, N_INSTANCE))
    print("每实例前缀缓存容量 %d token（单实例场景的 %.0f%%）"
          % (CACHE_TOKENS_PER_INSTANCE, 100.0 / N_INSTANCE))

    sep("1. 六种路由策略的对比")
    print("%-26s %12s %14s %14s %12s"
          % ("策略", "前缀命中率", "需prefill(token)", "实例负载不均衡",
             "相对轮询"))
    base = None
    results = {}
    for name, fn in POLICIES:
        r = summarize(run(fn, reqs))
        results[name] = r
        if base is None:
            base = r["need"]
        print("%-26s %11.1f%% %14d %13.2f %11.2fx"
              % (name, 100 * r["hit"], r["need"], r["imbalance"],
                 base / float(r["need"])))
    print("\n不均衡度 =（最忙实例的请求数 - 最闲的）/ 平均值。")
    print("轮询与随机的命中率最低：同一个 system prompt 的请求被打散到")
    print("所有实例，每个实例都要各存一份、各算一次。")

    sep("2. 命中率与实例数的关系（轮询 vs 前缀感知）")
    print("%10s %18s %18s %14s"
          % ("实例数", "轮询命中率", "前缀感知命中率", "计算量之比"))
    orig = N_INSTANCE
    for n in (1, 2, 4, 8, 16, 32):
        N_INSTANCE = n
        a = summarize(run(route_round_robin, reqs))
        b = summarize(run(route_prefix_aware, reqs))
        print("%10d %17.1f%% %17.1f%% %13.2fx"
              % (n, 100 * a["hit"], 100 * b["hit"],
                 a["need"] / float(b["need"])))
    N_INSTANCE = orig
    print("\n实例数越多，轮询的命中率下降越快（前缀被打散的程度与实例数成正比），")
    print("而前缀感知路由基本不受影响。因此**集群规模越大，路由策略越重要**。")

    sep("3. 前缀命中对 TTFT 的影响")
    print("以一个 2048 token 的 prompt 为例，命中不同比例时的 prefill 耗时：\n")
    print("%12s %14s %14s %12s" % ("命中比例", "需计算token", "prefill(ms)", "相对未命中"))
    full_t = rl.realistic(rl.prefill_step(M, HW, 2048)["t"])
    for frac in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        hit = int(2048 * frac)
        todo = 2048 - hit
        t = rl.realistic(rl.prefill_step(M, HW, todo, hit)["t"]) if todo else 0.0
        print("%11.0f%% %14d %14.1f %11.2fx"
              % (100 * frac, todo, t * 1e3,
                 full_t / t if t > 0 else float("inf")))
    print("\n命中 90% 时 prefill 耗时降到约五分之一。前缀感知路由的收益")
    print("直接体现在 TTFT 上，是服务层最有效的单项优化。")

    sep("4. 前缀感知路由的实现代价")
    print("上面的「查缓存」实现需要路由器知道每个实例缓存了什么，三种做法：\n")
    print("  1. 一致性哈希：按 prompt 前 N 个 token 哈希。")
    print("     无需状态，但无法处理「前缀相同长度不同」的情况，")
    print("     且实例增减时哈希环变动会导致缓存大面积失效。")
    print("  2. 路由器维护影子索引：记录每个实例缓存过哪些块哈希。")
    print("     命中率最高，代价是路由器要跟踪状态，且与实例的实际淘汰")
    print("     行为可能不一致（影子索引认为命中，实例上已被淘汰）。")
    print("  3. 实例上报：各实例周期性上报自己的缓存摘要（如布隆过滤器）。")
    print("     折中方案，有延迟但状态量小。")
    print("\n第 1 种的实测（本节表 1 中的「前缀哈希」行）已能取得接近")
    print("查缓存的效果，且实现最简单，是多数生产系统的起点。")

    sep("5. 负载均衡与前缀亲和的冲突")
    r_aware = summarize(run(route_prefix_aware, reqs))
    r_bal = summarize(run(route_prefix_aware_balanced, reqs))
    print("纯前缀感知     : 命中率 %.1f%%，不均衡度 %.2f"
          % (100 * r_aware["hit"], r_aware["imbalance"]))
    print("加负载上限     : 命中率 %.1f%%，不均衡度 %.2f"
          % (100 * r_bal["hit"], r_bal["imbalance"]))
    print("各实例请求数（纯前缀感知）: %s" % r_aware["served"])
    print("各实例请求数（加负载上限）: %s" % r_bal["served"])
    print("\n纯前缀感知只用了 4 个实例，其余 4 个完全空闲——因为负载中只有")
    print("4 个 system prompt，每个被固定到一个实例上。这在真实系统中")
    print("意味着一半的卡在空转。")
    print("加负载上限后 8 个实例均分请求，而命中率没有下降：热点前缀被")
    print("复制到多个实例上，每个都缓存一份，总缓存容量足以容纳。")
    print("命中率与均衡度并非总能兼得——缓存容量紧张时复制会导致淘汰增多。")
    print("生产实现通常是「命中长度 - 负载惩罚」的加权打分，而不是硬阈值。")


if __name__ == "__main__":
    main()
