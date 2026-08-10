#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 19 章 · 压测方法论

四部分：
  1. 闭环压测与开环压测的差别（压测中最常见的错误来源）
  2. 平均值掩盖长尾
  3. goodput 随并发的变化：找最优工作点
  4. 报告字段清单与自检
"""

import os
import random
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch02_roofline as rl        # noqa: E402

M = rl.MODELS["Llama-3-8B"]
HW = rl.HW["A100-80GB"]

BLOCK = 16
TOTAL_BLOCKS = 6000
MAX_SEQS = 128
TOKEN_BUDGET = 8192

SLO_TTFT = 2.0
SLO_TPOT = 0.05


def sep(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


class R(object):
    __slots__ = ("rid", "arrival", "plen", "olen", "computed", "generated",
                 "blocks", "t_first", "t_done", "itls", "last")

    def __init__(self, rid, arrival, plen, olen):
        self.rid, self.arrival, self.plen, self.olen = rid, arrival, plen, olen
        self.computed = self.generated = self.blocks = 0
        self.t_first = self.t_done = self.last = None
        self.itls = []

    def need(self, extra):
        return (self.computed + self.generated + extra + BLOCK - 1) // BLOCK


def gen_lengths(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = min(8192, max(32, int(rng.lognormvariate(6.4, 0.9))))
        o = min(2048, max(8, int(rng.expovariate(1.0 / 220))))
        out.append((p, o))
    return out


class Sim(object):
    """支持开环（按到达时刻）与闭环（固定并发）两种压测模式。"""

    def __init__(self, mode, param, n_req=400, seed=5):
        self.mode = mode          # "open" 或 "closed"
        self.param = param        # open: 到达率；closed: 并发数
        self.lens = gen_lengths(n_req, seed)
        self.n_req = n_req
        self.next_id = 0
        self.waiting = deque()
        self.running = []
        self.done = []
        self.free = TOTAL_BLOCKS
        self.now = 0.0
        self.preempted = 0
        rng = random.Random(seed + 1)
        if mode == "open":
            t = 0.0
            self.arrivals = []
            for i in range(n_req):
                t += rng.expovariate(param)
                self.arrivals.append(t)
            self.pending = deque(range(n_req))
        else:
            self.pending = deque(range(n_req))

    def spawn(self, i):
        p, o = self.lens[i]
        at = self.arrivals[i] if self.mode == "open" else self.now
        return R(i, at, p, o)

    def admit(self):
        if self.mode == "open":
            while (self.pending
                   and self.arrivals[self.pending[0]] <= self.now):
                self.waiting.append(self.spawn(self.pending.popleft()))
        else:
            # 闭环：始终维持 param 个在途请求
            inflight = len(self.waiting) + len(self.running)
            while self.pending and inflight < self.param:
                self.waiting.append(self.spawn(self.pending.popleft()))
                inflight += 1

    def alloc(self, r, want):
        d = want - r.blocks
        if d <= 0:
            return True
        if d > self.free:
            return False
        self.free -= d
        r.blocks = want
        return True

    def step(self):
        self.admit()
        budget = TOKEN_BUDGET
        prefill = []
        while self.waiting and len(self.running) + len(prefill) < MAX_SEQS:
            r = self.waiting[0]
            chunk = min(r.plen - r.computed, budget, 1024)
            if chunk <= 0:
                break
            if not self.alloc(r, r.need(chunk)):
                break
            self.waiting.popleft()
            prefill.append((r, chunk))
            budget -= chunk
            if budget <= 0:
                break

        while self.running:
            need = sum(max(0, x.need(1) - x.blocks) for x in self.running)
            if need <= self.free:
                break
            v = self.running.pop()
            self.free += v.blocks
            v.blocks = 0
            v.computed = v.generated = 0
            v.t_first = v.last = None
            self.waiting.appendleft(v)
            self.preempted += 1

        decode = [x for x in self.running if self.alloc(x, x.need(1))]

        if not prefill and not decode:
            if self.mode == "open" and self.pending:
                self.now = self.arrivals[self.pending[0]]
                return True
            if self.mode == "closed" and self.pending:
                return True
            return False

        dt = 0.0
        if decode:
            avg = sum(x.computed + x.generated for x in decode) / len(decode)
            dt += rl.realistic(rl.decode_step(M, HW, len(decode), avg)["t"])
        if prefill:
            tok = sum(c for _, c in prefill)
            pre = sum(r.computed for r, _ in prefill) / len(prefill)
            dt += rl.realistic(rl.prefill_step(
                M, HW, tok // len(prefill), int(pre), len(prefill))["t"])
        self.now += dt

        for r in decode:
            r.generated += 1
            r.itls.append(self.now - r.last)
            r.last = self.now
        for r, c in prefill:
            r.computed += c
            if r.computed >= r.plen:
                r.generated = 1
                r.t_first = r.last = self.now
                self.running.append(r)
            else:
                self.waiting.appendleft(r)

        for r in [x for x in self.running if x.generated >= x.olen]:
            r.t_done = self.now
            self.free += r.blocks
            r.blocks = 0
            self.running.remove(r)
            self.done.append(r)
        return True

    def run(self):
        g = 0
        while self.step():
            g += 1
            if g > 3000000:
                break
        return self


def stats(s):
    d = s.done
    ttft = [r.t_first - r.arrival for r in d]
    itl = []
    for r in d:
        itl.extend(r.itls)
    tok = sum(r.generated for r in d)
    good = 0
    for r in d:
        a = sum(r.itls) / len(r.itls) if r.itls else 0
        if (r.t_first - r.arrival) <= SLO_TTFT and a <= SLO_TPOT:
            good += r.generated
    return dict(n=len(d), span=s.now, thr=tok / s.now if s.now else 0,
                ttft_avg=sum(ttft) / len(ttft) if ttft else 0,
                ttft_p50=pct(ttft, .5), ttft_p95=pct(ttft, .95),
                ttft_p99=pct(ttft, .99),
                itl_avg=sum(itl) / len(itl) if itl else 0,
                itl_p50=pct(itl, .5), itl_p99=pct(itl, .99),
                good=good / s.now if s.now else 0, prem=s.preempted)


# ---------------------------------------------------------------- 1


def part1():
    sep("1. 闭环压测与开环压测")
    print("闭环：固定 N 个在途请求，一个完成立刻补一个（多数压测工具的默认）")
    print("开环：按固定到达率发送，不管系统是否处理得过来（真实流量的形态）\n")

    print("%-10s %10s %10s %12s %12s %12s %10s"
          % ("模式", "参数", "吞吐", "TTFT P50", "TTFT P99", "ITL P99", "goodput"))
    for n in (8, 32, 64, 128):
        r = stats(Sim("closed", n).run())
        print("%-10s %10d %10.0f %12.2f %12.2f %12.1f %10.0f"
              % ("闭环", n, r["thr"], r["ttft_p50"], r["ttft_p99"],
                 r["itl_p99"] * 1e3, r["good"]))
    for rate in (4, 8, 16, 32):
        r = stats(Sim("open", float(rate)).run())
        print("%-10s %10d %10.0f %12.2f %12.2f %12.1f %10.0f"
              % ("开环", rate, r["thr"], r["ttft_p50"], r["ttft_p99"],
                 r["itl_p99"] * 1e3, r["good"]))

    print("\n闭环压测的 TTFT 不会随并发无限增长：请求完成后才发下一个，")
    print("系统自动限流，队列长度恒等于 0。这掩盖了过载时的排队延迟。")
    print("开环压测在到达率超过系统能力时，队列会持续增长，TTFT 随之爆炸。")
    print("\n结论：")
    print("  测「系统的最大吞吐」——闭环可用，逐步加并发直到吞吐不再增长。")
    print("  测「在给定流量下能否满足 SLO」——必须用开环，因为真实用户")
    print("  不会等你处理完上一个才发下一个。")
    print("  只做闭环压测会得出「系统能扛住 N 并发」的乐观结论。")


# ---------------------------------------------------------------- 2


def part2():
    sep("2. 平均值掩盖长尾")
    r = stats(Sim("open", 12.0).run())
    print("同一次压测的 TTFT：")
    print("  平均值 : %.3f s" % r["ttft_avg"])
    print("  P50    : %.3f s" % r["ttft_p50"])
    print("  P95    : %.3f s" % r["ttft_p95"])
    print("  P99    : %.3f s" % r["ttft_p99"])
    print("  P99/P50: %.1f 倍" % (r["ttft_p99"] / max(r["ttft_p50"], 1e-9)))
    print("\nITL：")
    print("  平均值 : %.1f ms" % (r["itl_avg"] * 1e3))
    print("  P50    : %.1f ms" % (r["itl_p50"] * 1e3))
    print("  P99    : %.1f ms" % (r["itl_p99"] * 1e3))
    print("  P99/P50: %.1f 倍" % (r["itl_p99"] / max(r["itl_p50"], 1e-9)))
    print("\n抢占次数: %d" % r["prem"])
    print("\nP99 与 P50 相差数倍是推理服务的常态，来源是长请求、prefill 干扰")
    print("与抢占。只报平均值的压测结果无法用于 SLO 判断。")
    print("报告中至少需要 P50、P95、P99 三个分位数。")


# ---------------------------------------------------------------- 3


def part3():
    sep("3. 找最优工作点：goodput 随负载的变化")
    print("%10s %10s %12s %12s %12s %12s %8s"
          % ("到达率", "吞吐", "TTFT P95", "ITL P99", "goodput",
             "goodput占比", "抢占"))
    for rate in (2, 4, 6, 8, 10, 12, 16, 20, 30):
        r = stats(Sim("open", float(rate)).run())
        frac = 100.0 * r["good"] / r["thr"] if r["thr"] else 0
        print("%10d %10.0f %12.2f %12.1f %12.0f %11.0f%% %8d"
              % (rate, r["thr"], r["ttft_p95"], r["itl_p99"] * 1e3,
                 r["good"], frac, r["prem"]))
    print("\n吞吐随到达率上升到饱和后不再增长，而 goodput 会在某个点之后")
    print("下降——超出能力的请求虽然最终完成了，但已经违反 SLO，不计入")
    print("有效产出。这个拐点就是容量规划应当使用的工作点，而不是最大吞吐点。")


# ---------------------------------------------------------------- 4


def part4():
    sep("4. 压测报告的字段清单")
    fields = [
        ("环境", ["GPU 型号与数量", "驱动与 CUDA 版本", "引擎名称与版本",
                "并行配置 TP/PP", "宿主机 CPU 与内存"]),
        ("模型", ["模型名称与版本", "权重精度", "KV cache 精度",
                "量化方案与校准数据", "上下文长度上限"]),
        ("引擎配置", ["gpu_memory_utilization", "max_num_seqs",
                  "max_num_batched_tokens", "block_size",
                  "是否启用前缀缓存", "是否启用 chunked prefill 及 chunk 大小",
                  "是否启用 CUDA Graph", "是否启用投机解码及配置"]),
        ("负载", ["数据来源（真实日志 / 公开数据集 / 合成）",
                "输入长度 P50/P95/最大", "输出长度 P50/P95/最大",
                "压测模式（开环到达率 / 闭环并发数）", "总请求数与持续时间",
                "预热请求数"]),
        ("结果", ["输出吞吐 token/s", "总吞吐 token/s", "请求吞吐 req/s",
                "TTFT P50/P95/P99", "TPOT 或 ITL P50/P95/P99",
                "端到端延迟 P50/P95/P99", "goodput 及其 SLO 定义",
                "错误率", "前缀缓存命中率", "抢占次数", "峰值并发序列数"]),
    ]
    for cat, items in fields:
        print("\n%s：" % cat)
        for it in items:
            print("  - %s" % it)
    print("\n缺少任何一项，结果都无法与其他人的数据比较。")
    print("其中最常被遗漏的是：压测模式（开环还是闭环）、预热、")
    print("前缀缓存是否开启、以及负载的长度分布。")

    sep("5. 结果异常时的排查顺序")
    steps = [
        ("吞吐远低于估算", "对照第 02 章的 roofline 估算。差 3 倍以上说明"
                      "存在配置问题，先查 batch 是否上得去（看峰值并发序列数）"),
        ("并发上不去", "查 KV 池容量（第 05 章）与 max_num_seqs 配置；"
                  "看是否有抢占"),
        ("TTFT 高但队列为空", "prefill 计算本身慢：查输入长度、前缀命中率、"
                        "chunk 大小"),
        ("TTFT 高且队列很长", "容量不足：需要扩容或调整准入控制"),
        ("ITL 毛刺大", "prefill 干扰：查 chunked prefill 配置；查抢占次数"),
        ("吞吐正常但延迟差", "batch 过大：降低 max_num_seqs 换取延迟"),
        ("结果不可复现", "查是否有其他负载共享 GPU；查温度墙与功耗限制；"
                    "查预热是否充分"),
    ]
    print()
    for k, v in steps:
        print("  %-20s %s" % (k, v))


def main():
    part1()
    part2()
    part3()
    part4()


if __name__ == "__main__":
    main()
