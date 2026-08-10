#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 08 章 · 调度策略的对比

在同一负载上运行五种调度方式，对比吞吐、TTFT、ITL 与 goodput：
  1. 静态批处理（攒够一批一起跑到全部结束）
  2. 连续批处理 + prefill 优先（vLLM V0 的默认行为）
  3. 连续批处理 + chunked prefill（vLLM V1 的默认行为）
  4. 连续批处理 + 最短作业优先
  5. 连续批处理 + 准入控制（显存水位保护）

耗时模型复用第 02 章的 roofline 估算器。
"""

import os
import random
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch02_roofline as rl        # noqa: E402

M = rl.MODELS["Llama-3-8B"]
HW = rl.HW["A100-80GB"]

BLOCK_SIZE = 16
TOTAL_BLOCKS = 6000            # 96000 token 的 KV 空间
MAX_NUM_SEQS = 128
TOKEN_BUDGET = 8192            # 单步最多处理的 token 数
CHUNK_SIZE = 1024              # chunked prefill 的块大小

N_REQ = 300
ARRIVAL_RATE = 8.0

# SLO：用于计算 goodput
SLO_TTFT = 2.0                 # 秒
SLO_TPOT = 0.05                # 秒，请求的平均 ITL


def dec_time(batch, avg_len):
    return rl.realistic(rl.decode_step(M, HW, batch, avg_len)["t"])


def pre_time(tokens, prefix, n_seq):
    return rl.realistic(rl.prefill_step(M, HW, tokens, prefix, n_seq)["t"])


class Req(object):
    __slots__ = ("rid", "arrival", "plen", "olen", "computed", "generated",
                 "blocks", "t_first", "t_done", "itls", "last_token_time")

    def __init__(self, rid, arrival, plen, olen):
        self.rid, self.arrival = rid, arrival
        self.plen, self.olen = plen, olen
        self.computed = 0
        self.generated = 0
        self.blocks = 0
        self.t_first = None
        self.t_done = None
        self.itls = []
        self.last_token_time = None

    def need_blocks(self, extra):
        n = self.computed + self.generated + extra
        return (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    def cur_len(self):
        return self.computed + self.generated


def make_load(seed=5):
    rng = random.Random(seed)
    out, t = [], 0.0
    for i in range(N_REQ):
        t += rng.expovariate(ARRIVAL_RATE)
        # 输入对数正态，输出几何分布，贴近真实对话负载
        plen = min(8192, max(32, int(rng.lognormvariate(6.4, 0.9))))
        olen = min(2048, max(8, int(rng.expovariate(1.0 / 220))))
        out.append(Req(i, t, plen, olen))
    return out


class Sim(object):
    def __init__(self, policy):
        self.policy = policy
        self.reqs = make_load()
        self.pending = deque(self.reqs)
        self.waiting = deque()
        self.running = []
        self.done = []
        self.free = TOTAL_BLOCKS
        self.now = 0.0
        self.steps = 0
        self.preempted = 0
        self.prefill_steps = 0

    # -------------------------------------------------------- 显存

    def alloc(self, r, want):
        d = want - r.blocks
        if d <= 0:
            return True
        if d > self.free:
            return False
        self.free -= d
        r.blocks = want
        return True

    def release(self, r):
        self.free += r.blocks
        r.blocks = 0

    def preempt_last(self):
        r = self.running.pop()
        self.release(r)
        r.computed = 0
        r.generated = 0
        r.t_first = None
        r.last_token_time = None
        self.waiting.appendleft(r)
        self.preempted += 1

    # -------------------------------------------------------- 队列顺序

    def order_waiting(self):
        if self.policy == "sjf":
            # 最短作业优先：按 prompt 长度排序（输出长度不可知，只能用输入）
            items = sorted(self.waiting, key=lambda r: r.plen)
            self.waiting = deque(items)

    # -------------------------------------------------------- 一步

    def admit(self):
        while self.pending and self.pending[0].arrival <= self.now:
            self.waiting.append(self.pending.popleft())

    def step(self):
        self.admit()
        self.order_waiting()

        if self.policy == "static":
            return self.step_static()
        return self.step_continuous()

    # ---------------------------------------------------- 静态批处理

    def step_static(self):
        """攒满一批，跑到全部结束才换下一批。"""
        if not self.running:
            if not self.waiting:
                if self.pending:
                    self.now = self.pending[0].arrival
                    return True
                return False
            batch = []
            while (self.waiting and len(batch) < 32
                   and sum(r.plen for r in batch) + self.waiting[0].plen <= 16384):
                r = self.waiting[0]
                if not self.alloc(r, r.need_blocks(r.plen)):
                    break
                self.waiting.popleft()
                batch.append(r)
            if not batch:
                if self.running:
                    return True
                self.preempt_last() if self.running else None
                return bool(self.pending or self.waiting)
            tokens = sum(r.plen for r in batch)
            self.now += pre_time(tokens // len(batch), 0, len(batch))
            self.prefill_steps += 1
            for r in batch:
                r.computed = r.plen
                r.generated = 1
                r.t_first = self.now
                r.last_token_time = self.now
                self.running.append(r)
            self.steps += 1
            return True

        # 整批一起 decode，直到最长的那个结束
        avg = sum(r.cur_len() for r in self.running) / len(self.running)
        while self.running:
            need = sum(max(0, r.need_blocks(1) - r.blocks) for r in self.running)
            if need <= self.free:
                break
            self.preempt_last()
        if not self.running:
            return True
        dt = dec_time(len(self.running), avg)
        self.now += dt
        self.steps += 1
        for r in self.running:
            self.alloc(r, r.need_blocks(1))
            r.generated += 1
            r.itls.append(self.now - r.last_token_time)
            r.last_token_time = self.now
        if all(r.generated >= r.olen for r in self.running):
            for r in self.running:
                r.t_done = self.now
                self.release(r)
                self.done.append(r)
            self.running = []
        return True

    # ---------------------------------------------------- 连续批处理

    def step_continuous(self):
        chunked = self.policy == "chunked"
        budget = TOKEN_BUDGET
        prefill = []

        # 准入控制：显存低于水位时不再启动新的 prefill
        gate = (self.policy == "admission" and self.free < TOTAL_BLOCKS * 0.15)

        if not gate:
            while (self.waiting
                   and len(self.running) + len(prefill) < MAX_NUM_SEQS
                   and budget > 0):
                r = self.waiting[0]
                remain = r.plen - r.computed
                chunk = min(remain, budget, CHUNK_SIZE if chunked else remain)
                if chunk <= 0:
                    break
                if not self.alloc(r, r.need_blocks(chunk)):
                    break
                self.waiting.popleft()
                prefill.append((r, chunk))
                budget -= chunk
                if not chunked:
                    break_after = budget <= 0
                    if break_after:
                        break

        # chunked 模式下 decode 与 prefill 同批；否则 prefill 独占一步
        decode = []
        if chunked or not prefill:
            while self.running:
                need = sum(max(0, r.need_blocks(1) - r.blocks)
                           for r in self.running)
                if need <= self.free:
                    break
                self.preempt_last()
            for r in self.running:
                if self.alloc(r, r.need_blocks(1)):
                    decode.append(r)

        if not prefill and not decode:
            if self.pending:
                self.now = self.pending[0].arrival
                return True
            if self.waiting and self.running:
                return True
            if self.waiting:
                raise RuntimeError("block 池装不下单个请求")
            return False

        # 计算本步耗时
        dt = 0.0
        if decode:
            avg = sum(r.cur_len() for r in decode) / len(decode)
            dt += dec_time(len(decode), avg)
        if prefill:
            tok = sum(c for _, c in prefill)
            prefix = sum(r.computed for r, _ in prefill) / len(prefill)
            dt += pre_time(tok // len(prefill), int(prefix), len(prefill))
            self.prefill_steps += 1
        self.now += dt
        self.steps += 1

        for r in decode:
            r.generated += 1
            r.itls.append(self.now - r.last_token_time)
            r.last_token_time = self.now
        for r, chunk in prefill:
            r.computed += chunk
            if r.computed >= r.plen:
                r.generated = 1
                r.t_first = self.now
                r.last_token_time = self.now
                self.running.append(r)
            else:
                self.waiting.appendleft(r)

        for r in [x for x in self.running if x.generated >= x.olen]:
            r.t_done = self.now
            self.release(r)
            self.running.remove(r)
            self.done.append(r)
        return True

    def run(self):
        guard = 0
        while self.step():
            guard += 1
            if guard > 2000000:
                raise RuntimeError("模拟未收敛")
        return self


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def summarize(sim):
    d = sim.done
    ttft = [r.t_first - r.arrival for r in d]
    itl_all = []
    for r in d:
        itl_all.extend(r.itls)
    out_tok = sum(r.generated for r in d)
    good = 0
    for r in d:
        avg_itl = sum(r.itls) / len(r.itls) if r.itls else 0.0
        if (r.t_first - r.arrival) <= SLO_TTFT and avg_itl <= SLO_TPOT:
            good += r.generated
    return dict(
        n=len(d),
        span=sim.now,
        thr=out_tok / sim.now if sim.now else 0,
        ttft_p50=pct(ttft, 0.5), ttft_p99=pct(ttft, 0.99),
        itl_p50=pct(itl_all, 0.5), itl_p99=pct(itl_all, 0.99),
        good=good / sim.now if sim.now else 0,
        steps=sim.steps, pre=sim.prefill_steps, prem=sim.preempted)


POLICIES = [
    ("static", "静态批处理"),
    ("fcfs", "连续批处理 + prefill 优先"),
    ("chunked", "连续批处理 + chunked prefill"),
    ("sjf", "连续批处理 + 最短作业优先"),
    ("admission", "连续批处理 + 准入控制"),
]


def run_scenario(rate, label):
    global ARRIVAL_RATE
    ARRIVAL_RATE = rate
    print("\n" + "=" * 100)
    print("场景：%s（到达率 %.0f 请求/秒）" % (label, rate))
    print("=" * 100)
    print("%-28s %8s %10s %9s %9s %9s %9s %10s %7s"
          % ("策略", "完成", "吞吐", "TTFT P50", "TTFT P99",
             "ITL P50", "ITL P99", "goodput", "抢占"))
    print("%-28s %8s %10s %9s %9s %9s %9s %10s %7s"
          % ("", "", "tok/s", "s", "s", "ms", "ms", "tok/s", ""))
    print("-" * 100)
    groups_all = {}
    for key, name in POLICIES:
        s = Sim(key).run()
        r = summarize(s)
        print("%-28s %8d %10.0f %9.2f %9.2f %9.1f %9.1f %10.0f %7d"
              % (name, r["n"], r["thr"], r["ttft_p50"], r["ttft_p99"],
                 r["itl_p50"] * 1e3, r["itl_p99"] * 1e3, r["good"], r["prem"]))
        g = {"s": [], "m": [], "l": []}
        for q in s.done:
            k = "s" if q.plen < 300 else ("m" if q.plen <= 1500 else "l")
            g[k].append(q.t_first - q.arrival)
        groups_all[key] = g

    print("\n按 prompt 长度分组的 TTFT 中位数（秒）")
    print("%-28s %14s %14s %14s"
          % ("策略", "短(<300)", "中(300-1500)", "长(>1500)"))
    for key, name in POLICIES:
        g = groups_all[key]
        print("%-28s %14.2f %14.2f %14.2f"
              % (name, pct(g["s"], 0.5), pct(g["m"], 0.5), pct(g["l"], 0.5)))


def main():
    global CHUNK_SIZE, ARRIVAL_RATE
    print("负载: %d 请求，输入对数正态(中位约 600)，输出指数分布(均值 220)"
          % N_REQ)
    print("配置: block 池 %d（%d token），max_num_seqs %d，token 预算 %d，"
          % (TOTAL_BLOCKS, TOTAL_BLOCKS * BLOCK_SIZE, MAX_NUM_SEQS, TOKEN_BUDGET))
    print("      chunk 大小 %d" % CHUNK_SIZE)
    print("SLO : TTFT <= %.1fs 且 单请求平均 ITL <= %.0fms"
          % (SLO_TTFT, SLO_TPOT * 1e3))

    run_scenario(8.0, "接近饱和")
    run_scenario(30.0, "过载")

    print("\n解读")
    print("  静态批处理：一批内的短请求要等最长的那个结束才能释放位置，")
    print("    有效并发随时间衰减，TTFT 差两个量级，goodput 接近 0。")
    print("  prefill 优先：TTFT 最好，但 prefill 独占整步，ITL P99 高。")
    print("  chunked prefill：ITL P99 下降，TTFT 略增。")
    print("  最短作业优先：短请求 TTFT 改善，长请求 TTFT 变差（饥饿）。")
    print("    过载时这一效应最明显，见分组表的第三列。")
    print("  准入控制：显存低于水位时暂停接纳新 prefill，抢占次数下降，")
    print("    代价是 TTFT 变差——把等待从显存里挪到了队列里。")

    print("\n" + "=" * 100)
    print("chunked prefill 的 chunk 大小扫描")
    print("=" * 100)

    orig = CHUNK_SIZE
    print("%10s %10s %10s %10s %10s %10s"
          % ("chunk", "吞吐", "TTFT P50", "TTFT P99", "ITL P99", "goodput"))
    for c in (256, 512, 1024, 2048, 4096, 8192):
        CHUNK_SIZE = c
        r = summarize(Sim("chunked").run())
        print("%10d %10.0f %10.2f %10.2f %10.1f %10.0f"
              % (c, r["thr"], r["ttft_p50"], r["ttft_p99"],
                 r["itl_p99"] * 1e3, r["good"]))
    CHUNK_SIZE = orig
    print("\nchunk 越小 ITL 越平稳、TTFT 越差。goodput 在中间取得最大值，")
    print("具体位置取决于 SLO 的两个阈值。这是调参的依据。")


if __name__ == "__main__":
    main()
