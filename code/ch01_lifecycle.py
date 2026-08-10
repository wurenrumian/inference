#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 01 章 · 请求生命周期模拟

用离散事件的方式模拟一个推理引擎的调度循环，暴露三个排队点：
  A. 请求到达后等待进入 running 状态
  B. 调度器因 token 预算不足退回请求
  C. block 池耗尽

不做真实计算，用参数化的耗时模型代替，目的是观察 TTFT 的构成如何
随负载变化。修改下方常量后重新运行，对比 TTFT 中排队与计算的比例。
"""

import random
from collections import deque

# ---------------------------------------------------------------- 可调参数

NUM_REQUESTS = 40          # 请求总数
ARRIVAL_RATE = 12.0        # 每秒到达的请求数（泊松过程）
PROMPT_LEN_RANGE = (200, 2000)
OUTPUT_LEN_RANGE = (50, 400)

BLOCK_SIZE = 16            # 每个 block 容纳的 token 数
BLOCK_POOL_SIZE = 600      # block 池总量，600 * 16 = 9600 token 的 KV 空间
MAX_NUM_SEQS = 32          # 同时处于 running 的序列数上限
PREFILL_TOKEN_BUDGET = 2048  # 单步 prefill 最多处理的 token 数

# 耗时模型（秒）。数量级参考 7B 模型单卡，非实测值。
PREFILL_TIME_PER_TOKEN = 40e-6   # prefill 每 token 约 40 微秒
DECODE_STEP_BASE = 6e-3          # 一步 decode 的固定开销
DECODE_STEP_PER_SEQ = 0.25e-3    # 每多一个序列增加的耗时
SCHEDULE_OVERHEAD = 0.8e-3       # 每步调度与张量准备的 CPU 开销

SEED = 7

# ---------------------------------------------------------------- 数据结构


class Request(object):
    __slots__ = ("rid", "arrival", "prompt_len", "output_len",
                 "num_computed", "num_generated", "blocks",
                 "t_first_schedule", "t_first_token", "t_finish")

    def __init__(self, rid, arrival, prompt_len, output_len):
        self.rid = rid
        self.arrival = arrival
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.num_computed = 0      # 已完成 prefill 的 token 数
        self.num_generated = 0     # 已生成的 token 数
        self.blocks = 0            # 当前占用的 block 数
        self.t_first_schedule = None
        self.t_first_token = None
        self.t_finish = None

    @property
    def total_len(self):
        return self.prompt_len + self.num_generated

    def blocks_needed(self, extra_tokens):
        """再放入 extra_tokens 个 token 后总共需要多少 block。"""
        total = self.num_computed + self.num_generated + extra_tokens
        return (total + BLOCK_SIZE - 1) // BLOCK_SIZE


def make_requests():
    rng = random.Random(SEED)
    reqs = []
    t = 0.0
    for i in range(NUM_REQUESTS):
        t += rng.expovariate(ARRIVAL_RATE)
        reqs.append(Request(
            rid=i,
            arrival=t,
            prompt_len=rng.randint(*PROMPT_LEN_RANGE),
            output_len=rng.randint(*OUTPUT_LEN_RANGE),
        ))
    return reqs


# ---------------------------------------------------------------- 引擎循环


class Engine(object):
    def __init__(self, requests):
        self.pending = deque(requests)   # 尚未到达
        self.waiting = deque()           # 已到达，未进入 running
        self.running = []                # 正在执行
        self.finished = []
        self.free_blocks = BLOCK_POOL_SIZE
        self.now = 0.0
        self.step_id = 0
        self.num_preempted = 0
        self.log = []

    # ------------------------------------------------------------ 辅助

    def admit_arrivals(self):
        while self.pending and self.pending[0].arrival <= self.now:
            self.waiting.append(self.pending.popleft())

    def alloc(self, req, want_blocks):
        delta = want_blocks - req.blocks
        if delta <= 0:
            return True
        if delta > self.free_blocks:
            return False
        self.free_blocks -= delta
        req.blocks = want_blocks
        return True

    def release(self, req):
        self.free_blocks += req.blocks
        req.blocks = 0

    # ------------------------------------------------------------ 调度

    def schedule(self):
        """返回本步的 (prefill 列表, decode 列表)。

        策略与 vLLM V0 一致：prefill 优先。只要 waiting 队列中有请求能
        装下，本步就用于 prefill，decode 请求让路。这带来的后果是
        decode 请求的 TPOT 出现毛刺，见第 10 章。
        """
        prefill = []
        budget = PREFILL_TOKEN_BUDGET
        while self.waiting and len(self.running) + len(prefill) < MAX_NUM_SEQS:
            req = self.waiting[0]
            chunk = min(req.prompt_len - req.num_computed, budget)
            if chunk <= 0:
                break
            need = req.blocks_needed(chunk)
            if not self.alloc(req, need):
                break                      # 排队点 C：block 不足
            self.waiting.popleft()
            prefill.append((req, chunk))
            budget -= chunk
            if req.t_first_schedule is None:
                req.t_first_schedule = self.now
            if budget <= 0:
                break                      # 排队点 B：token 预算耗尽

        if prefill:
            return prefill, []

        # decode 阶段每个序列可能需要新增 1 个 block。若不足，抢占最晚加入
        # 的请求（丢弃其 KV，回到 waiting 队列重新 prefill）。这是第 09 章
        # 讲的 recompute 式抢占，此处用最简单的形式保证循环不会死锁。
        while self.running:
            need = sum(max(0, r.blocks_needed(1) - r.blocks) for r in self.running)
            if need <= self.free_blocks:
                break
            self.preempt(self.running[-1])

        decode = []
        for req in self.running:
            if self.alloc(req, req.blocks_needed(1)):
                decode.append(req)
        return [], decode

    def preempt(self, req):
        self.release(req)
        self.running.remove(req)
        req.num_computed = 0
        req.num_generated = 0
        req.t_first_token = None
        req.t_first_schedule = None
        self.waiting.appendleft(req)
        self.num_preempted += 1

    # ------------------------------------------------------------ 执行

    def step(self):
        self.admit_arrivals()
        prefill, decode = self.schedule()

        if not prefill and not decode:
            if self.pending:
                self.now = self.pending[0].arrival    # 空转，跳到下次到达
                return True
            if self.waiting:
                raise RuntimeError(
                    "block 池无法容纳单个请求：请调大 BLOCK_POOL_SIZE")
            return False

        if prefill:
            tokens = sum(c for _, c in prefill)
            dur = SCHEDULE_OVERHEAD + tokens * PREFILL_TIME_PER_TOKEN
            self.now += dur
            for req, chunk in prefill:
                req.num_computed += chunk
                if req.num_computed >= req.prompt_len:
                    req.num_generated = 1              # prefill 产出首 token
                    req.t_first_token = self.now
                    self.running.append(req)
                else:
                    self.waiting.appendleft(req)       # 分块未完，下步继续
            kind, size = "prefill", tokens
        else:
            dur = (SCHEDULE_OVERHEAD + DECODE_STEP_BASE
                   + DECODE_STEP_PER_SEQ * len(decode))
            self.now += dur
            for req in decode:
                req.num_generated += 1
            kind, size = "decode", len(decode)

        done = [r for r in self.running if r.num_generated >= r.output_len]
        for req in done:
            req.t_finish = self.now
            self.release(req)
            self.running.remove(req)
            self.finished.append(req)

        self.log.append((self.step_id, round(self.now, 4), kind, size,
                         len(self.running), len(self.waiting),
                         BLOCK_POOL_SIZE - self.free_blocks))
        self.step_id += 1
        return True

    def run(self):
        while self.step():
            pass


# ---------------------------------------------------------------- 输出


def main():
    reqs = make_requests()
    eng = Engine(reqs)
    eng.run()

    print("=" * 78)
    print("配置：block 池 %d（%d token），并发上限 %d，prefill 预算 %d token/步"
          % (BLOCK_POOL_SIZE, BLOCK_POOL_SIZE * BLOCK_SIZE,
             MAX_NUM_SEQS, PREFILL_TOKEN_BUDGET))
    print("=" * 78)

    print("\n迭代时间线（前 20 步）")
    print("%5s %9s %9s %7s %8s %8s %9s"
          % ("step", "时刻(s)", "类型", "规模", "running", "waiting", "已用block"))
    for row in eng.log[:20]:
        print("%5d %9.4f %9s %7d %8d %8d %9d" % row)

    print("\n请求明细（按到达顺序，前 15 条）")
    print("%4s %8s %7s %7s %9s %9s %9s %9s"
          % ("id", "到达", "输入", "输出", "TTFT", "排队", "prefill", "端到端"))
    for r in sorted(eng.finished, key=lambda x: x.rid)[:15]:
        queue = r.t_first_schedule - r.arrival
        compute = r.t_first_token - r.t_first_schedule
        ttft = r.t_first_token - r.arrival
        e2e = r.t_finish - r.arrival
        print("%4d %8.3f %7d %7d %9.3f %9.3f %9.3f %9.3f"
              % (r.rid, r.arrival, r.prompt_len, r.output_len,
                 ttft, queue, compute, e2e))

    ttfts = sorted(r.t_first_token - r.arrival for r in eng.finished)
    queues = [r.t_first_schedule - r.arrival for r in eng.finished]
    e2es = [r.t_finish - r.arrival for r in eng.finished]
    out_tokens = sum(r.output_len for r in eng.finished)

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    print("\n汇总")
    print("  完成请求数        : %d" % len(eng.finished))
    print("  总时长            : %.3f s" % eng.now)
    print("  输出吞吐          : %.1f token/s" % (out_tokens / eng.now))
    print("  TTFT  平均/P50/P99: %.3f / %.3f / %.3f s"
          % (sum(ttfts) / len(ttfts), pct(ttfts, 0.5), pct(ttfts, 0.99)))
    print("  其中排队占比      : %.1f%%"
          % (100.0 * sum(queues) / sum(ttfts)))
    print("  端到端平均        : %.3f s" % (sum(e2es) / len(e2es)))
    print("  迭代步数          : %d（prefill %d，decode %d）"
          % (len(eng.log),
             sum(1 for x in eng.log if x[2] == "prefill"),
             sum(1 for x in eng.log if x[2] == "decode")))
    print("  抢占次数          : %d（重算式，被抢占的请求需重新 prefill）"
          % eng.num_preempted)

    print("\n观察建议")
    print("  1. 默认配置处于过载状态：排队占 TTFT 的 97%，出现 14 次抢占。")
    print("     只把 ARRIVAL_RATE 降到 2，P50 TTFT 从 3.04 s 降到 0.06 s，")
    print("     但 P99 仍为 2.89 s 且抢占次数为 12，说明 block 池仍是约束。")
    print("  2. 只把 BLOCK_POOL_SIZE 提到 3000，抢占降为 0，吞吐提高约一倍。")
    print("     两项同时放宽后排队占比降到 14%，此时 TTFT 由 prefill 决定。")
    print("  3. 把 PREFILL_TOKEN_BUDGET 从 2048 降到 256（即 chunked prefill），")
    print("     观察 prefill 步数上升、单步耗时更均匀，见第 10 章。")


if __name__ == "__main__":
    main()
