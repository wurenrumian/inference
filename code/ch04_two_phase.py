#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 04 章 · prefill 与 decode 的对比

四组数据：
  1. batch 对两阶段吞吐的不同影响
  2. 混合批中 prefill 对 decode 的干扰
  3. 不同输入输出长度下两阶段的时间占比
  4. chunked prefill 的 chunk 大小对单步耗时分布的影响

耗时模型复用第 02 章的 roofline 估算器。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch02_roofline as rl        # noqa: E402

M = rl.MODELS["Llama-3-8B"]
HW = rl.HW["A100-80GB"]


def dec(batch, seqlen):
    """一步 decode 的实际耗时（秒）。"""
    return rl.realistic(rl.decode_step(M, HW, batch, seqlen)["t"])


def pre(tokens, prefix=0, n_seq=1):
    """一步 prefill 的实际耗时（秒）。

    tokens 为每序列新处理的 token 数，prefix 为已有 KV 长度。
    """
    return rl.realistic(rl.prefill_step(M, HW, tokens, prefix, n_seq)["t"])


def sep(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------- 1


def part1():
    sep("1. batch 对两阶段的作用相反（序列长度 2048）")
    print("%8s %12s %16s %14s %16s"
          % ("batch", "decode单步", "decode吞吐", "prefill耗时", "prefill吞吐"))
    print("%8s %12s %16s %14s %16s"
          % ("", "(ms)", "(token/s)", "(ms)", "(token/s)"))
    for b in (1, 2, 4, 8, 16, 32, 64, 128):
        td = dec(b, 2048)
        tp = pre(2048, n_seq=b)     # b 个各 2048 长度的独立 prompt 一起做
        print("%8d %12.2f %16.0f %14.1f %16.0f"
              % (b, td * 1e3, b / td, tp * 1e3, 2048 * b / tp))
    print("\ndecode 的吞吐随 batch 近似线性增长（权重访存被摊薄）；")
    print("prefill 的吞吐基本不变（已受算力限制，无可摊薄的成本）。")


# ---------------------------------------------------------------- 2


def part2():
    sep("2. 混合批的干扰：一步 decode 中混入 prefill")
    base = dec(32, 2048)
    print("纯 decode 步（batch 32，长度 2048）: %.2f ms" % (base * 1e3))
    print("\n%14s %14s %14s %10s"
          % ("混入prefill长度", "该步耗时(ms)", "ITL放大倍数", "用户感知"))
    for n in (0, 128, 512, 2048, 8192, 32768):
        t = base + (pre(n) if n else 0.0)
        note = "无感" if t / base < 1.5 else ("可感" if t / base < 5 else "卡顿")
        print("%14d %14.1f %14.1fx %10s" % (n, t * 1e3, t / base, note))
    print("\n毛刺的幅度与 prompt 长度成正比。这是 chunked prefill 要解决的问题。")


# ---------------------------------------------------------------- 3


def part3():
    sep("3. 两阶段的时间占比（batch 32，decode 单步按实际长度计算）")
    cases = [
        ("短问答", 128, 256),
        ("常规对话", 2048, 512),
        ("多轮对话", 8192, 512),
        ("长文摘要", 32768, 512),
        ("长文抽取", 32768, 64),
        ("代码补全", 8192, 32),
        ("长文生成", 2048, 4096),
    ]
    print("%-10s %8s %8s %12s %12s %12s %10s"
          % ("场景", "输入", "输出", "prefill(ms)", "decode(ms)",
             "合计(ms)", "prefill占比"))
    for name, n, m in cases:
        tp = pre(n)
        # decode 期间序列长度从 n 增长到 n+m，取中点估算
        td = dec(32, n + m // 2) * (m - 1)
        tot = tp + td
        print("%-10s %8d %8d %12.1f %12.1f %12.1f %9.1f%%"
              % (name, n, m, tp * 1e3, td * 1e3, tot * 1e3, 100 * tp / tot))
    print("\n输入长输出短的场景由 prefill 主导，反之由 decode 主导。")
    print("两类场景的优化重点不同，配置也不同。")


# ---------------------------------------------------------------- 4


def part4():
    sep("4. chunked prefill：chunk 大小对单步耗时的影响")
    prompt = 32768
    base_decode = dec(32, 2048)
    print("场景：一个 %d token 的 prompt，与 32 个正在 decode 的序列共存"
          % prompt)
    print("纯 decode 步耗时: %.2f ms\n" % (base_decode * 1e3))
    print("%10s %8s %14s %16s %16s"
          % ("chunk", "步数", "最大单步(ms)", "prefill总耗时(ms)", "TTFT相对一次性"))
    ref = None
    for chunk in (32768, 8192, 4096, 2048, 1024, 512, 256):
        steps = (prompt + chunk - 1) // chunk
        total, worst = 0.0, 0.0
        for k in range(steps):
            prefix = k * chunk
            n = min(chunk, prompt - prefix)
            t = base_decode + pre(n, prefix=prefix)
            total += t
            worst = max(worst, t)
        if ref is None:
            ref = total
        print("%10d %8d %14.1f %16.1f %15.2fx"
              % (chunk, steps, worst * 1e3, total * 1e3, total / ref))
    print("\nchunk 变小后单步耗时下降一个量级，ITL 更平稳；")
    print("代价是 prefill 的总耗时上升（每步都要重新读一遍模型权重），")
    print("即 TTFT 变差。chunk 大小是 TTFT 与 ITL 之间的调节旋钮。")
    print("\n注意：上表的 prefill 总耗时随 chunk 变小而上升，来自权重访存被")
    print("重复支付。chunk 足够大（超过 512）时 prefill 仍受算力限制，")
    print("这部分开销有限；chunk 过小则退化为访存受限，开销显著。")


def main():
    print("模型 Llama-3-8B，硬件 A100-80GB，数值为理论估算加经验系数")
    part1()
    part2()
    part3()
    part4()


if __name__ == "__main__":
    main()
