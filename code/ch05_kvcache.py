#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 05 章 · KV cache 的容量与显存预算

回答四个问题：
  1. 一个模型的 KV cache 每 token 占多少字节
  2. 给定卡型，装完权重后能容纳多少 token、多少并发
  3. MHA / MQA / GQA / MLA 的差别有多大
  4. KV 量化能换来多少并发
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch02_roofline as rl        # noqa: E402

MODELS = rl.MODELS
HW = rl.HW

# 运行时除权重与 KV 外的开销：激活、通信缓冲、CUDA context、碎片
RUNTIME_OVERHEAD_FRAC = 0.10


def kv_per_token(layers, kv_heads, head_dim, bits=16):
    """key 与 value 两份，单位字节。"""
    return 2 * layers * kv_heads * head_dim * bits / 8.0


def sep(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- 1


def part1():
    sep("1. 各模型的 KV cache 单价（fp16）")
    print("%-14s %8s %8s %8s %10s %14s %12s"
          % ("模型", "层数", "Q头", "KV头", "head_dim", "KB/token/序列",
             "2K序列(MB)"))
    for name, m in MODELS.items():
        b = kv_per_token(m["layers"], m["kv_heads"], m["head_dim"])
        print("%-14s %8d %8d %8d %10d %14.1f %12.1f"
              % (name, m["layers"], m["heads"], m["kv_heads"], m["head_dim"],
                 b / 1024, b * 2048 / 1e6))
    print("\n公式: 2 × 层数 × KV头数 × head_dim × 精度字节数")
    print("其中 2 表示 key 与 value 各一份。与 batch 无关，与 Q 头数无关。")


# ---------------------------------------------------------------- 2


def part2():
    sep("2. 显存预算：装完权重后还剩多少，能放多少并发")
    print("%-14s %-12s %8s %8s %10s %12s %10s"
          % ("模型", "硬件", "显存", "权重", "运行时", "KV可用", "KV token数"))
    rows = [("Llama-3-8B", "A100-80GB"), ("Llama-3-8B", "RTX4090"),
            ("Qwen2.5-7B", "RTX4090"), ("Llama-2-13B", "A100-80GB"),
            ("Qwen2.5-72B", "A100-80GB")]
    for mn, hn in rows:
        m, h = MODELS[mn], HW[hn]
        w = rl.weight_bytes(m, 16)
        over = h["mem"] * RUNTIME_OVERHEAD_FRAC
        kv = h["mem"] - w - over
        per = kv_per_token(m["layers"], m["kv_heads"], m["head_dim"])
        if kv <= 0:
            print("%-14s %-12s %7.0fG %7.1fG %9.1fG %12s %10s"
                  % (mn, hn, h["mem"] / 1e9, w / 1e9, over / 1e9,
                     "装不下", "-"))
            continue
        print("%-14s %-12s %7.0fG %7.1fG %9.1fG %11.1fG %10.0f"
              % (mn, hn, h["mem"] / 1e9, w / 1e9, over / 1e9,
                 kv / 1e9, kv / per))
    print("\n最后一列除以序列长度即为最大并发数。72B 单卡装不下，需要张量并行（第 15 章）。")


# ---------------------------------------------------------------- 3


def part3():
    sep("3. 最大并发随序列长度变化（Llama-3-8B on A100-80GB，fp16 KV）")
    m, h = MODELS["Llama-3-8B"], HW["A100-80GB"]
    kv_budget = h["mem"] - rl.weight_bytes(m, 16) - h["mem"] * RUNTIME_OVERHEAD_FRAC
    per = kv_per_token(m["layers"], m["kv_heads"], m["head_dim"])
    print("KV 预算 %.1f GB，单价 %.0f KB/token\n" % (kv_budget / 1e9, per / 1024))
    print("%12s %12s %14s %16s"
          % ("序列长度", "最大并发", "单序列KV(MB)", "该并发下的吞吐"))
    for s in (512, 2048, 8192, 32768, 131072):
        n = int(kv_budget / (per * s))
        if n < 1:
            print("%12d %12s %14.1f %16s"
                  % (s, "0（放不下）", per * s / 1e6, "-"))
            continue
        t = rl.realistic(rl.decode_step(m, h, min(n, 256), s)["t"])
        print("%12d %12d %14.1f %14.0f token/s"
              % (s, n, per * s / 1e6, min(n, 256) / t))
    print("\n序列长度每增加 4 倍，最大并发降到四分之一，吞吐随之下降。")
    print("这是长上下文服务成本高的直接原因。")


# ---------------------------------------------------------------- 4


def part4():
    sep("4. 注意力变体的对比（32 层，32 个 Q 头，head_dim 128）")
    layers, heads, hd = 32, 32, 128
    variants = [
        ("MHA（kv_heads=32）", 32, None),
        ("GQA-8（kv_heads=8）", 8, None),
        ("GQA-4（kv_heads=4）", 4, None),
        ("MQA（kv_heads=1）", 1, None),
        ("MLA（压缩维 512 + rope 64）", None, 512 + 64),
    ]
    m, h = MODELS["Llama-3-8B"], HW["A100-80GB"]
    kv_budget = h["mem"] - rl.weight_bytes(m, 16) - h["mem"] * RUNTIME_OVERHEAD_FRAC
    base = None
    print("%-28s %14s %10s %16s"
          % ("变体", "KB/token", "相对MHA", "8K序列的并发"))
    for name, kvh, mla_dim in variants:
        if mla_dim is None:
            b = kv_per_token(layers, kvh, hd)
        else:
            # MLA 缓存的是压缩后的隐向量，只有一份，不是 key/value 两份
            b = layers * mla_dim * 2
        if base is None:
            base = b
        print("%-28s %14.1f %9.2fx %16d"
              % (name, b / 1024, b / base, int(kv_budget / (b * 8192))))
    print("\nGQA 把 KV 降到 1/4（kv_heads 8 对 32），MQA 降到 1/32。")
    print("代价是模型表达能力，需要在预训练阶段就确定，不是部署时的选项。")
    print("MLA 把 KV 压缩到低维隐空间，解码时再投影回来，用计算换显存。")


# ---------------------------------------------------------------- 5


def part5():
    sep("5. KV 量化的收益（Llama-3-8B on A100-80GB，序列长度 8192）")
    m, h = MODELS["Llama-3-8B"], HW["A100-80GB"]
    kv_budget = h["mem"] - rl.weight_bytes(m, 16) - h["mem"] * RUNTIME_OVERHEAD_FRAC
    print("%-12s %12s %10s %14s %14s %14s"
          % ("KV 精度", "KB/token", "最大并发", "满并发单步", "满并发吞吐",
             "并发52时单步"))
    for label, bits in (("fp16", 16), ("fp8 / int8", 8), ("int4", 4)):
        per = kv_per_token(m["layers"], m["kv_heads"], m["head_dim"], bits)
        n = int(kv_budget / (per * 8192))
        t_full = rl.realistic(rl.decode_step(m, h, n, 8192, 16, bits)["t"])
        t_fix = rl.realistic(rl.decode_step(m, h, 52, 8192, 16, bits)["t"])
        print("%-12s %12.1f %10d %12.2fms %11.0f/s %12.2fms"
              % (label, per / 1024, n, t_full * 1e3, n / t_full, t_fix * 1e3))
    print("\n两列耗时对应两种用法：")
    print("  满并发：把省下的显存全部用于提高并发。单步耗时不变（KV 总字节数")
    print("          受预算限制，本来就已用满），吞吐随并发线性提高。")
    print("  固定并发：并发不变，单步耗时下降，即 TPOT 改善。")
    print("实际配置介于两者之间，由 SLO 决定。精度影响见第 13 章。")


# ---------------------------------------------------------------- 6


def part6():
    sep("6. KV cache 的布局与访存（torch CPU 验证）")
    try:
        import torch
    except ImportError:
        print("未安装 torch，跳过。")
        return

    n_blocks, block_size, kv_heads, head_dim = 64, 16, 8, 128
    # vLLM 的两种常见布局
    a = torch.zeros(n_blocks, block_size, kv_heads, head_dim)   # NHD
    b = torch.zeros(n_blocks, kv_heads, head_dim, block_size)   # HND
    print("布局 A (block, token, head, dim): %s，元素 %d"
          % (list(a.shape), a.numel()))
    print("布局 B (block, head, dim, token): %s，元素 %d"
          % (list(b.shape), b.numel()))
    print("\n两者元素总数相同，差别在于哪一维连续：")
    print("  布局 A：同一 token 的所有 head 连续 —— 写入新 token 时是一次连续写")
    print("  布局 B：同一 head 的所有 token 连续 —— 读取历史时是一次连续读")
    print("decode 阶段每步写 1 个 token、读全部历史，因此两者是写友好与")
    print("读友好的取舍。具体选择由 attention kernel 的实现决定。")

    print("\n单序列 KV cache 的实际大小验证：")
    m = MODELS["Llama-3-8B"]
    seq = 2048
    real = torch.zeros(2, m["layers"], seq, m["kv_heads"], m["head_dim"],
                       dtype=torch.float16)
    print("  张量形状 %s" % list(real.shape))
    print("  实际字节 %.1f MB" % (real.numel() * 2 / 1e6))
    print("  公式预期 %.1f MB"
          % (kv_per_token(m["layers"], m["kv_heads"], m["head_dim"])
             * seq / 1e6))


def main():
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
    print("\n观察建议")
    print("  1. 把 RUNTIME_OVERHEAD_FRAC 从 0.10 改为 0.05，看可用 KV 的变化。")
    print("     这对应 vLLM 的 --gpu-memory-utilization 参数。")
    print("  2. 第 3 节中把序列长度从 2048 改到 131072，观察并发数的下降幅度，")
    print("     这决定了长上下文服务的单位成本。")


if __name__ == "__main__":
    main()
