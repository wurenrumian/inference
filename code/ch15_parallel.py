#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 15 章 · 分布式推理的通信量与并行策略

四部分：
  1. 张量并行的通信量与延迟
  2. 显存分摊：多大的模型需要几张卡
  3. TP 与 PP 的对比
  4. 专家并行与 MoE 的特殊问题
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch02_roofline as rl        # noqa: E402
import ch05_kvcache as kv         # noqa: E402

HW = rl.HW["A100-80GB"]

LINKS = {
    "NVLink（同机 8 卡）": 600e9,
    "PCIe Gen4（同机）": 64e9,
    "RDMA 200Gb（跨机）": 25e9,
    "以太网 100Gb（跨机）": 12.5e9,
}


def sep(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def allreduce_bytes(elems, tp, dtype_bytes=2):
    """ring all-reduce 的单卡收发量：2 × (tp-1)/tp × 数据量。"""
    return 2.0 * (tp - 1) / tp * elems * dtype_bytes


# ---------------------------------------------------------------- 1


def part1():
    sep("1. 张量并行的通信量")
    print("张量并行把每层的权重按列或按行切开，每层需要 2 次 all-reduce：")
    print("  一次在 attention 的输出投影之后，一次在 MLP 的第二个矩阵乘之后。")
    print("通信的数据量 = batch × 本步 token 数 × hidden × 2 字节\n")

    M = rl.MODELS["Llama-3-8B"]
    L, H = M["layers"], M["hidden"]
    print("Llama-3-8B（%d 层，hidden %d），一步 decode，batch 32" % (L, H))
    tokens = 32
    per_step_elems = tokens * H * 2 * L        # 每层 2 次
    print("  单步通信量（TP=2）: %.2f MB"
          % (allreduce_bytes(per_step_elems, 2) / 1e6))
    print("  单步通信量（TP=4）: %.2f MB"
          % (allreduce_bytes(per_step_elems, 4) / 1e6))
    print("  单步通信量（TP=8）: %.2f MB"
          % (allreduce_bytes(per_step_elems, 8) / 1e6))

    print("\n%-22s %10s %12s %12s %14s"
          % ("链路", "TP", "通信量(MB)", "通信耗时(ms)", "占单步比例"))
    base_step = rl.realistic(rl.decode_step(M, HW, 32, 2048)["t"])
    for name, bw in LINKS.items():
        for tp in (2, 4, 8):
            b = allreduce_bytes(per_step_elems, tp)
            t = b / bw + L * 2 * 5e-6          # 加上每次通信的固定发起开销
            step = base_step / tp + t          # 计算被分摊，通信是额外的
            print("%-22s %10d %12.2f %12.2f %13.1f%%"
                  % (name, tp, b / 1e6, t * 1e3, 100 * t / step))
    print("\n跨机的 TP 通信耗时可以超过计算本身，因此张量并行通常限制在")
    print("单机内（NVLink 域）。跨机用流水并行或数据并行。")


# ---------------------------------------------------------------- 2


def part2():
    sep("2. 显存分摊：模型需要几张卡")
    print("单卡 %d GB，按 90%% 可用计算\n" % int(HW["mem"] / 1e9))
    print("%-14s %10s %10s %12s %14s %12s"
          % ("模型", "参数量", "fp16权重", "最小TP", "该TP下KV预算", "8K并发"))
    for name, M in rl.MODELS.items():
        w = rl.weight_bytes(M, 16)
        avail = HW["mem"] * 0.9
        tp = 1
        while w / tp + 4e9 > avail and tp < 16:   # 4GB 留给运行时
            tp *= 2
        kvb = (avail - w / tp - 4e9) * tp          # 全部卡的 KV 之和
        per = kv.kv_per_token(M["layers"], M["kv_heads"], M["head_dim"])
        print("%-14s %9.1fB %9.1fG %12d %13.1fG %12d"
              % (name, M["params"] / 1e9, w / 1e9, tp, kvb / 1e9,
                 int(kvb / (per * 8192))))
    print("\n张量并行不只是为了装下模型：即使装得下，增大 TP 也能")
    print("成倍扩大 KV 预算，从而提高并发。代价是通信开销与卡的利用率下降。")

    print("\nTP 对单请求延迟的影响（Llama-3-8B，batch 1，序列 2048）")
    M = rl.MODELS["Llama-3-8B"]
    L, H = M["layers"], M["hidden"]
    print("%8s %14s %14s %14s %12s"
          % ("TP", "计算(ms)", "通信(ms)", "合计(ms)", "相对TP=1"))
    base = None
    for tp in (1, 2, 4, 8):
        comp = rl.realistic(rl.decode_step(M, HW, 1, 2048)["t"]) / tp
        if tp == 1:
            comm = 0.0
        else:
            b = allreduce_bytes(1 * H * 2 * L, tp)
            comm = b / LINKS["NVLink（同机 8 卡）"] + L * 2 * 5e-6
        tot = comp + comm
        if base is None:
            base = tot
        print("%8d %14.2f %14.2f %14.2f %11.2fx"
              % (tp, comp * 1e3, comm * 1e3, tot * 1e3, base / tot))
    print("\nbatch 1 时 TP 能显著降低单请求延迟（权重访存被分摊到多卡），")
    print("这是低延迟场景使用大 TP 的理由。但卡的总吞吐并未按比例提高。")


# ---------------------------------------------------------------- 3


def part3():
    sep("3. 三种并行方式的对比")
    rows = [
        ("张量并行 TP", "每层的权重按维度切分", "每层 2 次 all-reduce",
         "降低单请求延迟；扩大 KV 预算", "通信频繁，需 NVLink"),
        ("流水并行 PP", "按层切分，每卡负责若干层", "层间传递激活，量小",
         "跨机可行；通信量小", "有流水气泡；单请求延迟不降"),
        ("数据并行 DP", "每卡一份完整模型，各自处理不同请求",
         "无（各自独立）", "线性扩展吞吐；实现最简单", "显存需求不降"),
    ]
    print("%-12s %-22s %-20s" % ("方式", "切分对象", "通信"))
    for r in rows:
        print("%-12s %-22s %-20s" % (r[0], r[1], r[2]))
    print()
    print("%-12s %-30s %-24s" % ("方式", "优点", "代价"))
    for r in rows:
        print("%-12s %-30s %-24s" % (r[0], r[3], r[4]))

    print("\n流水并行的气泡：")
    print("%10s %14s %16s %12s" % ("PP 级数", "单请求延迟", "气泡占比(batch 1)",
                                    "需要的microbatch"))
    for pp in (2, 4, 8):
        print("%10d %14s %15.0f%% %12d"
              % (pp, "不变", 100.0 * (pp - 1) / pp, pp))
    print("\nbatch 1 时 PP 的 %d 级中只有 1 级在工作，其余空转。" % 4)
    print("推理中缓解气泡的方式是连续批处理天然提供的请求流：")
    print("不同请求处于不同的流水级，只要并发足够就能填满。")
    print("因此 PP 在高并发下可用，在低并发下浪费严重。")

    print("\n选择顺序（经验规则）：")
    print("  1. 单卡装得下 → 用数据并行，多起几个实例")
    print("  2. 单卡装不下 → 先在单机内用 TP（NVLink 域内）")
    print("  3. 单机 8 卡仍装不下 → 跨机加 PP，机内保持 TP")
    print("  4. MoE 模型 → 加专家并行 EP，见第 4 节")


# ---------------------------------------------------------------- 4


def part4():
    sep("4. MoE 与专家并行")
    print("MoE 把 MLP 换成多个专家，每个 token 只激活其中几个。\n")
    n_exp, top_k, layers, hidden, inter = 64, 8, 32, 4096, 2048
    exp_bytes = 3 * hidden * inter * 2          # gate/up/down，fp16
    total = n_exp * exp_bytes * layers
    active = top_k * exp_bytes * layers
    print("配置：%d 个专家，每 token 激活 %d 个，%d 层" % (n_exp, top_k, layers))
    print("  全部专家的权重      : %.1f GB" % (total / 1e9))
    print("  单 token 激活的权重  : %.1f GB" % (active / 1e9))
    print("  激活比例            : %.1f%%" % (100.0 * active / total))

    print("\n关键问题：batch 增大时，被激活的专家集合迅速覆盖全部专家。")
    print("%10s %16s %18s %16s"
          % ("batch", "期望激活专家数", "需读取的权重(GB)", "相对单token"))
    for b in (1, 2, 4, 8, 16, 32, 64, 128):
        # 每个 token 独立选 top_k 个专家时，某个专家未被任何 token 选中的概率
        p_miss = (1.0 - float(top_k) / n_exp) ** b
        expected = n_exp * (1.0 - p_miss)
        rd = expected * exp_bytes * layers
        print("%10d %16.1f %18.1f %15.1fx"
              % (b, expected, rd / 1e9, rd / float(active)))
    print("\nbatch 32 时几乎全部专家都要读一遍，权重访存量接近稠密模型，")
    print("而计算量仍然只有激活部分。")

    part4b()


def part4b():
    """MoE 与同等激活参数量的稠密模型的算术强度对比。"""
    print("\n--- 算术强度对比：Mixtral-8x7B 类配置 ---")
    n_exp, top_k, layers = 8, 2, 32
    hidden, inter = 4096, 14336
    attn_params = layers * (4 * hidden * hidden)      # QKVO，粗略估计
    exp_params = 3 * hidden * inter                   # 一个专家
    total = attn_params + layers * n_exp * exp_params
    active = attn_params + layers * top_k * exp_params
    print("总参数 %.1fB，激活参数 %.1fB，激活比例 %.1f%%"
          % (total / 1e9, active / 1e9, 100.0 * active / total))
    print("\n%8s %12s %14s %12s %16s %10s"
          % ("batch", "激活专家数", "MoE访存(GB)", "MoE强度",
             "同激活量稠密强度", "比值"))
    for b in (1, 4, 8, 16, 32, 64, 128):
        cov = n_exp * (1.0 - (1.0 - float(top_k) / n_exp) ** b)
        mem = (attn_params + layers * cov * exp_params) * 2
        comp = 2.0 * active * b
        dense = comp / (active * 2.0)
        print("%8d %12.1f %14.1f %12.1f %16.1f %9.2f"
              % (b, cov, mem / 1e9, comp / mem, dense,
                 (comp / mem) / dense))
    print("\nbatch 足够大时访存量固定为全部参数量，计算量为激活参数量乘 batch，")
    print("因此：")
    print("    MoE 的算术强度 = batch × (激活参数量 / 总参数量)")
    print("本例的激活比例是 %.1f%%，所以算术强度只有同 batch 稠密模型的"
          % (100.0 * active / total))
    print("%.1f%%（表中最后一列）。**稀疏度越高，decode 阶段越吃亏**："
          % (100.0 * active / total))
    print("激活比例 5% 的配置，算术强度只有稠密模型的二十分之一。")
    print("这是 MoE 部署困难的根本原因。")
    print("\n注意 prefill 阶段没有这个问题：prefill 受算力限制，MoE 减少")
    print("计算量的收益直接兑现。因此 MoE 的 prefill 划算、decode 吃亏，")
    print("两侧的最优配置差异比稠密模型更大，更适合 PD 分离（第 10 章）。")

    print("\n专家并行（EP）：把专家分布到多张卡上，每张卡持有部分专家。")
    print("  通信：token 要发到持有对应专家的卡（all-to-all），算完再发回")
    print("  通信量与激活的 token 数和 hidden 成正比，与专家数无关")
    print("  难点是负载不均：热门专家所在的卡成为瓶颈，需要冗余或重分布")


def main():
    part1()
    part2()
    part3()
    part4()
    print("\n观察建议")
    print("  1. 第 1 节把链路换成跨机的 25 GB/s，观察 TP=8 的通信占比，")
    print("     这解释了为什么张量并行不跨机。")
    print("  2. 第 4 节把 batch 从 1 扫到 128，观察 MoE 的权重读取量如何")
    print("     从「只读激活部分」退化为「读全部专家」。")


if __name__ == "__main__":
    main()
