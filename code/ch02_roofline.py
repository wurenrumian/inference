#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 02 章 · roofline 估算器

用计算量与访存量估算 prefill 与 decode 的单步耗时，判定瓶颈在算力还是
显存带宽。所有数值为理论下界，不含 kernel 效率损失与框架开销；第 6 节
的经验系数在 EFFICIENCY 中给出。
"""

# ---------------------------------------------------------------- 硬件

HW = {
    #  名称        峰值算力(FLOPS, fp16)  显存带宽(B/s)  显存(B)
    "A100-80GB": dict(flops=312e12, bw=2.0e12, mem=80e9),
    "H100-SXM": dict(flops=990e12, bw=3.35e12, mem=80e9),
    "L20": dict(flops=119e12, bw=0.864e12, mem=48e9),
    "RTX4090": dict(flops=165e12, bw=1.008e12, mem=24e9),
}

# ---------------------------------------------------------------- 模型

MODELS = {
    "Llama-3-8B": dict(params=8.03e9, layers=32, hidden=4096,
                       heads=32, kv_heads=8, head_dim=128, vocab=128256),
    "Llama-2-13B": dict(params=13.0e9, layers=40, hidden=5120,
                        heads=40, kv_heads=40, head_dim=128, vocab=32000),
    "Qwen2.5-7B": dict(params=7.62e9, layers=28, hidden=3584,
                       heads=28, kv_heads=4, head_dim=128, vocab=152064),
    "Qwen2.5-72B": dict(params=72.7e9, layers=80, hidden=8192,
                        heads=64, kv_heads=8, head_dim=128, vocab=152064),
}

MODEL = "Llama-3-8B"
DEV = "A100-80GB"

EFFICIENCY = 0.7        # kernel 实际达到峰值的比例，经验值
FRAMEWORK_OVERHEAD = 2e-3   # 每步框架开销（秒）

# ---------------------------------------------------------------- 基本量


def weight_bytes(m, bits=16):
    return m["params"] * bits / 8.0


def kv_bytes_per_token(m, bits=16):
    """一个 token、一个序列的 KV cache 字节数（key 与 value 共两份）。"""
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * bits / 8.0


def decode_step(m, hw, batch, seqlen, wbits=16, kvbits=16):
    """一步 decode（batch 内每个序列各生成 1 个 token）的理论耗时。"""
    compute = 2.0 * m["params"] * batch
    mem_w = weight_bytes(m, wbits)
    mem_kv = kv_bytes_per_token(m, kvbits) * seqlen * batch
    mem = mem_w + mem_kv
    t_c = compute / hw["flops"]
    t_m = mem / hw["bw"]
    return dict(compute=compute, mem=mem, mem_w=mem_w, mem_kv=mem_kv,
                t_compute=t_c, t_mem=t_m, t=max(t_c, t_m),
                intensity=compute / mem,
                bound="compute" if t_c > t_m else "memory")


def prefill_step(m, hw, tokens, prefix=0, n_seq=1, wbits=16):
    """一步 prefill 的理论耗时。

    tokens : 每个序列本步新处理的 token 数
    prefix : 每个序列已有的 KV 长度（chunked prefill 的前序块，或前缀缓存命中）
    n_seq  : 本步一起处理的序列数

    attention 的计算量 = 新 token 数 × 需要注意到的位置数。新 token 之间
    是因果的（平均 tokens/2），加上对全部 prefix 的注意（prefix）。
    """
    per_seq_ctx = prefix + tokens / 2.0
    compute = n_seq * 2.0 * m["params"] * tokens
    attn = (n_seq * 2.0 * 2.0 * m["layers"] * m["heads"] * m["head_dim"]
            * tokens * per_seq_ctx)
    compute += attn
    mem_w = weight_bytes(m, wbits)
    # 写入新 KV；同时读取全部前缀 KV（每步各读一遍）
    mem_kv = kv_bytes_per_token(m) * n_seq * (tokens + prefix)
    act = 2.0 * tokens * n_seq * m["hidden"] * 2 * m["layers"] * 4
    mem = mem_w + mem_kv + act
    t_c = compute / hw["flops"]
    t_m = mem / hw["bw"]
    return dict(compute=compute, attn_flops=attn, mem=mem,
                t_compute=t_c, t_mem=t_m, t=max(t_c, t_m),
                intensity=compute / mem,
                bound="compute" if t_c > t_m else "memory")


def realistic(t):
    return t / EFFICIENCY + FRAMEWORK_OVERHEAD


# ---------------------------------------------------------------- 输出


def sep(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def main():
    m, hw = MODELS[MODEL], HW[DEV]

    sep("1. 机器平衡点：算力 / 带宽，单位 FLOPs per byte")
    print("%-12s %12s %12s %10s" % ("硬件", "算力(TFLOPS)", "带宽(TB/s)", "平衡点"))
    for name, h in HW.items():
        print("%-12s %12.0f %12.2f %10.0f"
              % (name, h["flops"] / 1e12, h["bw"] / 1e12, h["flops"] / h["bw"]))
    print("\n算术强度低于平衡点的算子受显存带宽限制，高于则受算力限制。")

    sep("2. %s 的静态参数" % MODEL)
    print("  参数量              : %.2f B" % (m["params"] / 1e9))
    print("  fp16 权重           : %.1f GB" % (weight_bytes(m, 16) / 1e9))
    print("  int8 权重           : %.1f GB" % (weight_bytes(m, 8) / 1e9))
    print("  int4 权重           : %.1f GB" % (weight_bytes(m, 4) / 1e9))
    print("  KV cache (fp16)     : %.1f KB / token / 序列"
          % (kv_bytes_per_token(m, 16) / 1024))
    print("  %s 装完 fp16 权重后剩余: %.1f GB"
          % (DEV, (hw["mem"] - weight_bytes(m, 16)) / 1e9))
    print("  可容纳的 KV token 数 : %.0f"
          % ((hw["mem"] * 0.9 - weight_bytes(m, 16)) / kv_bytes_per_token(m, 16)))

    sep("3. decode 的算术强度随 batch 变化（序列长度 1，忽略 KV 访存）")
    print("%8s %12s %10s %12s" % ("batch", "算术强度", "判定", "理论单步(ms)"))
    for b in (1, 4, 16, 32, 64, 128, 156, 256, 512):
        r = decode_step(m, hw, b, seqlen=1)
        print("%8d %12.1f %10s %12.2f"
              % (b, r["intensity"], r["bound"], r["t"] * 1e3))
    print("\ndecode 的算术强度约等于 batch 大小。在 %s 上，进入计算受限区" % DEV)
    print("需要 batch 超过 %.0f。" % (hw["flops"] / hw["bw"]))

    sep("4. decode 单步耗时：batch × 序列长度（%s，fp16）" % DEV)
    seqs = (512, 2048, 8192, 32768)
    print("%8s" % "batch", end="")
    for s in seqs:
        print("%14s" % ("len=%d" % s), end="")
    print()
    for b in (1, 8, 32, 64, 128):
        print("%8d" % b, end="")
        for s in seqs:
            r = decode_step(m, hw, b, s)
            print("%14s" % ("%.1f(%s)" % (realistic(r["t"]) * 1e3,
                                          "KV%.0f%%" % (100 * r["mem_kv"] / r["mem"]))),
                  end="")
        print()
    print("\n括号内为 KV cache 在总访存量中的占比。长序列大 batch 时 KV 访存")
    print("超过权重访存，此时 GQA 与 KV 量化的收益大于权重量化。")

    sep("5. 完整估算：%s，batch 32，序列长度 2048" % MODEL)
    r = decode_step(m, hw, 32, 2048)
    print("  权重访存            : %.2f GB → %.2f ms"
          % (r["mem_w"] / 1e9, r["mem_w"] / hw["bw"] * 1e3))
    print("  KV cache 访存       : %.2f GB → %.2f ms"
          % (r["mem_kv"] / 1e9, r["mem_kv"] / hw["bw"] * 1e3))
    print("  访存合计            : %.2f GB → %.2f ms"
          % (r["mem"] / 1e9, r["t_mem"] * 1e3))
    print("  计算量              : %.2f TFLOPs → %.2f ms"
          % (r["compute"] / 1e12, r["t_compute"] * 1e3))
    print("  瓶颈                : %s" % r["bound"])
    print("  理论单步            : %.2f ms" % (r["t"] * 1e3))
    print("  加效率系数 %.1f 与框架开销 %.0f ms → %.2f ms"
          % (EFFICIENCY, FRAMEWORK_OVERHEAD * 1e3, realistic(r["t"]) * 1e3))
    print("  每序列 TPOT         : %.2f ms" % (realistic(r["t"]) * 1e3))
    print("  系统输出吞吐        : %.0f token/s" % (32 / realistic(r["t"])))

    sep("6. 量化的收益（batch 32，序列长度 2048）")
    base = decode_step(m, hw, 32, 2048, 16, 16)
    print("%-22s %12s %12s %10s" % ("配置", "访存(GB)", "单步(ms)", "相对基线"))
    for label, wb, kb in (("fp16 权重 / fp16 KV", 16, 16),
                          ("int8 权重 / fp16 KV", 8, 16),
                          ("int4 权重 / fp16 KV", 4, 16),
                          ("fp16 权重 / fp8 KV", 16, 8),
                          ("int8 权重 / fp8 KV", 8, 8),
                          ("int4 权重 / fp8 KV", 4, 8)):
        r2 = decode_step(m, hw, 32, 2048, wb, kb)
        print("%-22s %12.2f %12.2f %10.2fx"
              % (label, r2["mem"] / 1e9, realistic(r2["t"]) * 1e3,
                 base["t"] / r2["t"]))
    print("\n量化在 decode 上的收益直接来自访存量的减少。注意这里只算了访存，")
    print("反量化本身的计算开销与精度损失未计入，见第 13 章。")

    sep("7. prefill：计算受限")
    print("%10s %14s %10s %12s %14s"
          % ("prompt长度", "算术强度", "判定", "理论(ms)", "attn计算占比"))
    for s in (128, 512, 2048, 8192, 32768):
        r3 = prefill_step(m, hw, s)
        print("%10d %14.0f %10s %12.1f %13.1f%%"
              % (s, r3["intensity"], r3["bound"], realistic(r3["t"]) * 1e3,
                 100 * r3["attn_flops"] / r3["compute"]))
    print("\nprefill 的算术强度远高于平衡点，因此受算力限制，量化权重收益有限。")
    print("attention 部分的计算量与长度的平方成正比，长 prompt 下占比迅速上升。")

    sep("8. 吞吐与延迟的权衡（序列长度 2048）")
    print("%8s %12s %14s %14s"
          % ("batch", "TPOT(ms)", "吞吐(token/s)", "吞吐边际增益"))
    prev = None
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        r4 = decode_step(m, hw, b, 2048)
        t = realistic(r4["t"])
        tp = b / t
        marg = "-" if prev is None else "%.2fx" % (tp / prev)
        print("%8d %12.2f %14.0f %14s" % (b, t * 1e3, tp, marg))
        prev = tp
    print("\n边际增益随 batch 增大而下降：KV 访存量与 batch 成正比，")
    print("因此权重访存被摊薄的收益逐渐被 KV 访存的增长抵消。")


if __name__ == "__main__":
    main()
