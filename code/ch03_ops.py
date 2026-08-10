#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 03 章 · 逐算子的计算量与访存量统计

第一部分按公式统计各算子在 prefill 与 decode 场景下的占比。
第二部分用 torch(CPU) 搭一个单层 Llama 结构，验证参数量与形状，
并测量 CPU 上的耗时分布作为参照。
"""

import time

# ---------------------------------------------------------------- 模型结构

LAYERS = 32
HIDDEN = 4096
HEADS = 32
KV_HEADS = 8            # 改为 32 即退化为 MHA
HEAD_DIM = 128
INTERMEDIATE = 14336
VOCAB = 128256
DTYPE_BYTES = 2

# 场景
SEQ_LEN = 8192          # 上下文长度
BATCH = 32              # decode 场景的 batch

HQ = HEADS * HEAD_DIM
HKV = KV_HEADS * HEAD_DIM

# ---------------------------------------------------------------- 公式统计


def gemm(t, k, n):
    """t 个 token 过一个 [k, n] 权重：返回 (计算量, 权重访存量)。"""
    return 2.0 * t * k * n, DTYPE_BYTES * k * n


def collect(phase):
    """phase 为 'prefill' 或 'decode'。返回算子清单。

    prefill: batch 1，处理 SEQ_LEN 个 token
    decode : batch BATCH，每序列处理 1 个 token，历史长度 SEQ_LEN
    """
    if phase == "prefill":
        b, s = 1, SEQ_LEN
        ctx = SEQ_LEN
    else:
        b, s = BATCH, 1
        ctx = SEQ_LEN
    t = b * s
    ops = []

    def add(name, flops, mem):
        ops.append(dict(name=name, flops=flops * LAYERS, mem=mem * LAYERS))

    f, m = gemm(t, HIDDEN, HQ + 2 * HKV)
    add("QKV 投影（合并）", f, m + DTYPE_BYTES * t * (HQ + 2 * HKV))
    f, m = gemm(t, HQ, HIDDEN)
    add("O 投影", f, m + DTYPE_BYTES * t * HIDDEN)
    f, m = gemm(t, HIDDEN, 2 * INTERMEDIATE)
    add("MLP gate+up（合并）", f, m + DTYPE_BYTES * t * 2 * INTERMEDIATE)
    f, m = gemm(t, INTERMEDIATE, HIDDEN)
    add("MLP down", f, m + DTYPE_BYTES * t * HIDDEN)

    if phase == "prefill":
        # 因果掩码使实际计算约为完整矩阵的一半
        attn_flops = 2.0 * 2.0 * HEADS * HEAD_DIM * s * s * 0.5 * b
        attn_mem = DTYPE_BYTES * 2 * HKV * s * b          # 写入 KV
    else:
        attn_flops = 2.0 * 2.0 * HEADS * HEAD_DIM * ctx * b
        attn_mem = DTYPE_BYTES * 2 * HKV * ctx * b        # 读取全部历史 KV
    add("attention（QK^T 与 AV）", attn_flops, attn_mem)

    add("RMSNorm ×2", 4.0 * t * HIDDEN, 2 * DTYPE_BYTES * t * HIDDEN * 2)
    add("RoPE", 6.0 * t * (HQ + HKV), 2 * DTYPE_BYTES * t * (HQ + HKV))
    add("SiLU 与逐元素乘", 5.0 * t * INTERMEDIATE,
        3 * DTYPE_BYTES * t * INTERMEDIATE)
    add("残差相加 ×2", 2.0 * t * HIDDEN, 3 * DTYPE_BYTES * t * HIDDEN * 2)

    # 非逐层部分
    f, m = gemm(b, HIDDEN, VOCAB)      # 只对每序列最后一个位置
    ops.append(dict(name="lm_head", flops=f, mem=m + DTYPE_BYTES * b * VOCAB))
    ops.append(dict(name="embedding 查表", flops=0.0,
                    mem=DTYPE_BYTES * t * HIDDEN))
    return ops


def report(phase):
    ops = collect(phase)
    tf = sum(o["flops"] for o in ops)
    tm = sum(o["mem"] for o in ops)
    if phase == "prefill":
        title = "prefill：batch 1，prompt 长度 %d" % SEQ_LEN
    else:
        title = "decode：batch %d，历史长度 %d" % (BATCH, SEQ_LEN)
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    print("%-24s %14s %8s %14s %8s"
          % ("算子", "计算量(GFLOPs)", "占比", "访存量(MB)", "占比"))
    for o in sorted(ops, key=lambda x: -x["mem"]):
        print("%-24s %14.2f %7.1f%% %14.1f %7.1f%%"
              % (o["name"], o["flops"] / 1e9, 100 * o["flops"] / tf,
                 o["mem"] / 1e6, 100 * o["mem"] / tm))
    print("%-24s %14.2f %7s %14.1f %7s"
          % ("合计", tf / 1e9, "", tm / 1e6, ""))
    print("算术强度: %.1f FLOPs/byte" % (tf / tm))

    # 按类别归并
    cat = {"矩阵乘权重": 0.0, "KV cache": 0.0, "激活与逐元素": 0.0}
    for o in ops:
        if o["name"].startswith("attention"):
            cat["KV cache"] += o["mem"]
        elif ("投影" in o["name"] or "MLP" in o["name"]
              or o["name"] == "lm_head"):
            cat["矩阵乘权重"] += o["mem"]
        else:
            cat["激活与逐元素"] += o["mem"]
    print("\n访存量按类别：")
    for k, v in cat.items():
        print("  %-14s %8.1f MB  %5.1f%%" % (k, v / 1e6, 100 * v / tm))
    return ops


# ---------------------------------------------------------------- torch 验证


def torch_check():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("\n未安装 torch，跳过第二部分。")
        return

    print("\n" + "=" * 76)
    print("torch(CPU) 验证：单层结构的参数量、形状与耗时分布")
    print("=" * 76)

    torch.manual_seed(0)
    h, i, hq, hkv = HIDDEN, INTERMEDIATE, HQ, HKV
    s = 512                      # CPU 上用较短序列，否则太慢

    qkv = nn.Linear(h, hq + 2 * hkv, bias=False)
    o = nn.Linear(hq, h, bias=False)
    gate_up = nn.Linear(h, 2 * i, bias=False)
    down = nn.Linear(i, h, bias=False)
    norm_w = torch.ones(h)

    params = sum(p.numel() for p in
                 list(qkv.parameters()) + list(o.parameters())
                 + list(gate_up.parameters()) + list(down.parameters()))
    print("单层参数量（不含归一化）: %.2f M" % (params / 1e6))
    print("公式预期                : %.2f M"
          % ((h * (hq + 2 * hkv) + hq * h + h * 2 * i + i * h) / 1e6))
    print("整模型 %d 层估计         : %.2f B  （不含 embedding 与 lm_head）"
          % (LAYERS, params * LAYERS / 1e9))
    print("加上 embedding 与 lm_head: %.2f B"
          % ((params * LAYERS + 2 * VOCAB * h) / 1e9))

    x = torch.randn(1, s, h)

    def rmsnorm(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * norm_w

    timings = {}

    def timed(name, fn, repeat=3):
        fn()
        t0 = time.perf_counter()
        for _ in range(repeat):
            out = fn()
        timings[name] = (time.perf_counter() - t0) / repeat
        return out

    xn = timed("RMSNorm", lambda: rmsnorm(x))
    qkv_out = timed("QKV 投影", lambda: qkv(xn))
    q, k, v = qkv_out.split([hq, hkv, hkv], dim=-1)
    q = q.view(1, s, HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(1, s, KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(1, s, KV_HEADS, HEAD_DIM).transpose(1, 2)
    rep = HEADS // KV_HEADS
    k = k.repeat_interleave(rep, dim=1)          # GQA：KV 头广播到 Q 头
    v = v.repeat_interleave(rep, dim=1)

    mask = torch.full((s, s), float("-inf")).triu(1)

    def attn():
        w = (q @ k.transpose(-1, -2)) / (HEAD_DIM ** 0.5) + mask
        return (w.softmax(-1) @ v)

    a = timed("attention", attn)
    a = a.transpose(1, 2).reshape(1, s, hq)
    ao = timed("O 投影", lambda: o(a))
    xr = ao + x
    xn2 = rmsnorm(xr)
    gu = timed("MLP gate+up", lambda: gate_up(xn2))
    g, u = gu.chunk(2, dim=-1)
    act = timed("SiLU 与乘", lambda: torch.nn.functional.silu(g) * u)
    out = timed("MLP down", lambda: down(act))
    out = out + xr

    print("\n形状检查（batch 1，序列 %d）" % s)
    print("  QKV 输出 : %s  （预期最后一维 %d）" % (list(qkv_out.shape), hq + 2 * hkv))
    print("  q        : %s" % list(q.shape))
    print("  k (广播后): %s  （GQA 把 %d 个 KV 头广播到 %d 个 Q 头）"
          % (list(k.shape), KV_HEADS, HEADS))
    print("  层输出   : %s  （与输入同形）" % list(out.shape))

    total = sum(timings.values())
    print("\nCPU 单层耗时分布（仅作算子清单的参照，不能推断 GPU 分布）")
    for name, t in sorted(timings.items(), key=lambda kv: -kv[1]):
        print("  %-14s %8.2f ms  %5.1f%%" % (name, t * 1e3, 100 * t / total))


def main():
    print("模型: %d 层, hidden %d, %d 头, %d KV 头, head_dim %d, I=%d, V=%d"
          % (LAYERS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTERMEDIATE, VOCAB))
    report("prefill")
    report("decode")
    torch_check()
    print("\n观察建议")
    print("  1. 把 SEQ_LEN 从 8192 改为 512 与 32768，看 attention 在 prefill")
    print("     计算量中的占比变化。")
    print("  2. 把 KV_HEADS 从 8 改为 32（MHA），看 decode 场景下 KV cache")
    print("     访存占比的变化，这是 GQA 的收益来源。")


if __name__ == "__main__":
    main()
