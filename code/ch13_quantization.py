#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 13 章 · 量化的误差实验

用 torch(CPU) 做真实的量化与反量化，测量误差：
  1. 量化粒度：per-tensor / per-channel / per-group
  2. 异常值对量化的影响
  3. 误差在矩阵乘中的传播
  4. KV cache 量化的误差
  5. 各方案的显存与访存收益
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch02_roofline as rl        # noqa: E402

torch.manual_seed(0)


def sep(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- 量化

def quant_dequant(x, bits, group=None, dim=-1, symmetric=True):
    """量化后立即反量化，返回重建的张量。

    group=None  : per-tensor（整个张量共用一组缩放参数）
    group=0     : per-channel（沿 dim 每一行一组）
    group=k     : 每 k 个元素一组
    """
    qmax = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
    shape = x.shape
    if group is None:
        xr = x.reshape(1, -1)
    elif group == 0:
        xr = x.reshape(shape[0], -1) if dim != 0 else x.reshape(-1, shape[-1]).T
    else:
        assert x.numel() % group == 0
        xr = x.reshape(-1, group)

    if symmetric:
        scale = xr.abs().amax(dim=-1, keepdim=True) / qmax
        scale = torch.clamp(scale, min=1e-12)
        q = torch.clamp(torch.round(xr / scale), -qmax - 1, qmax)
        out = q * scale
    else:
        lo = xr.amin(dim=-1, keepdim=True)
        hi = xr.amax(dim=-1, keepdim=True)
        scale = torch.clamp((hi - lo) / qmax, min=1e-12)
        zp = torch.round(-lo / scale)
        q = torch.clamp(torch.round(xr / scale) + zp, 0, qmax)
        out = (q - zp) * scale
    return out.reshape(shape)


def fp8_e4m3(x):
    """模拟 e4m3 格式：4 位指数 3 位尾数，动态范围大但精度低。"""
    x = x.to(torch.float32)
    sign = torch.sign(x)
    a = x.abs()
    a = torch.clamp(a, min=2 ** -9, max=448.0)      # e4m3 的可表示范围
    e = torch.floor(torch.log2(a))
    m = a / torch.pow(2.0, e)                        # 尾数在 [1, 2)
    m = torch.round(m * 8) / 8                       # 3 位尾数
    return sign * m * torch.pow(2.0, e)


def rel_err(a, b):
    return ((a - b).norm() / a.norm()).item()


# ---------------------------------------------------------------- 1


def part1():
    sep("1. 量化粒度对误差的影响（权重矩阵 4096 × 4096，正态分布）")
    w = torch.randn(4096, 4096)
    print("%-24s %8s %14s %16s"
          % ("方案", "位宽", "相对误差", "每 4096 元素的缩放参数数"))
    rows = [
        ("per-tensor 对称", 8, None),
        ("per-channel 对称", 8, 0),
        ("per-group 128 对称", 8, 128),
        ("per-tensor 对称", 4, None),
        ("per-channel 对称", 4, 0),
        ("per-group 128 对称", 4, 128),
        ("per-group 32 对称", 4, 32),
    ]
    for name, bits, g in rows:
        r = quant_dequant(w, bits, g)
        n_scale = 1 if g is None else (1 if g == 0 else 4096 // g)
        print("%-24s %8d %14.5f %16s"
              % (name, bits, rel_err(w, r),
                 "1/整个张量" if g is None else
                 ("1/行" if g == 0 else "%d/行" % n_scale)))
    print("\n粒度越细误差越小，代价是缩放参数本身占用的空间与访存。")
    print("per-group 128 的 int4 方案：每 128 个 4 位权重配一个 16 位缩放值，")
    print("等效位宽 = 4 + 16/128 = 4.125 位，是当前 int4 方案的常见配置。")


# ---------------------------------------------------------------- 2


def part2():
    sep("2. 异常值：量化误差的主要来源")
    w = torch.randn(64, 4096)
    base = rel_err(w, quant_dequant(w, 8, 0))
    print("原始权重（正态分布）per-channel int8 相对误差: %.5f\n" % base)

    print("%-30s %12s %12s" % ("异常值设置", "相对误差", "相对基准"))
    for n_out, mag in ((1, 10), (1, 50), (1, 100), (10, 50), (100, 50)):
        w2 = w.clone()
        idx = torch.randperm(w.numel())[:n_out]
        w2.view(-1)[idx] = mag * torch.sign(torch.randn(n_out))
        e = rel_err(w2, quant_dequant(w2, 8, 0))
        print("%-30s %12.5f %11.1fx"
              % ("%d 个元素为 ±%d 倍标准差" % (n_out, mag), e, e / base))
    print("\n误差随异常值的幅度与数量单调上升：缩放因子由最大绝对值决定，")
    print("一个远大于其他元素的值会把其余元素挤到很少的几个量化档位上。")
    print("本例中权重是正态分布，异常值是人为注入的；真实模型的某些层")
    print("天然存在幅度差异达两个量级的通道，误差比这里更严重。")
    print("\n这是 GPTQ、AWQ、SmoothQuant 等方案要解决的核心问题：")
    print("  GPTQ       : 逐列量化，用 Hessian 信息补偿已量化列带来的误差")
    print("  AWQ        : 识别重要通道并放大它们，量化后再缩回，等效于保护它们")
    print("  SmoothQuant: 把激活的异常值按通道迁移到权重上，两者都变得好量化")


# ---------------------------------------------------------------- 3


def part3():
    sep("3. 误差在矩阵乘中的传播")
    x = torch.randn(256, 4096)
    w = torch.randn(4096, 4096) / (4096 ** 0.5)
    ref = x @ w
    print("%-34s %14s %14s" % ("方案", "权重相对误差", "输出相对误差"))
    for name, fn in (
        ("int8 per-channel", lambda t: quant_dequant(t, 8, 0)),
        ("int4 per-channel", lambda t: quant_dequant(t, 4, 0)),
        ("int4 per-group 128", lambda t: quant_dequant(t, 4, 128)),
        ("fp8 e4m3", fp8_e4m3),
    ):
        wq = fn(w)
        out = x @ wq
        print("%-34s %14.5f %14.5f"
              % (name, rel_err(w, wq), rel_err(ref, out)))
    print("\n输出误差与权重误差在同一量级：矩阵乘是线性的，误差按输入范数")
    print("放大，不会被显著抵消。多层堆叠后误差累积，这是量化影响模型质量")
    print("的机制。实际评估必须用下游任务指标，权重误差只能作为初筛。")

    print("\n只量化权重与同时量化激活的区别：")
    for name, qw, qx in (("W8A16（只量化权重）", 8, None),
                         ("W8A8（权重与激活都量化）", 8, 8),
                         ("W4A16", 4, None),
                         ("W4A8", 4, 8)):
        wq = quant_dequant(w, qw, 0)
        xq = x if qx is None else quant_dequant(x, qx, 0)
        print("  %-26s 输出相对误差 %.5f" % (name, rel_err(ref, xq @ wq)))
    print("\nW8A16 只减少访存量（decode 受益），矩阵乘仍以 fp16 进行；")
    print("W8A8 还能用 int8 的乘法单元，prefill 也受益，但误差更大。")


# ---------------------------------------------------------------- 4


def part4():
    sep("4. KV cache 量化的误差")
    n, d = 2048, 128
    k = torch.randn(n, d)
    v = torch.randn(n, d)
    q = torch.randn(1, d)

    def attn(kk, vv):
        s = (q @ kk.T) / (d ** 0.5)
        return torch.softmax(s, dim=-1) @ vv

    ref = attn(k, v)
    print("%-30s %14s %16s" % ("KV 精度", "KV 相对误差", "attention 输出误差"))
    for name, fn in (
        ("fp8 e4m3", fp8_e4m3),
        ("int8 per-token", lambda t: quant_dequant(t, 8, 0)),
        ("int8 per-tensor", lambda t: quant_dequant(t, 8, None)),
        ("int4 per-token", lambda t: quant_dequant(t, 4, 0)),
        ("int4 per-group 32", lambda t: quant_dequant(t, 4, 32)),
    ):
        kq, vq = fn(k), fn(v)
        print("%-30s %14.5f %16.5f"
              % (name, rel_err(k, kq), rel_err(ref, attn(kq, vq))))
    print("\n三点结论：")
    print("  1. 粒度比位宽更关键：int8 per-token 的误差(0.0084)优于")
    print("     fp8 e4m3(0.0389)，因为后者只有 3 位尾数且不带缩放参数。")
    print("     fp8 的优势在于动态范围大、无需存缩放参数、有硬件支持。")
    print("  2. per-token 优于 per-tensor：KV 的数值分布随位置变化，")
    print("     整块共用缩放参数会放大误差。")
    print("  3. int4 KV 的误差比 int8 高一个量级，需要更细的粒度")
    print("     并用下游任务实测验证，不能只看重建误差。")


# ---------------------------------------------------------------- 5


def part5():
    sep("5. 收益：显存与访存")
    M, HW = rl.MODELS["Llama-3-8B"], rl.HW["A100-80GB"]
    print("Llama-3-8B，A100-80GB，batch 32，序列长度 8192\n")
    base = rl.decode_step(M, HW, 32, 8192, 16, 16)
    print("%-22s %12s %12s %12s %10s"
          % ("配置", "权重(GB)", "总访存(GB)", "单步(ms)", "相对基线"))
    for label, wb, kb in (("fp16 / fp16", 16, 16),
                          ("fp8 / fp8", 8, 8),
                          ("int8 / int8", 8, 8),
                          ("int4 / fp16", 4, 16),
                          ("int4 / int8", 4, 8)):
        r = rl.decode_step(M, HW, 32, 8192, wb, kb)
        print("%-22s %12.1f %12.1f %12.2f %9.2fx"
              % (label, rl.weight_bytes(M, wb) / 1e9, r["mem"] / 1e9,
                 rl.realistic(r["t"]) * 1e3, base["t"] / r["t"]))
    print("\n注意这是访存的理论收益。实际加速比通常低于此，因为：")
    print("  1. 反量化本身有计算开销（int4 需要解包、乘缩放、加零点）")
    print("  2. 低位宽的 kernel 效率可能不如 fp16 的成熟实现")
    print("  3. prefill 是计算受限的，权重量化对它几乎没有收益")
    print("因此报告量化收益时必须区分 prefill 与 decode，且要给出实测。")


def main():
    part1()
    part2()
    part3()
    part4()
    part5()
    print("\n观察建议")
    print("  1. 第 2 节把异常值倍数从 10 改到 100，观察误差的增长，")
    print("     这解释了为什么量化方案的核心是处理异常值。")
    print("  2. 第 4 节对比 per-token 与 per-tensor 的差距，说明 KV 量化")
    print("     必须按 token 或按 head 分组，不能整块共用缩放参数。")


if __name__ == "__main__":
    main()
