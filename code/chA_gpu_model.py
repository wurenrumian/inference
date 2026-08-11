#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""附录 A · GPU 执行模型的量化分析

六部分，全部可在本机计算，用于理解 kernel 层面的行为：
  1. 内存层次的容量、带宽、延迟
  2. GEMM 分块：为什么必须分块，tile 多大合适
  3. 延迟隐藏：需要多少 warp 才能让 SM 不空转
  4. occupancy：寄存器与共享内存如何限制并发 warp 数
  5. 访存合并：为什么访问模式比访问量更重要
  6. warp 调度的离散事件模拟

硬件参数以 A100 (SM80) 为准，属于量级参考，具体以厂商文档为准。
"""

import math

# ---------------------------------------------------------------- 硬件参数

A100 = dict(
    sm=108,                     # SM 数量
    warp_size=32,
    max_warps_per_sm=64,        # 每 SM 最多驻留的 warp 数
    max_threads_per_block=1024,
    regs_per_sm=65536,          # 32 位寄存器数
    smem_per_sm=164 * 1024,     # 共享内存字节数（可配置上限）
    smem_per_block_max=163 * 1024,
    clock=1.41e9,               # 约 1.41 GHz
    hbm_bw=2.0e12,              # 2 TB/s
    l2_bw=5.0e12,               # 约 2.5 倍 HBM，量级参考
    l2_size=40 * 1024 * 1024,
    fp16_tc_flops=312e12,       # tensor core fp16
    fp32_flops=19.5e12,         # 普通单元 fp32
)

# 延迟（时钟周期），量级参考
LATENCY = [
    ("寄存器", 1, "无需访存指令"),
    ("共享内存", 30, "无 bank conflict 时"),
    ("L1 / L2 命中", 200, "L2 命中"),
    ("HBM", 500, "L2 未命中"),
    ("NVLink 对端显存", 2000, "跨卡访问"),
    ("主机内存（PCIe）", 20000, "统一内存缺页"),
]


def sep(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- 1


def part1():
    sep("1. 内存层次：容量、带宽、延迟")
    hw = A100
    print("A100 单卡：%d 个 SM，每 SM 最多驻留 %d 个 warp（%d 个线程）"
          % (hw["sm"], hw["max_warps_per_sm"],
             hw["max_warps_per_sm"] * hw["warp_size"]))
    print("整卡最大并发线程数：%d\n"
          % (hw["sm"] * hw["max_warps_per_sm"] * hw["warp_size"]))

    print("%-20s %14s %16s %20s"
          % ("层级", "容量", "延迟(周期)", "说明"))
    caps = {
        "寄存器": "%d KB/SM" % (hw["regs_per_sm"] * 4 // 1024),
        "共享内存": "%d KB/SM" % (hw["smem_per_sm"] // 1024),
        "L1 / L2 命中": "%d MB (L2)" % (hw["l2_size"] // 1024 // 1024),
        "HBM": "80 GB",
        "NVLink 对端显存": "80 GB",
        "主机内存（PCIe）": "TB 级",
    }
    for name, cyc, note in LATENCY:
        print("%-20s %14s %16d %20s" % (name, caps[name], cyc, note))

    print("\n把延迟换算成「等待期间本可以做多少次运算」：")
    print("一个 SM 每周期可发射的 fp16 tensor core 运算约 %.0f 次"
          % (hw["fp16_tc_flops"] / hw["sm"] / hw["clock"]))
    per_sm_per_cycle = hw["fp16_tc_flops"] / hw["sm"] / hw["clock"]
    for name, cyc, _ in LATENCY[:4]:
        print("  等待一次%-14s（%5d 周期）= 浪费 %8.0f 次运算的时间"
              % (name, cyc, cyc * per_sm_per_cycle))
    print("\n这就是为什么 kernel 设计的核心是把数据留在尽量高的层级，")
    print("以及在等待访存时切换到其他 warp 继续计算（第 3 节）。")


# ---------------------------------------------------------------- 2


def part2():
    sep("2. GEMM 分块：为什么必须分块")
    M, K, N = 2048, 4096, 6144     # prefill 的 QKV 投影：2048 token
    print("场景：prefill 的 QKV 投影")
    print("  输入 A: [%d, %d]（%d 个 token，hidden %d）" % (M, K, M, K))
    print("  权重 B: [%d, %d]" % (K, N))
    print("  输出 C: [%d, %d]" % (M, N))
    flops = 2.0 * M * K * N
    print("  计算量: %.2f TFLOPs" % (flops / 1e12))

    print("\n不分块（每个输出元素独立计算）时的读取量：")
    print("  每个输出元素要读 A 的一行 %d 个 + B 的一列 %d 个" % (K, K))
    naive = 2.0 * M * N * K * 2      # fp16 两字节
    print("  总读取 = 2 × %d × %d × %d × 2 字节 = %.1f TB"
          % (M, N, K, naive / 1e12))
    print("  算术强度 = %.1f FLOPs/byte（远低于平衡点 156）"
          % (flops / naive))

    print("\n分块后：每个线程块负责输出的一个 [Tm, Tn] 小块，")
    print("读入 A 的 [Tm, K] 与 B 的 [K, Tn]，产出 Tm×Tn 个结果。")
    print("同一份数据被复用 Tm 或 Tn 次。\n")
    print("%10s %16s %16s %14s %12s"
          % ("tile 边长 T", "读取量(GB)", "相对不分块", "算术强度", "判定"))
    for T in (1, 8, 16, 32, 64, 128, 256):
        read = (M * N * K * (1.0 / T + 1.0 / T)) * 2
        ai = flops / read
        print("%10d %16.1f %15.3f %14.1f %12s"
              % (T, read / 1e9, read / naive, ai,
                 "访存受限" if ai < 156 else "计算受限"))
    print("\n算术强度约等于 tile 边长 T（严格地说是 2/(1/Tm+1/Tn)）。")
    print("分块把读取量从 206 GB 降到 1.6 GB（T=128），降低两个量级。")
    print("\n注意这个模型假设每个 tile 都从 HBM 读取。实际上权重会被多个")
    print("tile 复用并命中 L2（40 MB，足以容纳本例的 48 MB 权重的大部分），")
    print("因此真实的 HBM 访存量更低、算术强度更高。这也是为什么实际 kernel")
    print("用 T=128 左右就能接近算力上限，而不需要表中显示的 256 以上。")

    print("\ntile 不能无限大——它要放进共享内存：")
    print("%10s %20s %18s %14s"
          % ("tile 边长", "需要的共享内存(KB)", "A100 上限 163KB", "每 SM 可放几块"))
    for T in (32, 64, 128, 192, 256):
        # 双缓冲：同时保存当前块与预取的下一块
        need = 2 * (T * 32 + 32 * T) * 2        # 沿 K 方向每次取 32
        fit = A100["smem_per_sm"] // need if need else 0
        print("%10d %20.1f %18s %14d"
              % (T, need / 1024, "放得下" if need <= A100["smem_per_block_max"]
                 else "放不下", fit))
    print("\n共享内存容量决定了 tile 的上限，进而决定了算术强度的上限。")
    print("这是「即使算法允许，kernel 也达不到峰值」的一个具体原因。")


# ---------------------------------------------------------------- 3


def part3():
    sep("3. 延迟隐藏：需要多少 warp")
    hw = A100
    print("SM 上有多个 warp 同时驻留。某个 warp 发出访存指令后进入等待，")
    print("调度器立刻切换到另一个就绪的 warp。只要就绪的 warp 足够多，")
    print("访存延迟就被完全掩盖。\n")

    print("设一个 warp 每次访存后能做 W 个周期的计算，访存延迟 L 周期，")
    print("则需要 ceil(L / W) 个 warp 才能填满 SM。\n")
    print("%-14s %12s %16s %20s"
          % ("访存来源", "每次计算W", "需要的warp数", "A100 上限 64 够不够"))
    for L, name in ((30, "共享内存"), (200, "L2"), (500, "HBM")):
        for W in (1, 4, 16, 64):
            need = math.ceil(L / float(W))
            print("%-10s(%3d) %12d %16d %20s"
                  % (name, L, W, need,
                     "够" if need <= hw["max_warps_per_sm"]
                     else "不够，缺 %d 个" % (need - hw["max_warps_per_sm"])))
    print("\ndecode 的 GEMV 中，每读一个权重元素只做 1 次乘加（W 约等于 1），")
    print("因此需要约 500 个 warp 才能掩盖 HBM 延迟，而上限是 64。")
    print("这就是 decode 达不到理论带宽的直接原因——SM 大部分时间在等访存。")

    print("\n上表是「刚好填满」所需的 warp 数，是一个下界估计。")
    print("实际调度还要考虑计算单元本身的排队，第 6 节用离散事件模拟给出")
    print("更接近真实的利用率曲线。")
    print("\n结论：W 越小越难掩盖。decode 的 GEMV 中 W 约等于 1，")
    print("即使用满 64 个 warp 也远远不够，SM 大部分时间在等 HBM。")
    print("这是第 02 章那个 0.7 经验效率系数的来源之一。")


# ---------------------------------------------------------------- 4


def part4():
    sep("4. occupancy：什么限制了驻留的 warp 数")
    hw = A100
    print("每 SM 能驻留多少 warp，取三个限制中的最小值：")
    print("  硬件上限 64；寄存器总量 / 每线程寄存器数；")
    print("  共享内存总量 / 每线程块共享内存用量\n")
    print("%14s %16s %14s %14s %14s %12s"
          % ("每线程寄存器", "每块共享内存KB", "寄存器限制", "共享内存限制",
             "实际warp数", "occupancy"))
    for regs, smem_kb in ((32, 0), (64, 0), (128, 0), (255, 0),
                          (64, 16), (64, 48), (64, 96), (128, 48)):
        warps_per_block = 8                      # 假设每块 256 线程
        by_reg = hw["regs_per_sm"] // (regs * hw["warp_size"])
        if smem_kb:
            blocks = hw["smem_per_sm"] // (smem_kb * 1024)
            by_smem = blocks * warps_per_block
        else:
            by_smem = hw["max_warps_per_sm"]
        actual = min(hw["max_warps_per_sm"], by_reg, by_smem)
        print("%14d %16d %14d %14d %14d %11.0f%%"
              % (regs, smem_kb, by_reg, by_smem, actual,
                 100.0 * actual / hw["max_warps_per_sm"]))
    print("\nkernel 用的寄存器越多、共享内存越大，能同时驻留的 warp 越少，")
    print("延迟隐藏的能力越差。这是算子融合的一个隐藏代价：融合后的 kernel")
    print("需要更多寄存器保存中间结果，occupancy 下降，可能抵消融合的收益。")
    print("第 11 章说的「融合后的 kernel 占用的寄存器不能超限」即指此。")


# ---------------------------------------------------------------- 5


def part5():
    sep("5. 访存合并：访问模式比访问量更重要")
    print("显存事务的最小单位是 32 字节（一个 sector），实际常按 128 字节")
    print("的 cache line 计。一个 warp 的 32 个线程若访问连续地址，")
    print("硬件把它们合并成少数几次事务；若地址分散，则每个线程各发一次。\n")

    print("场景：一个 warp 的 32 个线程各读 4 字节（fp32）")
    print("%16s %16s %16s %16s"
          % ("访问模式", "实际需要(字节)", "传输(字节)", "带宽有效率"))
    useful = 32 * 4
    cases = [
        ("完全连续（步长1）", 128),
        ("步长 2", 256),
        ("步长 4", 512),
        ("步长 32", 32 * 32),
        ("完全随机", 32 * 32),
    ]
    for name, moved in cases:
        print("%16s %16d %16d %15.1f%%"
              % (name, useful, moved, 100.0 * useful / moved))

    print("\n对 KV cache 的影响：")
    bs = 16
    print("  分页后 KV 按 block 存放，block_size = %d" % bs)
    print("  块内的 %d 个 token 地址连续 → 合并访存" % bs)
    print("  跨块时地址跳跃 → 一次不连续访问")
    print("  因此每 %d 个 token 出现一次跳跃，占比 1/%d = %.1f%%"
          % (bs, bs, 100.0 / bs))
    print("\n这是第 06 章说的「block 太小会让访存过于分散」的量化形式：")
    print("%12s %20s %18s" % ("block_size", "跳跃占比", "相对连续布局的损失"))
    for b in (1, 4, 8, 16, 32, 64):
        jump = 100.0 / b
        # 粗略模型：跳跃处按半个事务浪费计
        loss = jump * 0.5
        print("%12d %19.1f%% %17.1f%%" % (b, jump, loss))
    print("\nblock_size 1（SGLang 的选择）在访存连续性上最差，")
    print("需要 kernel 侧做额外处理（例如把同一序列的块尽量分配在相邻位置）。")


# ---------------------------------------------------------------- 6


def part6():
    sep("6. warp 调度的离散事件模拟")
    print("模拟一个 SM 上 N 个 warp 交替执行「访存 → 计算」的过程，")
    print("统计计算单元的忙碌比例。\n")

    def simulate(n_warps, mem_latency, compute_cycles, n_iters=200):
        """每个 warp 循环执行：发出访存 → 等待 → 计算。

        SM 每周期只能让一个 warp 发射计算指令（简化模型）。
        """
        ready_at = [0] * n_warps          # 每个 warp 下次就绪的时刻
        remaining = [n_iters] * n_warps
        now = 0
        busy = 0
        while any(r > 0 for r in remaining):
            # 找出当前已就绪且还有工作的 warp
            cand = [i for i in range(n_warps)
                    if remaining[i] > 0 and ready_at[i] <= now]
            if not cand:
                nxt = min(ready_at[i] for i in range(n_warps)
                          if remaining[i] > 0)
                now = nxt
                continue
            w = cand[0]
            # 该 warp 占用计算单元 compute_cycles 个周期
            busy += compute_cycles
            now += compute_cycles
            remaining[w] -= 1
            # 之后发出下一次访存，等待 mem_latency
            ready_at[w] = now + mem_latency
        return busy / float(now) if now else 0.0

    print("访存延迟 500 周期（HBM），每次访存后计算 W 个周期")
    print("%10s %12s %12s %12s %12s"
          % ("warp 数", "W=1", "W=4", "W=16", "W=64"))
    for n in (1, 2, 4, 8, 16, 32, 64):
        row = [simulate(n, 500, w, 60) for w in (1, 4, 16, 64)]
        print("%10d %11.1f%% %11.1f%% %11.1f%% %11.1f%%"
              % (n, 100 * row[0], 100 * row[1], 100 * row[2], 100 * row[3]))
    print("\n表中为计算单元的忙碌比例。三个观察：")
    print("  1. W 固定时，warp 越多利用率越高，直到 warp 数 × W 接近延迟 500")
    print("  2. W=64 那一列在 8 个 warp 之后不再上升并略有回落，因为此时")
    print("     计算单元已经饱和，多出的 warp 只能排队等待发射")
    print("  3. warp 数固定在 64（硬件上限）时，W 决定了能否填满")
    print("\ndecode 的 GEMV 对应 W≈1 那一列：即使用满 64 个 warp，")
    print("计算单元的忙碌比例仍然只有个位数百分比。这与第 3 节的结论一致，")
    print("也再次说明 decode 的瓶颈不在算力。")


def part7():
    sep("7. 一步 decode 的 kernel 时间线")
    H, I, Hq, Hkv, L = 4096, 14336, 4096, 1024, 32
    B, S = 32, 2048
    eff, bw = 0.65, A100["hbm_bw"]
    launch_us = 3.0                     # 逐元素 kernel 的启动开销下界

    def t_us(nbytes):
        return nbytes / bw / eff * 1e6

    kv_per_layer = B * S * 2 * Hkv * 2  # 每层要读的 KV 字节数
    rows = [
        ("RMSNorm(融合残差)", B * H * 2 * 4),
        ("QKV 投影 (GEMV)", H * (Hq + 2 * Hkv) * 2),
        ("RoPE(融合KV写入)", B * (Hq + 2 * Hkv) * 2 * 2),
        ("PagedAttention", kv_per_layer),
        ("O 投影 (GEMV)", Hq * H * 2),
        ("RMSNorm(融合残差)", B * H * 2 * 4),
        ("MLP gate+up (GEMV)", H * 2 * I * 2),
        ("SiLU 与乘(融合)", B * 2 * I * 2 * 3),
        ("MLP down (GEMV)", I * H * 2),
    ]
    print("Llama-3-8B，batch %d，序列长度 %d，带宽利用率 %.0f%%\n"
          % (B, S, eff * 100))
    print("%-24s %12s %12s %10s" % ("kernel", "访存量(MB)", "耗时(us)", "占一层"))
    total = 0.0
    times = []
    for name, nb in rows:
        x = max(t_us(nb), launch_us)
        times.append(x)
        total += x
    for (name, nb), x in zip(rows, times):
        print("%-24s %12.1f %12.1f %9.1f%%"
              % (name, nb / 1e6, x, 100 * x / total))
    tot_bytes = sum(nb for _, nb in rows)
    print("%-24s %12.1f %12.1f %9s"
          % ("一层合计", tot_bytes / 1e6, total, "100%"))

    lm_head = t_us(1.05e9)
    sample = 300.0
    step = total * L + lm_head + sample
    print("\n%-24s %12.2f ms" % ("32 层", total * L / 1000))
    print("%-24s %12.2f ms" % ("lm_head", lm_head / 1000))
    print("%-24s %12.2f ms" % ("采样等", sample / 1000))
    print("%-24s %12.2f ms" % ("一步合计", step / 1000))

    import ch02_roofline as rl
    ref = rl.realistic(rl.decode_step(rl.MODELS["Llama-3-8B"],
                                      rl.HW["A100-80GB"], B, S)["t"]) * 1e3
    print("%-24s %12.2f ms （第 02 章按整体访存量估算）" % ("对照", ref))
    print("%-24s %12.1f%%" % ("两者相差", 100 * abs(step / 1000 - ref) / ref))

    print("\nkernel 启动开销的占比：")
    n_launch = len(rows) * L + 10
    for us in (5, 7, 10):
        print("  每次 %d us × %d 次 = %.2f ms，占一步 %.1f%%"
              % (us, n_launch, n_launch * us / 1000,
                 100 * n_launch * us / step))
    print("\n这部分与计算无关，是 CUDA Graph 要消除的目标。")
    print("模型越小、batch 越小，kernel 越快而启动开销不变，占比越高。")


def main():
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
    part7()
    print("\n" + "=" * 78)
    print("说明：硬件参数与延迟为量级参考，本机无 GPU，全部数值为计算结果")
    print("而非实测。真实值需用 nsys / ncu 在目标硬件上测量。")
    print("=" * 78)


if __name__ == "__main__":
    main()
