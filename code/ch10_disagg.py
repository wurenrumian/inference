#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 10 章 · PD 分离的资源配比与 KV 传输代价

四部分：
  1. 聚合部署下 prefill 与 decode 的资源竞争
  2. PD 分离的实例配比计算
  3. KV 传输的代价与链路要求
  4. 分层传输（逐层发送）带来的重叠收益
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch02_roofline as rl        # noqa: E402
import ch05_kvcache as kv         # noqa: E402

M = rl.MODELS["Llama-3-8B"]
HW = rl.HW["A100-80GB"]

LINKS = {
    "以太网 25Gb": 25e9 / 8,
    "以太网 100Gb": 100e9 / 8,
    "RDMA 200Gb": 200e9 / 8,
    "RDMA 400Gb": 400e9 / 8,
    "PCIe Gen4": 64e9,
    "NVLink": 600e9,
}


def kv_bytes(n):
    return kv.kv_per_token(M["layers"], M["kv_heads"], M["head_dim"]) * n


def sep(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- 1


def part1():
    sep("1. 聚合部署：prefill 与 decode 争抢同一张卡")
    print("场景：输入 %d，输出 %d，batch 32" % (4096, 256))
    tp = rl.realistic(rl.prefill_step(M, HW, 4096)["t"])
    td = rl.realistic(rl.decode_step(M, HW, 32, 4096)["t"])
    print("  一次 prefill        : %.1f ms" % (tp * 1e3))
    print("  一步 decode(batch32): %.1f ms" % (td * 1e3))
    print("  混在同一步          : %.1f ms（ITL 放大 %.1f 倍）"
          % ((tp + td) * 1e3, (tp + td) / td))
    print("\n聚合部署下每接纳一个新请求，正在 decode 的 32 个请求都要等。")
    print("请求到达率越高，decode 被打断的频率越高。")
    print("\n%14s %16s %18s" % ("到达率(req/s)", "prefill占用时间比", "decode有效时间比"))
    for rate in (1, 2, 5, 10, 20):
        busy = rate * tp
        print("%14d %15.1f%% %17.1f%%"
              % (rate, 100 * min(1.0, busy), 100 * max(0.0, 1 - busy)))
    print("\n到达率超过 1/prefill耗时 = %.1f req/s 时，卡上已无时间做 decode。"
          % (1.0 / tp))


# ---------------------------------------------------------------- 2


def part2():
    sep("2. PD 分离的实例配比")
    print("原理：分别算出单实例的 prefill 能力与 decode 能力，按负载求比例。\n")
    cases = [
        ("对话", 2048, 512, 32),
        ("长文摘要", 32768, 512, 8),
        ("代码补全", 8192, 32, 64),
        ("Agent", 16384, 256, 16),
    ]
    print("%-10s %8s %8s %8s %14s %14s %12s"
          % ("场景", "输入", "输出", "decode并发", "P实例吞吐",
             "D实例吞吐", "P:D 配比"))
    print("%-10s %8s %8s %8s %14s %14s %12s"
          % ("", "", "", "", "req/s", "req/s", ""))
    for name, n, m, batch in cases:
        tp = rl.realistic(rl.prefill_step(M, HW, n)["t"])
        p_rate = 1.0 / tp                                  # 每秒能 prefill 几个请求
        td = rl.realistic(rl.decode_step(M, HW, batch, n + m // 2)["t"])
        d_rate = batch / (td * m)                          # 每秒能完成几个请求
        # 要匹配同一到达率，实例数与单实例能力成反比
        ratio = d_rate / p_rate                            # P 实例数 : D 实例数
        if ratio >= 1:
            txt = "%.1f : 1" % ratio
        else:
            txt = "1 : %.1f" % (1.0 / ratio)
        print("%-10s %8d %8d %8d %14.2f %14.2f %12s"
              % (name, n, m, batch, p_rate, d_rate, txt))
    print("\n配比 = (1/prefill能力) : (1/decode能力)，即实例数与单实例能力成反比。")
    print("输入长输出短的场景需要更多 prefill 实例，")
    print("输入短输出长的场景需要更多 decode 实例。")
    print("聚合部署无法调整这个比例，这是 PD 分离最主要的收益。")


# ---------------------------------------------------------------- 3


def part3():
    sep("3. KV 传输的代价")
    print("PD 分离要求把 prefill 产出的全部 KV 传给 decode 实例。\n")
    print("%-14s %12s %12s %12s %12s"
          % ("链路", "2K序列", "8K序列", "32K序列", "128K序列"))
    print("%-14s %12s %12s %12s %12s"
          % ("", "(ms)", "(ms)", "(ms)", "(ms)"))
    for name, bw in LINKS.items():
        row = [kv_bytes(n) / bw * 1e3 for n in (2048, 8192, 32768, 131072)]
        print("%-14s %12.1f %12.1f %12.1f %12.1f" % tuple([name] + row))

    print("\n对照：同样长度的 prefill 计算耗时")
    print("%-14s %12s %12s %12s %12s" % ("", "2K", "8K", "32K", "128K"))
    row = [rl.realistic(rl.prefill_step(M, HW, n)["t"]) * 1e3
           for n in (2048, 8192, 32768, 131072)]
    print("%-14s %12.1f %12.1f %12.1f %12.1f" % tuple(["prefill 计算"] + row))

    print("\n判据：传输耗时应当远小于 prefill 耗时，否则分离得不偿失。")
    print("%-14s %12s %12s %12s %12s"
          % ("链路", "2K占比", "8K占比", "32K占比", "128K占比"))
    for name, bw in LINKS.items():
        cells = []
        for n in (2048, 8192, 32768, 131072):
            t_tx = kv_bytes(n) / bw
            t_pf = rl.realistic(rl.prefill_step(M, HW, n)["t"])
            cells.append("%.0f%%" % (100 * t_tx / t_pf))
        print("%-14s %12s %12s %12s %12s" % tuple([name] + cells))
    print("\n25Gb 以太网上传输耗时接近甚至超过 prefill 计算，不可用；")
    print("200Gb 以上的 RDMA 链路可以把传输压到 prefill 耗时的 10% 以内。")
    print("这是 PD 分离要求高速互联的原因。")


# ---------------------------------------------------------------- 4


def part4():
    sep("4. 分层传输：把传输与计算重叠")
    print("KV 是逐层产生的：第 1 层算完就可以开始传第 1 层的 KV，")
    print("不必等全部 %d 层算完。\n" % M["layers"])
    L = M["layers"]
    n = 8192
    t_pf = rl.realistic(rl.prefill_step(M, HW, n)["t"])
    print("序列长度 %d，prefill 计算 %.1f ms\n" % (n, t_pf * 1e3))
    print("%-14s %12s %14s %16s %10s"
          % ("链路", "传输(ms)", "串行TTFT(ms)", "逐层流水TTFT(ms)", "改善"))
    for name, bw in LINKS.items():
        t_tx = kv_bytes(n) / bw
        serial = t_pf + t_tx
        # 逐层流水：每层算完即传，只有最后一层的传输无法被计算掩盖
        overlap = t_pf + t_tx / L
        print("%-14s %12.1f %14.1f %16.1f %9.1f%%"
              % (name, t_tx * 1e3, serial * 1e3, overlap * 1e3,
                 100 * (1 - overlap / serial)))
    print("\n链路越慢，逐层流水的收益越大：25Gb 以太网上它把传输从关键路径")
    print("移除了三分之一的 TTFT，而 NVLink 上传输本来就可以忽略。")
    print("逐层传输把传输时间几乎完全隐藏，只剩最后一层的传输暴露在关键路径上。")
    print("代价是需要 %d 次小消息传输而不是 1 次大传输，对链路的小包性能"
          % M["layers"])
    print("有要求。RDMA 的 RC 传输在这种模式下优于基于 TCP 的实现，")
    print("因为它避免了每次传输的内核态开销与拷贝。")


# ---------------------------------------------------------------- 5


def part5():
    sep("5. PD 分离的适用条件")
    print("以下条件同时满足时分离才划算：\n")
    conds = [
        ("规模", "至少 4 张卡以上。2 卡分离会让每侧只剩 1 卡，失去批处理收益"),
        ("链路", "200Gb 以上 RDMA 或同机 NVLink，见第 3 节的占比表"),
        ("负载稳定", "输入输出长度分布稳定，否则配比需要频繁调整"),
        ("SLO 严格", "对 ITL 毛刺敏感。若只关心吞吐，聚合部署加 chunked "
                  "prefill 更简单"),
        ("运维能力", "需要额外的路由、KV 传输、故障处理，复杂度显著上升"),
    ]
    for k, v in conds:
        print("  %-10s %s" % (k, v))
    print("\n不满足时的替代方案：聚合部署 + chunked prefill。第 08 章的数据")
    print("显示它在接近饱和的负载下能把 ITL P99 从 342 ms 降到 189 ms，")
    print("而部署复杂度不变。")


def main():
    part1()
    part2()
    part3()
    part4()
    part5()
    print("\n观察建议")
    print("  1. 第 2 节改变输入输出长度，观察 P:D 配比的变化幅度。")
    print("     这个比例就是容量规划中两类实例的数量比。")
    print("  2. 第 3 节的占比表是选链路的依据：占比超过 20% 时，")
    print("     分离带来的收益会被传输开销抵消。")


if __name__ == "__main__":
    main()
