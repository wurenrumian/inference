#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 06 章 · block manager 的实现与显存利用率对比

包含：
  1. 一个可用的 block 分配器：free list、引用计数、写时复制
  2. 三种显存管理方式的利用率对比（按最大长度预留 / 按需连续 / 分页）
  3. block_size 对内部碎片与 block table 长度的影响
  4. fork 场景（并行采样、beam search）下的共享与写时复制
"""

import random

BLOCK_SIZE = 16


# ---------------------------------------------------------------- 分配器


class BlockAllocator(object):
    """物理 block 池。block 编号即物理位置，池外不感知具体地址。"""

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.free = list(range(num_blocks))
        self.ref = [0] * num_blocks
        self.peak_used = 0

    @property
    def num_free(self):
        return len(self.free)

    def allocate(self):
        if not self.free:
            raise MemoryError("block 池已耗尽")
        b = self.free.pop()
        self.ref[b] = 1
        self.peak_used = max(self.peak_used, self.num_blocks - len(self.free))
        return b

    def incref(self, b):
        self.ref[b] += 1

    def decref(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free.append(b)
        elif self.ref[b] < 0:
            raise RuntimeError("block %d 的引用计数为负" % b)


class Sequence(object):
    """一个序列的逻辑视图：token 列表 + block table。"""

    _next_id = 0

    def __init__(self, alloc, prompt_len=0):
        self.alloc = alloc
        self.sid = Sequence._next_id
        Sequence._next_id += 1
        self.num_tokens = 0
        self.block_table = []          # 逻辑块号 → 物理块号
        if prompt_len:
            self.append_tokens(prompt_len)

    # -------------------------------------------------------- 分配

    def _ensure_capacity(self):
        need = (self.num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        while len(self.block_table) < need:
            self.block_table.append(self.alloc.allocate())

    def append_tokens(self, n):
        """追加 n 个 token，按需分配 block。返回新增的 block 数。"""
        before = len(self.block_table)
        self.num_tokens += n
        self._ensure_capacity()
        return len(self.block_table) - before

    def append_one(self):
        """decode 一步：追加 1 个 token。写入前检查写时复制。"""
        idx = self.num_tokens // BLOCK_SIZE
        if idx < len(self.block_table):
            phys = self.block_table[idx]
            if self.alloc.ref[phys] > 1:      # 该 block 被多个序列共享
                new = self.alloc.allocate()   # 写时复制
                self.alloc.decref(phys)
                self.block_table[idx] = new
                self.num_tokens += 1
                return "copy-on-write: block %d → %d" % (phys, new)
        self.num_tokens += 1
        self._ensure_capacity()
        return None

    def fork(self):
        """派生一个共享全部已有 block 的新序列（并行采样 / beam search）。"""
        child = Sequence(self.alloc)
        child.num_tokens = self.num_tokens
        child.block_table = list(self.block_table)
        for b in child.block_table:
            self.alloc.incref(b)
        return child

    def free(self):
        for b in self.block_table:
            self.alloc.decref(b)
        self.block_table = []
        self.num_tokens = 0

    # -------------------------------------------------------- 视图

    def slot_of(self, pos):
        """逻辑位置 pos 对应的物理槽位编号，attention kernel 用它寻址。"""
        return self.block_table[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE

    def internal_fragmentation(self):
        """已分配但未使用的槽位数。"""
        return len(self.block_table) * BLOCK_SIZE - self.num_tokens


def sep(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------- 1


def part1():
    sep("1. 基本操作：分配、寻址、释放")
    alloc = BlockAllocator(num_blocks=32)
    s = Sequence(alloc, prompt_len=40)
    print("序列有 40 个 token，block_size=%d" % BLOCK_SIZE)
    print("  block table      : %s" % s.block_table)
    print("  占用 block 数     : %d（40 个 token 需要 ceil(40/16)=3 块）"
          % len(s.block_table))
    print("  内部碎片          : %d 个槽位" % s.internal_fragmentation())
    print("  逻辑位置 0  → 槽位 %d" % s.slot_of(0))
    print("  逻辑位置 20 → 槽位 %d" % s.slot_of(20))
    print("  逻辑位置 39 → 槽位 %d" % s.slot_of(39))
    print("  池中剩余 block   : %d" % alloc.num_free)
    print("\n物理 block 不连续，attention kernel 通过 block table 间接寻址。")
    s.free()
    print("释放后剩余 block : %d" % alloc.num_free)


# ---------------------------------------------------------------- 2


def gen_workload(seed=3, n_req=3000, max_len=4096):
    """生成请求序列：长度服从对数正态分布，随机的到达与结束顺序。"""
    rng = random.Random(seed)
    reqs = [min(max_len, max(16, int(rng.lognormvariate(6.0, 1.0))))
            for _ in range(n_req)]
    # 事件：(到达, 请求号) 与 (结束, 请求号)，结束时刻随机落在其后
    events = []
    for i, _ in enumerate(reqs):
        events.append((i, "arrive", i))
        events.append((i + rng.randint(20, 200), "finish", i))
    events.sort()
    return reqs, events


def part2():
    sep("2. 同一容量下三种管理方式能服务多少请求")
    max_len = 4096
    reqs, events = gen_workload(max_len=max_len)
    # 容量：按分页方式能容纳约 150 个平均长度序列来定
    avg = sum(reqs) / float(len(reqs))
    capacity = int(avg * 150)

    # --- 方式一：按 max_seq_len 预留
    used_a, live_a, rej_a, peak_a = 0, set(), 0, 0
    for _, kind, i in events:
        if kind == "arrive":
            if used_a + max_len <= capacity:
                used_a += max_len
                live_a.add(i)
                peak_a = max(peak_a, len(live_a))
            else:
                rej_a += 1
        elif i in live_a:
            live_a.discard(i)
            used_a -= max_len

    # --- 方式二：按实际长度分配连续空间，首次适配，会产生外部碎片
    free_list = [(0, capacity)]
    placed = {}
    rej_b, peak_b = 0, 0
    for _, kind, i in events:
        if kind == "arrive":
            ln = reqs[i]
            hit = -1
            for k, (st, sz) in enumerate(free_list):
                if sz >= ln:
                    hit = k
                    break
            if hit < 0:
                rej_b += 1
                continue
            st, sz = free_list[hit]
            placed[i] = (st, ln)
            if sz == ln:
                free_list.pop(hit)
            else:
                free_list[hit] = (st + ln, sz - ln)
            peak_b = max(peak_b, len(placed))
        elif i in placed:
            st, ln = placed.pop(i)
            free_list.append((st, ln))
            free_list.sort()
            merged = []
            for seg in free_list:                 # 合并相邻空闲区
                if merged and merged[-1][0] + merged[-1][1] == seg[0]:
                    merged[-1] = (merged[-1][0], merged[-1][1] + seg[1])
                else:
                    merged.append(seg)
            free_list = merged

    # --- 方式三：分页
    total_blocks = capacity // BLOCK_SIZE
    used_c, live_c, rej_c, peak_c = 0, {}, 0, 0
    for _, kind, i in events:
        if kind == "arrive":
            nb = (reqs[i] + BLOCK_SIZE - 1) // BLOCK_SIZE
            if used_c + nb <= total_blocks:
                used_c += nb
                live_c[i] = nb
                peak_c = max(peak_c, len(live_c))
            else:
                rej_c += 1
        elif i in live_c:
            used_c -= live_c.pop(i)

    n = len(reqs)
    print("%d 个请求，长度对数正态分布，平均 %.0f，最长 %d，上限 %d"
          % (n, avg, max(reqs), max_len))
    print("三种方式使用同一块容量：%d 个 token 的空间\n" % capacity)
    print("%-30s %10s %10s %12s" % ("方式", "拒绝数", "拒绝率", "峰值并发"))
    print("%-30s %10d %9.1f%% %12d"
          % ("按 max_seq_len 预留", rej_a, 100.0 * rej_a / n, peak_a))
    print("%-30s %10d %9.1f%% %12d"
          % ("按实际长度分配连续空间", rej_b, 100.0 * rej_b / n, peak_b))
    print("%-30s %10d %9.1f%% %12d"
          % ("分页（block_size=%d）" % BLOCK_SIZE, rej_c,
             100.0 * rej_c / n, peak_c))

    print("\n三种方式的浪费来源：")
    print("  预留式：内部碎片。按可能的最大长度预留，平均长度 %.0f 却占 %d，"
          % (avg, max_len))
    print("          浪费 %.0f%%。支持的上下文越长，浪费越大。"
          % (100.0 * (1 - avg / max_len)))
    print("  连续式：外部碎片。请求以不同顺序结束，空闲空间被切成小段，")
    print("          总量够但没有足够长的连续段。")
    print("  分页式：只剩每个序列最后一个 block 的未满部分，平均 %d 个槽位。"
          % (BLOCK_SIZE // 2))

    print("\n关于第二行需要说明：本模拟在分配时就知道该请求的最终长度，")
    print("因此外部碎片的影响有限（拒绝率 %.1f%%）。真实推理中输出长度"
          % (100.0 * rej_b / n))
    print("在生成结束前无法预知，连续分配只有两个选择：")
    print("  按上限预留 —— 退化为第一行，浪费 %.0f%%；"
          % (100.0 * (1 - avg / max_len)))
    print("  按当前长度分配，增长时搬移 —— 每次搬移要拷贝整个 KV，")
    print("    代价与序列长度成正比，且搬移期间无法计算。")
    print("**输出长度不可预知，是连续分配方案失效的根本原因。**")


# ---------------------------------------------------------------- 3


def part3():
    sep("3. block_size 的选择")
    global BLOCK_SIZE
    rng = random.Random(11)
    lens = [max(16, int(rng.lognormvariate(6.0, 1.0))) for _ in range(2000)]
    used = sum(lens)
    orig = BLOCK_SIZE
    print("%10s %14s %14s %16s %14s"
          % ("block_size", "占用槽位", "利用率", "平均内部碎片", "block table 长度"))
    for bs in (1, 4, 8, 16, 32, 64, 128, 256):
        BLOCK_SIZE = bs
        paged = sum(((l + bs - 1) // bs) * bs for l in lens)
        avg_frag = (paged - used) / len(lens)
        avg_tbl = sum((l + bs - 1) // bs for l in lens) / len(lens)
        print("%10d %14d %13.1f%% %16.1f %14.1f"
              % (bs, paged, 100.0 * used / paged, avg_frag, avg_tbl))
    BLOCK_SIZE = orig
    print("\nblock 越小，内部碎片越少，但 block table 越长：")
    print("  block table 本身要占显存，且 attention kernel 每次都要读它；")
    print("  block 太小还会让 KV 的物理访问过于分散，降低访存效率。")
    print("vLLM 默认取 16，兼顾两者。前缀共享的粒度也是 block，")
    print("  block 过大会降低前缀命中率（第 07 章）。")


# ---------------------------------------------------------------- 4


def part4():
    sep("4. fork 与写时复制（并行采样 n=4）")
    alloc = BlockAllocator(num_blocks=64)
    parent = Sequence(alloc, prompt_len=100)
    print("父序列 100 个 token，占用 %d 个 block: %s"
          % (len(parent.block_table), parent.block_table))
    print("池中剩余 %d 个 block" % alloc.num_free)

    kids = [parent.fork() for _ in range(4)]
    print("\nfork 出 4 个子序列后：")
    print("  池中剩余 %d 个 block（没有新增分配）" % alloc.num_free)
    print("  各子序列的 block table 与父序列相同")
    print("  最后一个 block 的引用计数: %d"
          % alloc.ref[parent.block_table[-1]])
    naive = 5 * len(parent.block_table)
    print("  不共享时需要 %d 个 block，共享后 %d 个，节省 %.0f%%"
          % (naive, len(parent.block_table),
             100.0 * (1 - len(parent.block_table) / float(naive))))

    print("\n各子序列各生成 1 个 token（写入的是同一个未满的 block）：")
    for i, k in enumerate(kids):
        msg = k.append_one()
        print("  子序列 %d: %s" % (i, msg if msg else "直接写入"))
    print("池中剩余 %d 个 block" % alloc.num_free)
    print("\n每个子序列写入时都发现该 block 被共享，各自复制一份，")
    print("共 4 次复制；父序列继续持有原块。")
    print("前面 %d 个已满的 block 不会被写入，仍然共享，引用计数为 5。"
          % (len(parent.block_table) - 1))
    print("已满 block 的引用计数: %d" % alloc.ref[parent.block_table[0]])

    for k in kids:
        k.free()
    parent.free()
    print("\n全部释放后剩余 %d 个 block（应等于池容量 64）" % alloc.num_free)
    assert alloc.num_free == 64, "引用计数有误，出现泄漏"
    print("引用计数校验通过，无泄漏。")


def main():
    part1()
    part2()
    part3()
    part4()
    print("\n观察建议")
    print("  1. 第 2 节把 max_len 从 4096 改为 32768，看预留方式的利用率")
    print("     如何随支持的上下文长度上升而崩塌。")
    print("  2. 第 3 节的表说明 block_size 是内部碎片与索引开销的折中。")


if __name__ == "__main__":
    main()
