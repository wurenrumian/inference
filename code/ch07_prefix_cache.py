#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 07 章 · 前缀缓存：哈希块表与基数树

实现两种前缀共享方案并在同一负载上对比：
  1. 哈希块表（vLLM 的做法）：块内容哈希 + 复用池 + LRU 淘汰
  2. 基数树（SGLang RadixAttention 的做法）：按 token 前缀建树

指标为 prefill 阶段被跳过的 token 比例，即节省的计算量。
"""

import random
from collections import OrderedDict

BLOCK_SIZE = 16


# ---------------------------------------------------------------- 负载


def make_workload(seed=1, n_req=800):
    """构造一个贴近真实的负载：

    - 4 个不同的 system prompt，长度 200-800，按 40/30/20/10 的比例出现
    - 30% 的请求是多轮对话的后续轮，共享前一轮的完整历史
    - 其余为独立的用户问题，长度 20-200
    """
    rng = random.Random(seed)
    sys_prompts = []
    tok = 100000
    for ln in (760, 420, 250, 210):
        sys_prompts.append(list(range(tok, tok + ln)))
        tok += ln

    weights = [0.4, 0.3, 0.2, 0.1]
    sessions = {}
    reqs, kinds = [], []
    for i in range(n_req):
        r = rng.random()
        acc, si = 0.0, 0
        for k, w in enumerate(weights):
            acc += w
            if r <= acc:
                si = k
                break
        sp = sys_prompts[si]

        if sessions and rng.random() < 0.30:      # 多轮对话的后续轮
            key = rng.choice(list(sessions.keys()))
            hist = sessions[key]
            new = [rng.randrange(1000, 90000) for _ in range(rng.randint(20, 120))]
            seq = hist + new
            full = seq + [rng.randrange(1000, 90000)
                          for _ in range(rng.randint(30, 200))]
            sessions[key] = full
            kinds.append("后续轮")
        else:                                     # 新会话
            new = [rng.randrange(1000, 90000) for _ in range(rng.randint(20, 200))]
            seq = sp + new
            full = seq + [rng.randrange(1000, 90000)
                          for _ in range(rng.randint(30, 200))]
            sessions[i] = full
            kinds.append("首轮")
        reqs.append((seq, full))
    return reqs, sys_prompts, kinds


# ---------------------------------------------------------------- 哈希块表


class HashedBlockCache(object):
    """块内容哈希 + LRU 复用池。

    每个块的哈希由「前一个块的哈希 + 本块的 token」共同决定，因此哈希
    相同即代表从序列开头到本块结尾的全部内容都相同。这是能直接用哈希
    判断前缀相同的原因。
    """

    def __init__(self, capacity_blocks, block_size=BLOCK_SIZE):
        self.capacity = capacity_blocks
        self.bs = block_size
        self.pool = OrderedDict()      # hash → 物理块号（LRU，末尾最新）
        self.next_block = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _block_hashes(self, tokens):
        """只有完整的块才参与哈希，最后不足一块的部分不缓存。"""
        hs, prev = [], 0
        for i in range(0, len(tokens) - self.bs + 1, self.bs):
            prev = hash((prev, tuple(tokens[i:i + self.bs])))
            hs.append(prev)
        return hs

    def lookup_and_insert(self, prompt, full=None):
        """用 prompt 查询命中长度，用 full（prompt + 已生成的输出）建缓存。

        请求结束时整个序列的 KV 都在池中，都可以被后续请求复用，因此
        插入的是 full 而不是 prompt。
        """
        if full is None:
            full = prompt
        matched = 0
        for h in self._block_hashes(prompt):
            if h in self.pool:
                self.pool.move_to_end(h)
                self.hits += 1
                matched += 1
            else:
                break
        for h in self._block_hashes(full):
            if h in self.pool:
                self.pool.move_to_end(h)
                continue
            self.misses += 1
            if len(self.pool) >= self.capacity:
                self.pool.popitem(last=False)      # 淘汰最久未用
                self.evictions += 1
            self.pool[h] = self.next_block
            self.next_block += 1
        return matched * self.bs


# ---------------------------------------------------------------- 基数树


class RadixNode(object):
    __slots__ = ("tokens", "children", "last_used")

    def __init__(self, tokens):
        self.tokens = tokens          # 本节点承载的 token 段（压缩路径）
        self.children = {}            # 首 token → 子节点
        self.last_used = 0


class RadixTree(object):
    """按 token 前缀建立的压缩前缀树。匹配粒度为 1 个 token。"""

    def __init__(self, capacity_tokens):
        self.root = RadixNode([])
        self.capacity = capacity_tokens
        self.size = 0
        self.clock = 0
        self.evicted = 0

    def match(self, tokens):
        """返回已缓存的最长前缀长度。"""
        node, i = self.root, 0
        self.clock += 1
        node.last_used = self.clock
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            seg = child.tokens
            j = 0
            while j < len(seg) and i + j < len(tokens) and seg[j] == tokens[i + j]:
                j += 1
            child.last_used = self.clock
            i += j
            if j < len(seg):          # 部分匹配，无法继续下探
                break
            node = child
        return i

    def insert(self, tokens):
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                leaf = RadixNode(tokens[i:])
                leaf.last_used = self.clock
                node.children[tokens[i]] = leaf
                self.size += len(leaf.tokens)
                break
            seg = child.tokens
            j = 0
            while j < len(seg) and i + j < len(tokens) and seg[j] == tokens[i + j]:
                j += 1
            if j < len(seg):          # 需要分裂节点
                tail = RadixNode(seg[j:])
                tail.children = child.children
                tail.last_used = child.last_used
                child.tokens = seg[:j]
                child.children = {seg[j]: tail}
            node = child
            node.last_used = self.clock
            i += j
        self._evict()

    def _evict(self):
        """容量超限时淘汰最久未用的叶子节点。"""
        while self.size > self.capacity:
            leaf, parent, key = self._oldest_leaf(self.root, None, None)
            if leaf is None or parent is None:
                break
            self.size -= len(leaf.tokens)
            self.evicted += len(leaf.tokens)
            del parent.children[key]

    def _oldest_leaf(self, node, parent, key):
        if not node.children:
            return node, parent, key
        best = (None, None, None)
        best_t = None
        for k, c in node.children.items():
            leaf, p, kk = self._oldest_leaf(c, node, k)
            if leaf is not None and (best_t is None or leaf.last_used < best_t):
                best, best_t = (leaf, p, kk), leaf.last_used
        return best


def sep(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------- 实验


def run_hash(reqs, capacity_blocks, bs=BLOCK_SIZE):
    c = HashedBlockCache(capacity_blocks, bs)
    total = saved = 0
    for prompt, full in reqs:
        total += len(prompt)
        saved += c.lookup_and_insert(prompt, full)
    return saved, total, c


def run_radix(reqs, capacity_tokens):
    tree = RadixTree(capacity_tokens)
    total = saved = 0
    for prompt, full in reqs:
        saved += tree.match(prompt)
        total += len(prompt)
        tree.insert(full)
    return saved, total, tree


def part1(reqs):
    sep("1. 三种方案在同一负载上的对比")
    total_tokens = sum(len(p) for p, _ in reqs)
    cap_tokens = 200000
    print("负载：%d 个请求，共 %d 个 prompt token" % (len(reqs), total_tokens))
    print("缓存容量：%d 个 token 的 KV 空间\n" % cap_tokens)

    print("%-24s %14s %12s %14s" % ("方案", "跳过的token", "节省比例", "剩余需prefill"))
    print("%-24s %14d %11.1f%% %14d" % ("无前缀缓存", 0, 0.0, total_tokens))

    s, t, c = run_hash(reqs, cap_tokens // BLOCK_SIZE)
    print("%-24s %14d %11.1f%% %14d"
          % ("哈希块表 (bs=%d)" % BLOCK_SIZE, s, 100.0 * s / t, t - s))

    s2, t2, tree = run_radix(reqs, cap_tokens)
    print("%-24s %14d %11.1f%% %14d"
          % ("基数树 (粒度 1 token)", s2, 100.0 * s2 / t2, t2 - s2))

    print("\n基数树的匹配粒度是 1 个 token，哈希块表是 %d 个 token，"
          % BLOCK_SIZE)
    print("因此基数树能多命中每个前缀末尾不足一块的部分。差距为 %.1f 个百分点。"
          % (100.0 * s2 / t2 - 100.0 * s / t))
    print("代价是树的维护开销与实现复杂度高于哈希表。")


def part2(reqs):
    sep("2. block_size 对命中率的影响（哈希块表）")
    print("%12s %14s %12s %16s"
          % ("block_size", "跳过的token", "节省比例", "相对 bs=1 的损失"))
    base = None
    for bs in (1, 4, 8, 16, 32, 64, 128):
        s, t, c = run_hash(reqs, 200000 // bs, bs)
        r = 100.0 * s / t
        if base is None:
            base = r
        print("%12d %14d %11.1f%% %15.1fpp" % (bs, s, r, base - r))
    print("\nblock 越大，前缀末尾不对齐的部分越多，命中率越低。")
    print("这是 SGLang 选择 token 粒度、vLLM 选择 16 的分歧点：")
    print("  token 粒度命中率高，但索引结构复杂、kernel 访存更分散；")
    print("  块粒度实现简单，损失有限。")


def part3(reqs):
    sep("3. 缓存容量对命中率的影响")
    print("%14s %14s %12s %14s"
          % ("容量(token)", "跳过的token", "节省比例", "淘汰的块数"))
    for cap in (2000, 10000, 50000, 100000, 200000, 500000, 2000000):
        s, t, c = run_hash(reqs, cap // BLOCK_SIZE)
        print("%14d %14d %11.1f%% %14d" % (cap, s, 100.0 * s / t, c.evictions))
    print("\n容量太小时热点前缀会被冷数据挤掉，命中率下降。")
    print("前缀缓存与 KV cache 共用同一块显存：缓存留得多，可用于")
    print("运行中请求的空间就少。这是一个需要按负载调的比例。")


def part4(reqs, sys_prompts, kinds):
    sep("4. 收益的来源分解：按请求类型")
    c = HashedBlockCache(200000 // BLOCK_SIZE, BLOCK_SIZE)
    stat = {}
    for (prompt, full), k in zip(reqs, kinds):
        m = c.lookup_and_insert(prompt, full)
        d = stat.setdefault(k, [0, 0, 0])
        d[0] += 1
        d[1] += len(prompt)
        d[2] += m
    print("%-10s %8s %14s %14s %12s"
          % ("类型", "请求数", "prompt token", "跳过的token", "节省比例"))
    for k in ("首轮", "后续轮"):
        n, tot, sv = stat[k]
        print("%-10s %8d %14d %14d %11.1f%%"
              % (k, n, tot, sv, 100.0 * sv / tot))
    tot = sum(v[1] for v in stat.values())
    sv = sum(v[2] for v in stat.values())
    print("%-10s %8d %14d %14d %11.1f%%"
          % ("合计", len(reqs), tot, sv, 100.0 * sv / tot))
    print("\n首轮请求的命中来自共享的 system prompt；")
    print("后续轮请求的命中来自上一轮的完整历史，因此比例接近 100%。")
    print("\n前缀缓存的收益完全取决于负载的重复度：")
    print("  相同 system prompt 的大量请求 —— 收益高")
    print("  多轮对话（每轮共享前面全部历史）—— 收益高")
    print("  完全独立的一次性请求 —— 收益为 0")
    print("因此报告前缀缓存的收益时必须说明负载构成，否则无法比较。")


def part5():
    sep("5. 正确性验证：命中不改变输出")
    a = [1, 2, 3, 4, 5, 6, 7, 8] * 4
    b = a[:16] + [99] * 16
    c = HashedBlockCache(1000, 8)
    print("序列 A 长度 %d，序列 B 与 A 共享前 16 个 token" % len(a))
    m1 = c.lookup_and_insert(a)
    m2 = c.lookup_and_insert(b)
    print("  A 入缓存，命中 %d 个 token（首次，应为 0）" % m1)
    print("  B 查询，命中 %d 个 token（应为 16）" % m2)
    assert m1 == 0 and m2 == 16, "前缀匹配长度不符合预期"

    d = list(a)
    d[3] = 12345                      # 改动第 4 个 token
    m3 = c.lookup_and_insert(d)
    print("  把 A 的第 4 个 token 改掉后再查，命中 %d（应为 0）" % m3)
    assert m3 == 0, "内容不同的块不应命中"
    print("\n块哈希由「前块哈希 + 本块 token」链式计算，因此任何一个 token")
    print("不同都会让其后所有块的哈希不同，不会误命中。")
    print("KV 只由 token 内容与位置决定，与采样参数无关，因此命中同一前缀")
    print("的请求即使 temperature 不同，复用 KV 也不会改变输出分布。")


def main():
    reqs, sys_prompts, kinds = make_workload()
    part1(reqs)
    part2(reqs)
    part3(reqs)
    part4(reqs, sys_prompts, kinds)
    part5()
    print("\n观察建议")
    print("  1. 把 make_workload 中多轮对话的比例从 0.30 改为 0，")
    print("     命中率会降到只剩 system prompt 的贡献。")
    print("  2. 把 4 个 system prompt 改为 100 个，模拟多租户场景，")
    print("     观察容量压力下命中率的变化。")


if __name__ == "__main__":
    main()
