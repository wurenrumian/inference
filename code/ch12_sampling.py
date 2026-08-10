#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 12 章 · 采样与约束解码

四部分：
  1. logits 处理链：温度、top-k、top-p、min-p、重复惩罚
  2. 参数对分布的影响
  3. 结构化输出：用状态机生成词表掩码
  4. 批内参数不一致带来的实现问题
"""

import numpy as np

np.random.seed(7)
np.seterr(all="ignore")

VOCAB = 32
TOKENS = ["<eos>", "{", "}", "[", "]", ":", ",", '"'] + \
         [chr(ord("a") + i) for i in range(24)]


def sep(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ---------------------------------------------------------------- 1


def apply_temperature(logits, t):
    if t <= 0:
        # 温度 0 等价于贪心：把最大值之外的全部压到负无穷
        out = np.full_like(logits, -np.inf)
        out[logits.argmax()] = 0.0
        return out
    return logits / t


def apply_top_k(logits, k):
    if k <= 0 or k >= len(logits):
        return logits
    thresh = np.partition(logits, -k)[-k]
    return np.where(logits < thresh, -np.inf, logits)


def apply_top_p(logits, p):
    """核采样：保留累积概率达到 p 的最小 token 集合。"""
    if p >= 1.0:
        return logits
    probs = softmax(logits)
    order = np.argsort(-probs)
    csum = np.cumsum(probs[order])
    # 保留使累积概率首次达到 p 的位置及其之前的全部 token
    cutoff = int(np.searchsorted(csum, p)) + 1
    keep = order[:cutoff]
    out = np.full_like(logits, -np.inf)
    out[keep] = logits[keep]
    return out


def apply_min_p(logits, mp):
    """保留概率不低于「最大概率 × mp」的 token。"""
    if mp <= 0:
        return logits
    probs = softmax(logits)
    return np.where(probs < probs.max() * mp, -np.inf, logits)


def apply_repetition_penalty(logits, generated, penalty):
    if penalty == 1.0 or not generated:
        return logits
    out = logits.copy()
    for t in set(generated):
        out[t] = out[t] / penalty if out[t] > 0 else out[t] * penalty
    return out


def part1():
    sep("1. logits 处理链的执行顺序")
    logits = np.array([5.0, 4.5, 4.0, 3.0, 2.0] + [0.5] * (VOCAB - 5))
    print("原始 logits 的前 6 个: %s" % np.round(logits[:6], 2))
    p0 = softmax(logits)
    print("原始概率的前 6 个    : %s" % np.round(p0[:6], 4))
    print("候选数（概率 > 1e-4）: %d\n" % (p0 > 1e-4).sum())

    steps = [
        ("重复惩罚 1.2（已生成过 token 0）",
         lambda x: apply_repetition_penalty(x, [0], 1.2)),
        ("温度 0.7", lambda x: apply_temperature(x, 0.7)),
        ("top-k = 5", lambda x: apply_top_k(x, 5)),
        ("top-p = 0.9", lambda x: apply_top_p(x, 0.9)),
    ]
    x = logits.copy()
    for name, fn in steps:
        x = fn(x)
        p = softmax(x)
        print("%-34s 候选数 %3d  最大概率 %.4f"
              % (name, int(np.isfinite(x).sum()), p.max()))
    print("\n顺序有影响：温度在 top-k / top-p 之前应用，因为后两者要基于")
    print("温度缩放后的概率来判断截断位置。重复惩罚作用在原始 logits 上。")
    print("引擎中这个顺序是固定的，不同引擎的顺序可能不同，会导致同样的")
    print("参数在不同引擎上产生不同的输出分布。")


# ---------------------------------------------------------------- 2


def part2():
    sep("2. 参数对候选集合的影响")
    logits = np.array([5.0, 4.5, 4.0, 3.0, 2.0] + [0.5] * (VOCAB - 5))

    print("温度的影响（不做截断）")
    print("%10s %12s %12s %14s" % ("温度", "最大概率", "候选数", "分布的熵"))
    for t in (0.0, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0):
        x = apply_temperature(logits, t)
        p = softmax(x)
        ent = -(p[p > 0] * np.log(p[p > 0])).sum()
        print("%10.1f %12.4f %12d %14.3f"
              % (t, p.max(), int((p > 1e-4).sum()), ent))

    print("\ntop-p 的影响（温度 1.0）")
    print("%10s %12s %14s" % ("top_p", "保留的候选数", "被截断的概率质量"))
    for p_ in (0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
        x = apply_top_p(logits, p_)
        kept = int(np.isfinite(x).sum())
        lost = 1.0 - softmax(logits)[np.isfinite(x)].sum()
        print("%10.2f %12d %16.4f" % (p_, kept, lost))

    print("\ntop-p 的候选数随分布形状变化：分布尖锐时保留很少的 token，")
    print("分布平坦时保留很多。这是它相对 top-k 的优势——自适应。")


# ---------------------------------------------------------------- 3


class JsonObjectFSM(object):
    """一个极简的状态机，只接受形如 {"ab":"cd","ef":"gh"} 的字符串。

    真实实现（Outlines、XGrammar）由正则或 JSON Schema 编译得到，
    状态数远多于此，原理相同：每个状态对应一个允许的 token 集合。
    """

    STATES = ["start", "in_key", "after_key", "expect_colon",
              "in_value", "after_value", "done"]

    def __init__(self):
        self.state = "start"
        self.depth = 0

    def allowed(self):
        letters = [i for i, t in enumerate(TOKENS) if t.isalpha()]
        quote = TOKENS.index('"')
        if self.state == "start":
            return [TOKENS.index("{")]
        if self.state == "in_key":
            return [quote]                       # 开引号
        if self.state == "after_key":
            return letters + [quote]             # 键名字符或闭引号
        if self.state == "expect_colon":
            return [TOKENS.index(":")]
        if self.state == "in_value":
            return [quote]
        if self.state == "after_value":
            return letters + [quote]
        if self.state == "done":
            return [TOKENS.index(","), TOKENS.index("}")]
        return []

    def step(self, tok):
        c = TOKENS[tok]
        if self.state == "start" and c == "{":
            self.state = "in_key"
        elif self.state == "in_key" and c == '"':
            self.state = "after_key"
        elif self.state == "after_key":
            self.state = "expect_colon" if c == '"' else "after_key"
        elif self.state == "expect_colon" and c == ":":
            self.state = "in_value"
        elif self.state == "in_value" and c == '"':
            self.state = "after_value"
        elif self.state == "after_value":
            self.state = "done" if c == '"' else "after_value"
        elif self.state == "done":
            self.state = "in_key" if c == "," else "end"
        return self.state


def part3():
    sep("3. 结构化输出：状态机生成的词表掩码")
    fsm = JsonObjectFSM()
    rng = np.random.RandomState(1)
    out = []
    print("%-14s %8s %-34s %s" % ("状态", "允许数", "允许的 token", "采样结果"))
    for _ in range(14):
        logits = rng.randn(VOCAB) * 2
        allow = fsm.allowed()
        if not allow:
            break
        mask = np.full(VOCAB, -np.inf)
        mask[allow] = 0.0
        masked = logits + mask                  # 掩码即加 -inf
        probs = softmax(masked)
        tok = int(rng.choice(VOCAB, p=probs))
        shown = "".join(TOKENS[i] for i in allow[:12])
        print("%-14s %8d %-34s %s"
              % (fsm.state, len(allow), shown[:32], TOKENS[tok]))
        out.append(TOKENS[tok])
        if fsm.step(tok) == "end":
            break
    print("\n生成结果: %s" % "".join(out))
    print("\n掩码把不合法的 token 的 logit 置为 -inf，采样时它们的概率为 0。")
    print("因此结构化输出不是「事后校验重试」，而是从根本上无法生成非法结果。")
    print("代价是：")
    print("  1. 每步都要计算当前状态的允许集合，需要预编译与缓存")
    print("  2. 掩码是一个词表大小的向量，词表 15 万时每步每请求 600 KB")
    print("  3. 若语法与模型的倾向冲突，会降低输出质量（模型想说的被禁止）")


# ---------------------------------------------------------------- 4


def part4():
    sep("4. 批内采样参数不一致的问题")
    print("同一个 batch 中不同请求的采样参数不同，例如：\n")
    reqs = [
        ("req0", dict(temperature=0.0, top_p=1.0, top_k=0)),
        ("req1", dict(temperature=0.7, top_p=0.9, top_k=50)),
        ("req2", dict(temperature=1.2, top_p=0.95, top_k=0)),
        ("req3", dict(temperature=0.7, top_p=0.9, top_k=50)),
    ]
    for name, p in reqs:
        print("  %-6s %s" % (name, p))
    print("\nlogits 张量的形状是 [batch, 词表]，但每一行要用不同的参数处理。")
    print("三种实现方式：")
    print("  逐请求循环   : 简单，但 batch 大时 kernel 启动次数与 batch 成正比")
    print("  分组批处理   : 参数相同的请求分为一组，组内向量化。上例中")
    print("                 req1 与 req3 参数相同，可合并，共 3 组")
    print("  全向量化     : 把参数本身做成张量（每行一个温度值），")
    print("                 用向量化的比较与掩码实现 top-k / top-p")
    print("\n引擎倾向于第三种。难点在 top-k：每行的 k 不同，需要用排序后")
    print("按行索引的方式实现，而不能直接用 topk 算子。")
    print("\n还有一个工程细节：贪心解码（温度 0）的结果应当可复现，但当它")
    print("与其他请求同批时，batch 大小的变化会改变矩阵乘的归约顺序，")
    print("浮点误差导致 argmax 可能不同。因此「相同输入相同输出」在批处理")
    print("推理中不是天然成立的，需要专门的确定性配置来保证。")


def main():
    part1()
    part2()
    part3()
    part4()
    print("\n观察建议")
    print("  1. 第 1 节调换处理顺序（先 top-k 再温度），观察候选集合的变化。")
    print("  2. 第 3 节把 logits 改为强烈偏向某个非法 token，观察掩码")
    print("     如何强制模型改变输出，以及这对输出质量的潜在影响。")


if __name__ == "__main__":
    main()
