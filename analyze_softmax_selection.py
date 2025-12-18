#!/usr/bin/env python3
"""
分析 Softmax 采样为什么选择了 Round 7
"""
import numpy as np

# 从日志中提取的分化潜力数据
candidates = [
    {'round': 1, 'potential': 0.0357},
    {'round': 2, 'potential': 0.0519},  # 最高
    {'round': 3, 'potential': 0.0150},
    {'round': 4, 'potential': 0.0318},
    {'round': 5, 'potential': 0.0484},  # 第2
    {'round': 6, 'potential': 0.0300},
    {'round': 7, 'potential': 0.0425},  # 被选中
    {'round': 8, 'potential': 0.0433},  # 第3
    {'round': 9, 'potential': 0.0106},
]

print("=" * 80)
print("Softmax 采样分析")
print("=" * 80)

# 提取分化潜力
potentials = np.array([c['potential'] for c in candidates])

print("\n分化潜力 (原始):")
for i, c in enumerate(candidates):
    print(f"  Round {c['round']}: {c['potential']:.4f}")

# Softmax 计算（与代码一致）
potentials_shifted = potentials - np.max(potentials)  # 避免溢出
exp_values = np.exp(potentials_shifted)
probabilities = exp_values / np.sum(exp_values)

print("\nSoftmax 概率分布:")
for i, c in enumerate(candidates):
    print(f"  Round {c['round']}: p={probabilities[i]:.4f} ({probabilities[i]*100:.2f}%), potential={c['potential']:.4f}")

print("\n排序后的概率 (从高到低):")
sorted_indices = np.argsort(probabilities)[::-1]
for rank, idx in enumerate(sorted_indices, 1):
    print(f"  {rank}. Round {candidates[idx]['round']}: p={probabilities[idx]:.4f} ({probabilities[idx]*100:.2f}%)")

print("\n分析:")
print(f"  最高分化潜力: Round 2 (0.0519), 概率 {probabilities[1]:.4f}")
print(f"  实际选中: Round 7 (0.0425), 概率 {probabilities[6]:.4f}")
print(f"  Round 7 的排名: 第{list(sorted_indices).index(6) + 1}名")

# 累计概率
cumulative = np.cumsum(sorted([probabilities[i] for i in sorted_indices]))
print("\n累计概率:")
for rank, (idx, cum_prob) in enumerate(zip(sorted_indices, cumulative), 1):
    print(f"  Top {rank}: {cum_prob:.4f} ({cum_prob*100:.2f}%)")

print("\n结论:")
print("  Softmax 是概率采样，不是贪心选择。")
print(f"  Round 7 虽然只排第{list(sorted_indices).index(6) + 1}名，但仍有 {probabilities[6]*100:.2f}% 的概率被选中。")
print("  这是合理的随机性，有助于探索和避免过早收敛。")

# 模拟多次采样
print("\n" + "=" * 80)
print("模拟 1000 次采样的分布:")
print("=" * 80)

num_simulations = 1000
selection_counts = np.zeros(len(candidates))

for _ in range(num_simulations):
    selected_idx = np.random.choice(len(candidates), p=probabilities)
    selection_counts[selected_idx] += 1

print("\n选中次数 (1000次模拟):")
sorted_by_count = sorted(enumerate(selection_counts), key=lambda x: -x[1])
for idx, count in sorted_by_count:
    expected = probabilities[idx] * num_simulations
    print(f"  Round {candidates[idx]['round']}: {int(count)}次 (期望: {expected:.1f}, 理论概率: {probabilities[idx]:.4f})")

print("=" * 80)
