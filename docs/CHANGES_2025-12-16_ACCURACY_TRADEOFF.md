# 分化选择准确率权衡修改说明

## 修改日期
2025-12-16 (晚)

## 修改内容

### 问题描述
原始的分化选择只基于**分化潜力**:
```python
selected = argmax_W(Split_Potential(W))
```

这会导致以下问题:
- **偏科严重但整体差**: 可能选择某个类别很强但整体准确率很低的workflow
- **忽视基础性能**: 没有考虑workflow的整体质量
- **专业化风险**: 基于差的workflow专业化,效果可能不理想

**示例场景**:
```
候选1: Round 3
  - Split_Potential = 0.10 (最高, 某类别很强)
  - Accuracy = 0.40 (整体很差)
  
候选2: Round 7
  - Split_Potential = 0.08 (略低)
  - Accuracy = 0.65 (整体很好)

问题: 原算法会选择Round 3，但基于40%准确率的workflow专业化，
      可能不如基于65%准确率的workflow专业化效果好
```

### 解决方案

添加**准确率权衡机制**:
```python
# 1. 归一化分化潜力
x = Split_Potential(W) / max(Split_Potential)

# 2. 计算修正分数（权衡潜力和准确率）
Adjusted_Score(W) = α · x + (1-α) · Accuracy

# 3. 选择修正分数最高的workflow
selected = argmax_W(Adjusted_Score(W))
```

**参数α控制权衡**:
- `α = 1.0`: 只看分化潜力 (原始行为)
- `α = 0.5`: 平衡潜力和准确率 (默认, 推荐)
- `α = 0.0`: 只看准确率 (退化为选最佳workflow)

**修改后示例**:
```
候选1: Round 3
  - Split_Potential = 0.10 (最高)
  - Accuracy = 0.40
  - Normalized_Potential = 0.10/0.10 = 1.0
  - Adjusted_Score = 0.5×1.0 + 0.5×0.40 = 0.70

候选2: Round 7
  - Split_Potential = 0.08
  - Accuracy = 0.65
  - Normalized_Potential = 0.08/0.10 = 0.8
  - Adjusted_Score = 0.5×0.8 + 0.5×0.65 = 0.725 ✓ 最高

选择: Round 7 (虽然潜力略低，但整体性能更好)
```

## 修改位置

### 1. 代码修改

**文件**: `scripts/enhanced_optimizer.py`

**Line 62-63** (添加参数):
```python
# === Differentiation Selection Weights ===
alpha_split_potential: float = 0.5,  # 分化潜力权重 (0.5 = 平衡潜力和准确率)
```

**Line 109-110** (保存参数):
```python
# Differentiation selection weights
self.alpha_split_potential = alpha_split_potential  # 分化潜力权重
```

**Line 500-530** (更新函数文档):
```python
"""
选择策略（权衡潜力和准确率）：
1. 计算每个workflow的分化潜力 Split_Potential(W)
2. 归一化: x = Split_Potential(W) / max(Split_Potential)
3. 修正分数: Adjusted_Score = α·x + (1-α)·Acc_global
4. 选择修正分数最高的workflow

参数α控制权衡:
- α=1.0: 只看分化潜力(可能选择"偏科"但性能差的workflow)
- α=0.5: 平衡潜力和准确率(默认)
- α=0.0: 只看准确率(退化为选择最佳workflow)

算法寻找"高性能偏科专家"：既有专业化潜力，整体性能也不能太差
"""
```

**Line 650-690** (选择逻辑修改):
```python
# 归一化分化潜力（以最大值归一化）
max_potential = max(potentials)
normalized_potentials = [p / max_potential if max_potential > 0 else 0 for p in potentials]

# 计算修正分数: α * normalized_potential + (1-α) * accuracy
alpha = self.alpha_split_potential
adjusted_scores = [
    alpha * norm_pot + (1 - alpha) * acc
    for norm_pot, acc in zip(normalized_potentials, accuracies)
]

# 记录修正分数到candidates
for i, cand in enumerate(candidates):
    cand['normalized_potential'] = normalized_potentials[i]
    cand['adjusted_score'] = adjusted_scores[i]

# 选择修正分数最高的工作流
selected_idx = adjusted_scores.index(max(adjusted_scores))
```

**Line 700-730** (日志输出):
```python
logger.info(f"Differentiation candidate ranking (top 3 by adjusted score):")
logger.info(f"  (α={alpha:.2f} for potential, {1-alpha:.2f} for accuracy)")
sorted_indices = sorted(range(len(candidates)), 
                       key=lambda i: candidates[i]['adjusted_score'], 
                       reverse=True)[:3]
for rank, idx in enumerate(sorted_indices, 1):
    c = candidates[idx]
    logger.info(f"  {rank}. Round {c['round']}: "
               f"adjusted={c['adjusted_score']:.4f} "
               f"(potential={c['split_potential']:.4f}, "
               f"norm_pot={c['normalized_potential']:.4f}, "
               f"acc={c['acc_global']:.4f}), "
               f"category={c['target_category']}")
```

**Line 735-750** (选择结果日志):
```python
logger.info(f"SELECTED: Round {selected['round']} for specialization")
logger.info(f"  Target Category: {selected['target_category']}")
logger.info(f"  Adjusted Score: {selected['adjusted_score']:.4f}")
logger.info(f"    ├─ Normalized Potential: {selected['normalized_potential']:.4f} (weight={alpha:.2f})")
logger.info(f"    └─ Global Accuracy: {selected['acc_global']:.4f} (weight={1-alpha:.2f})")
logger.info(f"  Raw Split Potential: {selected['split_potential']:.4f}")
```

### 2. 文档更新

#### 2.1 DIFFERENTIATION_OPERATION.md

**Line 300-380** (选择策略章节完全重写):
```markdown
### 3.2 选择策略 (准确率权衡)

**问题**: 只看分化潜力可能选择"偏科"但整体性能差的workflow

**解决方案**: 权衡分化潜力和整体准确率

[详细的算法步骤和示例]

**参数α的作用**:
- α = 1.0: 只看分化潜力
- α = 0.5: 平衡潜力和准确率 (推荐)
- α = 0.0: 只看准确率

**示例**:
[具体的计算示例，展示3个候选的对比]
```

#### 2.2 QUICK_REFERENCE.md

**Line 75-95** (公式速查更新):
```markdown
# 选择策略（权衡潜力和准确率）
x = Split_Potential(W) / max(Split_Potential)  # 归一化
Adjusted_Score(W) = α · x + (1-α) · Acc_global
selected_workflow = argmax_W(Adjusted_Score(W))

# α: 潜力权重 (默认0.5)
#   - α=1.0: 只看潜力
#   - α=0.5: 平衡
#   - α=0.0: 只看准确率
```

**Line 145-165** (新增参数表):
```markdown
### 2.3 分化选择

| 参数 | 符号 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `alpha_split_potential` | α | 0.5 | 0.0-1.0 | 分化潜力权重 |

**调参建议**:
- α=1.0: 只看分化潜力 (可能选择"偏科"但性能差的)
- α=0.5: 平衡潜力和准确率 (推荐)
- α=0.0: 只看准确率 (退化为选最佳workflow)

**权衡公式**:
Adjusted_Score = α × (Potential/MaxPotential) + (1-α) × Accuracy
```

**Line 280** (命令行参数):
```bash
python run.py \
  --alpha_split_potential 0.5
```

#### 2.3 CODE_CONSISTENCY_CHECK.md

**Line 105-125** (参数对照表更新):
```markdown
| `alpha_split_potential` | 0.5 | 0.5 | ✅ | 分化潜力权重 (新增于2025-12-16) |

**修改历史**:
- **2025-12-16 (晚)**: 添加准确率权衡机制
  - 新参数: alpha_split_potential (默认0.5)
  - 公式: Adjusted_Score = α·(Potential/MaxPotential) + (1-α)·Accuracy
  - 目的: 防止选择"偏科严重但整体差"的workflow
  - 效果: 权衡专业化潜力和整体性能
```

## 影响分析

### 对分化选择的影响

1. **不再优先选择极端偏科**:
   - 之前: 某类别很强但整体很差的workflow会被选中
   - 现在: 需要同时考虑专业化潜力和整体性能

2. **基于更好的基础进行专业化**:
   - 之前: 可能基于40%准确率的workflow专业化
   - 现在: 倾向于选择整体性能更好的workflow

3. **更稳健的专业化**:
   - 专业化后的workflow有更好的基础
   - 减少"过度专业化"的风险

### 参数α的作用

**α = 1.0 (纯潜力导向)**:
- 行为: 与原始算法相同，只看分化潜力
- 适合: 希望激进专业化，不在乎基础性能
- 风险: 可能选择整体差但某类别强的workflow

**α = 0.5 (平衡, 默认)**:
- 行为: 平衡专业化潜力和整体性能
- 适合: 大多数场景
- 效果: 既考虑专业化价值，也确保基础性能

**α = 0.0 (纯性能导向)**:
- 行为: 只看整体准确率，忽略分化潜力
- 适合: 希望保守专业化
- 退化: 相当于总是选择最佳workflow进行分化

### 实际运行影响

**预期变化**:
- 分化选择更倾向于**高性能且有专业化潜力**的workflow
- 避免选择"偏科但整体差"的workflow
- 专业化后的workflow质量更高

**数值示例**:
```
场景1: 三个候选
  R3: pot=0.10, acc=0.40 → adj=0.70 (潜力高但性能差)
  R5: pot=0.06, acc=0.58 → adj=0.67 (中庸)
  R7: pot=0.07, acc=0.62 → adj=0.75 ✓ (平衡最好)

α=1.0时会选R3, α=0.5时会选R7
```

**兼容性**:
- ✅ 完全向后兼容（默认α=0.5）
- ✅ 可以通过设置α=1.0恢复原始行为
- ✅ 不影响已有workflow
- ✅ 只影响未来的分化选择

## 调参建议

### 根据数据集特点

**数据集很难，整体性能都较低**:
```python
alpha_split_potential = 0.7  # 更看重潜力
# 因为整体性能都低，更应该关注专业化机会
```

**数据集较简单，整体性能都较高**:
```python
alpha_split_potential = 0.3  # 更看重准确率
# 因为性能都高，选择最好的基础进行专业化
```

**通用场景**:
```python
alpha_split_potential = 0.5  # 平衡（默认）
```

### 根据优化阶段

**早期阶段 (Round 1-10)**:
```python
alpha_split_potential = 0.3  # 保守
# 早期workflow质量差异大，优先选择性能好的
```

**中期阶段 (Round 10-20)**:
```python
alpha_split_potential = 0.5  # 平衡（默认）
# 中期workflow质量趋同，平衡考虑
```

**后期阶段 (Round 20+)**:
```python
alpha_split_potential = 0.7  # 激进
# 后期性能提升困难，更关注专业化机会
```

## 验证建议

### 单元测试

```python
def test_adjusted_score_calculation():
    """测试修正分数计算"""
    # 场景: 高潜力低准确率 vs 低潜力高准确率
    potentials = [0.10, 0.06]
    accuracies = [0.40, 0.65]
    
    # α=0.5
    max_pot = max(potentials)
    norm_pots = [p/max_pot for p in potentials]
    alpha = 0.5
    
    adj1 = alpha * norm_pots[0] + (1-alpha) * accuracies[0]
    adj2 = alpha * norm_pots[1] + (1-alpha) * accuracies[1]
    
    # adj1 = 0.5*1.0 + 0.5*0.40 = 0.70
    # adj2 = 0.5*0.6 + 0.5*0.65 = 0.725
    assert adj2 > adj1  # 应该选择更平衡的
```

### 集成测试

```bash
# 测试不同α值
for alpha in 0.0 0.3 0.5 0.7 1.0; do
  echo "Testing alpha=$alpha"
  python run.py \
    --dataset MATH \
    --max_rounds 10 \
    --alpha_split_potential $alpha \
    > logs/test_alpha_${alpha}.log 2>&1
done

# 比较选择结果
grep "SELECTED: Round" logs/test_alpha_*.log
```

### 日志检查

```bash
# 查看修正分数计算
grep "Differentiation candidate ranking" logs/AFlow.log -A 5

# 查看选择结果
grep "SELECTED:" logs/AFlow.log -A 10

# 验证权衡是否生效
# 检查是否选择了更平衡的候选
```

## 总结

这次修改通过添加**准确率权衡机制**,解决了只看分化潜力可能导致的"偏科严重但整体差"的问题。

**关键改进**:
- ✅ 防止选择"偏科但性能差"的workflow
- ✅ 确保专业化基于良好的基础性能
- ✅ 通过参数α灵活控制权衡程度
- ✅ 保持算法的可解释性

**核心公式**:
```
Adjusted_Score = α × (Potential/MaxPotential) + (1-α) × Accuracy
```

**参数控制**:
- α=1.0: 激进专业化 (只看潜力)
- α=0.5: 平衡策略 (默认, 推荐)
- α=0.0: 保守专业化 (只看准确率)

**效果**:
- 分化选择更加稳健
- 专业化workflow质量更高
- 避免基于差的workflow进行专业化

**后续**:
- 监控不同α值的效果
- 根据数据集特点调整α
- 考虑动态调整α (早期保守, 后期激进)
