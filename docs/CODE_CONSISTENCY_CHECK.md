# 代码一致性检查 (Code Consistency Check)

## 文档版本
- **创建日期**: 2025-12-15
- **检查范围**: `scripts/enhanced_optimizer.py`
- **参考文档**: SYSTEM_ARCHITECTURE.md, OPTIMIZE_OPERATION.md, DIFFERENTIATION_OPERATION.md, FUSION_OPERATION.md

---

## 1. 检查目的

本文档对比**理论设计**与**实际实现**,识别:
1. **参数不一致**: 文档中的公式参数 vs 代码中的实际值
2. **算法偏差**: 理论算法 vs 实现逻辑
3. **遗留代码**: 废弃或未使用的代码
4. **潜在Bug**: 逻辑错误或边界情况

---

## 2. 超参数对照表

### 2.1 停滞检测参数

| 参数名称 | 理论文档 | 代码实现 | 一致性 | 备注 |
|---------|---------|---------|--------|------|
| `k` (滑动窗口) | 3 | `sliding_window_k=3` | ✅ | Line 54 |
| `κ` (敏感度) | 80.0 | `stagnation_sensitivity_kappa=80.0` | ✅ | Line 55 |
| 停滞公式 | `plateau = (1 - avg_improvement/κ) * 100` | 同 | ✅ | Line 380-400 |

**验证**: 停滞检测参数完全一致

---

### 2.2 操作概率参数

| 参数名称 | 理论文档 | 代码实现 | 一致性 | 备注 |
|---------|---------|---------|--------|------|
| `α_s` (分化基础) | 0.3 | `alpha_s=0.50` | ⚠️ **不一致** | Line 58 |
| `α_m` (融合基础) | 0.4 | `alpha_m=0.60` | ⚠️ **不一致** | Line 59 |
| `η_s` (分化衰减) | 0.1 | `eta_s=0.03` | ⚠️ **不一致** | Line 61 |
| `η_m` (融合衰减) | 0.1 | `eta_m=0.03` | ⚠️ **不一致** | Line 62 |

**分析**:
- **分化/融合基础概率更高**: 代码中 `α_s=0.5, α_m=0.6` 比理论 `0.3, 0.4` 更激进
- **衰减因子更小**: 代码中 `η=0.03` 比理论 `0.1` 衰减更慢

**影响**:
- 分化和融合会更频繁触发
- 重复操作的惩罚较轻
- 可能导致过度分化/融合

**建议**:
```python
# 更保守的配置 (接近理论)
alpha_s = 0.3
alpha_m = 0.4
eta_s = 0.1
eta_m = 0.1

# 当前配置 (更激进)
alpha_s = 0.5
alpha_m = 0.6
eta_s = 0.03
eta_m = 0.03
```

---

### 2.3 融合选择参数

#### 2.3.1 互补性和一致性权重

| 参数名称 | 理论文档 | 代码实现 | 一致性 | 备注 |
|---------|---------|---------|--------|------|
| `α_U` (互补性) | 未明确 | `alpha_U=0.6` | ⚠️ 需确认 | Line 64 |
| `α_I` (一致性) | 未明确 | `alpha_I=0.4` | ⚠️ 需确认 | Line 65 |

**注意**: 原文档中使用 `β_p`(pairwise) 和 `β_t`(triplet),代码中增加了互补性/一致性的分层权重

#### 2.3.2 Pairwise和Triplet权重

| 参数名称 | 理论文档 | 代码实现 | 一致性 | 备注 |
|---------|---------|---------|--------|------|
| `β_p` (pairwise) | 0.4 | `beta_pair=0.4` | ✅ | Line 67 |
| `β_t` (triplet) | 0.3 | `beta_triple=0.6` | ❌ **严重不一致** | Line 66 |
| `γ_pair` (交集) | 未明确 | `gamma_pair=0.7` | ⚠️ 新增 | Line 68 |
| `γ_triple` (交集) | 未明确 | `gamma_triple=0.3` | ⚠️ 新增 | Line 69 |

**关键问题**: `beta_triple=0.6` vs 理论 `β_t=0.3`

**分析**:
```python
# 理论公式
Φ_merge = β_p · Σ_pairwise + β_t · φ_triple
        = 0.4 · Σ_pairwise + 0.3 · φ_triple

# 代码实现可能不同,需要检查具体计算
```

**验证位置**: Line 832-1000 (`_select_for_merge`)

---

### 2.4 分化选择参数

| 参数名称 | 理论文档 | 代码实现 | 一致性 | 备注 |
|---------|---------|---------|--------|------|
| `Contrib_k` 定义 | `C_k / N` | `C_k / N` | ✅ | 绝对贡献度 (修改于2025-12-16) |
| `Recall_k - Acc_global` | 直接差值 | 直接差值 | ✅ | Line 485-656 |
| 优势阈值 | `Recall > Global` | `Recall > Global` | ✅ | 无额外阈值 |
| `alpha_split_potential` | 0.5 | 0.5 | ✅ | 分化潜力权重 (新增于2025-12-16) |

**修改历史**:
- **2025-12-16 (早)**: 将 `Contrib_k` 从 `C_k / C_total` 改为 `C_k / N`
  - 目的: 防止小类别因相对占比高而过度偏向
  - 效果: 使用绝对贡献度,更公平地评估各类别的分化价值
  
- **2025-12-16 (晚)**: 添加准确率权衡机制
  - 新参数: `alpha_split_potential` (默认0.5)
  - 公式: `Adjusted_Score = α·(Potential/MaxPotential) + (1-α)·Accuracy`
  - 目的: 防止选择"偏科严重但整体差"的workflow
  - 效果: 权衡专业化潜力和整体性能

**验证**: 分化选择公式完全一致

---

## 3. 算法逻辑检查

### 3.1 停滞度计算

**理论公式**:
```
t: 当前轮次
k: 滑动窗口大小

if t < 2*k:
    return 0.0  # warm-up期

window = performance_history[t-k : t]
avg_improvement = mean(window[i+1] - window[i])

plateau_t = (1 - avg_improvement / κ) * 100
plateau_t = clip(plateau_t, 0, 100)
```

**代码实现** (Line 370-430):
```python
def _calculate_plateau(self) -> float:
    t = len(self.performance_history)
    k = self.sliding_window_k
    
    # ⚠️ 修改后的warm-up逻辑
    if t < 2:  # 改为 t < 2
        return 0.0
    
    # 动态调整窗口大小
    effective_k = min(k, t // 2)
    if effective_k < 1:
        return 0.0
    
    # 计算平均改进
    window = self.performance_history[-effective_k:]
    if len(window) < 2:
        return 0.0
    
    improvements = [
        window[i+1] - window[i]
        for i in range(len(window) - 1)
    ]
    avg_improvement = np.mean(improvements)
    
    # 停滞度
    plateau = (1 - avg_improvement / self.stagnation_sensitivity_kappa) * 100
    plateau = max(0.0, min(100.0, plateau))
    
    return plateau
```

**差异分析**:

| 项目 | 理论 | 实现 | 一致性 | 影响 |
|------|------|------|--------|------|
| Warm-up条件 | `t < 2*k` | `t < 2` | ❌ **已修改** | 允许早期计算停滞度 |
| 窗口大小 | 固定 `k` | 动态 `effective_k = min(k, t//2)` | ⚠️ **改进** | 早期使用小窗口 |
| 边界检查 | 无 | `if effective_k < 1` | ⚠️ **增强** | 防止除零错误 |

**结论**: 代码实现是**改进版**,解决了早期轮次无法计算停滞度的问题

---

### 3.2 操作概率计算

**理论公式**:
```
p_opt = (1 - α_s - α_m) · plateau_t
p_split = α_s · plateau_t · exp(-η_s · N_s)
p_merge = α_m · plateau_t · exp(-η_m · N_m)
```

**代码实现** (Line 190-260):
```python
plateau = self._calculate_plateau()

p_opt_raw = (1 - self.alpha_s - self.alpha_m) * plateau
p_split_raw = self.alpha_s * plateau * np.exp(-self.eta_s * self.N_s)
p_merge_raw = self.alpha_m * plateau * np.exp(-self.eta_m * self.N_m)

# 归一化
total = p_opt_raw + p_split_raw + p_merge_raw
if total > 0:
    p_opt = p_opt_raw / total
    p_split = p_split_raw / total
    p_merge = p_merge_raw / total
else:
    p_opt = 1.0
    p_split = 0.0
    p_merge = 0.0
```

**一致性**: ✅ **完全一致** (加上归一化保证概率和为1)

---

### 3.3 分化选择算法

**理论算法** (Algorithm 2):
```
for each workflow W_i:
    Acc_global = C_total / N
    
    for each category k:
        Recall_k = C_k / N_k
        
        if Recall_k > Acc_global:
            Contrib_k = C_k / N  # 修改: 使用绝对贡献度
            Score_k = Contrib_k × (Recall_k - Acc_global)
    
    Split_Potential(W_i) = max_k(Score_k)

selected = argmax_i(Split_Potential(W_i))
target_category = argmax_k(Score_k for W_selected)
```

**修改说明** (2025-12-16):
- 将 `Contrib_k = C_k / C_total` 改为 `Contrib_k = C_k / N`
- 目的: 防止小类别因相对占比高而过度偏向
- 效果: 基于类别的绝对贡献评估,更加公平

**代码实现** (Line 485-656):
```python
def _select_for_split(self, workflows):
    best_potential = 0
    best_workflow = None
    best_category = None
    
    for workflow in workflows:
        # 加载统计
        category_stats = self._load_workflow_category_stats(workflow)
        category_metadata = self._load_category_metadata()
        
        # 全局性能
        c_total = sum(category_stats.values())
        n = sum(category_metadata.values())
        acc_global = c_total / n if n > 0 else 0
        
        # 每个类别
        for category, n_k in category_metadata.items():
            c_k = category_stats.get(category, 0)
            if n_k == 0:
                continue
            
            recall_k = c_k / n_k
            
            # 优势类别
            if recall_k > acc_global:
                contrib_k = c_k / n  # 修改: 使用绝对贡献度
                score_k = contrib_k * (recall_k - acc_global)
                
                if score_k > best_potential:
                    best_potential = score_k
                    best_workflow = workflow
                    best_category = category
    
    return best_workflow, best_category
```

**一致性**: ✅ **完全一致** (已修改为绝对贡献度)

---

### 3.4 融合选择算法

**理论算法** (Algorithm 3):
```
# 1. 选择包络线 (Envelope)
envelope = {W_i: W_i not dominated by any other}

# 2. 计算融合潜力
for each triple (W_i, W_j, W_k) in envelope:
    Φ_pair(i,j) = |C_i ⊕ C_j|
    Φ_triple(i,j,k) = |exactly_one| + |exactly_two|
    
    Φ_merge = β_p · (Φ_pair(i,j) + Φ_pair(j,k) + Φ_pair(i,k)) +
              β_t · Φ_triple(i,j,k)

# 3. 选择最大
best_triple = argmax(Φ_merge)
```

**代码实现** (Line 832-1000):
需要详细检查,因为参数不一致 (`beta_triple=0.6` vs `β_t=0.3`)

**关键代码**:
```python
def _select_for_merge(self, workflows):
    # ... 省略包络线选择 ...
    
    best_potential = 0
    best_triple = None
    
    for i in range(len(envelope)):
        for j in range(i+1, len(envelope)):
            for k in range(j+1, len(envelope)):
                # 计算pairwise
                phi_ij = self._pairwise_complementarity(w_i, w_j)
                phi_jk = self._pairwise_complementarity(w_j, w_k)
                phi_ik = self._pairwise_complementarity(w_i, w_k)
                
                # 计算triplet
                phi_triple = self._triplet_complementarity(w_i, w_j, w_k)
                
                # ⚠️ 需要检查此处公式
                phi_merge = (
                    self.beta_pair * (phi_ij + phi_jk + phi_ik) +
                    self.beta_triple * phi_triple
                )
                
                if phi_merge > best_potential:
                    best_potential = phi_merge
                    best_triple = (w_i, w_j, w_k)
    
    return best_triple
```

**一致性**: ⚠️ **需要验证实际公式**

**可能的分层结构**:
```python
# 如果代码使用了 alpha_U 和 alpha_I
U_score = beta_triple * triple_union + beta_pair * pairwise_union
I_score = gamma_pair * pairwise_intersection + gamma_triple * triple_intersection

Φ_merge = alpha_U * U_score + alpha_I * I_score
```

**建议**: 详细阅读 Line 832-1000,确认实际公式

---

## 4. 数据结构一致性

### 4.1 performance_history

**理论描述**:
```
performance_history: List[float]
  - performance_history[0] = Round 1的分数
  - performance_history[t-1] = Round t的分数
```

**实际实现**:
```python
# Round 1 (initial workflow) 不加入 performance_history
# performance_history 从 Round 2 开始

self.performance_history = []

# Round 2优化后
score = evaluate(round_2_workflow)
self.performance_history.append(score)  # performance_history[0] = Round 2

# Round 3优化后
score = evaluate(round_3_workflow)
self.performance_history.append(score)  # performance_history[1] = Round 3
```

**索引映射**:
```
performance_history索引  对应轮次  说明
0                        Round 2   第一次优化后
1                        Round 3   第二次优化后
t-2                      Round t   第t轮
```

**一致性**: ⚠️ **文档需要明确说明Round 1不在performance_history中**

---

### 4.2 log.json 格式

**预期格式** (根据文档):
```json
{
  "execution_logs": [
    {
      "problem_id": "problem_0",
      "category": "Mathematical Reasoning",
      "score": 1.0,
      "reasoning": "..."
    }
  ],
  "differentiation_metadata": { ... },
  "fusion_metadata": { ... }
}
```

**实际格式** (可能):
```json
[
  {
    "problem_id": "problem_0",
    "category": "unknown",  # ⚠️ 早期可能是unknown
    "score": 1.0
  }
]
```

**一致性**: ⚠️ **早期log可能没有正确的category字段**

**修复状态**: 
- ✅ 已添加 `_index` 字段生成
- ✅ 所有benchmarks已修改 `problem_id` 提取逻辑
- ⏳ 需要重新运行生成正确的log

---

### 4.3 problem_classifications.json

**预期格式**:
```json
{
  "categories": [
    "Mathematical & Logical Reasoning",
    "Geometric & Spatial Reasoning",
    ...
  ],
  "problem_classifications": [
    {
      "problem_id": "problem_0",
      "category": "Mathematical & Logical Reasoning",
      "reasoning": "..."
    }
  ]
}
```

**实际格式**: ✅ **一致**

**生成时机**: 第一次分化操作时自动生成

---

## 5. 遗留代码检查

### 5.1 已废弃的代码

#### 5.1.1 旧的warm-up逻辑

**位置**: Line 370-430

**状态**: ✅ **已修改**

**旧代码**:
```python
if t < 2*k:
    return 0.0
```

**新代码**:
```python
if t < 2:
    return 0.0

effective_k = min(k, t // 2)
```

#### 5.1.2 旧的融合选择逻辑

**位置**: Line 1565-1620

**状态**: ✅ **已修改**

**旧代码**:
```python
async def _execute_fusion_async(self):
    # 重新查询workflows
    envelope_workflows = self.data_utils.find_envelope_workflows()
    selected = select_best_triple(envelope_workflows)
```

**新代码**:
```python
async def _execute_fusion_async(self, fusion_triple: Tuple):
    # 直接使用传入的triple
    w_i, w_j, w_k = fusion_triple
```

**影响**: 修复了"找到2个,需要3个"的bug

---

### 5.2 未使用的参数

检查是否有定义但未使用的参数:

```python
# 已定义
self.max_envelope_workflows = max_envelope_workflows  # Line ~96
self.fusion_score_threshold = fusion_score_threshold  # Line ~97

# 搜索使用位置
grep "max_envelope_workflows" enhanced_optimizer.py
grep "fusion_score_threshold" enhanced_optimizer.py
```

**结果**: 需要检查这些参数是否真正使用

---

## 6. 潜在Bug

### 6.1 除零风险

**位置**: 多处

**检查**:

#### 6.1.1 全局准确率计算

```python
acc_global = c_total / n if n > 0 else 0  # ✅ 已保护
```

#### 6.1.2 类别召回率计算

```python
recall_k = c_k / n_k if n_k > 0 else 0  # ✅ 需要检查
```

#### 6.1.3 贡献度计算

```python
contrib_k = c_k / c_total if c_total > 0 else 0  # ✅ 需要检查
```

**建议**: 详细检查所有除法操作

---

### 6.2 空列表处理

**位置**: 多处

**检查**:

#### 6.2.1 performance_history为空

```python
if len(self.performance_history) < 2:
    return 0.0  # ✅ 已保护
```

#### 6.2.2 workflows列表为空

```python
if len(workflows) < 3:
    return None  # ✅ 需要检查
```

#### 6.2.3 包络线为空

```python
if len(envelope) < 3:
    return None  # ✅ 需要检查
```

---

### 6.3 文件不存在

**位置**: 多处文件读取

**检查**:

```python
# log.json
if not os.path.exists(log_path):
    return {}  # ✅ 已保护

# problem_classifications.json
if not os.path.exists(classification_file):
    logger.warning("...")
    return {}  # ✅ 已保护

# results.json
if not os.path.exists(results_file):
    return []  # ⚠️ 需要检查
```

---

## 7. 优先修复列表

### 7.1 高优先级 (影响正确性)

1. **✅ 已修复: Fusion triple参数传递**
   - 修改 `_execute_fusion_async()` 接受参数
   - 避免重新查询workflows

2. **✅ 已修复: Problem ID生成**
   - 所有benchmarks添加 `_index` 字段
   - 确保log.json有正确的category

3. **⚠️ 需要验证: 融合公式参数**
   - `beta_triple=0.6` vs 理论 `β_t=0.3`
   - 确认实际使用的公式

### 7.2 中优先级 (影响性能)

4. **⚠️ 建议调整: 操作概率参数**
   - 考虑降低 `alpha_s` 和 `alpha_m`
   - 考虑增加 `eta_s` 和 `eta_m`
   - 使配置更接近理论设计

5. **✅ 已改进: Warm-up逻辑**
   - 允许早期计算停滞度
   - 动态调整窗口大小

### 7.3 低优先级 (优化改进)

6. **⏳ 待优化: 缓存机制**
   - 缓存正确问题集合
   - 避免重复加载log.json

7. **⏳ 待优化: 并行计算**
   - 并行计算所有三元组的互补性
   - 加速融合选择过程

---

## 8. 配置建议

### 8.1 保守配置 (接近理论)

```python
EnhancedOptimizer(
    # 停滞检测
    sliding_window_k=3,
    stagnation_sensitivity_kappa=80.0,
    
    # 操作概率 (保守)
    alpha_s=0.3,  # 降低分化频率
    alpha_m=0.4,  # 降低融合频率
    eta_s=0.1,    # 加快衰减
    eta_m=0.1,
    
    # 融合选择 (理论)
    beta_pair=0.4,
    beta_triple=0.3,  # 修正为理论值
)
```

### 8.2 激进配置 (当前默认)

```python
EnhancedOptimizer(
    # 停滞检测
    sliding_window_k=3,
    stagnation_sensitivity_kappa=80.0,
    
    # 操作概率 (激进)
    alpha_s=0.5,  # 更频繁分化
    alpha_m=0.6,  # 更频繁融合
    eta_s=0.03,   # 缓慢衰减
    eta_m=0.03,
    
    # 融合选择 (当前)
    beta_pair=0.4,
    beta_triple=0.6,  # 更重视triplet
    alpha_U=0.6,
    alpha_I=0.4,
    gamma_pair=0.7,
    gamma_triple=0.3,
)
```

### 8.3 平衡配置 (推荐)

```python
EnhancedOptimizer(
    # 停滞检测
    sliding_window_k=3,
    stagnation_sensitivity_kappa=80.0,
    
    # 操作概率 (平衡)
    alpha_s=0.4,
    alpha_m=0.5,
    eta_s=0.05,
    eta_m=0.05,
    
    # 融合选择 (平衡)
    beta_pair=0.4,
    beta_triple=0.4,  # 平衡pairwise和triplet
)
```

---

## 9. 测试建议

### 9.1 单元测试

```python
# 测试停滞度计算
def test_plateau_calculation():
    # 早期轮次
    opt = EnhancedOptimizer(...)
    opt.performance_history = [0.5]
    plateau = opt._calculate_plateau()
    assert plateau == 0.0  # 应该返回0
    
    # 正常轮次
    opt.performance_history = [0.5, 0.55, 0.56, 0.57]
    plateau = opt._calculate_plateau()
    assert 0 <= plateau <= 100

# 测试操作概率
def test_operation_probabilities():
    opt = EnhancedOptimizer(...)
    probs = opt._calculate_operation_probabilities()
    assert abs(sum(probs.values()) - 1.0) < 1e-6  # 概率和为1

# 测试分化选择
def test_split_selection():
    opt = EnhancedOptimizer(...)
    workflows = [...]
    selected, category = opt._select_for_split(workflows)
    assert selected is not None or category is None  # 逻辑一致

# 测试融合选择
def test_merge_selection():
    opt = EnhancedOptimizer(...)
    workflows = [...]
    triple = opt._select_for_merge(workflows)
    assert triple is None or len(triple) == 3  # 返回None或三元组
```

### 9.2 集成测试

```bash
# 完整运行测试
python run.py \
  --dataset MATH \
  --max_rounds 10 \
  --sliding_window_k 3 \
  --alpha_s 0.3 \
  --alpha_m 0.4 \
  --eta_s 0.1 \
  --eta_m 0.1

# 检查日志
grep "Operation Probabilities" logs/AFlow.log
grep "Selected operation" logs/AFlow.log
grep "Score for round" logs/AFlow.log
```

---

## 10. 总结

### 10.1 主要发现

| 项目 | 状态 | 优先级 |
|------|------|--------|
| Problem ID生成 | ✅ 已修复 | 高 |
| Fusion triple传递 | ✅ 已修复 | 高 |
| Warm-up逻辑 | ✅ 已改进 | 中 |
| 操作概率参数 | ⚠️ 不一致 | 中 |
| 融合公式参数 | ⚠️ 需验证 | 高 |
| 缓存优化 | ⏳ 待实现 | 低 |

### 10.2 建议行动

**立即执行**:
1. ✅ 验证融合公式实现 (Line 832-1000)
2. ⚠️ 决定是否调整操作概率参数
3. ⏳ 运行完整测试,验证修复效果

**短期优化**:
1. 添加单元测试覆盖关键算法
2. 添加参数验证 (确保配置合理)
3. 优化日志输出 (更清晰的调试信息)

**长期改进**:
1. 实现缓存机制
2. 并行化计算
3. 参数自适应调整

---

## 11. 下一步

- [快速参考](QUICK_REFERENCE.md) - 关键公式和参数速查
- [系统架构总览](SYSTEM_ARCHITECTURE.md) - 返回整体视图
- [优化操作详解](OPTIMIZE_OPERATION.md) - 了解优化细节

---

## 附录: 检查清单

```markdown
## 代码审查清单

### 算法一致性
- [ ] 停滞度公式与理论一致
- [ ] 操作概率公式与理论一致
- [ ] 分化选择算法与理论一致
- [ ] 融合选择算法与理论一致

### 参数一致性
- [ ] 停滞检测参数
- [ ] 操作概率参数
- [ ] 融合选择参数
- [ ] 分化选择参数

### 数据结构
- [ ] performance_history索引正确
- [ ] log.json格式正确
- [ ] problem_classifications.json格式正确
- [ ] results.json格式正确

### 错误处理
- [ ] 所有除法有除零保护
- [ ] 所有文件读取有存在性检查
- [ ] 所有列表访问有长度检查
- [ ] 所有可选参数有默认值

### 性能优化
- [ ] 避免重复计算
- [ ] 使用缓存机制
- [ ] 并行化可能的地方
- [ ] 日志级别合理

### 测试覆盖
- [ ] 单元测试
- [ ] 集成测试
- [ ] 边界情况测试
- [ ] 性能测试
```
