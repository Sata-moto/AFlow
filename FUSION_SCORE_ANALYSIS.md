# 融合分数计算机制分析

## 当前实现（两路融合）

### 1. 选择阶段 (`_select_for_fuse`)

**位置**: `scripts/enhanced_optimizer.py` 第641-740行

**算法**: Algorithm 3: SelectForFuse (两路版本)

**步骤**:

#### Step 1: 筛选候选集
```python
top_M = min(5, len(workflow_results))  # 选择前5个
sorted_workflows = sorted(workflow_results, 
                          key=lambda w: w.get('avg_score', 0.0), 
                          reverse=True)
candidates = sorted_workflows[:top_M]
```

#### Step 2: 成对评分
```python
for i, w_i in enumerate(candidates):
    for j, w_j in enumerate(candidates):
        if i >= j:  # 避免重复
            continue
        
        # 互补性 (Complementarity)
        phi_U = max(score_i, score_j) * 1.1
        
        # 共识性 (Consensus)
        phi_I = min(score_i, score_j)
        
        # 惩罚因子 (避免重复融合)
        penalty = 0.1 if already_attempted else 1.0
        
        # 融合势函数
        phi_merge = (alpha_U * phi_U + alpha_I * phi_I) * penalty
```

**超参数**:
- `alpha_U = 0.6` (互补性权重)
- `alpha_I = 0.4` (共识性权重)
- `top_M = 5` (候选集大小)

**输出**: `(w1, w2)` - 两个工作流

### 2. 基准分数计算 (`_execute_single_fusion`)

**位置**: `scripts/enhanced_optimizer.py` 第1095行

**当前代码**:
```python
w1, w2 = fusion_pair
min_envelope_score = min(w1.get('avg_score', 0.0), w2.get('avg_score', 0.0))
```

**逻辑**: 取两个工作流中的**最小分数**作为基准

**问题**: 
- ❌ **硬编码为两个工作流** (`w1, w2`)
- ❌ 无法处理三个或更多工作流
- ❌ 使用 `min()` 只能接受两个参数（当前实现）

### 3. 融合成功判断

**位置**: `scripts/enhanced_optimizer.py` 第1143行

**当前代码**:
```python
if fusion_score > min_envelope_score + self.fusion_score_threshold:
    logger.info(f"Fusion successful!")
```

**判断标准**:
```
融合成功 ⟺ fusion_score > min(source_scores) + threshold
```

**逻辑**:
- 融合后的工作流必须**超过所有源工作流中最差的那个**
- 加上一个阈值 `fusion_score_threshold`（默认0.0）

### 4. 融合分数评估

**位置**: `scripts/enhanced_optimizer.py` 第1121行

**代码**:
```python
fusion_score = await self.evaluation_utils.evaluate_graph(
    self, directory, self.validation_rounds, data, initial=False
)
```

**逻辑**:
- 直接在测试集上评估融合后的工作流
- 使用标准的评估流程
- **与源工作流数量无关** ✅ 已经支持任意数量

---

## 问题总结

### ✅ 支持任意多工作流的部分

1. **融合评估** (`evaluate_graph`): 只评估最终融合结果，与源数量无关
2. **融合元数据保存**: 使用列表推导式，支持任意数量
3. **融合执行**: LLM接受任意数量的工作流输入

### ❌ 不支持任意多工作流的部分

1. **`_select_for_fuse` 返回值**: 
   - 硬编码返回 `tuple` (两个元素)
   - 遍历逻辑是成对的 `for i, for j`

2. **`min_envelope_score` 计算**:
   - 硬编码 `min(w1, w2)`
   - 无法处理 `w1, w2, w3`

3. **融合对解包**:
   - `w1, w2 = fusion_pair` 期望恰好两个元素

---

## 修改方案：支持三路融合

### 修改1: `_select_for_fuse` 改为三路版本

**当前**:
```python
def _select_for_fuse(self, workflow_results: List[Dict]) -> tuple:
    # ...
    for i, w_i in enumerate(candidates):
        for j, w_j in enumerate(candidates):
            if i >= j:
                continue
            # 计算 phi_merge for pair (i, j)
    return (w_i, w_j)  # 返回2元组
```

**修改为**:
```python
def _select_for_fuse(self, workflow_results: List[Dict]) -> tuple:
    # ...
    for i, w_i in enumerate(candidates):
        for j, w_j in enumerate(candidates):
            for k, w_k in enumerate(candidates):
                if i >= j or j >= k:  # 避免重复，保证 i < j < k
                    continue
                # 计算 phi_merge for triple (i, j, k)
    return (w_i, w_j, w_k)  # 返回3元组
```

**评分函数修改**:
```python
# 互补性：三路并集
phi_U = max(score_i, score_j, score_k) * 1.15  # 稍微提高系数

# 共识性：三路交集
phi_I = min(score_i, score_j, score_k)

# 惩罚因子
temp_workflows = [
    {"round": round_i, ...},
    {"round": round_j, ...},
    {"round": round_k, ...}
]
penalty = 0.1 if already_attempted else 1.0

# 融合势函数（与理论一致）
phi_merge = (alpha_U * phi_U + alpha_I * phi_I) * penalty
```

### 修改2: `min_envelope_score` 计算

**当前**:
```python
w1, w2 = fusion_pair
min_envelope_score = min(w1.get('avg_score', 0.0), w2.get('avg_score', 0.0))
```

**修改为**:
```python
# 支持任意数量的工作流
envelope_scores = [w.get('avg_score', 0.0) for w in fusion_pair]
min_envelope_score = min(envelope_scores)
```

或者明确三个：
```python
w1, w2, w3 = fusion_pair
min_envelope_score = min(
    w1.get('avg_score', 0.0), 
    w2.get('avg_score', 0.0),
    w3.get('avg_score', 0.0)
)
```

### 修改3: 日志和元数据

**当前**:
```python
logger.info(f"Selected fusion pair: Round {best_pair[0].get('round', 0)} + "
           f"Round {best_pair[1].get('round', 0)}")
```

**修改为**:
```python
rounds_str = " + ".join([f"Round {w.get('round', 0)}" for w in best_pair])
logger.info(f"Selected fusion triple: {rounds_str}")
```

或者明确三个：
```python
logger.info(f"Selected fusion triple: Round {best_pair[0].get('round', 0)} + "
           f"Round {best_pair[1].get('round', 0)} + "
           f"Round {best_pair[2].get('round', 0)}")
```

---

## 理论依据

### 三路并集（互补性）

$$\Phi_U = \text{Cov}(C_i \cup C_j \cup C_k)$$

**物理意义**: 三个工作流联合能解决的问题总数

**近似**: `max(score_i, score_j, score_k)`
- 假设分数与覆盖率正相关
- 取最高分作为上界
- 乘以系数 1.15 奖励多样性

### 三路交集（共识性）

$$\Phi_I = \text{Cov}(C_i \cap C_j \cap C_k)$$

**物理意义**: 三个工作流都能解决的问题数

**近似**: `min(score_i, score_j, score_k)`
- 三个都能解决的问题不会超过最弱的那个
- 取最低分作为下界

### 融合势函数

$$\Phi_{merge} = (\alpha_U \Phi_U + \alpha_I \Phi_I) \cdot \text{Penalty}(i,j,k)$$

**平衡**:
- `alpha_U = 0.6`: 更重视互补性（覆盖更多问题）
- `alpha_I = 0.4`: 同时考虑共识性（保证可靠性）

**惩罚**:
- 如果三元组已融合过: `penalty = 0.1`
- 否则: `penalty = 1.0`

---

## 评估标准

### 基准分数 (Baseline)

```
min_envelope_score = min(scores of source workflows)
```

**为什么用最小值？**
- 融合至少要比**最差的源工作流**好
- 否则融合没有意义（直接用最好的源工作流即可）
- 这是**保守策略**，确保融合有价值

### 成功条件

```
fusion_score > min_envelope_score + threshold
```

**逻辑**:
1. 融合后分数 > 所有源工作流中最差的
2. 加上一个安全边际 `threshold`（默认0.0）

**示例**（三路融合）:
- W1: 0.82
- W2: 0.79
- W3: 0.85
- `min_envelope_score = 0.79`
- `threshold = 0.0`
- **要求**: `fusion_score > 0.79`

### 理想情况

```
fusion_score > max(source_scores)
```

**更严格的标准**: 融合应该比**最好的源工作流**还要好

**但当前未使用**，因为：
- 融合可能在不同问题上表现更好
- 平均分数可能略低但覆盖面更广
- 过于严格会导致大量融合被拒绝

---

## 总结

### 当前状态

| 组件 | 是否支持任意多工作流 | 说明 |
|------|---------------------|------|
| `_select_for_fuse` | ❌ | 硬编码成对遍历 |
| `min_envelope_score` | ❌ | 硬编码 `min(w1, w2)` |
| `fusion_score` 评估 | ✅ | 只评估融合结果 |
| 融合元数据保存 | ✅ | 使用列表 |
| LLM 融合执行 | ✅ | 接受任意数量 |

### 需要修改的地方

1. **`_select_for_fuse` 方法** (第641-740行)
   - 三层循环 `for i, for j, for k`
   - 返回3元组 `(w1, w2, w3)`
   - 更新日志信息

2. **`_execute_single_fusion` 方法** (第1089-1095行)
   - 改为 `w1, w2, w3 = fusion_pair`
   - 或 `min(w.get('avg_score') for w in fusion_pair)`

3. **日志和提示**
   - "fusion pair" → "fusion triple"
   - "两个工作流" → "三个工作流"

### 修改后效果

✅ 完全支持三路融合  
✅ 与理论算法一致  
✅ 可扩展到 N 路融合（如需要）
