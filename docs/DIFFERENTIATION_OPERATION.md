# 分化操作详解 (DIFFERENTIATION Operation)

## 文档版本
- **创建日期**: 2025-12-15
- **相关代码**: `scripts/enhanced_optimizer.py` (Line 485-1270)
- **依赖文档**: [系统架构总览](SYSTEM_ARCHITECTURE.md)

---

## 1. 操作概述

### 1.1 核心目标
分化操作通过**类别分析**识别workflow的优势领域,针对特定问题类别创建**专业化分支**,牺牲通用性换取局部高性能。

### 1.2 设计理念
- **精准定位**: 基于统计数据找到优势类别
- **有的放矢**: 针对特定类别优化,而非泛化改进
- **理论驱动**: 使用数学公式量化分化潜力

### 1.3 触发条件
- **概率采样**: 根据停滞度和历史分化次数计算概率
- **典型场景**: `plateau_t > 0.3` 时概率显著提升
- **公式**: `p_split = α_s · plateau_t · exp(-η_s · N_s)`

---

## 2. 完整流程

### 2.1 流程图

```
┌────────────────────────────────────────────────────────────────┐
│                DIFFERENTIATION 操作流程                         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  预处理: 确保问题分类存在                                        │
│                                                                 │
│  首次分化时执行:                                                 │
│  1. 检查 problem_classifications.json 是否存在                 │
│  2. 如不存在,调用 LLM 对所有问题分类                            │
│  3. 生成类别列表和问题-类别映射                                  │
│                                                                 │
│  示例输出:                                                      │
│  {                                                              │
│    "categories": [                                             │
│      "Mathematical & Logical Reasoning",                       │
│      "Geometric & Spatial Reasoning",                          │
│      "Combinatorial Counting & Enumeration",                   │
│      "Dynamic Programming & Recursion"                         │
│    ],                                                          │
│    "problem_classifications": [                                │
│      {"problem_id": "problem_0", "category": "...", ...}       │
│    ]                                                           │
│  }                                                              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第1阶段: 选择分化对象 (_select_for_split)                      │
│                                                                 │
│  对每个workflow W_i:                                            │
│    1. 加载 log.json → 获取每道题的结果和类别                    │
│    2. 统计全局性能:                                             │
│       - C_total: 总答对题目数                                   │
│       - N: 总题目数                                             │
│       - Acc_global = C_total / N                               │
│                                                                 │
│    3. 对每个类别 k:                                             │
│       - C_k: 该类别答对题目数                                   │
│       - N_k: 该类别总题目数                                     │
│       - Recall_k = C_k / N_k (类别召回率)                       │
│       - Contrib_k = C_k / C_total (类别贡献度)                  │
│                                                                 │
│    4. 计算分化潜力:                                             │
│       if Recall_k > Acc_global:  # 有优势                      │
│         Score_k = Contrib_k × (Recall_k - Acc_global)          │
│                                                                 │
│    5. 工作流分化潜力:                                            │
│       Split_Potential(W_i) = max_k(Score_k)                    │
│                                                                 │
│  选择策略 (确定性):                                             │
│    selected = argmax_i(Split_Potential(W_i))                   │
│    target_category = argmax_k(Score_k for W_selected)          │
│                                                                 │
│  返回: (selected_workflow, target_category)                    │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第2阶段: 加载目标类别样本                                       │
│                                                                 │
│  1. 从 problem_classifications.json 筛选目标类别的problem_id   │
│  2. 从验证集加载对应的完整问题                                   │
│  3. 限制样本数量 (默认10个)                                     │
│  4. 格式化为示例列表:                                           │
│     [                                                          │
│       {                                                        │
│         "problem": "问题描述",                                  │
│         "solution": "标准答案"                                  │
│       },                                                       │
│       ...                                                      │
│     ]                                                          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第3阶段: LLM生成专业化Workflow                                 │
│                                                                 │
│  输入:                                                          │
│  - 基础workflow (selected_workflow的graph.py + prompt.py)     │
│  - 目标类别描述                                                 │
│  - 该类别的样本问题                                             │
│  - 分化提示词模板                                               │
│                                                                 │
│  LLM任务:                                                       │
│  - 分析类别特点                                                 │
│  - 识别该类别需要的特殊处理                                      │
│  - 生成针对该类别优化的workflow                                 │
│  - 可以牺牲其他类别的性能                                        │
│                                                                 │
│  输出:                                                          │
│  - 专业化的 graph.py                                           │
│  - 专业化的 prompt.py                                          │
│  - 说明专业化的理由                                             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第4阶段: 创建并评估新Workflow                                  │
│                                                                 │
│  1. 创建 round_N/ 目录                                         │
│  2. 保存专业化的 graph.py 和 prompt.py                         │
│  3. 记录分化元数据到 log.json                                   │
│  4. 在**完整**验证集上评估                                      │
│     (不只是目标类别,而是全部问题)                               │
│  5. 记录:                                                       │
│     - 全局性能可能下降                                          │
│     - 目标类别性能提升                                          │
│     - 其他类别性能下降                                          │
│                                                                 │
│  更新:                                                          │
│  - N_s += 1 (累计分化次数)                                      │
│  - category_last_differentiation[category] = current_round    │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 关键代码位置

#### 2.2.1 主入口
**文件**: `scripts/enhanced_optimizer.py`
**行数**: 281-295

```python
elif operation == 'differentiate':
    logger.info("=" * 80)
    logger.info(f"Executing DIFFERENTIATE operation for round {self.round + 1}")
    logger.info("=" * 80)
    
    # 执行分化
    score = await self._differentiate()
    
    if score is not None:
        # 分化成功
        self.N_s += 1  # 增加分化计数
```

#### 2.2.2 分化选择算法
**文件**: `scripts/enhanced_optimizer.py`
**行数**: 485-656

```python
def _select_for_split(self, workflows: List[Dict]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Algorithm 2: SelectForSplit - 选择最适合分化的workflow和目标类别
    
    基于类别级别的性能分析,选择具有最大分化潜力的workflow
    
    公式:
        对于workflow W_i和类别k:
        - Acc_global = C_total / N (全局准确率)
        - Recall_k = C_k / N_k (类别召回率)
        - Contrib_k = C_k / N (类别绝对贡献度)
        
        如果 Recall_k > Acc_global (该类别有优势):
            Score_split(W_i, k) = Contrib_k × (Recall_k - Acc_global)
        
        Split_Potential(W_i) = max_k(Score_split(W_i, k))
    
    注意: Contrib_k使用绝对贡献度(C_k/N)而非相对贡献度(C_k/C_total)，
          避免小类别因高占比而过度偏向
    
    返回:
        (selected_workflow, target_category): 最佳分化对象和目标类别
        如果没有合适的分化对象,返回 (None, None)
    """
```

#### 2.2.3 专业化Workflow生成
**文件**: `scripts/enhanced_optimizer.py`
**行数**: 1217-1270

```python
async def _execute_single_differentiation(self) -> float:
    """
    执行单次分化操作
    
    返回:
        float: 专业化workflow的分数,失败返回None
    """
```

---

## 3. 核心算法详解

### 3.1 分化潜力计算

**目标**: 量化每个workflow在各类别的专业化价值

**公式推导**:

```
给定:
- C_total: workflow总答对题目数
- N: 总题目数  
- C_k: 类别k答对题目数
- N_k: 类别k总题目数

定义:
- Acc_global = C_total / N        (全局准确率)
- Recall_k = C_k / N_k            (类别召回率)
- Contrib_k = C_k / N             (类别绝对贡献度)

注意: 使用绝对贡献度而非相对贡献度(C_k/C_total)
      - 相对贡献度: 小类别答对几题就占比很高,容易过度偏向
      - 绝对贡献度: 基于该类别对整体分数的实质贡献,更加公平

优势类别判定:
  if Recall_k > Acc_global:
      该类别是优势类别
      (该类别的表现好于平均水平)

分化价值:
  Score_split(W, k) = Contrib_k × (Recall_k - Acc_global)
  
  解释:
  - Contrib_k: 该类别对总分的绝对贡献(C_k/N)
  - (Recall_k - Acc_global): 超出平均水平的程度(优势)
  - 两者相乘: 高绝对贡献且高优势的类别分化价值最大

工作流分化潜力:
  Split_Potential(W) = max_k(Score_split(W, k))
```

**实际计算示例**:

```
Round 5 分析:
  总体: 69/119 correct, Acc_global = 0.5798
  N = 119 (总题目数)

  类别1: Mathematical Reasoning
    - C_k = 45, N_k = 60
    - Recall_k = 45/60 = 0.7500 ✓ (> Acc_global)
    - Contrib_k = 45/119 = 0.3782 (绝对贡献度)
    - Score_k = 0.3782 × (0.7500 - 0.5798) = 0.0644

  类别2: Geometric Reasoning  
    - C_k = 15, N_k = 30
    - Recall_k = 15/30 = 0.5000 ✗ (< Acc_global)
    - 无分化价值 (Score_k = 0)

  类别3: Combinatorial Counting
    - C_k = 9, N_k = 20
    - Recall_k = 9/20 = 0.4500 ✗ (< Acc_global)
    - 无分化价值

  类别4: Dynamic Programming
    - C_k = 0, N_k = 9
    - Recall_k = 0/9 = 0.0000 ✗ (< Acc_global)
    - 无分化价值

结果:
  Split_Potential(Round 5) = 0.0644
  Target_Category = "Mathematical Reasoning"

对比旧公式 (C_k/C_total):
  - 旧 Contrib_k = 45/69 = 0.6522
  - 旧 Score_k = 0.6522 × 0.1702 = 0.1110
  - 新 Score_k = 0.0644 (降低了42%, 更合理地反映该类别的实质贡献)
```

### 3.2 选择策略 (准确率权衡)

**问题**: 只看分化潜力可能选择"偏科"但整体性能差的workflow

**解决方案**: 权衡分化潜力和整体准确率

```python
# 1. 计算所有workflows的分化潜力
potentials = []
accuracies = []
for workflow in workflows:
    split_potential = calculate_split_potential(workflow)
    accuracy = workflow.acc_global
    potentials.append(split_potential)
    accuracies.append(accuracy)

# 2. 归一化分化潜力（以最大值归一化）
max_potential = max(potentials)
normalized_potentials = [p / max_potential for p in potentials]

# 3. 计算修正分数（权衡潜力和准确率）
alpha = 0.5  # 默认: 平衡权重
adjusted_scores = [
    alpha * norm_pot + (1 - alpha) * acc
    for norm_pot, acc in zip(normalized_potentials, accuracies)
]

# 4. 选择修正分数最高的workflow
selected_idx = adjusted_scores.index(max(adjusted_scores))
selected_workflow = workflows[selected_idx]

# 5. 找到该workflow的最佳目标类别
target_category = find_best_category(selected_workflow)
```

**参数α的作用**:
- `α = 1.0`: 只看分化潜力 (可能选择"偏科"workflow)
- `α = 0.5`: 平衡潜力和准确率 (推荐)
- `α = 0.0`: 只看准确率 (退化为选择最佳workflow)

**为什么需要权衡?**
- 防止选择"偏科严重但整体差"的workflow
- 确保分化后的专业化分支有良好的基础
- 平衡专业化深度和整体性能

**示例**:
```
候选1: Round 3
  - Split_Potential = 0.08 (最高)
  - Accuracy = 0.45
  - Normalized_Potential = 0.08/0.08 = 1.0
  - Adjusted_Score = 0.5×1.0 + 0.5×0.45 = 0.725

候选2: Round 5
  - Split_Potential = 0.06
  - Accuracy = 0.58 (更高)
  - Normalized_Potential = 0.06/0.08 = 0.75
  - Adjusted_Score = 0.5×0.75 + 0.5×0.58 = 0.665

候选3: Round 7
  - Split_Potential = 0.07
  - Accuracy = 0.62 (最高)
  - Normalized_Potential = 0.07/0.08 = 0.875
  - Adjusted_Score = 0.5×0.875 + 0.5×0.62 = 0.748 ✓ 最高

选择: Round 7 (既有较高潜力,准确率也最好)
```

### 3.3 类别元数据加载

**问题**: 如何知道每个问题属于哪个类别?

**解决方案**: 首次分化时生成 `problem_classifications.json`

**代码位置**: `scripts/enhanced_optimizer.py`, Line 730-794

```python
def _load_category_metadata(self) -> Dict[str, int]:
    """
    加载问题分类元数据
    
    返回: {category: count} - 每个类别的问题数量
    """
    classification_file = f"{self.root_path}/workflows/problem_classifications.json"
    
    if not os.path.exists(classification_file):
        logger.warning("Problem classifications not found!")
        return {}
    
    try:
        with open(classification_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理嵌套数组格式
        if "problem_classifications" in data:
            classifications = data["problem_classifications"]
        else:
            classifications = data
        
        # 统计每个类别的问题数量
        category_counts = {}
        for item in classifications:
            category = item.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return category_counts
    
    except Exception as e:
        logger.error(f"Failed to load category metadata: {e}")
        return {}
```

**数据格式**:
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
      "reasoning": "分类理由..."
    },
    ...
  ]
}
```

### 3.4 Workflow类别统计

**目标**: 统计某个workflow在各类别的正确数量

**代码位置**: `scripts/enhanced_optimizer.py`, Line 795-830

```python
def _load_workflow_category_stats(self, workflow: Dict) -> Dict[str, int]:
    """
    加载工作流在各类别的统计
    
    从 log.json 加载,统计每个类别答对了多少题
    
    返回: {"category_A": C_k, "category_B": C_k, ...}
    """
    round_num = workflow.get('round', 0)
    log_path = f"{self.root_path}/workflows/round_{round_num}/log.json"
    
    if not os.path.exists(log_path):
        return {}
    
    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    # 处理两种log格式
    if isinstance(log_data, dict) and "execution_logs" in log_data:
        entries = log_data["execution_logs"]
    else:
        entries = log_data
    
    # 统计每个类别的正确数量
    category_counts = {}
    for entry in entries:
        category = entry.get('category', 'unknown')
        score = entry.get('score', 0.0)
        
        # 只统计正确的 (score >= 0.5)
        if score >= 0.5:
            category_counts[category] = category_counts.get(category, 0) + 1
    
    return category_counts
```

**关键点**: 
- 依赖 `log.json` 中的 `category` 字段
- **必须**在评估时正确记录category (见Problem ID修复)
- 只统计 `score >= 0.5` 的题目

---

## 4. 专业化Workflow生成

### 4.1 LLM提示词设计

**目标**: 指导LLM生成针对特定类别优化的workflow

**提示词结构**:
```python
differentiation_prompt = f"""
You are creating a SPECIALIZED workflow for {target_category} problems.

## Base Workflow
The current general-purpose workflow:
```python
{base_graph_code}
```

```python
{base_prompt_code}
```

## Target Category: {target_category}
Category Description: {category_description}

## Sample Problems from This Category
{example_problems}

## Performance Analysis
- Current performance on this category: {recall_k:.2%}
- Global performance: {acc_global:.2%}
- This category contributes {contrib_k:.2%} of total correct answers

## Your Task
Create a SPECIALIZED workflow optimized specifically for {target_category}.

IMPORTANT:
1. Focus ONLY on solving {target_category} problems well
2. You MAY sacrifice performance on other categories
3. Add domain-specific reasoning steps
4. Customize the prompt for this category's characteristics
5. It's OK if general performance drops - we want HIGH performance on THIS category

## Output Format
Provide the specialized workflow:

```python
# graph.py
<specialized graph code>
```

```python
# prompt.py  
<specialized prompt code>
```

Explanation:
<explain how your specialization helps {target_category}>
"""
```

**关键设计点**:
1. **明确专业化目标**: "Focus ONLY on..."
2. **允许性能权衡**: "You MAY sacrifice..."  
3. **提供样本**: 让LLM理解类别特点
4. **领域特定**: "Add domain-specific reasoning"

### 4.2 样本问题选择

**代码位置**: `scripts/enhanced_optimizer.py`, Line 1217-1270

```python
# 从problem_classifications筛选目标类别
target_problems = [
    p for p in classifications 
    if p.get('category') == target_category
]

# 限制数量 (默认10个)
sampled_problems = target_problems[:10]

# 从验证集加载完整问题
example_problems = []
for p in sampled_problems:
    problem_id = p['problem_id']
    # 根据problem_id从数据集加载完整问题
    full_problem = load_problem_by_id(problem_id)
    example_problems.append(full_problem)
```

**为什么需要样本?**
- LLM需要理解类别的具体特点
- 避免抽象的类别描述
- 提供具体的优化目标

### 4.3 专业化代价

**预期效果**:
```
分化前 (Round 5):
  全局: 69/119 correct (58.0%)
  Math:  45/60 correct (75.0%)  ← 优势类别
  Geo:   15/30 correct (50.0%)
  Comb:   9/20 correct (45.0%)
  DP:     0/9  correct (0.0%)

分化后 (Round 6):
  全局: 60/119 correct (50.4%) ↓ 下降7.6%
  Math:  50/60 correct (83.3%) ↑ 提升8.3% ✓
  Geo:    7/30 correct (23.3%) ↓ 下降26.7%
  Comb:   3/20 correct (15.0%) ↓ 下降30.0%
  DP:     0/9  correct (0.0%)  - 持平
```

**这是正常的!** 
- 专业化必然牺牲通用性
- 目标类别性能提升
- 其他类别性能下降
- 全局分数可能下降

**长期策略**:
- 对每个类别都创建专业化分支
- 最后通过融合操作整合各分支优势
- 形成"专家集成"架构

---

## 5. 分化元数据

### 5.1 记录分化信息

**位置**: `log.json` 中添加分化元数据

```json
{
  "differentiation_metadata": {
    "source_workflow": "round_5",
    "target_category": "Mathematical & Logical Reasoning",
    "split_potential": 0.1110,
    "category_stats": {
      "C_k": 45,
      "N_k": 60,
      "Recall_k": 0.7500,
      "Contrib_k": 0.6522
    },
    "timestamp": "2025-12-15T10:15:30"
  },
  "execution_logs": [...]
}
```

### 5.2 跟踪分化历史

**目的**: 避免对同一个类别重复分化

**实现**: `category_last_differentiation`

```python
# 记录每个类别最后一次分化的轮次
self.category_last_differentiation = {
    "Mathematical Reasoning": 6,
    "Geometric Reasoning": 8,
    ...
}

# 在选择时考虑冷却时间
min_gap = 3  # 至少间隔3轮
if current_round - last_diff_round < min_gap:
    # 跳过该类别或降低优先级
    pass
```

---

## 6. 典型执行场景

### 6.1 成功分化案例

```
[Round 5 → Round 6 分化]

选择阶段:
  - 评估了11个workflows
  - Round 5 有最高分化潜力: 0.1110
  - 目标类别: "Mathematical & Logical Reasoning"
  - 该类别: 45/60 correct (75.0%), 贡献65.2%

样本加载:
  - 加载10个数学推理问题
  - 包括代数、几何证明、数论等

LLM生成:
  - 添加了显式的代数推导步骤
  - 增强了等式变换逻辑
  - 专门处理数学符号

评估结果:
  Round 6:
    - 全局: 60/119 (50.4%) ↓ -7.6%
    - 数学: 50/60 (83.3%) ↑ +8.3% ✓
  
  分化成功! 目标类别性能显著提升
```

### 6.2 无合适分化对象

```
[Round 10 分化尝试]

选择阶段:
  - 评估了13个workflows
  - 所有workflows的分析结果:
    * Round 1: 无优势类别
    * Round 2: 无优势类别  
    * Round 3: 无优势类别
    * ...
    * Round 13: 无优势类别

  问题: 所有workflows在所有类别都没有超过全局水平

原因:
  - 可能所有workflows都很"平均"
  - 各类别表现都接近全局水平
  - 没有明显的专业化机会

结果:
  返回 (None, None)
  分化操作失败,自动重试或跳过
```

### 6.3 多次分化策略

```
Round 6: 分化 → Mathematical Reasoning
  → 创建数学专家分支

Round 8: 分化 → Geometric Reasoning  
  → 创建几何专家分支

Round 12: 分化 → Combinatorial Counting
  → 创建组合专家分支

Round 15: 融合 (Round 6, 8, 12)
  → 整合三个专家的优势
  → 创建综合性强的workflow
```

---

## 7. 关键参数

### 7.1 分化概率参数

| 参数 | 默认值 | 说明 | 影响 |
|------|--------|------|------|
| `α_s` | 0.3 | 分化基础权重 | 控制分化基础概率 |
| `η_s` | 0.1 | 分化衰减率 | 控制重复分化惩罚 |
| `N_s` | 动态 | 累计分化次数 | 影响衰减计算 |

**概率计算**:
```python
p_split_raw = α_s · plateau_t · exp(-η_s · N_s)
```

**示例**:
```
plateau_t = 0.5, N_s = 0:
  p_split = 0.3 × 0.5 × exp(0) = 0.15 (15%)

plateau_t = 0.8, N_s = 2:
  p_split = 0.3 × 0.8 × exp(-0.2) = 0.197 (19.7%)

plateau_t = 0.9, N_s = 5:
  p_split = 0.3 × 0.9 × exp(-0.5) = 0.164 (16.4%)
```

### 7.2 分化执行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_example_problems` | 10 | 提供给LLM的样本数 |
| `min_category_advantage` | 0.0 | 最小优势阈值 |
| `category_cooldown` | 0 | 类别分化冷却轮数 |

---

## 8. 代码映射

### 8.1 主要方法

```python
class EnhancedOptimizer:
    # 分化主流程
    async def _differentiate(self):
        """Line 1162-1215"""
        pass
    
    # 选择分化对象
    def _select_for_split(self, workflows):
        """Line 485-656"""
        pass
    
    # 执行单次分化
    async def _execute_single_differentiation(self):
        """Line 1217-1270"""
        pass
    
    # 加载类别元数据
    def _load_category_metadata(self):
        """Line 730-794"""
        pass
    
    # 加载workflow类别统计
    def _load_workflow_category_stats(self, workflow):
        """Line 795-830"""
        pass
```

### 8.2 数据流

```
results.json ────┐
                 │
log.json ────────┼────→ _load_workflow_category_stats()
                 │      └─→ {category: correct_count}
                 │
problem_classifications.json ─┐
                 │             │
                 ├─────────────┼→ _load_category_metadata()
                 │             └─→ {category: total_count}
                 │
                 ├────→ _select_for_split()
                 │      ├─ 计算 Acc_global, Recall_k, Contrib_k
                 │      ├─ 计算 Score_split(W, k)
                 │      └─→ (selected_workflow, target_category)
                 │
selected_workflow + target_category ─┐
                                      │
problem_classifications.json ────────┼→ 加载目标类别样本
                                      │
                                      ├→ LLM生成专业化workflow
                                      │
                                      └→ 评估并保存
```

---

## 9. 最佳实践

### 9.1 何时使用分化

**适合场景**:
- ✅ 停滞度 > 0.3
- ✅ 某些类别表现明显好于平均
- ✅ 通用优化遇到瓶颈

**不适合场景**:
- ❌ 早期轮次 (数据不足)
- ❌ 所有类别表现都很平均
- ❌ 已经对所有类别都分化过

### 9.2 分化策略建议

1. **循序渐进**: 先对贡献度最高的类别分化
2. **保留基线**: 不要删除通用workflow
3. **定期融合**: 每3-5个专业化分支后执行融合
4. **监控效果**: 跟踪目标类别的性能提升

### 9.3 调试技巧

```bash
# 1. 检查问题分类
cat workspace/DATASET/workflows/problem_classifications.json | jq '.categories'

# 2. 查看某轮的类别统计
cat workspace/DATASET/workflows/round_5/log.json | \
  jq '[.[] | select(.category == "Mathematical Reasoning")] | length'

# 3. 检查分化日志
grep "Split potential" logs/AFlow.log
grep "Target Category" logs/AFlow.log
```

---

## 10. 常见问题

### Q1: 为什么分化后全局分数下降?
**A**: 这是正常的! 分化牺牲通用性换取局部高性能。预期目标类别提升,其他类别下降。

### Q2: 如何选择分化时机?
**A**: 当 `plateau_t > 0.3` 且有明显优势类别时。系统会自动根据概率采样。

### Q3: 可以对同一类别多次分化吗?
**A**: 可以,但有衰减惩罚。建议先对不同类别分化,再考虑重复分化。

### Q4: 分化失败怎么办?
**A**: 系统会自动重试(最多3次)。如果持续失败,可能:
- 没有合适的分化对象
- LLM生成的代码有问题
- 跳过本轮,继续下一轮

### Q5: 如何评估分化效果?
**A**: 
```python
# 比较分化前后目标类别的性能
before_recall = C_k_before / N_k
after_recall = C_k_after / N_k
improvement = after_recall - before_recall

# 期望: improvement > 0.05 (5%提升)
```

---

## 11. 下一步

- [融合操作详解](FUSION_OPERATION.md) - 了解如何整合专业化分支
- [优化操作详解](OPTIMIZE_OPERATION.md) - 了解通用优化策略
- [系统架构总览](SYSTEM_ARCHITECTURE.md) - 返回整体视图

---

## 附录: 完整算法伪代码

```python
def differentiate():
    """分化操作完整流程"""
    
    # 阶段1: 选择分化对象
    workflows = load_all_workflows()
    selected_workflow, target_category = select_for_split(workflows)
    
    if selected_workflow is None:
        return None  # 无合适对象
    
    # 阶段2: 加载目标类别样本
    classifications = load_problem_classifications()
    target_problems = [
        p for p in classifications 
        if p['category'] == target_category
    ][:10]
    
    # 阶段3: 生成专业化workflow
    base_code = load_workflow_code(selected_workflow)
    specialized_code = llm_generate_specialized(
        base_code=base_code,
        target_category=target_category,
        example_problems=target_problems
    )
    
    # 阶段4: 评估
    save_workflow(current_round + 1, specialized_code)
    score = evaluate_on_full_validation_set()
    
    # 阶段5: 更新元数据
    N_s += 1
    category_last_differentiation[target_category] = current_round
    
    return score

def select_for_split(workflows):
    """选择最佳分化对象"""
    best_potential = 0
    best_workflow = None
    best_category = None
    
    for workflow in workflows:
        # 加载统计数据
        category_stats = load_category_stats(workflow)
        category_metadata = load_category_metadata()
        
        # 计算全局性能
        c_total = sum(category_stats.values())
        n = sum(category_metadata.values())
        acc_global = c_total / n
        
        # 对每个类别计算分化潜力
        for category, n_k in category_metadata.items():
            c_k = category_stats.get(category, 0)
            recall_k = c_k / n_k
            
            if recall_k > acc_global:  # 有优势
                contrib_k = c_k / c_total
                score_k = contrib_k * (recall_k - acc_global)
                
                if score_k > best_potential:
                    best_potential = score_k
                    best_workflow = workflow
                    best_category = category
    
    return best_workflow, best_category
```
