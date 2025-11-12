# 分化策略详解 (Differentiation Strategy)

## 1. 策略概述

分化策略将通用工作流特化为针对特定问题类型的专家工作流，通过专业化提升在特定领域的性能。

### 核心思想
```
通用工作流 + 目标问题类型 + 示例 → LLM特化 → 专家工作流
```

### 生物学类比
```
通用物种 → 适应特定生态位 → 专门物种

例如：
- 通用鸟类 → 适应水域捕鱼 → 鱼鹰
- 通用工作流 → 适应数学问题 → 数学专家工作流
```

## 2. 完整流程伪代码

```python
async def differentiate_workflow(round):
    """
    工作流分化的完整流程
    
    Returns:
        score: 分化后工作流的评估分数
    """
    
    # ====== 第1步：触发条件检查 ======
    if not enable_differentiation:
        return None
    
    # 动态概率检查（随轮次增加）
    current_prob = calculate_differentiation_probability(round)
    if random.random() > current_prob:
        return None  # 概率检查失败
    
    print(f"触发分化策略 (概率: {current_prob:.2%})")
    
    # ====== 第2步：选择源工作流 ======
    # 获取所有历史工作流的汇总信息
    workflow_summaries = get_workflow_summaries()
    
    # 更新分化计数信息
    for workflow in workflow_summaries:
        workflow.differentiation_count = get_differentiation_count(workflow.round)
    
    # 计算选择权重（考虑分数和分化次数）
    for workflow in workflow_summaries:
        # 分化次数越多，权重越低（避免过度分化）
        penalty = 0.95 ** workflow.differentiation_count
        workflow.selection_weight = workflow.score * penalty
    
    # MCTS 选择
    source_workflow = mcts_select(workflow_summaries)
    
    print(f"选择 Round {source_workflow.round} 作为源工作流")
    print(f"  原始分数: {source_workflow.score:.4f}")
    print(f"  已分化次数: {source_workflow.differentiation_count}")
    
    # ====== 第3步：选择目标问题类型 ======
    # 获取所有问题分类统计
    category_stats = get_category_statistics()
    
    # 示例输出：
    # {
    #     "Mathematical Derivation": {
    #         "count": 62,
    #         "description": "需要数学推导和计算",
    #         "last_differentiation_round": 5,
    #         "differentiation_count": 2
    #     },
    #     "Code Generation": {
    #         "count": 45,
    #         "description": "需要生成可执行代码",
    #         "last_differentiation_round": -1,  # 从未分化
    #         "differentiation_count": 0
    #     },
    #     ...
    # }
    
    # 计算每个类别的优先级
    for category, stats in category_stats.items():
        # 因素1: 问题数量（更多问题的类别优先级更高）
        count_factor = stats['count'] / total_problems
        
        # 因素2: 上次分化时间（越久未分化优先级越高）
        if stats['last_differentiation_round'] == -1:
            time_factor = 1.0  # 从未分化，最高优先级
        else:
            rounds_since = round - stats['last_differentiation_round']
            time_factor = min(1.0, rounds_since / 5)  # 5轮后完全恢复
        
        # 因素3: 分化次数（分化越少优先级越高）
        frequency_factor = 0.8 ** stats['differentiation_count']
        
        # 综合优先级
        stats['priority'] = count_factor * 0.5 + time_factor * 0.3 + frequency_factor * 0.2
    
    # 选择优先级最高的类别
    target_category = max(category_stats.items(), key=lambda x: x[1]['priority'])[0]
    category_info = category_stats[target_category]
    
    print(f"选择目标类别: {target_category}")
    print(f"  问题数量: {category_info['count']}")
    print(f"  优先级分数: {category_info['priority']:.4f}")
    
    # ====== 第4步：获取示例问题 ======
    # 从该类别中随机选择3个问题作为示例
    example_problems = get_problems_by_category(target_category, limit=3)
    
    print(f"获取了 {len(example_problems)} 个示例问题")
    
    # ====== 第5步：构建分化提示词并调用LLM ======
    differentiation_prompt = create_differentiation_prompt(
        source_workflow, target_category, example_problems
    )
    
    response = await optimize_llm.call_with_format(
        differentiation_prompt,
        format_schema=GraphOptimize
    )
    
    # ====== 第6步：保存并评估 ======
    new_round = round + 1
    save_differentiated_workflow(new_round, response, target_category)
    
    avg_score = await evaluate_workflow(new_round)
    
    print(f"分化工作流评估完成: {avg_score:.4f}")
    
    return avg_score
```

## 3. 问题分类系统

### 3.1 分类流程（逐个分类）

```python
async def classify_all_problems(validation_data):
    """
    在优化开始前对所有问题进行分类
    采用逐个分类策略，动态创建类别
    """
    classifications = {}
    existing_categories = {}
    
    for i, problem in enumerate(validation_data):
        # 构建分类提示词
        prompt = f"""
已有类别：
{format_existing_categories(existing_categories)}

待分类问题：
{format_problem(problem)}

要求：
1. 从解决方法层面分类（不是内容主题）
2. 如果属于已有类别，选择该类别
3. 如果不属于，创建新类别
4. 避免过于具体的类别名称

返回：category, description, is_new
"""
        
        response = await llm(prompt)
        
        # 更新分类
        classifications[problem['id']] = response['category']
        
        if response['is_new']:
            existing_categories[response['category']] = {
                "description": response['description'],
                "count": 1
            }
        else:
            existing_categories[response['category']]['count'] += 1
    
    return classifications
```

### 3.2 类别选择策略

```python
def select_target_category(category_stats, round, history):
    """
    选择最适合当前分化的目标类别
    
    综合考虑三个因素：
    1. 问题数量（30%） - 更多问题 = 更重要
    2. 上次分化时间（40%） - 避免短期重复
    3. 分化频率（30%） - 避免过度分化
    """
    for category, stats in category_stats.items():
        # 计算优先级分数
        count_factor = normalize(stats['count'])
        time_factor = calculate_time_factor(category, round, history)
        freq_factor = 0.8 ** history.get(category, {}).get('count', 0)
        
        stats['priority'] = (
            count_factor * 0.3 +
            time_factor * 0.4 +
            freq_factor * 0.3
        )
    
    return max(category_stats, key=lambda x: x['priority'])
```

## 4. 动态概率机制

```python
def calculate_differentiation_probability(round, max_rounds, base_prob):
    """
    计算当前轮次的分化概率
    
    策略：
    - 前 75% 轮次：从 base_prob 线性增长到 2*base_prob
    - 后 25% 轮次：保持 2*base_prob
    
    示例（base_prob=0.3, max_rounds=20）：
      Round 2:  prob = 0.30
      Round 8:  prob = 0.45
      Round 15: prob = 0.60 (达到最大)
      Round 20: prob = 0.60 (保持)
    """
    threshold = int(max_rounds * 0.75)
    
    if round <= threshold:
        progress = (round - 2) / (threshold - 2)
        prob = base_prob * (1 + progress)
    else:
        prob = 2 * base_prob
    
    return min(prob, 2 * base_prob)
```

## 5. 分化提示词核心要点

```python
differentiation_prompt = f"""
### 核心原则：保持结构，调整专业化

目标：将通用工作流特化为 {target_category} 专家

要求：
1. 保持工作流结构
   - 不改变操作符数量和连接关系
   - 不添加或删除步骤

2. 调整操作符角色
   - 将抽象描述转为该类别的具体描述
   - 调整提示词强调该类别特点
   - 优化参数适应该类别需求

3. 示例对比
   通用版：Analyst: "Analyze the problem"
   特化版：Analyst: "Identify mathematical relationships and constraints"

类别信息：
{category_description}

示例问题：
{format_examples(example_problems)}

当前工作流：
{source_graph}
{source_prompt}

请返回特化后的 graph 和 prompt
"""
```

## 6. 效果示例

```
通用工作流 (Round 5):
  总体: 65%
  数学: 60%
  代码: 70%
  
↓ 分化（针对数学）

数学专家 (Round 6):
  总体: 68% (+3%)
  数学: 75% (+15%)  ← 目标提升
  代码: 62% (-8%)   ← 可能下降

价值：
- 数学问题占50% → 总体+3%显著
- 创建专家工作流 → 可用于融合
```

## 7. 与其他策略协同

### 分化 + 优化
```
Round 5: 分化 → 数学专家
Round 6: 优化 → 改进数学专家
```

### 分化 + 融合
```
Round 3: 分化 → 数学专家
Round 5: 分化 → 代码专家
Round 7: 融合 → 综合专家
```

## 8. 调优建议

| 参数 | 太低 | 推荐 | 太高 |
|------|------|------|------|
| 基础概率 | <0.2 缺乏多样性 | 0.3-0.4 | >0.5 过度分化 |
| 最大轮次 | <3 多样性不足 | 5-7 | >8 过度碎片化 |
| 示例数量 | 1 代表性不足 | 3 | >5 提示词过长 |

---

**核心价值**：通过创建专门化工作流，在特定领域实现突破性提升。
