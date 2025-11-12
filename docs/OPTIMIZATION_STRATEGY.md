# 优化策略详解 (Optimization Strategy)

## 1. 策略概述

优化策略是 AFlow 框架的基础策略，通过分析历史经验和错误案例，逐步改进工作流设计。

### 核心思想
```
当前工作流 + 历史经验 + 错误分析 → LLM改进 → 新工作流
```

## 2. 完整流程伪代码

```python
async def optimize_workflow(round):
    """
    工作流优化的完整流程
    
    Args:
        round: 当前轮次
    
    Returns:
        score: 优化后工作流的评估分数
    """
    
    # ====== 第1步：选择父工作流 ======
    # 从历史工作流中选择表现最好的几个
    top_workflows = get_top_workflows(sample=3)
    
    # 使用 MCTS 算法选择一个作为父工作流
    # 考虑因素：分数、探索次数、UCB值
    parent_workflow = mcts_select(top_workflows)
    
    print(f"选择 Round {parent_workflow.round} 作为父工作流")
    print(f"  分数: {parent_workflow.score:.4f}")
    
    # ====== 第2步：加载父工作流内容 ======
    # 读取父工作流的代码和提示词
    parent_graph = load_graph_code(parent_workflow.round)
    parent_prompt = load_prompt_code(parent_workflow.round)
    
    # ====== 第3步：收集历史经验 ======
    # 从所有轮次的 experience.json 中提取改进经验
    all_experiences = load_all_experiences()
    
    # 过滤和格式化与当前父工作流相关的经验
    relevant_experience = format_experience_for_prompt(
        all_experiences, 
        parent_round=parent_workflow.round
    )
    
    # 经验示例：
    # """
    # Round 3 → Round 4 (分数: 0.65 → 0.72):
    #   修改: 添加了推理步骤验证机制
    #   
    # Round 5 → Round 6 (分数: 0.70 → 0.75):
    #   修改: 优化了答案提取逻辑
    # """
    
    # ====== 第4步：分析错误案例 ======
    # 从父工作流的 log.json 中提取失败案例
    error_cases = load_error_cases(parent_workflow.round, limit=3)
    
    # 错误案例示例：
    # """
    # 案例1:
    #   输入: "计算 15 * 23 + 7"
    #   预期输出: "352"
    #   实际输出: "345"
    #   错误: 计算错误
    #   
    # 案例2: ...
    # """
    
    # ====== 第5步：构建优化提示词 ======
    optimization_prompt = f"""
你是一个工作流优化专家。请根据以下信息改进工作流：

### 当前工作流性能
- 轮次: Round {parent_workflow.round}
- 准确率: {parent_workflow.score:.2%}
- 已解决问题: {len(parent_workflow.solved_problems)}

### 历史改进经验
{relevant_experience}

### 当前工作流代码

**图结构 (graph.py):**
```python
{parent_graph}
```

**提示词 (prompt.py):**
```python
{parent_prompt}
```

### 典型错误案例
{error_cases}

### 可用操作符
{operator_descriptions}

### 优化要求
1. 分析当前工作流的不足之处
2. 针对错误案例提出改进方案
3. 确保修改不会破坏已有的成功模式
4. 生成改进后的完整代码

请返回：
1. modification: 详细的修改说明
2. graph: 改进后的图结构代码
3. prompt: 改进后的提示词代码
"""
    
    # ====== 第6步：调用 LLM 生成改进方案 ======
    while True:
        # 使用优化专用的大模型
        response = await optimize_llm.call_with_format(
            optimization_prompt,
            format_schema=GraphOptimize  # Pydantic 模型
        )
        
        # 检查修改的独特性（避免重复之前的修改）
        is_unique = check_modification_uniqueness(
            response['modification'],
            all_experiences,
            parent_workflow.round
        )
        
        if is_unique:
            break
        else:
            print("修改与历史重复，重新生成...")
    
    # ====== 第7步：保存新工作流 ======
    new_round = round + 1
    new_directory = f"workspace/{dataset}/workflows/round_{new_round}"
    create_directory(new_directory)
    
    # 保存代码文件
    save_file(f"{new_directory}/graph.py", response['graph'])
    save_file(f"{new_directory}/prompt.py", response['prompt'])
    save_file(f"{new_directory}/__init__.py", "")
    
    # 保存经验记录（初始版本，不含评估后的分数）
    experience = {
        "father": parent_workflow.round,
        "before": parent_workflow.score,
        "modification": response['modification'],
        "after": None  # 待评估后填充
    }
    save_json(f"{new_directory}/experience.json", experience)
    
    # ====== 第8步：评估新工作流 ======
    # 动态加载新工作流
    new_graph = dynamic_import(f"{new_directory}/graph.py")
    
    # 在验证集上评估（运行 validation_rounds 次）
    scores = []
    for i in range(validation_rounds):
        result = await evaluate_workflow(new_graph, validation_data)
        scores.append(result.score)
    
    avg_score = mean(scores)
    
    print(f"新工作流评估完成:")
    print(f"  平均分数: {avg_score:.4f}")
    print(f"  提升幅度: {avg_score - parent_workflow.score:+.4f}")
    
    # ====== 第9步：更新经验文件 ======
    experience['after'] = avg_score
    save_json(f"{new_directory}/experience.json", experience)
    
    # 保存评估结果到 results.json
    append_result({
        "round": new_round,
        "score": avg_score,
        "solved_problems": result.solved_problems,
        "cost": result.total_cost
    })
    
    return avg_score
```

## 3. 关键组件详解

### 3.1 父工作流选择（MCTS）

```python
def mcts_select(workflows):
    """
    使用 Monte Carlo Tree Search 选择父工作流
    
    平衡探索（exploration）和利用（exploitation）
    """
    for workflow in workflows:
        # 计算 UCB (Upper Confidence Bound) 分数
        exploitation = workflow.score  # 利用：已知的好分数
        exploration = sqrt(log(total_trials) / workflow.visit_count)  # 探索：尝试少的
        
        ucb_score = exploitation + C * exploration
        workflow.selection_weight = ucb_score
    
    # 根据权重进行概率性选择
    selected = random.choices(
        workflows, 
        weights=[w.selection_weight for w in workflows]
    )[0]
    
    return selected
```

**选择逻辑**：
- 高分工作流更可能被选中（利用）
- 尝试次数少的工作流也有机会（探索）
- 避免陷入局部最优

### 3.2 经验格式化

```python
def format_experience_for_prompt(experiences, parent_round):
    """
    将历史经验格式化为 LLM 可理解的文本
    """
    formatted = []
    
    for exp in experiences:
        # 只包含与父工作流相关的经验链
        if exp['father'] == parent_round or exp['round'] in ancestry(parent_round):
            improvement = exp['after'] - exp['before']
            formatted.append(f"""
Round {exp['father']} → Round {exp['round']} (分数: {exp['before']:.2%} → {exp['after']:.2%}):
  修改: {exp['modification']}
  效果: {improvement:+.2%}
""")
    
    return "\n".join(formatted)
```

### 3.3 错误案例提取

```python
def load_error_cases(round_num, limit=3):
    """
    从 log.json 中提取典型错误案例
    """
    log_file = f"workflows/round_{round_num}/log.json"
    all_cases = load_json(log_file)
    
    # 筛选错误案例
    error_cases = [
        case for case in all_cases 
        if case['prediction'] != case['expected_output']
    ]
    
    # 随机选择有代表性的案例
    selected = random.sample(error_cases, min(limit, len(error_cases)))
    
    # 格式化
    formatted = []
    for i, case in enumerate(selected, 1):
        formatted.append(f"""
案例 {i}:
  输入: {case['input'][:100]}...
  预期输出: {case['expected_output']}
  实际输出: {case['prediction']}
  问题分析: {analyze_error(case)}
""")
    
    return "\n".join(formatted)
```

## 4. 优化策略的优势与局限

### 优势
✓ **稳定性高**：每次基于已验证的好工作流改进  
✓ **可追溯**：清晰的父子关系和改进历史  
✓ **针对性强**：基于实际错误案例改进  
✓ **经验累积**：历史改进知识不断积累

### 局限
✗ **局部优化**：难以跳出当前工作流的设计范式  
✗ **缺乏多样性**：总是在相似的方向上改进  
✗ **增量式改进**：难以实现突破性提升

## 5. 与其他策略的协同

### 与分化的协同
```
优化创建通用工作流 → 分化创建专家工作流 → 优化改进专家工作流
```

### 与融合的协同
```
优化创建多个好工作流 → 融合整合优势 → 优化微调融合结果
```

## 6. 实际效果示例

```
Round 1 → Round 2: +5%  (添加了验证步骤)
Round 2 → Round 3: +3%  (优化了提示词)
Round 3 → Round 4: +7%  (改进了答案提取)
Round 4 → Round 5: +2%  (修复了边界情况)
Round 5 → Round 6: +4%  (增强了推理链)

累计提升: ~21%
特点: 稳定、持续、可预测
```

## 7. 调优建议

### 父工作流采样数 (`sample`)
- **小值 (1-2)**：快速收敛，可能局部最优
- **大值 (5-10)**：探索充分，但计算开销大
- **推荐**：3-5

### 错误案例数量
- **太少 (<2)**：改进针对性不足
- **太多 (>5)**：提示词过长，重点不突出
- **推荐**：3

### 验证轮次
- **少 (1-2)**：评估不稳定
- **多 (>10)**：计算成本高
- **推荐**：5

---

**核心价值**：优化策略是框架的基石，提供稳定、可靠的性能提升路径。
