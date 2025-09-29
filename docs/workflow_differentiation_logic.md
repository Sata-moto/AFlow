# AFlow 工作流分化执行逻辑详细说明

## 概述

工作流分化（Workflow Differentiation）是 AFlow 系统的一个关键特性，它通过创建现有工作流的专门化版本来探索解决方案空间的不同区域。分化过程基于特定方向对高性能工作流进行改进，以提升在特定问题类型或策略上的表现。

## 1. 分化触发条件

### 1.1 启用条件检查
分化执行需要满足以下所有条件（在 `EnhancedOptimizer._should_attempt_differentiation()` 中检查）：

```python
# 基础启用检查
- enable_differentiation = True  # 分化功能已启用
- round >= 2  # 不在第一轮执行（需要有历史数据）

# 资源限制检查  
- differentiation_rounds_used < max_differentiation_rounds  # 默认最大5轮
- len(workflows_data) >= 2  # 至少有2个工作流可供选择

# 随机概率检查
- random.random() <= differentiation_probability  # 默认30%概率
```

### 1.2 配置参数
```python
# 分化相关配置参数
enable_differentiation: bool = True                    # 是否启用分化
differentiation_probability: float = 0.3              # 分化概率（vs常规优化）
max_differentiation_rounds: int = 5                   # 每个优化周期最大分化轮数
```

## 2. 候选工作流选择

### 2.1 选择策略
分化候选选择基于性能分析（在 `WorkflowDifferentiation.analyze_differentiation_candidates()` 中实现）：

**高性能候选（前3名）**：
- **选择原因**：`high_performance`
- **建议方向**：`["strategy_diversification", "problem_type_specialization"]`
- **优先级**：1
- **目标**：在已有优势基础上进一步专业化

**中等性能候选（前50%）**：
- **选择原因**：`moderate_performance`  
- **建议方向**：`["algorithmic_approach_variation", "complexity_adaptation"]`
- **优先级**：2
- **目标**：通过算法变化提升性能

### 2.2 选择流程
```python
# 1. 按分数排序所有工作流
sorted_workflows = sorted(workflow_results, key=lambda x: x.get("avg_score", 0), reverse=True)

# 2. 分类候选工作流
for i, workflow in enumerate(sorted_workflows):
    if i < 3:  # 前3名 - 高性能
        candidates.append({
            "workflow": workflow,
            "reason": "high_performance",
            "suggested_directions": ["strategy_diversification", "problem_type_specialization"],
            "priority": 1
        })
    elif i < len(sorted_workflows) // 2:  # 中位数以上 - 中等性能
        candidates.append({
            "workflow": workflow, 
            "reason": "moderate_performance",
            "suggested_directions": ["algorithmic_approach_variation", "complexity_adaptation"],
            "priority": 2
        })

# 3. 选择最高优先级候选（通常是最高分工作流）
selected_candidate = candidates[0]["workflow"]
```

## 3. 分化方向与种类

### 3.1 可用分化方向
系统定义了5种主要分化方向：

#### 3.1.1 问题类型专业化 (`problem_type_specialization`)
**目标**：针对特定类型问题进行专门优化

**数学问题**：
- 专攻特定数学领域（代数、几何、微积分、组合数学等）
- 关注问题模式、定理应用、领域特定推理策略

**编程问题**：
- 专攻特定编程范式（算法、数据结构、优化等）
- 关注代码模式、效率考虑、实现策略

**问答问题**：
- 专攻特定推理类型（事实性、推理性、分析性等）
- 关注信息处理模式、推理链构建

#### 3.1.2 策略多样化 (`strategy_diversification`)
**目标**：引入替代性问题解决策略

**数学策略**：
- 视觉/几何方法 vs 代数方法
- 构造性证明 vs 反证法
- 递归思维 vs 迭代思维
- 模式识别 vs 第一性原理

**编程策略**：
- 自底向上 vs 自顶向下设计
- 迭代 vs 递归解决方案
- 空间优化 vs 时间优化方法
- 函数式 vs 命令式范式

**问答策略**：
- 演绎 vs 归纳推理
- 整体性 vs 分析性思维
- 上下文驱动 vs 规则驱动方法
- 多视角分析

#### 3.1.3 算法方法变化 (`algorithmic_approach_variation`)
**目标**：引入不同的算法范式和计算方法

**数学算法**：
- 符号计算、数值方法、图论思维
- 优化技术、概率推理

**编程算法**：
- 动态规划、贪心算法、分治策略
- 回溯法、机器学习方法

**问答算法**：
- 语义分析、逻辑推理、证据聚合
- 多步推理链

#### 3.1.4 复杂度适应 (`complexity_adaptation`)
**目标**：适应不同复杂度级别，实现分层处理

**分层策略**：
- 简单情况：直接方法
- 中等情况：增强推理
- 复杂情况：高级技术和验证

#### 3.1.5 错误模式处理 (`error_pattern_handling`)
**目标**：专门处理特定错误模式和失败场景

**错误处理策略**：
- 验证步骤引入
- 替代解决路径
- 错误模式识别
- 基于性能分析的特定错误处理

### 3.2 分化方向选择逻辑

```python
def select_differentiation_direction(workflow, existing_directions, performance_gaps):
    # 1. 过滤已使用的方向
    available_directions = [d for d in all_directions if d not in existing_directions]
    
    # 2. 基于工作流性能选择偏好方向
    workflow_score = workflow.get("avg_score", 0.0)
    
    if workflow_score > 0.7:        # 高性能工作流
        preferred = ["strategy_diversification", "problem_type_specialization"]
    elif workflow_score > 0.4:      # 中等性能工作流  
        preferred = ["algorithmic_approach_variation", "complexity_adaptation"]
    else:                          # 低性能工作流
        preferred = ["error_pattern_handling", "complexity_adaptation"]
    
    # 3. 选择第一个可用的偏好方向
    for direction in preferred:
        if direction in available_directions:
            return direction
    
    # 4. 备用方案：选择第一个可用方向
    return available_directions[0] if available_directions else "strategy_diversification"
```

## 4. 分化过程详述

### 4.1 整体执行流程

```python
async def _attempt_differentiation() -> float:
    """完整的分化执行流程"""
    
    # 1. 记录分化尝试
    self.differentiation_rounds_used += 1
    
    # 2. 获取工作流数据并生成候选
    workflow_results = self.data_utils.load_results(f"{self.root_path}/workflows")
    round_summaries = self.workflow_manager.get_round_summaries(workflow_results)
    candidates = self.differentiation_processor.analyze_differentiation_candidates(round_summaries)
    
    # 3. 选择最佳候选和分化方向
    selected_candidate = candidates[0]["workflow"]  # 最高优先级
    differentiation_direction = self.differentiation_processor.select_differentiation_direction(
        selected_candidate, self.used_differentiation_directions, performance_gaps=[]
    )
    
    # 4. 记录使用的方向
    self.used_differentiation_directions.append(differentiation_direction)
    
    # 5. 加载源工作流内容
    source_workflow_content = await self.workflow_manager.load_workflow_content(selected_candidate["round"])
    operator_description = self.graph_utils.load_operators_description(self.operators)
    
    # 6. 创建分化工作流
    differentiation_response = await self.differentiation_processor.create_differentiated_workflow(
        source_workflow=source_workflow_content,
        differentiation_direction=differentiation_direction,
        operator_description=operator_description
    )
    
    # 7. 保存分化工作流
    next_round = self.round + 1
    success = self.differentiation_processor.save_differentiated_workflow_direct(
        differentiation_response, selected_candidate, differentiation_direction, 
        next_round, self.root_path, self.graph_utils, self.experience_utils
    )
    
    # 8. 评估分化结果
    self.graph = self.graph_utils.load_graph(next_round, graph_path)
    data = self.data_utils.load_results(graph_path)
    differentiation_score = await self.evaluation_utils.evaluate_graph(
        self, directory, self.validation_rounds, data, initial=False
    )
    
    # 9. 保存分化元数据
    self.differentiation_processor.save_differentiation_metadata(
        source_workflow=selected_candidate,
        differentiated_workflow=differentiation_response,
        differentiation_direction=differentiation_direction,
        target_round=next_round,
        differentiation_score=differentiation_score
    )
    
    return differentiation_score
```

### 4.2 LLM调用和提示生成

**提示生成过程**：
```python
# 1. 提取源工作流信息
source_prompt = source_workflow.get("prompt", "")
source_graph = source_workflow.get("graph", "")  
source_score = source_workflow.get("score", 0.0)
solved_problems = source_workflow.get("solved_problems", set())

# 2. 生成方向特定指导
direction_guidance = self._get_direction_guidance(differentiation_direction, question_type, performance_gaps)

# 3. 构建完整提示
differentiation_prompt = f"""
You are an expert in {question_type} problem-solving workflow design. 
Your task is to create a SPECIALIZED version of an existing workflow by differentiating it in a specific direction.

## Current Workflow Analysis  
**Dataset**: {dataset}
**Current Score**: {source_score:.4f}
**Problems Solved**: {len(solved_problems)} problems
**Specialization Direction**: {differentiation_direction}

## Source Workflow
### Current Prompt:
{source_prompt}

### Current Graph:  
{source_graph}

## Specialization Objective
{direction_guidance}

## Available Operators
{operator_description}

## Output Format
<modification>[详细的分化描述]</modification>
<graph>[完整的Python类定义]</graph>
<prompt>[更新后的提示]</prompt>
"""
```

### 4.3 结果解析和验证

```python
# 1. LLM响应解析
response = await self._call_differentiation_llm(differentiation_prompt)
parsed_fields = CodeProcessor.extract_fields_from_response(
    response, ['modification', 'graph', 'prompt']
)

# 2. 代码清理（处理```符号问题）
cleaned_graph = CodeProcessor.clean_code_content(parsed_fields.get('graph', ''))
cleaned_prompt = CodeProcessor.clean_code_content(parsed_fields.get('prompt', ''))

# 3. 结构化输出
return {
    "modification": parsed_fields.get('modification', ''),
    "graph": cleaned_graph, 
    "prompt": cleaned_prompt
}
```

## 5. 分化结果保存和管理

### 5.1 工作流文件保存

分化工作流保存到下一轮目录：`{root_path}/workflows/round_{next_round}/`

**保存内容**：
- `graph.py` - 分化后的工作流图实现
- `prompt.py` - 分化后的提示
- `__init__.py` - Python包初始化文件
- `experience.json` - 经验记录文件
- `log.json` - 执行日志文件

### 5.2 经验文件生成

```python
# experience.json 结构
{
    "father_node": source_workflow["round"],           # 源工作流轮次
    "before": source_workflow["avg_score"],           # 分化前分数
    "after": null,                                    # 分化后分数（评估后填入）
    "modification": differentiation_modification,      # 分化描述
    "timestamp": current_timestamp
}

# differentiation_modification 格式
f"Workflow Differentiation: Specialized from round {source_workflow['round']} "
f"(score: {source_workflow['avg_score']:.4f}) in direction '{differentiation_direction}'. "
f"{differentiation_response.get('modification', 'Enhanced with targeted specialization.')[:200]}"
```

### 5.3 日志文件生成

```python
# log.json 结构
{
    "differentiation_metadata": {
        "timestamp": current_time,
        "differentiation_type": "workflow_specialization",
        "source_workflow": {
            "round": source_workflow["round"],
            "score": source_workflow["avg_score"], 
            "solved_problems": len(source_workflow.get("solved_problems", []))
        },
        "differentiation_direction": differentiation_direction,
        "strategy": f"Specialized the workflow from round {source_workflow['round']} to focus on {differentiation_direction}"
    },
    "execution_logs": []  # 运行时填入实际执行日志
}
```

### 5.4 分化元数据保存

```python
# differentiation_metadata_{target_round}.json 结构  
{
    "differentiation_timestamp": current_time,
    "source_workflow": {
        "round": source_workflow.get("round"),
        "score": source_workflow.get("avg_score"),
        "solved_problems_count": len(source_workflow.get("solved_problems", []))
    },
    "differentiation_direction": differentiation_direction,
    "target_round": target_round,
    "differentiation_score": differentiation_score,
    "modification_summary": differentiated_workflow.get("modification", "")[:500]
}
```

## 6. 评估和性能跟踪

### 6.1 分化结果评估
```python
# 1. 加载分化后的工作流
self.graph = self.graph_utils.load_graph(next_round, graph_path)

# 2. 使用标准评估流程
differentiation_score = await self.evaluation_utils.evaluate_graph(
    self, directory, self.validation_rounds, data, initial=False
)

# 3. 更新经验文件
if experience_data.get("after") is None:
    self.experience_utils.update_experience(directory, experience_data, differentiation_score)
```

### 6.2 性能追踪指标

**基础指标**：
- 分化前后分数对比
- 问题解决数量变化
- 特定问题类型性能提升

**追踪机制**：
- `differentiation_rounds_used` - 已使用分化轮数
- `used_differentiation_directions` - 已使用分化方向列表
- 分化元数据文件 - 详细记录每次分化的结果

## 7. 优化策略优先级

在 `EnhancedOptimizer.optimize()` 中，策略执行优先级为：

```python
# 优先级 1: 工作流融合
if self._should_attempt_fusion():
    score = loop.run_until_complete(self._attempt_fusion())
    if score is not None: break

# 优先级 2: 工作流分化  
if score is None and self._should_attempt_differentiation():
    score = loop.run_until_complete(self._attempt_differentiation())
    if score is not None: break

# 优先级 3: 常规优化
if score is None:
    score = loop.run_until_complete(self._optimize_graph())
```

## 8. 关键设计要点

### 8.1 避免重复分化
- 跟踪已使用的分化方向 (`used_differentiation_directions`)
- 方向耗尽时重置列表
- 限制最大分化轮数避免过度专业化

### 8.2 保证分化质量
- 基于性能选择候选工作流
- 方向特定的指导生成
- 代码清理确保语法正确性

### 8.3 灵活性和扩展性
- 模块化分化处理器设计
- 可配置的分化参数
- 支持自定义分化方向

### 8.4 性能分析整合
- 性能缺陷分析指导分化方向
- 历史数据用于候选选择
- 分化效果持续跟踪

## 9. 总结

AFlow的工作流分化机制是一个复杂而精妙的系统，它通过以下方式增强优化过程：

1. **智能候选选择**：基于性能数据选择最有潜力的工作流进行分化
2. **多样化方向**：5种不同的分化方向覆盖问题解决的各个维度
3. **自适应策略**：根据工作流性能自动选择最合适的分化方向
4. **完整的生命周期管理**：从候选选择到结果评估的全流程自动化
5. **性能跟踪**：详细记录分化过程和结果，支持持续改进

这个机制使得AFlow能够在保持现有高性能工作流的同时，系统性地探索解决方案空间的不同区域，从而实现更全面和robust的问题求解能力。
