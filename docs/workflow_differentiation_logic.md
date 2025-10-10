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

### 2.1 重新设计的选择策略

**重要改进**：解决了原有选择逻辑的三个关键问题：

1. **避免分化已分化工作流**：检测并降权已经被分化出来的工作流
2. **不总是选择最好的工作流**：基于综合权重而非单纯性能分数进行选择
3. **考虑分化次数惩罚**：每次分化会降低该工作流未来被选择的概率

### 2.2 新的权重计算逻辑

```python
def analyze_differentiation_candidates(workflow_results):
    for workflow in workflow_results:
        # 1. 检查是否为分化工作流
        is_differentiated = self._is_differentiated_workflow(workflow)
        
        # 2. 获取基础性能分数
        base_score = workflow.get("avg_score", 0.0)
        
        # 3. 计算被分化次数惩罚
        differentiation_count = workflow.get("differentiation_count", 0)
        differentiation_penalty = differentiation_count * 0.05  # 每次分化减少5%权重
        
        # 4. 计算最终权重分数
        if is_differentiated:
            # 已分化工作流大幅降权，避免二次分化
            final_weight = base_score * 0.1  # 降到10%
        else:
            # 未分化工作流根据性能和分化次数计算权重
            final_weight = base_score * (1.0 - differentiation_penalty)
        
        # 5. 筛选条件：权重>0.3且分化次数<3
        if final_weight > 0.3 and differentiation_count < 3:
            candidates.append({
                "workflow": workflow,
                "final_weight": final_weight,  # 用于排序的关键指标
                "differentiation_count": differentiation_count,
                "is_differentiated": is_differentiated
            })
    
    # 按最终权重排序，而不是简单按分数排序
    candidates.sort(key=lambda x: x["final_weight"], reverse=True)
```

### 2.3 分化工作流检测

系统通过多种方式识别已分化的工作流：

```python
def _is_differentiated_workflow(workflow):
    indicators = [
        workflow.get("is_differentiated", False),
        workflow.get("differentiation_source") is not None,
        workflow.get("father_node_type") == "differentiation",
        "differentiation" in workflow.get("creation_type", "").lower(),
        "specialized" in workflow.get("description", "").lower(),
        # 检查分化元数据文件存在性
        os.path.exists(f"differentiation_metadata_{round}.json"),
        # 检查experience.json中的modification字段
        "differentiation" in experience_data.get('modification', '').lower()
    ]
    return any(indicators)
```

### 2.4 权重分类系统

**高优先级候选（权重 > 0.7）**：
- **特征**：高性能且未被过度分化
- **优先级**：1（最高）
- **分化原因**：`high_performance_specialization`

**中等优先级候选（0.3 < 权重 ≤ 0.7）**：
- **特征**：中等性能或被分化1-2次的高性能工作流
- **优先级**：2
- **分化原因**：`moderate_performance_specialization`

**低优先级候选（权重 ≤ 0.3 或已分化）**：
- **特征**：性能较低或已经被分化的工作流
- **优先级**：3（基本不考虑）
- **分化原因**：`secondary_differentiation` 或 `low_performance_rescue`

## 3. 分化方向与种类

### 3.1 重新设计的分化方向

**重要变更**：基于对分化本质的重新理解，我们将分化方向大幅简化，专注于真正的差异化而非优化。

#### 3.1.1 问题类型专业化 (`problem_type_specialization`) - 唯一保留的核心方向
**目标**：针对特定类型问题进行专门优化，创造真正的工作流差异化

**数学问题**：
- 专攻特定数学领域（代数、几何、微积分、组合数学等）
- 关注问题模式、定理应用、领域特定推理策略

**编程问题**：
- 专攻特定编程范式（算法、数据结构、优化等）
- 关注代码模式、效率考虑、实现策略

**问答问题**：
- 专攻特定推理类型（事实性、推理性、分析性等）
- 关注信息处理模式、推理链构建

#### 3.1.2 移除的"伪分化"方向
以下方向已被移除，因为它们本质上是优化而非分化，应该整合到常规优化过程中：

**已移除的方向**：
- ~~`strategy_diversification`~~ → 应该是常规优化的策略选择
- ~~`algorithmic_approach_variation`~~ → 应该是常规优化的算法探索
- ~~`complexity_adaptation`~~ → 应该是常规优化的适应性改进
- ~~`error_pattern_handling`~~ → 应该是常规优化的错误处理改进

### 3.2 简化的分化方向选择

```python
def select_differentiation_direction(workflow, existing_directions, performance_gaps):
    # 简化：直接返回唯一的真正分化方向
    return "problem_type_specialization"
```

## 4. 分化过程详述

### 4.1 改进的执行流程

```python
async def _attempt_differentiation() -> float:
    """改进后的分化执行流程，解决原有问题"""
    
    # 1. 记录分化尝试
    self.differentiation_rounds_used += 1
    
    # 2. 获取工作流数据并生成轮次摘要
    workflow_results = self.data_utils.load_results(f"{self.root_path}/workflows")
    round_summaries = self.workflow_manager.get_round_summaries(workflow_results)
    
    # 3. 更新工作流分化次数信息
    self._update_differentiation_counts(round_summaries)
    
    # 4. 分析候选工作流（考虑分化次数和权重）
    candidates = self.differentiation_processor.analyze_differentiation_candidates(round_summaries)
    
    # 5. 选择最佳候选（基于权重而非单纯分数）
    selected_candidate = candidates[0]["workflow"]  # 最高权重候选
    logger.info(f"Selected workflow from round {selected_candidate['round']}")
    logger.info(f"  Base score: {selected_candidate.get('avg_score', 0):.4f}")
    logger.info(f"  Final weight: {candidates[0].get('final_weight', 0):.4f}")
    logger.info(f"  Differentiation count: {candidates[0].get('differentiation_count', 0)}")
    logger.info(f"  Is differentiated: {candidates[0].get('is_differentiated', False)}")
    
    # 6. 更新该工作流的分化次数
    source_round = selected_candidate['round']
    self.workflow_differentiation_counts[source_round] = self.workflow_differentiation_counts.get(source_round, 0) + 1
    
    # 7. 选择分化方向（固定为问题类型专业化）
    differentiation_direction = self.differentiation_processor.select_differentiation_direction(
        selected_candidate, self.used_differentiation_directions, performance_gaps=[]
    )  # 返回 "problem_type_specialization"
    
    # 8. 加载源工作流内容
    source_workflow_content = await self.workflow_manager.load_workflow_content(selected_candidate["round"])
    operator_description = self.graph_utils.load_operators_description(self.operators)
    
    # 9. 创建分化工作流（专注于问题类型专业化）
    differentiation_response = await self.differentiation_processor.create_differentiated_workflow(
        source_workflow=source_workflow_content,
        differentiation_direction=differentiation_direction,
        operator_description=operator_description
    )
    
    # 10-12. 保存、评估、记录元数据（保持不变）
    # ...
    
    return differentiation_score
```

### 4.2 分化次数跟踪机制

```python
class EnhancedOptimizer:
    def __init__(self):
        # 新增：跟踪每个工作流被分化的次数
        self.workflow_differentiation_counts = {}  # {round: count}
    
    def _update_differentiation_counts(self, round_summaries):
        """将分化次数信息注入到工作流数据中"""
        for summary in round_summaries:
            round_num = summary.get('round')
            if round_num is not None:
                # 注入分化次数
                current_count = self.workflow_differentiation_counts.get(round_num, 0)
                summary['differentiation_count'] = current_count
                
                # 检查是否为分化工作流
                summary['is_differentiated'] = self._check_if_differentiated_workflow(round_num)
    
    def _check_if_differentiated_workflow(self, round_num):
        """检查工作流是否为分化产生的"""
        # 检查分化元数据文件
        metadata_file = f"differentiation_metadata_{round_num}.json"
        if os.path.exists(os.path.join(workflows_dir, metadata_file)):
            return True
            
        # 检查experience.json中的modification字段
        experience_file = f"round_{round_num}/experience.json"
        if os.path.exists(os.path.join(workflows_dir, experience_file)):
            with open(experience_file, 'r') as f:
                experience_data = json.load(f)
                modification = experience_data.get('modification', '').lower()
                return 'differentiation' in modification or 'specialized' in modification
        
        return False
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

## 9. 总结与关键改进

AFlow的工作流分化机制经过重新设计，解决了原有的关键问题，现在更加专注于真正的差异化：

### 9.1 核心设计原则

1. **真正的分化 vs 伪分化**：只保留 `problem_type_specialization`，移除其他优化性质的"分化"方向
2. **智能候选选择**：基于综合权重而非单纯性能进行选择，避免总是分化同一个工作流
3. **分化次数控制**：通过权重衰减机制避免过度分化，每次分化降低5%权重
4. **已分化工作流识别**：通过多种机制识别并降权已分化的工作流，避免二次分化

### 9.2 解决的关键问题

**问题1：避免分化已分化工作流**
- ✅ **解决方案**：通过元数据文件和experience.json检测已分化工作流
- ✅ **效果**：已分化工作流权重降至10%，基本不会被选中

**问题2：避免总是选择最佳工作流**  
- ✅ **解决方案**：基于综合权重（性能×分化惩罚）选择候选
- ✅ **效果**：高性能但已被多次分化的工作流会被降权

**问题3：控制单个工作流分化次数**
- ✅ **解决方案**：引入 `workflow_differentiation_counts` 跟踪机制
- ✅ **效果**：每次分化增加5%权重惩罚，最多分化3次后排除

### 9.3 权重计算公式

```python
# 核心权重计算逻辑
if is_differentiated:
    final_weight = base_score * 0.1  # 已分化：大幅降权
else:
    differentiation_penalty = differentiation_count * 0.05  # 每次分化5%惩罚
    final_weight = base_score * (1.0 - differentiation_penalty)

# 筛选条件
if final_weight > 0.3 and differentiation_count < 3:
    # 纳入候选
```

### 9.4 分化方向简化

- **保留**：`problem_type_specialization` - 真正的工作流差异化
- **移除**：`strategy_diversification`, `algorithmic_approach_variation`, `complexity_adaptation`, `error_pattern_handling` 
- **原因**：后者本质上是优化而非分化，应整合到常规优化过程中

### 9.5 系统优势

1. **专注性**：只做真正的分化，不与优化功能重叠
2. **智能性**：避免无效的重复分化和过度分化
3. **平衡性**：在探索多样性和避免无效分化间取得平衡
4. **可追踪性**：完整记录每个工作流的分化历史
5. **扩展性**：为未来添加新的真正分化方向提供框架

这个重新设计的分化机制真正实现了工作流差异化的目标，同时避免了与常规优化功能的冗余，使整个AFlow系统更加清晰和高效。
