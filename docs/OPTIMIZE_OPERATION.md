# 优化操作详解 (OPTIMIZE Operation)

## 文档版本
- **创建日期**: 2025-12-15
- **相关代码**: `scripts/enhanced_optimizer.py`, `scripts/optimizer.py`
- **依赖文档**: [系统架构总览](SYSTEM_ARCHITECTURE.md)

---

## 1. 操作概述

### 1.1 核心目标
优化操作是AFlow的**基础操作**,通过分析历史失败案例,利用LLM生成改进方案,创建性能更好的新workflow。

### 1.2 设计理念
- **经验驱动**: 从失败中学习,避免重复错误
- **渐进式改进**: 基于最佳workflow进行增量优化
- **知识积累**: 保留完整的优化历史和经验轨迹

### 1.3 触发条件
- **概率采样**: 根据 `_calculate_operation_probabilities()` 计算的概率选择
- **通常概率**: 在性能正常提升时 > 80%
- **强制触发**: 前几轮因 `plateau=0` 导致 p_opt=1.0

---

## 2. 完整流程

### 2.1 流程图

```
┌────────────────────────────────────────────────────────────────┐
│                   OPTIMIZE 操作流程                             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第1阶段: 选择基础Workflow                                      │
│                                                                 │
│  1. 加载 results.json                                          │
│  2. 按 avg_score 降序排序                                       │
│  3. 选择分数最高的round                                         │
│                                                                 │
│  示例: Round 2 (score=0.8140) > Round 1 (score=0.7326)       │
│  → 选择 Round 2 作为优化基础                                    │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第2阶段: 提取失败经验                                          │
│                                                                 │
│  1. 加载 round_2/log.json                                      │
│  2. 筛选失败案例 (score < 0.5)                                  │
│  3. 分析错误模式                                                │
│  4. 生成 experience.json:                                      │
│     {                                                          │
│       "before": {                                              │
│         "method": "当前workflow描述",                           │
│         "performance": "失败案例统计"                           │
│       },                                                       │
│       "failure_examples": [                                    │
│         {                                                      │
│           "problem": "问题描述",                                │
│           "error": "错误原因",                                  │
│           "expected": "期望答案"                                │
│         }                                                      │
│       ]                                                        │
│     }                                                          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第3阶段: 格式化经验数据                                        │
│                                                                 │
│  1. 调用 Formatter.format_experience()                         │
│  2. 将所有轮次的experience合并                                  │
│  3. 生成 processed_experience.json                             │
│  4. 包含完整的优化历史轨迹                                       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第4阶段: LLM生成改进方案                                       │
│                                                                 │
│  输入:                                                          │
│  - 当前workflow (round_2/graph.py + prompt.py)                │
│  - processed_experience.json (失败案例)                        │
│  - 优化提示词模板                                               │
│                                                                 │
│  LLM任务:                                                       │
│  - 分析失败原因                                                 │
│  - 识别改进机会                                                 │
│  - 生成新的graph.py和prompt.py                                 │
│                                                                 │
│  输出:                                                          │
│  - 改进后的workflow代码                                         │
│  - 修改说明                                                     │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第5阶段: 创建新Workflow                                        │
│                                                                 │
│  1. 创建 round_3/ 目录                                         │
│  2. 保存 graph.py (新workflow代码)                             │
│  3. 保存 prompt.py (新提示词)                                   │
│  4. 保存 __init__.py (模块初始化)                              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第6阶段: 评估新Workflow                                        │
│                                                                 │
│  1. 在验证集上执行新workflow                                    │
│  2. 生成 log.json (详细执行日志)                                │
│  3. 计算 avg_score                                             │
│  4. 生成 experience.json (本轮经验)                            │
│  5. 更新 results.json                                          │
│  6. 更新 performance_history[]                                 │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
                        返回新workflow分数
```

### 2.2 关键代码位置

#### 2.2.1 主入口
**文件**: `scripts/enhanced_optimizer.py`
**行数**: 261-280

```python
if operation == 'optimize':
    logger.info("=" * 80)
    logger.info(f"Executing OPTIMIZE operation for round {self.round + 1}")
    logger.info("=" * 80)
    
    # 执行优化
    score = await self._optimize_graph(mode)
    
    if score is not None:
        # 优化成功,继续下一轮
        pass
```

#### 2.2.2 优化核心逻辑
**文件**: `scripts/optimizer.py`
**行数**: 72-174

```python
async def _optimize_graph(self, mode: OptimizerType = "Graph") -> float:
    """
    优化工作流的核心方法
    
    Args:
        mode: 优化模式 ("Graph" 或其他)
    
    Returns:
        float: 新workflow的评估分数
    """
    # 第1轮特殊处理
    if self.round == 1:
        # 评估初始workflow
        directory = self.graph_utils.create_round_directory(graph_path, self.round)
        self.graph = self.graph_utils.load_graph(self.round, graph_path)
        avg_score = await self.evaluation_utils.evaluate_graph(
            self, directory, validation_n, data, initial=True
        )
        # 不return,继续创建round 2
    
    # 循环直到生成有效的workflow
    while True:
        # 1. 创建新round目录
        directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)
        
        # 2. 格式化历史经验
        self.formatter.format_experience(...)
        
        # 3. 选择最佳workflow作为基础
        results = self.data_utils.load_results(graph_path)
        selected_round = self._select_best_workflow(results)
        
        # 4. 调用LLM生成改进方案
        await self.graph_opt.optimize(...)
        
        # 5. 检查生成的workflow是否有效
        self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)
        if self.graph is None:
            # 重试
            continue
        
        # 6. 评估新workflow
        avg_score = await self.evaluation_utils.evaluate_graph(
            self, directory, validation_n, data, initial=False
        )
        
        return avg_score
```

---

## 3. 详细阶段分析

### 3.1 阶段1: 选择基础Workflow

**目标**: 选择历史上表现最好的workflow作为优化起点

**算法**:
```python
def _select_best_workflow(self, results):
    """选择分数最高的workflow"""
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return sorted_results[0]['round']
```

**为什么不总是选择最新的workflow?**
- 最新workflow可能因为分化/融合导致性能下降
- 选择最佳workflow确保从最好的起点出发
- 避免在错误方向上继续优化

**示例**:
```
Results:
  Round 1: 0.7326
  Round 2: 0.8140  ← 最高分,被选中
  Round 3: 0.7907
  Round 4: 0.8605  ← 最新,但可能还不稳定

选择: Round 2 作为优化基础
```

### 3.2 阶段2: 提取失败经验

**目标**: 从历史执行中识别失败模式和错误原因

**代码位置**: `scripts/optimizer_utils/experience_utils.py`

**提取逻辑**:
```python
def extract_experience(self, log_path, graph_description):
    """从log.json提取失败案例"""
    log_data = load_json(log_path)
    
    failure_examples = []
    for entry in log_data:
        if entry['score'] < 0.5:  # 失败案例
            failure_examples.append({
                'problem': entry['question'],
                'model_output': entry['model_output'],
                'right_answer': entry['right_answer'],
                'error_analysis': entry['judge_explanation']
            })
    
    # 生成experience.json
    experience = {
        'before': {
            'method': graph_description,
            'performance': f"Failed on {len(failure_examples)} problems"
        },
        'failure_examples': failure_examples[:10]  # 限制数量
    }
    
    return experience
```

**经验数据结构**:
```json
{
  "before": {
    "method": "Current workflow uses step-by-step reasoning...",
    "performance": "Failed on 23 out of 86 problems"
  },
  "after": {
    "score": 0.8140,
    "timestamp": "2025-12-15T10:10:39"
  },
  "failure_examples": [
    {
      "problem": "A convex pentagon has two congruent acute angles...",
      "model_output": "The answer is 120 degrees",
      "right_answer": "135",
      "error_analysis": "Model miscalculated the angle sum formula"
    },
    {
      "problem": "How many base-10 integers are exactly 4 digits...",
      "model_output": "There are 10 such integers",
      "right_answer": "9",
      "error_analysis": "Off-by-one error in counting"
    }
  ]
}
```

### 3.3 阶段3: 格式化经验数据

**目标**: 将多轮经验合并为结构化格式,便于LLM理解

**代码位置**: `scripts/formatter.py`

**格式化逻辑**:
```python
def format_experience(self, workflow_path, round_num):
    """合并所有轮次的经验"""
    all_experiences = []
    
    # 收集所有round的experience.json
    for r in range(1, round_num + 1):
        exp_path = f"{workflow_path}/round_{r}/experience.json"
        if exists(exp_path):
            exp = load_json(exp_path)
            all_experiences.append({
                'round': r,
                'score': exp['after']['score'],
                'failures': exp['failure_examples']
            })
    
    # 生成processed_experience.json
    processed = {
        'optimization_history': all_experiences,
        'key_insights': extract_common_patterns(all_experiences),
        'improvement_suggestions': generate_suggestions(all_experiences)
    }
    
    save_json(f"{workflow_path}/processed_experience.json", processed)
```

**processed_experience.json 示例**:
```json
{
  "optimization_history": [
    {
      "round": 1,
      "score": 0.7326,
      "key_failures": [
        "Geometry problems: 5/10 failed",
        "Combinatorics: 3/8 failed"
      ]
    },
    {
      "round": 2,
      "score": 0.8140,
      "improvements": "Better handling of multi-step problems",
      "remaining_issues": "Still struggles with geometric proofs"
    }
  ],
  "key_insights": [
    "Geometric problems require explicit diagram analysis",
    "Combinatorial counting needs systematic case enumeration"
  ],
  "improvement_suggestions": [
    "Add explicit geometric reasoning steps",
    "Implement structured counting framework"
  ]
}
```

### 3.4 阶段4: LLM生成改进方案

**目标**: 利用LLM分析经验并生成改进的workflow

**代码位置**: `scripts/operators.py` - `GraphOptimizer.optimize()`

**提示词结构**:
```python
optimization_prompt = f"""
You are an AI workflow optimizer. Your task is to improve the given workflow 
based on failure analysis.

## Current Workflow
```python
{current_graph_code}
```

```python
{current_prompt_code}
```

## Performance History
{processed_experience}

## Current Performance
- Score: {current_score}
- Solved: {solved_count}/{total_problems}
- Failed Categories: {failure_categories}

## Failure Examples
{failure_examples}

## Your Task
1. Analyze why the current workflow fails on these examples
2. Identify specific weaknesses in the reasoning process
3. Propose concrete improvements to address these weaknesses
4. Generate improved `graph.py` and `prompt.py`

## Requirements
- Maintain the same function signatures
- Keep the workflow structure compatible
- Focus on addressing the identified failure patterns
- Provide clear comments explaining your improvements

## Output Format
Please provide the improved workflow code:

```python
# graph.py
<improved graph code>
```

```python
# prompt.py
<improved prompt code>
```

Explanation of improvements:
<your explanation>
"""
```

**LLM响应解析**:
```python
def parse_llm_response(response_text):
    """从LLM响应中提取代码"""
    # 提取graph.py
    graph_match = re.search(r'```python\s*\n# graph\.py\n(.*?)\n```', 
                           response_text, re.DOTALL)
    graph_code = graph_match.group(1) if graph_match else None
    
    # 提取prompt.py
    prompt_match = re.search(r'```python\s*\n# prompt\.py\n(.*?)\n```', 
                            response_text, re.DOTALL)
    prompt_code = prompt_match.group(1) if prompt_match else None
    
    return graph_code, prompt_code
```

### 3.5 阶段5: 创建新Workflow

**目标**: 将LLM生成的代码保存为新的workflow

**代码位置**: `scripts/optimizer_utils/graph_utils.py`

**创建逻辑**:
```python
def create_round_directory(self, workflow_path, round_num):
    """创建新round目录并保存workflow"""
    directory = f"{workflow_path}/round_{round_num}"
    os.makedirs(directory, exist_ok=True)
    
    # 保存graph.py
    graph_file = f"{directory}/graph.py"
    with open(graph_file, 'w', encoding='utf-8') as f:
        f.write(graph_code)
    
    # 保存prompt.py
    prompt_file = f"{directory}/prompt.py"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt_code)
    
    # 创建__init__.py
    init_file = f"{directory}/__init__.py"
    with open(init_file, 'w') as f:
        f.write("")
    
    return directory
```

**目录结构**:
```
workspace/MATH/workflows/round_3/
├── __init__.py
├── graph.py          # 新的workflow代码
├── prompt.py         # 新的提示词
├── log.json          # 评估后生成
├── experience.json   # 评估后生成
└── 0.8256_timestamp.csv  # 评估后生成
```

### 3.6 阶段6: 评估新Workflow

**目标**: 在验证集上执行新workflow并记录结果

**代码位置**: `scripts/optimizer_utils/evaluation_utils.py`

**评估流程**:
```python
async def evaluate_graph(self, optimizer, directory, validation_rounds, data, initial=False):
    """评估workflow性能"""
    # 1. 加载验证数据
    validation_data = await benchmark.load_data(validation_rounds)
    
    # 2. 在验证集上执行workflow
    results = await benchmark.run_evaluation(
        agent=optimizer.graph,
        va_list=validation_rounds,
        max_concurrent_tasks=30
    )
    
    # 3. 提取评估结果
    avg_score = results[0]  # 平均分数
    total_cost = results[2]  # 总成本
    solved_count = results[3]  # 解决问题数
    
    # 4. 保存到results.json
    self.data_utils.save_result(
        optimizer.root_path,
        optimizer.round if initial else optimizer.round + 1,
        avg_score,
        total_cost,
        solved_count
    )
    
    # 5. 提取经验
    log_path = f"{directory}/log.json"
    if os.path.exists(log_path):
        graph_description = extract_graph_description(optimizer.graph)
        experience = self.experience_utils.extract_experience(
            log_path, graph_description
        )
        self.experience_utils.save_experience(directory, experience)
    
    # 6. 更新经验的after字段
    self.experience_utils.update_experience(directory, experience, avg_score)
    
    return avg_score
```

---

## 4. 优化策略分析

### 4.1 经验利用策略

**为什么关注失败案例?**
1. **信息量大**: 失败案例明确指出workflow的弱点
2. **可操作性强**: 比成功案例更容易找到改进方向
3. **避免重复**: 记录历史失败避免重蹈覆辙

**为什么限制失败案例数量?**
```python
failure_examples[:10]  # 只保留前10个
```
- 避免提示词过长
- 关注最典型的失败模式
- 提高LLM处理效率

### 4.2 渐进式改进策略

**为什么总是基于最佳workflow优化?**
```
Round 2: 0.8140  ← 最佳
Round 3: 0.7907  ← 基于Round 2优化,但下降
Round 4: 0.8605  ← 仍然基于Round 2(最佳)优化,成功提升!
```

**优势**:
- 确保从最好的起点出发
- 即使某次优化失败,下次仍可从好的起点重试
- 避免在错误方向上越走越远

### 4.3 重试机制

**为什么需要while True循环?**
```python
while True:
    # 生成新workflow
    self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)
    
    if self.graph is None:
        # LLM生成的代码有语法错误或无法加载
        logger.warning("Generated workflow is invalid, retrying...")
        continue  # 重试
    
    # workflow有效,评估并返回
    return avg_score
```

**保护措施**:
- 设置最大重试次数(通常3-5次)
- 记录每次失败的原因
- 如果持续失败,可能需要人工介入

---

## 5. 关键参数

### 5.1 优化相关参数

| 参数 | 默认值 | 说明 | 影响 |
|------|--------|------|------|
| `validation_rounds` | `[0, 1, 2, ...]` | 验证集题目索引 | 评估样本量 |
| `max_retries` | 3 | workflow生成最大重试次数 | 容错能力 |
| `max_failure_examples` | 10 | 记录的失败案例数量 | 提示词长度 |
| `experience_window` | 所有历史 | 使用的经验轮数 | 记忆容量 |

### 5.2 性能阈值

| 阈值 | 值 | 说明 |
|------|-----|------|
| `failure_threshold` | 0.5 | 低于此分数视为失败 |
| `improvement_threshold` | 0.01 | 显著改进的最小提升 |

---

## 6. 典型执行场景

### 6.1 正常优化流程

```
[Round 2 → Round 3 优化]

输入:
- 基础workflow: Round 2 (score=0.8140)
- 失败案例: 16个失败问题
- 历史经验: Round 1-2的完整经验

LLM分析:
- 发现: 几何题目推理不足
- 改进: 添加显式几何分析步骤

输出:
- 新workflow: Round 3
- 评估分数: 0.8256 (+1.4%)
- 几何题目: 8/10 → 9/10 ✓
```

### 6.2 优化失败场景

```
[Round 3 → Round 4 优化失败]

输入:
- 基础workflow: Round 2 (score=0.8140, 最佳)
  (注意: 不是Round 3, 因为Round 3分数下降了)
- 失败案例: 包含组合计数问题

LLM尝试:
- 改进: 添加系统化计数框架
- 问题: 引入了过于复杂的逻辑

输出:
- 新workflow: Round 4
- 评估分数: 0.7523 (-7.6%) ✗
- 发现: 过度优化导致性能下降

下一轮:
- 仍然基于 Round 2 (最佳) 重新优化
- 避免重复Round 4的错误
```

### 6.3 第1轮特殊处理

```
[Round 1 初始化]

特殊逻辑:
if self.round == 1:
    # 先评估初始workflow
    evaluate_initial_workflow()
    # 然后继续创建round 2
    # (不返回,继续执行优化循环)

结果:
- Round 1: 初始workflow的评估结果
- Round 2: 基于Round 1优化的新workflow

为什么这样设计?
- 确保有初始baseline
- 第一次优化立即开始
- 符合"evaluate then optimize"的循环
```

---

## 7. 代码映射

### 7.1 主要类和方法

#### EnhancedOptimizer
```python
class EnhancedOptimizer(Optimizer):
    # 文件: scripts/enhanced_optimizer.py
    
    # 优化入口
    async def _optimize_graph(self, mode="Graph"):
        # Line 继承自Optimizer
        return await super()._optimize_graph(mode)
```

#### Optimizer (父类)
```python
class Optimizer:
    # 文件: scripts/optimizer.py
    
    # Line 72-174
    async def _optimize_graph(self, mode="Graph"):
        """核心优化逻辑"""
        pass
    
    # Line 176-192
    def _select_best_workflow(self, results):
        """选择最佳workflow"""
        pass
```

#### GraphOptimizer
```python
class GraphOptimizer:
    # 文件: scripts/operators.py
    
    # Line 156-280
    async def optimize(self, round_number, selected_round, ...):
        """调用LLM生成改进方案"""
        pass
```

#### EvaluationUtils
```python
class EvaluationUtils:
    # 文件: scripts/optimizer_utils/evaluation_utils.py
    
    # Line 23-85
    async def evaluate_graph(self, optimizer, directory, ...):
        """评估workflow性能"""
        pass
```

#### ExperienceUtils
```python
class ExperienceUtils:
    # 文件: scripts/optimizer_utils/experience_utils.py
    
    # Line 18-68
    def extract_experience(self, log_path, graph_description):
        """提取失败经验"""
        pass
    
    # Line 70-88
    def save_experience(self, directory, experience):
        """保存经验到experience.json"""
        pass
    
    # Line 90-110
    def update_experience(self, directory, experience, score):
        """更新经验的after字段"""
        pass
```

### 7.2 数据流映射

```
results.json  ────┐
                  │
                  ├───→ _select_best_workflow()
                  │     └─→ selected_round
                  │
log.json      ────┤
                  ├───→ extract_experience()
                  │     └─→ experience.json
                  │
experience.json ──┤
                  ├───→ format_experience()
                  │     └─→ processed_experience.json
                  │
processed_exp ────┤
current_graph ────┼───→ GraphOptimizer.optimize()
                  │     └─→ new_graph.py, new_prompt.py
                  │
new_graph ────────┤
validation_set ───┼───→ evaluate_graph()
                  │     └─→ log.json, new_score
                  │
new_score ────────┴───→ update results.json
                       update performance_history
```

---

## 8. 性能优化建议

### 8.1 加速优化循环

1. **并行评估**: 
   ```python
   max_concurrent_tasks=30  # 同时评估30个问题
   ```

2. **缓存验证数据**:
   ```python
   # 避免重复加载
   self.validation_data = await benchmark.load_data(validation_rounds)
   ```

3. **增量经验提取**:
   ```python
   # 只提取新增的失败案例
   new_failures = extract_new_failures(current_log, previous_log)
   ```

### 8.2 提升优化质量

1. **更丰富的经验数据**:
   ```python
   # 不仅记录失败,也记录接近失败的案例
   borderline_cases = [e for e in log if 0.3 < e['score'] < 0.6]
   ```

2. **分类失败模式**:
   ```python
   failures_by_type = {
       'calculation_error': [],
       'logic_error': [],
       'interpretation_error': []
   }
   ```

3. **渐进式提示词**:
   ```python
   # 根据历史表现调整提示词的详细程度
   if consecutive_improvements < 2:
       prompt_detail_level = "high"  # 更详细的指导
   ```

---

## 9. 常见问题

### Q1: 为什么优化有时会导致性能下降?
**A**: 
- LLM可能过度优化导致复杂化
- 某些改进在部分问题上有效,但损害了通用性
- 解决方案: 总是保留最佳workflow,即使优化失败也可以重新开始

### Q2: 如何避免陷入局部最优?
**A**:
- 停滞度检测会触发分化/融合操作
- 分化探索专业化方向
- 融合引入其他workflows的视角

### Q3: 优化操作的成本如何?
**A**:
- LLM调用成本: 每次优化1-2次API调用
- 评估成本: 取决于验证集大小
- 时间成本: 通常5-10分钟/轮

### Q4: 如何调试优化失败?
**A**:
```python
# 1. 检查LLM生成的代码
cat workspace/DATASET/workflows/round_N/graph.py

# 2. 检查经验数据
cat workspace/DATASET/workflows/processed_experience.json

# 3. 检查评估日志
cat workspace/DATASET/workflows/round_N/log.json
```

### Q5: 优化操作何时停止?
**A**:
- 达到max_rounds
- 收敛检测触发(连续Top-K分数变化<阈值)
- 手动停止

---

## 10. 最佳实践

### 10.1 配置建议

```yaml
# 基础配置
max_rounds: 20              # 充足的优化轮数
validation_rounds: [0-99]   # 合理的验证集大小
max_concurrent_tasks: 30    # 平衡速度和资源

# 优化参数
max_failure_examples: 10    # 典型失败案例数
experience_window: all      # 使用全部历史经验
```

### 10.2 监控指标

```python
# 关键监控
- 每轮优化耗时
- 平均分数趋势
- 失败案例数量变化
- LLM生成质量(是否频繁重试)
```

### 10.3 异常处理

```python
try:
    score = await self._optimize_graph(mode)
except Exception as e:
    logger.error(f"Optimization failed: {e}")
    # 降级策略: 跳过本轮,继续下一轮
    score = previous_best_score
```

---

## 11. 下一步

- [分化操作详解](DIFFERENTIATION_OPERATION.md) - 了解如何针对特定类别专业化
- [融合操作详解](FUSION_OPERATION.md) - 了解如何合并多个workflows
- [系统架构总览](SYSTEM_ARCHITECTURE.md) - 返回整体视图

---

## 附录: 完整代码示例

### A.1 简化的优化流程

```python
async def optimize_workflow(current_round, dataset):
    """简化的优化流程示例"""
    
    # 1. 选择最佳workflow
    results = load_results()
    best_round = max(results, key=lambda x: x['score'])['round']
    
    # 2. 加载workflow和经验
    current_workflow = load_workflow(best_round)
    experiences = load_all_experiences()
    
    # 3. 格式化提示词
    prompt = f"""
    Current workflow:
    {current_workflow}
    
    Historical failures:
    {experiences}
    
    Please improve the workflow.
    """
    
    # 4. 调用LLM
    response = await llm_client.generate(prompt)
    new_workflow = parse_workflow(response)
    
    # 5. 保存并评估
    save_workflow(current_round + 1, new_workflow)
    score = await evaluate_workflow(new_workflow, dataset)
    
    return score
```

### A.2 经验提取示例

```python
def extract_failures(log_path):
    """提取失败案例"""
    log = load_json(log_path)
    
    failures = []
    for entry in log:
        if entry['score'] < 0.5:
            failures.append({
                'problem': entry['question'],
                'error': entry['model_output'],
                'expected': entry['right_answer'],
                'reason': entry['judge_explanation']
            })
    
    return failures[:10]  # 限制数量
```
