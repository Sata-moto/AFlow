# Cost统计功能修改说明

## 修改日期
2025-12-16

## 修改内容

### 需求描述
增加一个总体的 cost 统计功能，统计每一轮消耗的 cost 并在最后累加。

### 问题分析
之前系统中已经有 cost 统计：
- `Benchmark.run_evaluation()` 每轮评估后会输出 `Total Cost`
- 但 `EnhancedOptimizer` 没有累加和汇总这些 cost

需要实现：
1. ✅ 每轮记录cost（来自evaluation）
2. ✅ 累加总cost
3. ✅ 在优化结束时输出cost统计报告

### 解决方案

#### 核心思路
Cost主要来自**执行评估时的LLM调用**（Benchmark.run_evaluation），而不是优化时的LLM调用。因此：
- 在 `evaluate_graph` 方法中返回 `total_cost`
- 在各个操作方法中传递 `cost`
- 在主循环中累加 `cost`

#### 修改详情

### 1. 修改 `EvaluationUtils.evaluate_graph()` 

**文件**: `scripts/optimizer_utils/evaluation_utils.py`

**修改位置**: Line 31-60

**修改内容**:
```python
async def evaluate_graph(self, optimizer, directory, validation_n, data, initial=False):
    evaluator = Evaluator(eval_path=directory)
    sum_score = 0
    sum_total_cost = 0  # 新增：累加total_cost
    all_solved_problems = set()

    # Repeat the test validation_n times to get the average
    for i in range(validation_n):
        score, avg_cost, total_cost, solved_problems = await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
            directory,
            is_test=False,
        )

        # ... 省略其他代码 ...

        sum_score += score
        sum_total_cost += total_cost  # 新增：累加cost

    # Store the union of all solved problems for this round
    optimizer.current_round_solved_problems = all_solved_problems
    
    return sum_score / validation_n, sum_total_cost  # 修改：返回tuple
```

**效果**: 
- `evaluate_graph` 现在返回 `(avg_score, total_cost)` 元组
- `total_cost` 是所有validation runs的总cost

### 2. 修改 `_optimize_graph()` 方法

**文件**: `scripts/enhanced_optimizer.py`

**修改位置1**: Line 1827-1829（第一轮评估）
```python
if self.round == 1:
    directory = self.graph_utils.create_round_directory(graph_path, self.round)
    self.graph = self.graph_utils.load_graph(self.round, graph_path)
    avg_score, total_cost = await self.evaluation_utils.evaluate_graph(
        self, directory, validation_n, data, initial=True
    )
    return avg_score, total_cost  # 修改：返回tuple
```

**修改位置2**: Line 1901-1907（正常轮次评估）
```python
# Evaluate the graph
avg_score, total_cost = await self.evaluation_utils.evaluate_graph(
    self, directory, validation_n, data, initial=False
)

# Update the current round score in the experience file
self.experience_utils.update_experience(directory, experience, avg_score)

return avg_score, total_cost  # 修改：返回tuple
```

**效果**: 
- `_optimize_graph` 现在返回 `(score, cost)` 元组

### 3. 修改 `_attempt_differentiation()` 方法

**文件**: `scripts/enhanced_optimizer.py`

**修改位置**: Line 1488-1513

```python
data = self.data_utils.load_results(graph_path)
differentiation_score, total_cost = await self.evaluation_utils.evaluate_graph(
    self, directory, self.validation_rounds, data, initial=False
)

# ... 省略其他代码 ...

logger.info(f"Problem type specialization completed with score: {differentiation_score:.4f}")
return differentiation_score, total_cost  # 修改：返回tuple
```

**效果**: 
- `_attempt_differentiation` 现在返回 `(score, cost)` 元组

### 4. 修改 `_attempt_fusion()` 方法

**文件**: `scripts/enhanced_optimizer.py`

**修改位置**: Line 1648-1698

```python
# Evaluate using standard evaluation process
fusion_score, total_cost = await self.evaluation_utils.evaluate_graph(
    self, directory, self.validation_rounds, data, initial=False
)

# ... 省略检查和保存逻辑 ...

# Check if fusion meets threshold
if fusion_score > min_envelope_score + self.fusion_score_threshold:
    logger.info(f"3-way fusion successful! ...")
    # ... 省略其他代码 ...
    return fusion_score, total_cost  # 修改：返回tuple
else:
    logger.info(f"Fusion score {fusion_score:.4f} below threshold ...")
    # ... 省略其他代码 ...
    return fusion_score, total_cost  # 修改：返回tuple
```

**效果**: 
- `_attempt_fusion` 现在返回 `(score, cost)` 元组

### 5. 修改主optimize循环

**文件**: `scripts/enhanced_optimizer.py`

**修改位置**: Line 263-310

**关键修改**:
```python
# 执行选定的操作
if operation == 'differentiate':
    if self.enable_differentiation:
        result = self._attempt_with_retry(
            lambda: loop.run_until_complete(self._attempt_differentiation()),
            "differentiation", 3
        )
        if result is not None:
            score, round_cost = result  # 解包tuple
            self.N_s += 1
        else:
            score, round_cost = loop.run_until_complete(self._optimize_graph())
    else:
        score, round_cost = loop.run_until_complete(self._optimize_graph())

elif operation == 'fuse':
    if self.enable_fusion:
        if self._check_fusion_preconditions():
            result = self._attempt_with_retry(
                lambda: loop.run_until_complete(self._attempt_fusion()),
                "fusion", 3
            )
            if result is not None:
                score, round_cost = result  # 解包tuple
                self.N_m += 1
            else:
                score, round_cost = loop.run_until_complete(self._optimize_graph())
        else:
            score, round_cost = loop.run_until_complete(self._optimize_graph())
    else:
        score, round_cost = loop.run_until_complete(self._optimize_graph())

else:  # operation == 'optimize'
    score, round_cost = loop.run_until_complete(self._optimize_graph())
```

**累加cost**:
```python
self.round += 1
logger.info(f"Score for round {self.round}: {score}")

# 记录本轮的cost（已经从操作中获取）
self.round_costs.append({
    'round': self.round,
    'cost': round_cost,
    'score': score
})
self.total_cost += round_cost
logger.info(f"Cost for round {self.round}: ${round_cost:.4f} (Total: ${self.total_cost:.4f})")
```

**效果**: 
- 所有操作现在都返回 `(score, cost)` 元组
- 每轮cost被正确记录到 `self.round_costs`
- 总cost累加到 `self.total_cost`
- 每轮输出当前cost和累计cost

### 6. 删除废弃的 `_get_round_cost()` 方法

**文件**: `scripts/enhanced_optimizer.py`

**删除原因**:
之前尝试从LLM的usage tracker中获取cost，但实际上：
- 主要cost来自evaluation（执行workflow）
- 通过 `evaluate_graph` 返回的 `total_cost` 更准确
- 不需要从多个LLM实例中追踪usage

**删除的方法**:
```python
def _get_round_cost(self) -> float:
    """
    获取本轮优化的cost
    ...
    """
    # 删除整个方法实现（约30行代码）
```

### 7. Cost统计报告（已存在）

**文件**: `scripts/enhanced_optimizer.py`

**方法**: `_print_cost_summary()`（Line 394-468）

这个方法已经存在，会在优化结束时输出：
- 每轮的score和cost
- 总cost
- 平均cost
- 按操作类型分类的cost（optimize/differentiate/fuse）

**输出示例**:
```
================================================================================
COST SUMMARY
================================================================================
Round      Score           Cost ($)       
----------------------------------------
1          0.6500          $0.3088        
2          0.6700          $0.4314        
3          0.7000          $0.3070        
...
================================================================================
Total Cost: $2.5432
Average Cost per Round: $0.3179
================================================================================

Cost Breakdown by Operation Type:
  Optimize:       8 rounds, $1.8342 (72.1%)
  Differentiate:  2 rounds, $0.4590 (18.1%)
  Fuse:          2 rounds, $0.2500 (9.8%)
================================================================================
```

## 影响分析

### 对现有功能的影响

1. **方法签名变化**:
   - `evaluate_graph()`: 返回值从 `float` 变为 `(float, float)`
   - `_optimize_graph()`: 返回值从 `float` 变为 `(float, float)`
   - `_attempt_differentiation()`: 返回值从 `float` 变为 `(float, float)`
   - `_attempt_fusion()`: 返回值从 `float` 变为 `(float, float)`

2. **向后兼容性**:
   - ⚠️ 不兼容：子类覆盖这些方法时需要修改返回值
   - ⚠️ 不兼容：外部调用这些方法时需要修改接收方式
   - ✅ 兼容：`Optimizer`基类没有修改，baseline optimizer仍然正常工作

3. **数据结构变化**:
   - `self.round_costs` 现在包含准确的evaluation cost
   - `self.total_cost` 累加所有轮次的cost

### 优势

1. **准确性**:
   - Cost来自实际的evaluation（benchmark执行）
   - 避免了从多个LLM实例追踪usage的复杂性

2. **可追溯性**:
   - 每轮cost清晰记录
   - 可以分析不同操作的cost差异

3. **简洁性**:
   - 删除了复杂的 `_get_round_cost` 方法
   - Cost追踪逻辑集中在evaluation层

### 测试建议

#### 单元测试
```python
def test_cost_tracking():
    """测试cost追踪功能"""
    # 创建optimizer
    optimizer = EnhancedOptimizer(...)
    
    # 运行几轮优化
    optimizer.optimize("Graph")
    
    # 验证cost记录
    assert len(optimizer.round_costs) > 0
    assert all('cost' in r for r in optimizer.round_costs)
    assert optimizer.total_cost > 0
    
    # 验证累加正确
    calculated_total = sum(r['cost'] for r in optimizer.round_costs)
    assert abs(calculated_total - optimizer.total_cost) < 0.01
```

#### 集成测试
```bash
# 运行完整优化并检查日志
python run_enhanced.py --dataset GSM8K --max_rounds 5

# 检查每轮cost输出
grep "Cost for round" logs/AFlow.log

# 检查总cost输出
grep "Total Cost:" logs/AFlow.log
```

#### 回归测试
```bash
# 测试baseline optimizer仍然正常工作
python run.py --dataset GSM8K --max_rounds 3

# 应该没有错误，因为Optimizer基类没有修改
```

## 使用示例

### 查看实时cost

运行优化时，每轮会输出：
```
2025-12-16 18:09:19 - INFO - Score for round 8: 0.7234
2025-12-16 18:09:19 - INFO - Cost for round 8: $0.9241 (Total: $4.5632)
```

### 查看最终统计

优化结束时，会输出完整的cost报告：
```python
optimizer = EnhancedOptimizer(...)
optimizer.optimize("Graph")

# 会自动调用 _print_cost_summary() 输出报告
```

### 编程访问cost数据

```python
# 获取每轮的cost
for round_info in optimizer.round_costs:
    print(f"Round {round_info['round']}: "
          f"Score={round_info['score']:.4f}, "
          f"Cost=${round_info['cost']:.4f}")

# 获取总cost
print(f"Total cost: ${optimizer.total_cost:.4f}")

# 计算平均cost
avg_cost = optimizer.total_cost / len(optimizer.round_costs)
print(f"Average cost per round: ${avg_cost:.4f}")
```

### 分析cost趋势

```python
import matplotlib.pyplot as plt

# 绘制cost趋势图
rounds = [r['round'] for r in optimizer.round_costs]
costs = [r['cost'] for r in optimizer.round_costs]

plt.plot(rounds, costs, marker='o')
plt.xlabel('Round')
plt.ylabel('Cost ($)')
plt.title('Cost per Round')
plt.grid(True)
plt.savefig('cost_trend.png')
```

## 注意事项

### 1. Cost来源
- ✅ **主要cost**: Evaluation时执行workflow的LLM调用（由Benchmark统计）
- ⚠️ **不包括**: Optimization时生成新workflow的LLM调用（通常很小）
- 📝 如果需要包括optimization cost，可以在`_optimize_graph`中额外追踪

### 2. Validation runs
- Cost是所有validation runs的总和
- 如果 `validation_rounds=3`，每轮cost = 3次evaluation的总cost
- 这反映了实际的完整评估成本

### 3. 失败处理
- 如果某轮操作失败，cost设为0.0
- 失败的retry不会额外计入cost（因为没有执行evaluation）

### 4. 并发评估
- 如果使用并发评估（`max_concurrent_tasks`），cost仍然准确
- Benchmark的cost统计自动处理并发情况

## 总结

这次修改实现了完整的cost追踪功能：

**核心改进**:
- ✅ 从evaluation准确获取每轮cost
- ✅ 累加总cost并实时输出
- ✅ 在优化结束时生成详细报告
- ✅ 删除了复杂且不准确的usage tracker方案

**数据流**:
```
Benchmark.run_evaluation()
  └─> evaluator.graph_evaluate() [返回total_cost]
      └─> EvaluationUtils.evaluate_graph() [累加并返回total_cost]
          └─> _optimize_graph/_attempt_fusion/_attempt_differentiation [返回(score, cost)]
              └─> optimize() 主循环 [记录cost, 累加total_cost, 输出日志]
                  └─> _print_cost_summary() [生成最终报告]
```

**效果**:
- 每轮显示当前cost和累计cost
- 优化结束时显示完整统计
- 可以分析不同操作的cost差异
- 准确反映实际的evaluation成本
