# AFlow 增强优化框架概览

## 1. 总体架构

AFlow 增强优化框架集成了三种互补的工作流演化策略：

```
┌─────────────────────────────────────────────────────────────┐
│                    增强优化器 (EnhancedOptimizer)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   优化策略    │  │   分化策略    │  │   融合策略    │    │
│  │ Optimization │  │Differentiation│  │   Fusion     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                 │                  │             │
│         └─────────────────┼──────────────────┘             │
│                          │                                 │
│                   ┌──────▼──────┐                         │
│                   │ 策略选择逻辑  │                         │
│                   └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## 2. 三种策略简介

### 2.1 优化策略 (Optimization)
- **目标**：通过分析历史经验，逐步改进单个工作流
- **触发**：作为默认策略，在融合和分化都不满足条件时执行
- **特点**：局部改进，稳定提升

### 2.2 分化策略 (Differentiation)
- **目标**：将通用工作流特化为针对特定问题类型的专家工作流
- **触发**：基于动态概率（随轮次增加而提高）
- **特点**：创造多样性，专注特定领域

### 2.3 融合策略 (Fusion)
- **目标**：合并多个互补工作流的优势，创建综合性工作流
- **触发**：检测到包络工作流（envelope workflows）时
- **特点**：综合优势，突破性能瓶颈

## 3. 执行流程

### 主循环伪代码

```python
def optimize(dataset, max_rounds):
    """主优化循环"""
    
    # 初始化阶段
    round = 1
    if enable_differentiation:
        # 问题分类（仅执行一次）
        classify_all_problems(validation_data)
    
    # 第一轮：评估初始工作流
    evaluate_initial_workflow()
    round = 2
    
    # 迭代优化循环
    while round <= max_rounds:
        
        # === 策略选择（按优先级检查） ===
        
        # 优先级 1: 检查是否应该融合
        if should_attempt_fusion():
            score = attempt_fusion()
            if score is not None:
                round += 1
                continue
        
        # 优先级 2: 检查是否应该分化
        if should_attempt_differentiation():
            score = attempt_differentiation()
            if score is not None:
                round += 1
                continue
        
        # 优先级 3: 默认优化
        score = attempt_optimization()
        round += 1
        
        # 检查收敛
        if check_convergence():
            break
```

### 策略优先级说明

```
融合 > 分化 > 优化

原因：
- 融合：机会稀少（需要多个互补工作流），优先利用
- 分化：概率触发，创造多样性
- 优化：默认策略，保证稳定进步
```

## 4. 关键特性

### 4.1 问题分类系统
```python
# 在优化开始前执行一次
classifications = classify_problems(validation_data)
# 结果示例：
# {
#     "problem_0": {
#         "category": "Mathematical Derivation",
#         "description": "需要数学推导和计算"
#     },
#     "problem_1": {
#         "category": "Code Generation",
#         "description": "需要生成可执行代码"
#     },
#     ...
# }
```

### 4.2 动态分化概率
```python
# 概率随轮次增加
if round <= 3/4 * max_rounds:
    prob = base_prob * (1 + progress_ratio)
else:
    prob = 2 * base_prob  # 达到最大值并保持
```

### 4.3 包络工作流检测
```python
# 找到在不同问题子集上表现最好的工作流
envelope_workflows = find_workflows_with_complementary_strengths()
# 示例：
# Workflow A: 擅长数学问题
# Workflow B: 擅长代码问题
# Workflow C: 擅长推理问题
```

## 5. 协同作用机制

```
时间线示例（20轮优化）：

Round 1:  [初始评估] 
Round 2:  [优化] → 改进通用能力
Round 3:  [分化] → 创建数学专家工作流
Round 4:  [优化] → 改进数学工作流
Round 5:  [分化] → 创建代码专家工作流
Round 6:  [优化] → 改进代码工作流
Round 7:  [融合] → 合并数学+代码专家 → 性能跃升！
Round 8:  [优化] → 微调融合工作流
Round 9:  [分化] → 在融合基础上再分化
Round 10: [优化] → 继续改进
...

关键观察：
1. 分化创造多样性和专业性
2. 优化稳定提升性能
3. 融合在合适时机整合优势，实现突破
```

## 6. 性能指标

### 评估维度
- **准确率**：解决问题的比例
- **成本效率**：每个问题的平均调用成本
- **多样性**：工作流在不同问题类型上的表现分布
- **融合收益**：融合后相比融合前的性能提升

### 预期效果
```
基础优化：      +10-15% (稳定提升)
+ 分化策略：    +15-25% (专业化带来的提升)
+ 融合策略：    +25-40% (整合优势的跃升)
```

## 7. 关键参数配置

```python
# 分化相关
enable_differentiation = True
differentiation_probability = 0.3  # 基础概率
max_differentiation_rounds = 5     # 最大分化轮次

# 融合相关
enable_fusion = True
fusion_start_round = 5             # 从第5轮开始尝试融合
fusion_interval_rounds = 2         # 融合间隔至少2轮
max_envelope_workflows = 3         # 最多融合3个工作流

# 优化相关
max_rounds = 20                    # 总轮次
validation_rounds = 5              # 每轮验证次数
```

## 8. 下一步阅读

- [优化策略详解](./OPTIMIZATION_STRATEGY.md) - 详细了解优化过程
- [分化策略详解](./DIFFERENTIATION_STRATEGY.md) - 详细了解分化机制
- [融合策略详解](./FUSION_STRATEGY.md) - 详细了解融合过程
- [协同机制详解](./SYNERGY_MECHANISM.md) - 详细了解三者如何协同工作

## 9. 快速开始

```bash
# 运行完整的增强优化流程
python run_enhanced.py \
    --dataset MIXED \
    --max_rounds 20 \
    --enable_differentiation true \
    --enable_fusion true \
    --differentiation_probability 0.3
```

---

**核心理念**：工作流生成不是单一的全局搜索，而是融合、分化、优化三种机制协同作用的演化过程。
