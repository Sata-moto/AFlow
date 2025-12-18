# AFlow 系统架构总览

## 文档版本
- **创建日期**: 2025-12-15
- **版本**: v2.0 (理论化重构版本)
- **状态**: 当前稳定版本

---

## 1. 系统概述

### 1.1 核心目标
AFlow 是一个**自动化工作流优化系统**,通过三种核心操作(优化、分化、融合)不断迭代改进工作流,以提升问题求解性能。

### 1.2 设计理念
1. **理论驱动**: 所有操作决策基于数学公式和理论分析
2. **自适应**: 根据性能停滞度动态调整操作概率
3. **数据驱动**: 基于历史经验和问题分类指导优化
4. **模块化**: 清晰分离优化、分化、融合三种操作

### 1.3 系统特点
- ✅ **无需人工干预**: 全自动决策和执行
- ✅ **性能导向**: 基于验证集实时评估
- ✅ **经验积累**: 记录失败案例指导后续优化
- ✅ **多策略融合**: 结合优化、分化、融合三种策略

---

## 2. 整体架构

### 2.1 主控制流程

```
┌─────────────────────────────────────────────────────────────┐
│                    EnhancedOptimizer                         │
│                     (主控制器)                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │   while round < max_rounds:            │
        │                                        │
        │   1. 计算操作概率                       │
        │      - _calculate_plateau()            │
        │      - _calculate_operation_probs()    │
        │                                        │
        │   2. 采样选择操作                       │
        │      - np.random.choice([opt,split,fuse]) │
        │                                        │
        │   3. 执行对应操作                       │
        │      - OPTIMIZE: _optimize_graph()     │
        │      - DIFFERENTIATE: _differentiate() │
        │      - FUSE: _fuse()                   │
        │                                        │
        │   4. 评估新workflow                     │
        │      - evaluate_graph()                │
        │      - 更新 performance_history         │
        │                                        │
        │   5. 收敛检查                           │
        │      - check_convergence()             │
        │                                        │
        │   round += 1                           │
        └───────────────────────────────────────┘
```

### 2.2 模块层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                          应用层                              │
│  run_enhanced.py - 命令行入口                                │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                         控制层                               │
│  EnhancedOptimizer - 主优化器                                │
│  - 操作概率计算                                               │
│  - 操作选择与执行                                             │
│  - 轮次管理                                                  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  优化操作     │   │  分化操作     │   │  融合操作     │
│              │   │              │   │              │
│ GraphOptimizer│  │ DifferentiationProcessor │ │ FusionProcessor│
│ - 图优化      │   │ - 类别分析    │   │ - 三路融合    │
│ - 经验提取    │   │ - 专业化分化   │   │ - 互补性计算  │
└──────────────┘   └──────────────┘   └──────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                         工具层                               │
│  - WorkflowManager: workflow管理                            │
│  - ProblemClassifier: 问题分类                              │
│  - GraphUtils: 图操作工具                                    │
│  - DataUtils: 数据加载工具                                   │
│  - EvaluationUtils: 评估工具                                │
│  - ExperienceUtils: 经验管理                                │
│  - ConvergenceUtils: 收敛检测                               │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                         数据层                               │
│  - Benchmarks: 数据集评测接口                                │
│  - Datasets: 验证集数据                                      │
│  - Workflows: 工作流存储                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 数据流向

### 3.1 优化循环数据流

```
┌──────────────┐
│ 初始Workflow  │
│  (round_1)   │
└──────┬───────┘
       │
       ↓
┌──────────────────────────────────────────────────────────┐
│  评估循环                                                  │
│                                                           │
│  1. 在验证集上执行 → 生成 log.json                         │
│     - question, answer, score, category                  │
│                                                           │
│  2. 计算平均分数 → 更新 results.json                       │
│     - round, avg_score, solved_count                     │
│                                                           │
│  3. 提取失败经验 → 生成 experience.json                    │
│     - before: {method, performance}                      │
│     - after: {score, timestamp}                          │
│                                                           │
│  4. 更新性能历史 → performance_history[]                   │
│     - 用于停滞度计算                                        │
└──────────────────────────────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────────────────────┐
│  操作概率计算                                              │
│                                                           │
│  1. 计算停滞度 plateau_t                                   │
│     - 比较最近k轮 vs 之前k轮                                │
│     - Logistic映射: 1/(1 + exp(κ·Δ_t))                   │
│                                                           │
│  2. 计算原始概率                                           │
│     - p_opt = 1 - p_split - p_fuse                       │
│     - p_split = α_s · plateau · exp(-η_s · N_s)          │
│     - p_fuse = α_m · plateau · exp(-η_m · N_m)           │
│                                                           │
│  3. 归一化 → 概率分布 π_t                                  │
└──────────────────────────────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────────────────────┐
│  操作执行                                                  │
│                                                           │
│  OPTIMIZE:                                                │
│  ├─ 加载 processed_experience.json                        │
│  ├─ 调用 LLM 生成改进方案                                  │
│  └─ 创建 round_N+1/ 新workflow                            │
│                                                           │
│  DIFFERENTIATE:                                           │
│  ├─ 分析 log.json 中各类别表现                             │
│  ├─ 选择优势类别 (Recall_k > Acc_global)                  │
│  ├─ 加载该类别样本                                         │
│  └─ 创建专业化 round_N+1/                                 │
│                                                           │
│  FUSE:                                                    │
│  ├─ 计算所有三元组的 Φ_merge                               │
│  ├─ 选择最佳组合 (Round i, j, k)                          │
│  ├─ 调用 LLM 融合三个workflows                            │
│  └─ 创建融合 round_N+1/                                   │
└──────────────────────────────────────────────────────────┘
       │
       └──────→ 循环继续
```

### 3.2 关键数据结构

#### 3.2.1 results.json
```json
[
  {
    "round": 1,
    "avg_score": 0.7326,
    "total_cost": 0.00647,
    "solved_count": 63,
    "total_problems": 86,
    "timestamp": "2025-12-15T10:09:59"
  }
]
```

#### 3.2.2 log.json
```json
[
  {
    "question": "问题描述",
    "right_answer": "正确答案",
    "model_output": "模型输出",
    "score": 1.0,
    "judge_explanation": "评判理由",
    "problem_id": "problem_0",
    "category": "Mathematical & Logical Reasoning"
  }
]
```

#### 3.2.3 experience.json
```json
{
  "before": {
    "method": "当前workflow的描述",
    "performance": "在失败案例上的表现"
  },
  "after": {
    "score": 0.8140,
    "timestamp": "2025-12-15T10:10:39"
  },
  "failure_examples": [
    {
      "problem": "失败的问题",
      "error": "错误原因",
      "expected": "期望输出"
    }
  ]
}
```

#### 3.2.4 problem_classifications.json
```json
{
  "categories": [
    "Mathematical & Logical Reasoning",
    "Geometric & Spatial Reasoning",
    "Combinatorial Counting & Enumeration",
    "Dynamic Programming & Recursion"
  ],
  "category_descriptions": {
    "Mathematical & Logical Reasoning": "描述..."
  },
  "problem_classifications": [
    {
      "problem_id": "problem_0",
      "category": "Mathematical & Logical Reasoning",
      "reasoning": "分类理由"
    }
  ]
}
```

---

## 4. 关键机制

### 4.1 性能停滞检测

**目的**: 判断当前优化是否进入瓶颈,决定是否需要探索性操作(分化/融合)

**算法**:
```python
def _calculate_plateau():
    # 1. 获取性能历史
    t = len(performance_history)
    k = sliding_window_k  # 默认2
    
    # 2. Warm-up检查
    if t < 2:
        return 0.0  # 历史数据不足
    
    # 3. 动态窗口
    effective_k = min(k, t // 2)
    if effective_k < 1:
        return 0.0
    
    # 4. 计算最近和之前的最大性能
    R_recent = max(history[t - effective_k : t])
    R_prev = max(history[t - 2*effective_k : t - effective_k])
    
    # 5. Logistic映射
    delta_t = R_recent - R_prev
    plateau_t = 1.0 / (1.0 + exp(κ * delta_t))
    
    # plateau_t ∈ [0, 1]
    # 接近0: 性能快速提升 (不停滞)
    # 接近0.5: 性能平稳 (轻度停滞)
    # 接近1: 性能下降 (严重停滞)
    
    return plateau_t
```

**关键参数**:
- `sliding_window_k = 2`: 滑动窗口大小
- `stagnation_sensitivity_kappa = 10`: 停滞敏感度(κ)

### 4.2 操作概率计算

**公式**:
```
p_split_raw = α_s · plateau_t · exp(-η_s · N_s)
p_fuse_raw = α_m · plateau_t · exp(-η_m · N_m)
p_opt_raw = 1 - p_split_raw - p_fuse_raw

π_t = Normalize({p_opt, p_split, p_fuse})
```

**参数含义**:
- `α_s = 0.3`: 分化基础权重
- `α_m = 0.2`: 融合基础权重
- `η_s = 0.1`: 分化衰减率(控制重复分化的惩罚)
- `η_m = 0.15`: 融合衰减率(控制重复融合的惩罚)
- `N_s`: 累计分化次数
- `N_m`: 累计融合次数

**设计逻辑**:
1. **停滞度主导**: `plateau_t` 越大,探索性操作概率越高
2. **操作历史惩罚**: 指数衰减避免过度使用某种操作
3. **动态平衡**: 自动在exploitation(优化)和exploration(分化/融合)间切换

### 4.3 操作选择策略

**原则**: 所有选择都基于确定性argmax,避免随机性

#### 4.3.1 优化操作选择
```python
# 从历史workflows中选择最高分作为优化基础
sorted_rounds = sorted(results, key=lambda x: x['score'], reverse=True)
selected_round = sorted_rounds[0]  # 选择分数最高的
```

#### 4.3.2 分化操作选择
```python
# 计算每个workflow的分化潜力
for workflow in workflows:
    for category in categories:
        if Recall_k > Acc_global:  # 有优势
            score_k = Contrib_k * (Recall_k - Acc_global)
    
    split_potential = max(score_k for all categories)

# 确定性选择
selected_idx = potentials.index(max(potentials))  # argmax
```

#### 4.3.3 融合操作选择
```python
# 评估所有可能的三元组
for (i, j, k) in all_triples:
    Φ_U = 互补性指标
    Φ_I = 一致性指标
    Φ_merge = α_U * Φ_U + α_I * Φ_I

# 确定性选择
best_triple = max(all_triples, key=lambda t: t.Φ_merge)
```

---

## 5. 文件组织

### 5.1 目录结构

```
AFlow/
├── run_enhanced.py              # 主程序入口
├── config/
│   └── config2.example.yaml     # 配置模板
├── scripts/
│   ├── enhanced_optimizer.py    # 核心优化器
│   ├── optimizer.py             # 基础优化器(父类)
│   ├── evaluator.py             # 评估器
│   ├── formatter.py             # 格式化工具
│   ├── interface.py             # LLM接口
│   ├── workflow_fusion.py       # 融合处理器
│   ├── operator_an.py           # 操作符分析
│   ├── operators.py             # 操作符定义
│   ├── optimizer_utils/         # 优化器工具集
│   │   ├── convergence_utils.py  # 收敛检测
│   │   ├── data_utils.py         # 数据工具
│   │   ├── evaluation_utils.py   # 评估工具
│   │   ├── experience_utils.py   # 经验管理
│   │   └── graph_utils.py        # 图操作工具
│   ├── prompts/                 # 提示词模板
│   │   ├── fusion_prompt.py      # 融合提示词
│   │   ├── optimize_prompt.py    # 优化提示词
│   │   └── prompt.py             # 基础提示词
│   └── utils/                   # 通用工具
│       ├── common.py             # 通用函数
│       ├── code.py               # 代码工具
│       └── sanitize.py           # 清理工具
├── benchmarks/
│   ├── benchmark.py             # 基准测试基类
│   ├── math.py                  # MATH数据集
│   ├── gsm8k.py                 # GSM8K数据集
│   ├── drop.py                  # DROP数据集
│   ├── hotpotqa.py              # HotpotQA数据集
│   ├── humaneval.py             # HumanEval数据集
│   └── mbpp.py                  # MBPP数据集
├── data/
│   └── datasets/                # 数据集文件
│       ├── math_validate.jsonl
│       ├── gsm8k_validate.jsonl
│       └── ...
└── workspace/
    └── {DATASET}/               # 每个数据集的工作空间
        └── workflows/
            ├── results.json      # 所有轮次结果
            ├── processed_experience.json  # 处理后的经验
            ├── problem_classifications.json  # 问题分类
            ├── fusion_metadata_N.json     # 融合元数据
            ├── round_1/          # 第1轮workflow
            │   ├── graph.py      # 工作流代码
            │   ├── prompt.py     # 提示词
            │   ├── log.json      # 执行日志
            │   ├── experience.json  # 经验数据
            │   └── 0.7326_timestamp.csv  # 结果文件
            ├── round_2/          # 第2轮workflow
            └── ...
```

### 5.2 关键文件说明

| 文件 | 作用 | 创建时机 |
|------|------|----------|
| `results.json` | 记录所有轮次的评估结果 | 每轮评估后更新 |
| `log.json` | 记录每道题的详细执行情况 | 每轮评估时创建 |
| `experience.json` | 记录失败案例和经验 | 每轮评估后创建 |
| `processed_experience.json` | 格式化的经验数据 | 优化操作前创建 |
| `problem_classifications.json` | 问题分类结果 | 首次分化前创建(一次性) |
| `fusion_metadata_N.json` | 融合操作的元数据 | 每次融合后创建 |
| `graph.py` | 工作流的可执行代码 | 每轮创建新workflow时 |
| `prompt.py` | 工作流使用的提示词 | 每轮创建新workflow时 |

---

## 6. 执行流程示例

### 6.1 典型执行序列

```
Round 1: 
  - 操作: OPTIMIZE (初始化,实际是评估初始workflow)
  - 分数: 0.7326
  - 说明: 评估初始workflow,创建round_1/的评估结果

Round 2:
  - Plateau: 0.0000 (历史数据不足)
  - 概率: p_opt=1.0, p_split=0.0, p_fuse=0.0
  - 操作: OPTIMIZE
  - 基于: Round 1 (分数最高)
  - 分数: 0.8140 (+11.1%)

Round 3:
  - Plateau: 0.0000 (t=2, 仍在warm-up)
  - 概率: p_opt=1.0, p_split=0.0, p_fuse=0.0
  - 操作: OPTIMIZE
  - 基于: Round 2
  - 分数: 0.7907 (-2.9%)

Round 4:
  - Plateau: 0.0000 (t=3, effective_k=1, 但仍在warm-up)
  - 概率: p_opt=1.0, p_split=0.0, p_fuse=0.0
  - 操作: OPTIMIZE
  - 基于: Round 2
  - 分数: 0.8605 (+8.8%)

Round 5:
  - Plateau: 0.0890 (t=4, 开始计算停滞度)
  - 概率: p_opt=0.9021, p_split=0.0534, p_fuse=0.0445
  - 操作: OPTIMIZE (概率采样)
  - 基于: Round 4
  - 分数: 0.8488 (-1.4%)

Round 6:
  - Plateau: 0.3586 (性能停滞)
  - 概率: p_opt=0.8565, p_split=0.0954, p_fuse=0.0481
  - 操作: DIFFERENTIATE (概率采样选中)
  - 基于: Round 5
  - 目标类别: "Mathematical & Logical Reasoning"
  - 分数: 0.5000 (-41.1%, 专业化代价)

Round 7:
  - Plateau: 0.9704 (严重停滞,分数下降)
  - 概率: p_opt=0.6874, p_split=0.2461, p_fuse=0.0665
  - 操作: DIFFERENTIATE (高概率再次选中)
  - 基于: Round 4
  - 目标类别: "Geometric & Spatial Reasoning"
  - 分数: 0.4884 (-43.2%)

Round 8:
  - 操作: FUSE (高停滞度触发融合)
  - 融合: (Round 2, Round 4, Round 5)
  - 分数: 0.8953 (+4.0%, 融合成功!)
```

### 6.2 操作决策逻辑

```
if plateau_t < 0.3:
    # 性能正常提升
    → 高概率选择 OPTIMIZE (exploitation)
    → 继续当前优化方向

elif 0.3 <= plateau_t < 0.7:
    # 性能停滞
    → 提高 DIFFERENTIATE 概率
    → 尝试专业化分支

else:  # plateau_t >= 0.7
    # 严重停滞
    → 提高 FUSE 概率
    → 融合多个workflows获得突破
```

---

## 7. 性能指标

### 7.1 评估指标

| 指标 | 定义 | 计算方式 |
|------|------|----------|
| **avg_score** | 平均分数 | Σ(score) / N |
| **solved_count** | 解决问题数 | Σ(score >= 0.5) |
| **Acc_global** | 全局准确率 | C_total / N |
| **Recall_k** | 类别召回率 | C_k / N_k |
| **Contrib_k** | 类别贡献度 | C_k / C_total |

### 7.2 停滞度解释

| Plateau | 含义 | 典型操作 |
|---------|------|----------|
| 0.0 - 0.2 | 快速提升 | OPTIMIZE (90%+) |
| 0.2 - 0.4 | 轻度停滞 | OPTIMIZE (80%), SPLIT (15%), FUSE (5%) |
| 0.4 - 0.6 | 中度停滞 | OPTIMIZE (70%), SPLIT (20%), FUSE (10%) |
| 0.6 - 0.8 | 严重停滞 | OPTIMIZE (60%), SPLIT (25%), FUSE (15%) |
| 0.8 - 1.0 | 性能下降 | SPLIT (30%), FUSE (20%), OPTIMIZE (50%) |

---

## 8. 扩展接口

### 8.1 添加新数据集

1. 在 `benchmarks/` 创建新benchmark类
2. 继承 `BaseBenchmark`
3. 实现 `evaluate_problem()` 方法
4. 在 `run_enhanced.py` 注册数据集

### 8.2 添加新操作

1. 在 `EnhancedOptimizer` 添加操作方法
2. 在 `_calculate_operation_probabilities()` 添加概率计算
3. 在主循环添加操作分支

### 8.3 自定义评估器

1. 修改 `EvaluationUtils.evaluate_graph()`
2. 自定义评分逻辑
3. 更新 `log.json` 格式

---

## 9. 常见问题

### Q1: 为什么前几轮都是OPTIMIZE?
**A**: 前几轮 `plateau_t = 0` (历史数据不足),导致 `p_split = p_fuse = 0`,只能选择OPTIMIZE。

### Q2: 分化为什么会导致性能下降?
**A**: 分化会创建专业化workflow,牺牲通用性换取专业领域的高性能。在验证集上可能表现下降,但在目标类别上表现提升。

### Q3: 融合如何工作?
**A**: 融合选择3个互补的workflows,通过LLM整合它们的优势,创建一个综合性能更好的新workflow。

### Q4: 如何调整探索/利用平衡?
**A**: 调整以下参数:
- 增大 `α_s`, `α_m`: 更多探索
- 减小 `η_s`, `η_m`: 降低重复操作惩罚
- 增大 `κ`: 对停滞更敏感

### Q5: 收敛如何判断?
**A**: 连续轮次的Top-K分数变化小于阈值,且满足最小轮次要求。

---

## 10. 下一步阅读

- [优化操作详解](OPTIMIZE_OPERATION.md) - 了解优化算法和经验利用
- [分化操作详解](DIFFERENTIATION_OPERATION.md) - 了解类别分析和专业化
- [融合操作详解](FUSION_OPERATION.md) - 了解三路融合和互补性计算
- [快速参考](QUICK_REFERENCE.md) - 关键参数和公式速查

---

## 附录: 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| 工作流 | Workflow | 一个完整的问题求解方法,包含graph.py和prompt.py |
| 停滞度 | Plateau | 衡量优化进展停滞程度的指标,[0,1] |
| 分化 | Differentiation | 针对特定问题类别创建专业化workflow |
| 融合 | Fusion | 合并多个workflow的优势创建新workflow |
| 召回率 | Recall | 某类别中答对题目的比例 |
| 贡献度 | Contribution | 某类别答对题目占总答对题目的比例 |
| 互补性 | Complementarity | 多个workflows覆盖不同问题的程度 |
| 一致性 | Consensus | 多个workflows都能答对的问题的程度 |
