# 融合操作详解 (FUSION Operation)

## 文档版本
- **创建日期**: 2025-12-15
- **相关代码**: `scripts/enhanced_optimizer.py` (Line 658-1161)
- **依赖文档**: [系统架构总览](SYSTEM_ARCHITECTURE.md)

---

## 1. 操作概述

### 1.1 核心目标
融合操作通过**三路组合**整合多个workflows的优势,创建综合性更强的新workflow,实现"1+1+1>3"的协同效应。

### 1.2 设计理念
- **互补性**: 选择在不同问题上表现好的workflows
- **包络线**: 选择帕累托最优集合 (envelope workflows)
- **三路融合**: 考虑三个workflows的pairwise和triplet互补性

### 1.3 触发条件
- **概率采样**: 根据停滞度和历史融合次数计算概率
- **前置条件**: 至少有3个workflows可用
- **公式**: `p_merge = α_m · plateau_t · exp(-η_m · N_m)`

---

## 2. 完整流程

### 2.1 流程图

```
┌────────────────────────────────────────────────────────────────┐
│                  FUSION 操作流程                               │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  预检查: 确认条件满足                                           │
│                                                                 │
│  1. 检查可用workflow数量 >= 3                                   │
│  2. 如果不足,返回None                                           │
│                                                                 │
│  示例:                                                          │
│    Round 2: 只有2个workflows → 跳过融合                        │
│    Round 5: 有5个workflows → 可以执行融合                      │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第1阶段: 选择包络线Workflows (_select_envelope_for_merge)     │
│                                                                 │
│  目标: 找到帕累托最优集合                                       │
│                                                                 │
│  算法:                                                          │
│    1. 加载所有workflows的 log.json                             │
│    2. 对每个workflow,提取问题级别的分数:                       │
│       scores[workflow_id][problem_id] = score                  │
│                                                                 │
│    3. 帕累托支配关系:                                           │
│       A dominates B if:                                        │
│         ∀ problem: score_A[p] >= score_B[p]                   │
│         ∃ problem: score_A[p] > score_B[p]                    │
│                                                                 │
│    4. 包络线 = 所有不被支配的workflows                         │
│       (任何一个workflow在至少一道题上是最好的)                  │
│                                                                 │
│  示例:                                                          │
│    Problem 1: W1=1.0, W2=0.8, W3=0.9, W4=0.7                  │
│    Problem 2: W1=0.6, W2=1.0, W3=0.8, W4=0.9                  │
│    Problem 3: W1=0.7, W2=0.8, W3=1.0, W4=0.6                  │
│                                                                 │
│    结果: Envelope = {W1, W2, W3}                               │
│    理由: W4在所有题上都不是最优 → 被支配                       │
│                                                                 │
│  输出: envelope_workflows (List[Dict])                         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第2阶段: 选择最佳三元组 (_select_for_merge)                    │
│                                                                 │
│  目标: 在包络线中找到互补性最强的3个workflows                   │
│                                                                 │
│  算法:                                                          │
│    对于每个三元组 (W_i, W_j, W_k):                             │
│                                                                 │
│      1. 计算Pairwise互补性:                                    │
│         Φ_pair(i,j) = |{p: W_i correct, W_j wrong}| +         │
│                       |{p: W_j correct, W_i wrong}|            │
│                       (两个workflows互相弥补的题目数)           │
│                                                                 │
│      2. 计算Triplet互补性:                                     │
│         Φ_triple(i,j,k) = |{p: exactly_one_correct}| +        │
│                           |{p: exactly_two_correct}|           │
│                           (需要投票才能正确的题目数)            │
│                                                                 │
│      3. 总融合潜力:                                             │
│         Φ_merge = β_p · (Φ_pair(i,j) + Φ_pair(j,k) +         │
│                          Φ_pair(i,k)) +                       │
│                   β_t · Φ_triple(i,j,k)                       │
│                                                                 │
│         其中: β_p = 0.4 (pairwise权重)                        │
│               β_t = 0.3 (triplet权重)                         │
│                                                                 │
│      4. 选择 argmax_triple(Φ_merge)                            │
│                                                                 │
│  输出: (W_i, W_j, W_k) - 最佳三元组                            │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第3阶段: LLM生成融合Workflow                                   │
│                                                                 │
│  输入:                                                          │
│  - 三个workflows的完整代码 (graph.py + prompt.py)              │
│  - 互补性分析报告 (哪些题需要融合)                              │
│  - 融合策略提示                                                 │
│                                                                 │
│  LLM任务:                                                       │
│  1. 分析三个workflows的差异:                                   │
│     - 推理策略差异                                              │
│     - prompt设计差异                                            │
│     - 结构差异                                                  │
│                                                                 │
│  2. 识别互补模式:                                               │
│     - W1擅长代数,W2擅长几何 → 合并两种reasoning                │
│     - W1使用CoT,W2使用PoT → 根据问题类型选择                   │
│     - W1详细,W2简洁 → 平衡详细度                               │
│                                                                 │
│  3. 设计融合策略:                                               │
│     - 串行融合: Step1用W1, Step2用W2, Step3用W3               │
│     - 并行融合: 同时运行三个,投票决定                          │
│     - 条件融合: 根据问题特征路由                                │
│     - 混合融合: 结合以上策略                                    │
│                                                                 │
│  输出:                                                          │
│  - 融合后的 graph.py                                           │
│  - 融合后的 prompt.py                                          │
│  - 融合说明文档                                                 │
└────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  第4阶段: 评估融合效果                                          │
│                                                                 │
│  1. 创建 round_N/ 目录                                         │
│  2. 保存融合后的代码                                            │
│  3. 记录融合元数据到 log.json:                                 │
│     - source_workflows: [round_i, round_j, round_k]           │
│     - fusion_potential: Φ_merge                                │
│     - pairwise_metrics: {...}                                  │
│     - triplet_metrics: {...}                                   │
│                                                                 │
│  4. 在完整验证集上评估                                          │
│                                                                 │
│  5. 预期效果:                                                   │
│     - 总分 >= max(score_i, score_j, score_k)                  │
│     - 互补题目的正确率显著提升                                  │
│     - 可能牺牲少量原本正确的题目                                │
│                                                                 │
│  更新:                                                          │
│  - N_m += 1 (累计融合次数)                                     │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 关键代码位置

#### 2.2.1 主入口
**文件**: `scripts/enhanced_optimizer.py`
**行数**: 296-308

```python
elif operation == 'fuse':
    logger.info("=" * 80)
    logger.info(f"Executing FUSE operation for round {self.round + 1}")
    logger.info("=" * 80)
    
    # 执行融合
    score = await self._merge()
    
    if score is not None:
        # 融合成功
        self.N_m += 1  # 增加融合计数
```

#### 2.2.2 包络线选择
**文件**: `scripts/enhanced_optimizer.py`
**行数**: 658-729

```python
def _select_envelope_for_merge(self) -> List[Dict]:
    """
    选择包络线workflows (帕累托最优集合)
    
    包络线: 在至少一道题上表现最优的workflows集合
    
    算法:
        1. 提取每个workflow在每道题上的分数
        2. 对于每个workflow,检查是否被其他workflow支配
        3. 不被支配的workflows构成包络线
    
    支配关系:
        A dominates B if:
            ∀ problem: score_A[p] >= score_B[p]
            ∃ problem: score_A[p] > score_B[p]
    
    返回:
        List[Dict]: 包络线workflows
    """
```

#### 2.2.3 三元组选择
**文件**: `scripts/enhanced_optimizer.py`
**行数**: 832-1000

```python
def _select_for_merge(self, workflows: List[Dict]) -> Optional[Tuple[Dict, Dict, Dict]]:
    """
    Algorithm 3: SelectForMerge - 选择最佳三元组
    
    基于pairwise和triplet互补性,选择融合潜力最大的三个workflows
    
    公式:
        Φ_pair(i,j) = |C_i ⊕ C_j|  (对称差)
        Φ_triple(i,j,k) = |exactly_one| + |exactly_two|
        
        Φ_merge = β_p · (Φ_pair(i,j) + Φ_pair(j,k) + Φ_pair(i,k)) +
                  β_t · Φ_triple(i,j,k)
    
    返回:
        (W_i, W_j, W_k): 最佳三元组,失败返回None
    """
```

#### 2.2.4 融合执行
**文件**: `scripts/enhanced_optimizer.py`
**行数**: 1565-1620

```python
async def _execute_fusion_async(self, fusion_triple: Tuple) -> bool:
    """
    执行融合操作
    
    参数:
        fusion_triple: 选中的三元组 (W_i, W_j, W_k)
    
    返回:
        bool: 是否成功
    """
```

---

## 3. 核心算法详解

### 3.1 包络线选择 (Envelope Selection)

**目标**: 识别帕累托最优集合

**为什么需要包络线?**
- 减少搜索空间 (从O(n³)到O(k³), k << n)
- 排除被支配的workflows
- 确保选择的都是"最优"候选

**算法实现**:

```python
def select_envelope(workflows):
    """选择包络线workflows"""
    
    # 1. 加载问题级别的分数
    scores = {}  # scores[workflow_id][problem_id] = score
    for workflow in workflows:
        log = load_log(workflow)
        scores[workflow['id']] = {
            entry['problem_id']: entry['score']
            for entry in log
        }
    
    # 2. 检查支配关系
    envelope = []
    for i, w_i in enumerate(workflows):
        is_dominated = False
        
        for j, w_j in enumerate(workflows):
            if i == j:
                continue
            
            # 检查w_j是否支配w_i
            dominates = True
            strictly_better_on_some = False
            
            for problem_id in scores[w_i['id']].keys():
                score_i = scores[w_i['id']][problem_id]
                score_j = scores[w_j['id']][problem_id]
                
                if score_j < score_i:
                    dominates = False
                    break
                if score_j > score_i:
                    strictly_better_on_some = True
            
            if dominates and strictly_better_on_some:
                is_dominated = True
                break
        
        if not is_dominated:
            envelope.append(w_i)
    
    return envelope
```

**示例**:

```
有5个workflows:
  Problem 1: W1=1.0, W2=0.8, W3=0.9, W4=0.7, W5=0.6
  Problem 2: W1=0.6, W2=1.0, W3=0.8, W4=0.9, W5=0.7
  Problem 3: W1=0.7, W2=0.8, W3=1.0, W4=0.6, W5=0.8
  Problem 4: W1=0.9, W2=0.7, W3=0.6, W4=1.0, W5=0.7

分析:
  - W1: 在Problem 1最优 → 不被支配 ✓
  - W2: 在Problem 2最优 → 不被支配 ✓
  - W3: 在Problem 3最优 → 不被支配 ✓
  - W4: 在Problem 4最优 → 不被支配 ✓
  - W5: 在所有问题上都不是最优 → 被支配 ✗

结果: Envelope = {W1, W2, W3, W4}
```

### 3.2 互补性计算

#### 3.2.1 Pairwise互补性

**定义**: 两个workflows能互相弥补的题目数量

**公式**:
```
C_i = {p: W_i correct on p}  (W_i答对的题目集合)
C_j = {p: W_j correct on p}

Φ_pair(i, j) = |C_i ⊕ C_j|
             = |C_i \ C_j| + |C_j \ C_i|
             = |{p: W_i correct, W_j wrong}| +
               |{p: W_j correct, W_i wrong}|
```

**直观理解**:
- W_i答对但W_j答错: W_i可以帮助W_j
- W_j答对但W_i答错: W_j可以帮助W_i
- 两个和越大,互补性越强

**代码实现**:

```python
def pairwise_complementarity(w_i, w_j):
    """计算pairwise互补性"""
    correct_i = set(get_correct_problems(w_i))
    correct_j = set(get_correct_problems(w_j))
    
    # 对称差
    symmetric_diff = correct_i ^ correct_j
    return len(symmetric_diff)
```

**示例**:

```
W1 correct: {1, 2, 3, 5, 7, 9}     (6题)
W2 correct: {1, 3, 4, 6, 8, 9}     (6题)

W1独有: {2, 5, 7}                  (3题)
W2独有: {4, 6, 8}                  (3题)

Φ_pair(1, 2) = 3 + 3 = 6
```

#### 3.2.2 Triplet互补性

**定义**: 三个workflows需要"投票"才能正确的题目数量

**公式**:
```
对于每道题p,统计正确的workflows数量:

n_correct(p) = |{i: W_i correct on p}|

Exactly one correct: n_correct(p) = 1
Exactly two correct: n_correct(p) = 2

Φ_triple(i,j,k) = |{p: n_correct(p) = 1}| +
                  |{p: n_correct(p) = 2}|
```

**为什么关注1和2?**
- **1个正确**: 少数派正确,需要信任少数
- **2个正确**: 多数派正确,投票可决定
- **0个正确**: 三个都错,融合也无济于事
- **3个正确**: 已经全对,不需要融合

**代码实现**:

```python
def triplet_complementarity(w_i, w_j, w_k):
    """计算triplet互补性"""
    correct_i = get_correct_problems(w_i)
    correct_j = get_correct_problems(w_j)
    correct_k = get_correct_problems(w_k)
    
    # 统计每道题有多少个workflow答对
    all_problems = set(correct_i.keys()) | set(correct_j.keys()) | set(correct_k.keys())
    
    exactly_one = 0
    exactly_two = 0
    
    for problem in all_problems:
        n_correct = (
            (problem in correct_i) +
            (problem in correct_j) +
            (problem in correct_k)
        )
        
        if n_correct == 1:
            exactly_one += 1
        elif n_correct == 2:
            exactly_two += 1
    
    return exactly_one + exactly_two
```

**示例**:

```
Problem 1: W1✓ W2✓ W3✗ → 2个正确
Problem 2: W1✓ W2✗ W3✓ → 2个正确
Problem 3: W1✗ W2✓ W3✓ → 2个正确
Problem 4: W1✓ W2✗ W3✗ → 1个正确
Problem 5: W1✗ W2✗ W3✓ → 1个正确
Problem 6: W1✗ W2✗ W3✗ → 0个正确
Problem 7: W1✓ W2✓ W3✓ → 3个正确

Exactly one: {4, 5} → 2题
Exactly two: {1, 2, 3} → 3题

Φ_triple = 2 + 3 = 5
```

#### 3.2.3 总融合潜力

**公式**:
```
Φ_merge(i,j,k) = β_p · [Φ_pair(i,j) + Φ_pair(j,k) + Φ_pair(i,k)] +
                 β_t · Φ_triple(i,j,k)
```

**参数**:
- `β_p = 0.4`: Pairwise权重
- `β_t = 0.3`: Triplet权重

**计算示例**:

```
三元组 (W1, W2, W3):

Pairwise:
  Φ_pair(1,2) = 6
  Φ_pair(2,3) = 8
  Φ_pair(1,3) = 5
  Total pairwise = 6 + 8 + 5 = 19

Triplet:
  Φ_triple(1,2,3) = 5

Total:
  Φ_merge = 0.4 × 19 + 0.3 × 5
          = 7.6 + 1.5
          = 9.1
```

### 3.3 选择策略

**目标**: 在包络线中找到互补性最强的三元组

**算法**:
```python
def select_best_triple(envelope_workflows):
    """选择最佳三元组"""
    
    best_potential = 0
    best_triple = None
    
    # 遍历所有三元组组合
    for i in range(len(envelope_workflows)):
        for j in range(i+1, len(envelope_workflows)):
            for k in range(j+1, len(envelope_workflows)):
                w_i = envelope_workflows[i]
                w_j = envelope_workflows[j]
                w_k = envelope_workflows[k]
                
                # 计算融合潜力
                phi_ij = pairwise_complementarity(w_i, w_j)
                phi_jk = pairwise_complementarity(w_j, w_k)
                phi_ik = pairwise_complementarity(w_i, w_k)
                phi_triple = triplet_complementarity(w_i, w_j, w_k)
                
                phi_merge = (
                    BETA_P * (phi_ij + phi_jk + phi_ik) +
                    BETA_T * phi_triple
                )
                
                if phi_merge > best_potential:
                    best_potential = phi_merge
                    best_triple = (w_i, w_j, w_k)
    
    return best_triple, best_potential
```

**复杂度**:
- 包络线大小: k
- 三元组数量: C(k,3) = k(k-1)(k-2)/6
- 时间复杂度: O(k³ × n), n为问题数量

**优化**: 
- 限制包络线大小 (k <= 10)
- 缓存正确问题集合
- 并行计算互补性

---

## 4. LLM融合策略

### 4.1 提示词设计

**目标**: 指导LLM生成综合性workflow

**提示词结构**:

```python
fusion_prompt = f"""
You are creating a UNIFIED workflow by merging three complementary workflows.

## Three Source Workflows

### Workflow 1 (Round {round_i})
Score: {score_i}/119

```python
# graph.py
{graph_code_i}
```

```python
# prompt.py
{prompt_code_i}
```

### Workflow 2 (Round {round_j})
Score: {score_j}/119

```python
# graph.py
{graph_code_j}
```

```python
# prompt.py
{prompt_code_j}
```

### Workflow 3 (Round {round_k})
Score: {score_k}/119

```python
# graph.py
{graph_code_k}
```

```python
# prompt.py
{prompt_code_k}
```

## Complementarity Analysis

Pairwise metrics:
- Workflow 1 ⊕ Workflow 2: {phi_ij} problems
  * Workflow 1 unique: {unique_i_vs_j} problems
  * Workflow 2 unique: {unique_j_vs_i} problems

- Workflow 2 ⊕ Workflow 3: {phi_jk} problems
  * Workflow 2 unique: {unique_j_vs_k} problems
  * Workflow 3 unique: {unique_k_vs_j} problems

- Workflow 1 ⊕ Workflow 3: {phi_ik} problems
  * Workflow 1 unique: {unique_i_vs_k} problems
  * Workflow 3 unique: {unique_k_vs_i} problems

Triplet metrics:
- Exactly one correct: {exactly_one} problems
- Exactly two correct: {exactly_two} problems

Total fusion potential: {phi_merge:.2f}

## Your Task

Create a UNIFIED workflow that combines the strengths of all three.

FUSION STRATEGIES:

1. **Serial Fusion**: Chain the workflows sequentially
   - Use Workflow 1's approach first
   - If uncertain, try Workflow 2's approach
   - Final check with Workflow 3's approach

2. **Parallel Fusion**: Run multiple approaches and vote
   - Execute all three strategies
   - Compare results
   - Choose the most confident answer

3. **Conditional Fusion**: Route based on problem characteristics
   - If problem has X feature, use Workflow 1
   - If problem has Y feature, use Workflow 2
   - Otherwise, use Workflow 3

4. **Hybrid Fusion**: Combine the above strategies
   - Use different fusion strategies for different steps

IMPORTANT:
- Aim for score > max({score_i}, {score_j}, {score_k})
- Focus on complementary strengths
- The unified workflow should be MORE capable than any individual workflow

## Output Format

```python
# graph.py
<unified graph code>
```

```python
# prompt.py
<unified prompt code>
```

Explanation:
<explain your fusion strategy and how it combines the three workflows>
"""
```

### 4.2 融合策略类型

#### 4.2.1 串行融合 (Serial Fusion)

**概念**: 将三个workflows按顺序连接

**示例**:
```python
class UnifiedGraph:
    def __init__(self):
        self.stage1 = Workflow1_Reasoner()
        self.stage2 = Workflow2_Verifier()
        self.stage3 = Workflow3_Refiner()
    
    def __call__(self, problem):
        # Stage 1: 初步推理
        result1 = self.stage1(problem)
        
        # Stage 2: 验证
        verified = self.stage2(problem, result1)
        
        # Stage 3: 精炼
        final = self.stage3(problem, verified)
        
        return final
```

**优点**:
- 结构清晰
- 易于理解
- 可以逐步改进

**缺点**:
- 错误会传播
- 计算成本高

#### 4.2.2 并行融合 (Parallel Fusion)

**概念**: 同时运行三个workflows,投票决定

**示例**:
```python
class UnifiedGraph:
    def __init__(self):
        self.workflow1 = Workflow1()
        self.workflow2 = Workflow2()
        self.workflow3 = Workflow3()
    
    def __call__(self, problem):
        # 并行执行
        result1 = self.workflow1(problem)
        result2 = self.workflow2(problem)
        result3 = self.workflow3(problem)
        
        # 投票
        votes = [result1, result2, result3]
        final = majority_vote(votes)
        
        return final
```

**优点**:
- 鲁棒性强
- 可以纠错

**缺点**:
- 计算成本最高
- 需要投票机制

#### 4.2.3 条件融合 (Conditional Fusion)

**概念**: 根据问题特征路由到不同workflow

**示例**:
```python
class UnifiedGraph:
    def __init__(self):
        self.workflow1 = Workflow1()  # 擅长代数
        self.workflow2 = Workflow2()  # 擅长几何
        self.workflow3 = Workflow3()  # 擅长组合
        self.router = ProblemRouter()
    
    def __call__(self, problem):
        # 路由
        problem_type = self.router.classify(problem)
        
        if problem_type == 'algebra':
            return self.workflow1(problem)
        elif problem_type == 'geometry':
            return self.workflow2(problem)
        else:
            return self.workflow3(problem)
```

**优点**:
- 高效
- 针对性强

**缺点**:
- 需要准确的路由
- 可能遗漏混合类型

#### 4.2.4 混合融合 (Hybrid Fusion)

**概念**: 结合多种策略

**示例**:
```python
class UnifiedGraph:
    def __init__(self):
        self.workflow1 = Workflow1()
        self.workflow2 = Workflow2()
        self.workflow3 = Workflow3()
        self.router = ProblemRouter()
    
    def __call__(self, problem):
        # Stage 1: 路由决定主要方法
        problem_type = self.router.classify(problem)
        
        if problem_type == 'algebra':
            primary = self.workflow1(problem)
            # 用workflow2验证
            verified = self.workflow2.verify(problem, primary)
            return verified
        
        elif problem_type == 'geometry':
            # 并行运行workflow2和workflow3,投票
            result2 = self.workflow2(problem)
            result3 = self.workflow3(problem)
            return majority_vote([result2, result3])
        
        else:
            # 串行: workflow1 → workflow2 → workflow3
            r1 = self.workflow1(problem)
            r2 = self.workflow2(problem, r1)
            r3 = self.workflow3(problem, r2)
            return r3
```

---

## 5. 融合元数据

### 5.1 记录融合信息

**位置**: `log.json` 中添加融合元数据

```json
{
  "fusion_metadata": {
    "source_workflows": ["round_5", "round_8", "round_11"],
    "fusion_potential": 9.1,
    "pairwise_metrics": {
      "5_8": {"phi": 6, "unique_5": 3, "unique_8": 3},
      "8_11": {"phi": 8, "unique_8": 4, "unique_11": 4},
      "5_11": {"phi": 5, "unique_5": 2, "unique_11": 3}
    },
    "triplet_metrics": {
      "exactly_one": 2,
      "exactly_two": 3,
      "phi_triple": 5
    },
    "fusion_strategy": "hybrid",
    "timestamp": "2025-12-15T11:30:45"
  },
  "execution_logs": [...]
}
```

### 5.2 跟踪融合历史

**目的**: 避免重复融合相同的组合

**实现**: 记录已融合的三元组

```python
self.fused_triples = set()

# 融合前检查
triple_id = tuple(sorted([w1['round'], w2['round'], w3['round']]))
if triple_id in self.fused_triples:
    logger.info(f"Triple {triple_id} already fused, skipping")
    return None

# 融合后记录
self.fused_triples.add(triple_id)
```

---

## 6. 典型执行场景

### 6.1 成功融合案例

```
[Round 12 融合]

包络线选择:
  - 共有11个workflows
  - 包络线: {Round 5, 8, 9, 11} (4个)

三元组选择:
  - 评估了C(4,3)=4个三元组
  - 最佳: (Round 5, 8, 11)
  
互补性分析:
  Pairwise:
    R5 ⊕ R8: 12题 (R5独有6题, R8独有6题)
    R8 ⊕ R11: 15题 (R8独有7题, R11独有8题)
    R5 ⊕ R11: 10题 (R5独有5题, R11独有5题)
    Total: 37题
  
  Triplet:
    Exactly one: 5题
    Exactly two: 8题
    Total: 13题
  
  Fusion Potential:
    Φ_merge = 0.4 × 37 + 0.3 × 13 = 18.7

融合策略: 混合融合
  - 代数题: 用Round 5 (专长)
  - 几何题: Round 8和Round 11并行+投票
  - 其他题: 串行 R5→R8→R11

评估结果:
  Round 12:
    - 总分: 75/119 (63.0%) ← 新最高分!
    - 比最好的单个workflow (R11: 69/119) 提升6题
    - 互补题目正确率: 从50%提升到85%
  
融合成功! ✓
```

### 6.2 workflows不足

```
[Round 3 融合尝试]

预检查:
  - 可用workflows: 2个 (Round 1, 2)
  - 需要: >= 3个
  
结果:
  跳过融合,返回None
  日志: "Insufficient workflows for fusion: need 3, have 2"
```

### 6.3 无明显互补性

```
[Round 15 融合尝试]

包络线选择:
  - 包络线: {Round 5, 8, 11, 12, 14} (5个)

三元组评估:
  - 所有三元组的互补性都很低
  - 最佳三元组: (R12, R14, R5), Φ_merge = 2.3
  
原因:
  - 这些workflows已经很相似
  - 大部分题目的答案一致
  - 少量互补题目不足以justify融合

决策:
  虽然返回了三元组,但融合潜力过低
  可能不会带来显著提升
  
结果:
  继续融合,但效果可能有限
  Round 16: 73/119 (比R12的75下降了2题)
```

---

## 7. 关键参数

### 7.1 融合概率参数

| 参数 | 默认值 | 说明 | 影响 |
|------|--------|------|------|
| `α_m` | 0.4 | 融合基础权重 | 控制融合基础概率 |
| `η_m` | 0.1 | 融合衰减率 | 控制重复融合惩罚 |
| `N_m` | 动态 | 累计融合次数 | 影响衰减计算 |

**概率计算**:
```python
p_merge_raw = α_m · plateau_t · exp(-η_m · N_m)
```

### 7.2 互补性参数

| 参数 | 默认值 | 说明 | 理论依据 |
|------|--------|------|----------|
| `β_p` | 0.4 | Pairwise权重 | 强调两两互补 |
| `β_t` | 0.3 | Triplet权重 | 考虑三方协同 |

**调整建议**:
```python
# 强调pairwise (适合差异大的workflows)
β_p = 0.5, β_t = 0.2

# 强调triplet (适合需要投票的场景)
β_p = 0.3, β_t = 0.4

# 平衡 (默认)
β_p = 0.4, β_t = 0.3
```

### 7.3 选择阈值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_fusion_potential` | 0.0 | 最小融合潜力阈值 |
| `max_envelope_size` | 无限制 | 包络线最大大小 |
| `min_workflows_for_fusion` | 3 | 融合所需最少workflows |

---

## 8. 代码映射

### 8.1 主要方法

```python
class EnhancedOptimizer:
    # 融合主流程
    async def _merge(self):
        """Line 1003-1051"""
        pass
    
    # 选择包络线
    def _select_envelope_for_merge(self):
        """Line 658-729"""
        pass
    
    # 选择最佳三元组
    def _select_for_merge(self, workflows):
        """Line 832-1000"""
        pass
    
    # 执行融合
    async def _execute_fusion_async(self, fusion_triple):
        """Line 1565-1620"""
        pass
    
    # 计算pairwise互补性
    def _pairwise_complementarity(self, w_i, w_j):
        """内部实现,Line ~870"""
        pass
    
    # 计算triplet互补性
    def _triplet_complementarity(self, w_i, w_j, w_k):
        """内部实现,Line ~920"""
        pass
```

### 8.2 数据流

```
results.json ────┐
                 │
log.json ────────┼────→ _select_envelope_for_merge()
                 │      ├─ 提取问题级别分数
                 │      ├─ 计算支配关系
                 │      └─→ envelope_workflows
                 │
envelope_workflows ──→ _select_for_merge()
                 │      ├─ 计算所有三元组的互补性
                 │      ├─ Φ_pair(i,j) for all pairs
                 │      ├─ Φ_triple(i,j,k) for all triples
                 │      ├─ Φ_merge = β_p·Σ_pair + β_t·φ_triple
                 │      └─→ best_triple (W_i, W_j, W_k)
                 │
best_triple ──→ _execute_fusion_async(fusion_triple)
                 │      ├─ 加载三个workflows的代码
                 │      ├─ 生成融合提示词
                 │      ├─ LLM生成融合workflow
                 │      ├─ 保存到round_N/
                 │      └─→ 评估并返回分数
```

---

## 9. 最佳实践

### 9.1 何时使用融合

**适合场景**:
- ✅ 有3个以上workflows可用
- ✅ 停滞度 > 0.4 (优化和分化都遇到瓶颈)
- ✅ workflows之间有明显差异

**不适合场景**:
- ❌ workflows数量 < 3
- ❌ workflows非常相似
- ❌ 早期轮次 (优先优化和分化)

### 9.2 融合策略建议

1. **优先融合专业化分支**: 对不同类别的专家进行融合
2. **避免重复融合**: 记录已融合的组合
3. **监控融合效果**: 如果连续失败,调整参数或策略
4. **定期清理**: 移除表现差的融合结果

### 9.3 调试技巧

```bash
# 1. 查看包络线
grep "Envelope workflows" logs/AFlow.log

# 2. 查看互补性分析
grep "Complementarity" logs/AFlow.log
grep "Φ_merge" logs/AFlow.log

# 3. 检查融合元数据
cat workspace/DATASET/workflows/round_12/log.json | jq '.fusion_metadata'

# 4. 比较融合前后
# 融合源workflows
cat workspace/DATASET/workflows/results.json | \
  jq '.[] | select(.round == 5 or .round == 8 or .round == 11)'

# 融合结果
cat workspace/DATASET/workflows/results.json | \
  jq '.[] | select(.round == 12)'
```

---

## 10. 常见问题

### Q1: 为什么融合后分数反而下降?
**A**: 可能原因:
1. LLM融合策略不当 (串行传播错误)
2. 互补性计算有误 (选择的三元组不合适)
3. 三个workflows实际上很相似
4. 融合增加了复杂度,引入新错误

**解决方案**:
- 检查融合元数据,验证互补性
- 调整融合策略 (尝试并行+投票)
- 提高互补性阈值 (`min_fusion_potential`)

### Q2: 包络线太大怎么办?
**A**: 
- 限制包络线大小: `max_envelope_size = 10`
- 在包络线中优先选择:
  * 分数最高的workflows
  * 最近创建的workflows
  * 多样性最大的workflows

### Q3: 如何选择融合策略?
**A**: 根据workflows特点:
- **差异大**: 条件融合 (根据问题类型路由)
- **差异小**: 串行融合 (逐步改进)
- **混合**: 并行融合 (投票决定)

### Q4: 融合会丢失原workflows吗?
**A**: 不会! 原workflows仍然保留在各自的round目录中,可以继续参与后续操作。

### Q5: 可以融合超过3个workflows吗?
**A**: 当前实现只支持3-way fusion。如果需要更多:
- 先做3-way融合得到中间结果
- 再用中间结果和其他workflows融合
- 形成层次化融合

---

## 11. 下一步

- [分化操作详解](DIFFERENTIATION_OPERATION.md) - 了解如何创建专业化分支
- [优化操作详解](OPTIMIZE_OPERATION.md) - 了解通用优化策略
- [系统架构总览](SYSTEM_ARCHITECTURE.md) - 返回整体视图

---

## 附录: 完整算法伪代码

```python
def fuse():
    """融合操作完整流程"""
    
    # 预检查
    workflows = load_all_workflows()
    if len(workflows) < 3:
        logger.info("Insufficient workflows for fusion")
        return None
    
    # 阶段1: 选择包络线
    envelope = select_envelope(workflows)
    logger.info(f"Envelope size: {len(envelope)}")
    
    if len(envelope) < 3:
        logger.info("Envelope too small")
        return None
    
    # 阶段2: 选择最佳三元组
    best_triple, phi_merge = select_best_triple(envelope)
    
    if best_triple is None:
        logger.info("No suitable triple found")
        return None
    
    w_i, w_j, w_k = best_triple
    logger.info(f"Selected triple: ({w_i['round']}, {w_j['round']}, {w_k['round']})")
    logger.info(f"Fusion potential: {phi_merge:.2f}")
    
    # 阶段3: 生成融合workflow
    code_i = load_workflow_code(w_i)
    code_j = load_workflow_code(w_j)
    code_k = load_workflow_code(w_k)
    
    fused_code = llm_generate_fusion(
        code_i=code_i,
        code_j=code_j,
        code_k=code_k,
        phi_merge=phi_merge
    )
    
    # 阶段4: 评估
    save_workflow(current_round + 1, fused_code)
    score = evaluate_on_full_validation_set()
    
    # 阶段5: 更新元数据
    N_m += 1
    fused_triples.add((w_i['round'], w_j['round'], w_k['round']))
    
    return score

def select_envelope(workflows):
    """选择包络线workflows"""
    scores = load_all_problem_scores(workflows)
    
    envelope = []
    for i, w_i in enumerate(workflows):
        is_dominated = False
        
        for j, w_j in enumerate(workflows):
            if i == j:
                continue
            
            # 检查w_j是否支配w_i
            if dominates(w_j, w_i, scores):
                is_dominated = True
                break
        
        if not is_dominated:
            envelope.append(w_i)
    
    return envelope

def select_best_triple(envelope):
    """选择最佳三元组"""
    best_potential = 0
    best_triple = None
    
    for i in range(len(envelope)):
        for j in range(i+1, len(envelope)):
            for k in range(j+1, len(envelope)):
                w_i, w_j, w_k = envelope[i], envelope[j], envelope[k]
                
                # 计算互补性
                phi_ij = pairwise_complementarity(w_i, w_j)
                phi_jk = pairwise_complementarity(w_j, w_k)
                phi_ik = pairwise_complementarity(w_i, w_k)
                phi_triple = triplet_complementarity(w_i, w_j, w_k)
                
                # 总融合潜力
                phi_merge = (
                    BETA_P * (phi_ij + phi_jk + phi_ik) +
                    BETA_T * phi_triple
                )
                
                if phi_merge > best_potential:
                    best_potential = phi_merge
                    best_triple = (w_i, w_j, w_k)
    
    return best_triple, best_potential
```
