# 分化对象选择算法改进方案

## 当前问题分析

### 现有算法（`_select_for_split`）的问题

**当前方法**：
```python
Φ_split(i) = L_i · [λ₁(1-R_i) + λ₂D_i] · exp(-η_s·s_i)
```

**问题**：
1. **忽略问题分类信息**：没有利用问题类别（category）的统计信息
2. **缺乏细粒度分析**：只看整体性能，不看子问题领域的表现
3. **无法识别"偏科专家"**：无法发现某个工作流在特定子领域有优势
4. **分化方向不明确**：不知道应该针对哪个子问题类别进行分化

## 新算法设计：优势特征提取模型

### 核心思想

**从"整体性能评估"转变为"子领域优势识别"**

识别那些**"对工作流总分贡献巨大，且在特定子领域表现显著优于全局水平"**的工作流。

### 关键指标

#### 1. 全局正确率 (Global Accuracy)
```
Acc_global = C_total / N
```
- `C_total`: 工作流正确解决的问题总数
- `N`: 数据集总大小
- **作用**：基准线，子领域必须显著超过此值才有分化价值

#### 2. 子问题召回率 (Category Recall)
```
Recall_k = C_k / N_k
```
- `C_k`: 工作流正确解决第 k 类问题的数量
- `N_k`: 数据集中第 k 类问题的总数
- **作用**：衡量工作流在该特定领域的"统治力"

#### 3. 子问题贡献度 (Category Contribution)
```
Contrib_k = C_k / C_total
```
- **作用**：衡量该类问题在工作流"功劳簿"上的比重

### 分化潜力公式

```
Score_split(W) = max_k [ Contrib_k · (Recall_k - Acc_global) ]
```

**物理含义**：
- `(Recall_k - Acc_global)`: **特化红利** - 子领域超过整体的表现
- `Contrib_k`: **身份确认** - 该子领域在工作流中的重要性
- 取 max：找到最具分化潜力的子领域

### 算法流程

```python
def select_workflow_to_split(workflows, dataset_meta):
    """
    Args:
        workflows: 所有工作流列表
        dataset_meta: {"category_A": N_k, ...} 每类问题的总数
    
    Returns:
        selected_workflow: 选中的工作流
        target_category: 应该针对的子问题类别
    """
    candidates = []
    
    total_problems = sum(dataset_meta.values())
    
    for W in workflows:
        # 1. 获取分类统计
        correct_counts = W.category_statistics  # {"category_A": C_k, ...}
        c_total = sum(correct_counts.values())
        
        if c_total == 0:
            continue
        
        # 2. 计算全局正确率（基准线）
        acc_global = c_total / total_problems
        
        # 3. 遍历所有子问题类别，找最大优势
        max_score = -1.0
        best_category = None
        
        for category, n_k in dataset_meta.items():
            if n_k == 0:
                continue
            
            c_k = correct_counts.get(category, 0)
            
            # 子问题召回率
            recall_k = c_k / n_k
            
            # 只考虑超过全局水平的子领域
            if recall_k > acc_global:
                # 子问题贡献度
                contrib_k = c_k / c_total
                
                # 分化潜力得分
                score_k = contrib_k * (recall_k - acc_global)
                
                if score_k > max_score:
                    max_score = score_k
                    best_category = category
        
        # 4. 记录该工作流的分化潜力
        final_score = max(0.0, max_score)
        
        candidates.append({
            'workflow': W,
            'split_potential': final_score,
            'target_category': best_category,
            'acc_global': acc_global
        })
    
    # 5. Softmax 采样选择
    scores = [c['split_potential'] for c in candidates]
    probabilities = softmax(scores)
    selected_idx = np.random.choice(len(candidates), p=probabilities)
    
    return candidates[selected_idx]['workflow'], candidates[selected_idx]['target_category']
```

## 数值案例验证

### 场景1：偏科专家（高分化价值）

**数据集**：N=100（A类50题，B类50题）

**工作流 W1**：答对40个A，答对10个B
- C_total = 50
- Acc_global = 0.5

**针对A类**：
- Recall_A = 40/50 = 0.8
- Contrib_A = 40/50 = 0.8
- Score_A = 0.8 × (0.8 - 0.5) = **0.24**

**针对B类**：
- Recall_B = 10/50 = 0.2
- Score_B = 0.2 × (0.2 - 0.5) = -0.06

**结果**：分化潜力 = **0.24**，目标类别 = **A类**

### 场景2：平庸混合体（低分化价值）

**工作流 W2**：答对25个A，答对25个B
- C_total = 50
- Acc_global = 0.5

**针对A类**：
- Recall_A = 25/50 = 0.5
- Score_A = 0.5 × (0.5 - 0.5) = **0**

**结果**：分化潜力 = **0**（没有优势领域）

### 对比结论

虽然 W1 和 W2 总分相同（都是50分），但算法成功识别：
- W1 是被B类拖累的A类专家，有很高分化价值
- W2 平庸全面，分化无意义

## 代码修改清单

### 1. 数据结构准备

**需要添加**：
- 问题类别元数据：每类问题的总数
- 工作流的分类统计：每个工作流在各类别的表现

**位置**：
- `scripts/problem_classifier.py` - 已有分类系统
- `scripts/optimizer_utils/data_utils.py` - 加载分类统计

### 2. 核心方法重写

**文件**：`scripts/enhanced_optimizer.py`

**方法**：`_select_for_split(workflow_results)`

**主要修改**：
```python
# 旧版（第487-590行）
def _select_for_split(self, workflow_results: List[Dict]) -> Dict:
    # 基于 L_i, R_i, D_i 的势函数
    # 没有利用问题分类信息
    ...

# 新版
def _select_for_split(self, workflow_results: List[Dict]) -> tuple:
    """
    返回：(选中的工作流, 目标分化类别)
    """
    # 1. 加载问题分类元数据
    category_metadata = self._load_category_metadata()
    total_problems = sum(category_metadata.values())
    
    # 2. 为每个工作流计算分化潜力
    candidates = []
    for workflow in workflow_results:
        # 2.1 加载该工作流的分类统计
        category_stats = self._load_workflow_category_stats(workflow)
        c_total = sum(category_stats.values())
        
        if c_total == 0:
            continue
        
        # 2.2 计算全局正确率
        acc_global = c_total / total_problems
        
        # 2.3 遍历所有类别，找最大优势
        max_score = -1.0
        best_category = None
        
        for category, n_k in category_metadata.items():
            c_k = category_stats.get(category, 0)
            recall_k = c_k / n_k if n_k > 0 else 0
            
            if recall_k > acc_global:
                contrib_k = c_k / c_total
                score_k = contrib_k * (recall_k - acc_global)
                
                if score_k > max_score:
                    max_score = score_k
                    best_category = category
        
        final_score = max(0.0, max_score)
        candidates.append({
            'workflow': workflow,
            'split_potential': final_score,
            'target_category': best_category,
            ...
        })
    
    # 3. Softmax 采样
    ...
    
    return selected_workflow, target_category
```

### 3. 辅助方法新增

**需要添加的方法**：

#### a. `_load_category_metadata()`
```python
def _load_category_metadata(self) -> Dict[str, int]:
    """
    加载问题分类元数据
    
    Returns:
        {"category_A": N_k, "category_B": N_k, ...}
    """
    # 从 problem_classifier 获取
    stats = self.problem_classifier.get_category_statistics()
    return {cat: info['count'] for cat, info in stats.items()}
```

#### b. `_load_workflow_category_stats(workflow)`
```python
def _load_workflow_category_stats(self, workflow: Dict) -> Dict[str, int]:
    """
    加载工作流在各类别的统计
    
    Returns:
        {"category_A": C_k, "category_B": C_k, ...}
    """
    # 从 log.json 或 results.json 中加载
    # 需要在评估时记录每个问题的类别标签
    round_num = workflow['round']
    log_path = f"{self.root_path}/workflows/round_{round_num}/log.json"
    
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # 统计每个类别的正确数量
    category_counts = {}
    for entry in log_data:
        category = entry.get('category', 'unknown')
        is_correct = entry.get('is_correct', False)
        
        if is_correct:
            category_counts[category] = category_counts.get(category, 0) + 1
    
    return category_counts
```

### 4. 评估系统修改

**文件**：`scripts/optimizer_utils/evaluation_utils.py`

**修改**：在 `evaluate_graph()` 时记录问题类别

```python
# 评估每个问题时，附加类别标签
for problem in validation_data:
    problem_id = problem['id']
    category = self.problem_classifier.get_problem_category(problem_id)
    
    result = await evaluate_single_problem(problem)
    result['category'] = category  # 添加类别标签
    
    log_data.append(result)
```

### 5. 分化执行修改

**文件**：`scripts/enhanced_optimizer.py`

**方法**：`_execute_single_differentiation()`

**修改**：传递目标类别给分化处理器

```python
async def _execute_single_differentiation(self) -> float:
    # 旧版：selected_workflow = self._select_for_split(round_summaries)
    # 新版：
    selected_workflow, target_category = self._select_for_split(round_summaries)
    
    logger.info(f"Selected workflow from round {source_round}")
    logger.info(f"Target specialization category: {target_category}")
    
    # 传递给分化处理器
    differentiation_response = await self.differentiation_processor.create_differentiated_workflow(
        source_workflow=source_workflow_content,
        differentiation_direction="problem_type_specialization",
        operator_description=operator_description,
        target_round=next_round,
        target_category=target_category,  # 明确目标类别
        category_examples=example_problems
    )
```

### 6. 日志系统修改

**位置**：`scripts/enhanced_optimizer.py` → `_select_for_split()`

**新增日志**：
```python
logger.info(f"Category Metadata: {category_metadata}")
logger.info(f"Total problems: {total_problems}")

for candidate in candidates:
    logger.info(f"  Round {candidate['round']}:")
    logger.info(f"    Acc_global: {candidate['acc_global']:.4f}")
    logger.info(f"    Best category: {candidate['target_category']}")
    logger.info(f"    Split potential: {candidate['split_potential']:.4f}")
    
    # 详细类别分析
    for cat, stats in candidate['category_analysis'].items():
        logger.info(f"      {cat}: Recall={stats['recall']:.4f}, "
                   f"Contrib={stats['contrib']:.4f}, Score={stats['score']:.4f}")
```

## 数据流依赖

```
问题分类系统 (problem_classifier)
    ↓
问题类别元数据 (category_metadata)
    ↓
工作流评估时记录类别标签 (log.json)
    ↓
工作流分类统计 (category_stats)
    ↓
分化潜力计算 (split_potential)
    ↓
目标类别选择 (target_category)
    ↓
针对性分化 (specialized workflow)
```

## 实现优先级

### 阶段1：基础数据支持（必需）
- [ ] 修改评估系统，记录问题类别标签
- [ ] 实现 `_load_category_metadata()`
- [ ] 实现 `_load_workflow_category_stats()`

### 阶段2：核心算法重写（核心）
- [ ] 重写 `_select_for_split()` 方法
- [ ] 添加详细的类别分析日志
- [ ] 返回 (workflow, target_category) 元组

### 阶段3：集成与验证（完善）
- [ ] 修改 `_execute_single_differentiation()` 传递目标类别
- [ ] 更新分化处理器使用目标类别
- [ ] 端到端测试验证

### 阶段4：优化与扩展（可选）
- [ ] 添加多目标分化（选择top-k类别）
- [ ] 引入时间衰减（近期表现更重要）
- [ ] 自适应阈值（根据数据集特性调整）

## 潜在问题与解决方案

### 问题1：问题分类信息缺失

**问题**：早期轮次可能没有分类信息

**解决方案**：
- 在第一轮优化前完成问题分类（已实现）
- 评估时强制记录类别标签
- 缺失时使用 "unknown" 类别

### 问题2：类别不平衡

**问题**：某些类别问题数量很少

**解决方案**：
- 设置最小类别大小阈值（如 N_k >= 5）
- 对小类别问题上采样
- 使用归一化的 Recall 而非绝对数量

### 问题3：所有类别都没有优势

**问题**：工作流在所有类别都低于平均水平

**解决方案**：
- 分化潜力为 0，该工作流不会被选中
- Softmax 会自动选择其他有潜力的工作流
- 如果所有工作流都为 0，退化为随机选择

### 问题4：计算开销

**问题**：需要遍历所有工作流×所有类别

**解决方案**：
- 缓存分类统计结果
- 只在分化操作触发时计算
- 预先过滤低性能工作流（top-k candidates）

## 总结

### 核心改进点

1. **细粒度分析**：从整体性能到子领域优势
2. **明确分化方向**：知道应该针对哪个类别
3. **理论支撑**：基于 Recall、Contribution 的数学模型
4. **可解释性**：清晰的优势识别逻辑

### 与现有系统的兼容性

- ✅ 利用已有的问题分类系统
- ✅ 与概率控制机制正交（互不影响）
- ✅ 保留 Softmax 采样避免确定性
- ✅ 日志系统易于调试分析

### 预期效果

- 选择真正有分化价值的"偏科专家"
- 避免分化平庸的全面型工作流
- 分化后的工作流针对性更强
- 系统整体覆盖率提升
