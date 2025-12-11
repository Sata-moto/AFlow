# 分化对象选择算法修改确认清单

## 概述

将分化对象选择从**基于整体性能的势函数**改为**基于子问题优势的特征提取模型**。

核心公式：`Score_split(W) = max_k [Contrib_k · (Recall_k - Acc_global)]`

## 需要修改的文件和具体位置

### 📁 文件1：`scripts/optimizer_utils/evaluation_utils.py`

#### 修改点1.1：记录问题类别标签

**位置**：`evaluate_graph()` 方法内的问题评估循环

**当前代码** （约第150-200行）：
```python
for i in range(len(validation_data)):
    problem = validation_data[i]
    result = await self.evaluate_single_problem(...)
    log_data.append({
        'problem_id': problem_id,
        'is_correct': is_correct,
        'output': output,
        ...
    })
```

**修改为**：
```python
for i in range(len(validation_data)):
    problem = validation_data[i]
    problem_id = problem.get('id') or problem.get('problem_id') or i
    
    # 获取问题类别（如果分类系统可用）
    category = 'unknown'
    if hasattr(optimizer, 'problem_classifier') and optimizer.problem_classifier:
        classifications = optimizer.problem_classifier.load_classifications()
        category = classifications.get(str(problem_id), 'unknown')
    
    result = await self.evaluate_single_problem(...)
    log_data.append({
        'problem_id': problem_id,
        'category': category,  # 新增：问题类别
        'is_correct': is_correct,
        'output': output,
        ...
    })
```

**作用**：为后续统计提供类别信息

---

### 📁 文件2：`scripts/enhanced_optimizer.py`

#### 修改点2.1：添加辅助方法 - 加载类别元数据

**位置**：在 `_calculate_diversity()` 方法后添加

**新增代码**：
```python
def _load_category_metadata(self) -> Dict[str, int]:
    """
    加载问题分类元数据：每个类别有多少问题
    
    Returns:
        Dict[str, int]: {"category_A": N_k, "category_B": N_k, ...}
    """
    if not self.enable_differentiation or not hasattr(self, 'problem_classifier'):
        return {}
    
    try:
        stats = self.problem_classifier.get_category_statistics()
        category_metadata = {cat: info['count'] for cat, info in stats.items()}
        logger.info(f"Loaded category metadata: {len(category_metadata)} categories")
        for cat, count in category_metadata.items():
            logger.debug(f"  {cat}: {count} problems")
        return category_metadata
    except Exception as e:
        logger.warning(f"Failed to load category metadata: {e}")
        return {}
```

#### 修改点2.2：添加辅助方法 - 加载工作流分类统计

**位置**：在 `_load_category_metadata()` 方法后添加

**新增代码**：
```python
def _load_workflow_category_stats(self, workflow: Dict) -> Dict[str, int]:
    """
    加载工作流在各类别的统计：每个类别答对了多少题
    
    Args:
        workflow: 工作流信息字典
        
    Returns:
        Dict[str, int]: {"category_A": C_k, "category_B": C_k, ...}
    """
    round_num = workflow.get('round', 0)
    log_path = f"{self.root_path}/workflows/round_{round_num}/log.json"
    
    if not os.path.exists(log_path):
        logger.warning(f"Log file not found: {log_path}")
        return {}
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        # 统计每个类别的正确数量
        category_counts = {}
        for entry in log_data:
            if isinstance(entry, dict):
                category = entry.get('category', 'unknown')
                is_correct = entry.get('is_correct', False)
                
                if is_correct:
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        logger.debug(f"Round {round_num} category stats: {category_counts}")
        return category_counts
        
    except Exception as e:
        logger.warning(f"Failed to load workflow category stats for round {round_num}: {e}")
        return {}
```

#### 修改点2.3：完全重写 `_select_for_split()` 方法

**位置**：当前第487-590行

**当前方法签名**：
```python
def _select_for_split(self, workflow_results: List[Dict]) -> Dict:
```

**新方法签名**：
```python
def _select_for_split(self, workflow_results: List[Dict]) -> tuple:
```

**完整新实现**：
```python
def _select_for_split(self, workflow_results: List[Dict]) -> tuple:
    """
    理论化分化目标选择 (Algorithm 2: SelectForSplit - 优势特征提取模型)
    
    基于子问题优势识别最适合分化的工作流：
    Score_split(W) = max_k [Contrib_k · (Recall_k - Acc_global)]
    
    其中：
    - Acc_global = C_total / N：全局正确率（基准线）
    - Recall_k = C_k / N_k：子问题k的召回率（统治力）
    - Contrib_k = C_k / C_total：子问题k的贡献度（重要性）
    
    算法寻找"偏科专家"：在某子领域表现显著优于整体水平，且该子领域是其主要得分来源
    
    Args:
        workflow_results: 所有工作流的结果列表
        
    Returns:
        tuple: (选中的工作流, 目标分化类别)
               如果没有合适的工作流，返回 (None, None)
    """
    logger.info("=" * 80)
    logger.info("SelectForSplit: Dominant Feature Extraction for Specialization")
    logger.info("=" * 80)
    
    if not workflow_results or len(workflow_results) == 0:
        logger.warning("No workflows available for split selection")
        return None, None
    
    # Step 1: 加载问题分类元数据
    category_metadata = self._load_category_metadata()
    
    if not category_metadata:
        logger.warning("No category metadata available, falling back to simple selection")
        # 回退到简单选择：选择性能最好但未被过度分化的工作流
        best_workflow = max(workflow_results, key=lambda w: w.get('avg_score', 0.0))
        return best_workflow, None
    
    total_problems = sum(category_metadata.values())
    logger.info(f"Dataset: {total_problems} total problems across {len(category_metadata)} categories")
    
    # Step 2: 为每个工作流计算分化潜力
    candidates = []
    
    for workflow in workflow_results:
        round_num = workflow.get('round', 0)
        
        # 2.1 加载该工作流的分类统计
        category_stats = self._load_workflow_category_stats(workflow)
        c_total = sum(category_stats.values())
        
        if c_total == 0:
            logger.debug(f"Round {round_num}: No correct answers, skipping")
            continue
        
        # 2.2 计算全局正确率（基准线）
        acc_global = c_total / total_problems
        
        # 2.3 遍历所有类别，找最大优势
        max_score = -1.0
        best_category = None
        category_analysis = {}
        
        for category, n_k in category_metadata.items():
            if n_k == 0:
                continue
            
            c_k = category_stats.get(category, 0)
            
            # 子问题召回率
            recall_k = c_k / n_k
            
            # 只考虑超过全局水平的子领域（有优势）
            if recall_k > acc_global:
                # 子问题贡献度
                contrib_k = c_k / c_total
                
                # 分化潜力得分 = 贡献度 × 相对优势
                score_k = contrib_k * (recall_k - acc_global)
                
                category_analysis[category] = {
                    'c_k': c_k,
                    'recall': recall_k,
                    'contrib': contrib_k,
                    'advantage': recall_k - acc_global,
                    'score': score_k
                }
                
                if score_k > max_score:
                    max_score = score_k
                    best_category = category
            else:
                # 记录劣势类别（用于调试）
                category_analysis[category] = {
                    'c_k': c_k,
                    'recall': recall_k,
                    'contrib': c_k / c_total if c_total > 0 else 0,
                    'advantage': recall_k - acc_global,
                    'score': 0.0  # 无分化价值
                }
        
        # 2.4 记录该工作流的最终分化潜力
        final_score = max(0.0, max_score)
        
        # 获取已分化次数（用于日志）
        s_i = self.workflow_differentiation_counts.get(round_num, 0)
        
        candidates.append({
            'workflow': workflow,
            'round': round_num,
            'score': workflow.get('avg_score', 0.0),
            'c_total': c_total,
            'acc_global': acc_global,
            'split_potential': final_score,
            'target_category': best_category,
            'category_analysis': category_analysis,
            'differentiation_count': s_i
        })
        
        # 详细日志
        logger.info(f"Round {round_num} (already split {s_i} times):")
        logger.info(f"  Overall: {c_total}/{total_problems} correct, Acc_global={acc_global:.4f}")
        logger.info(f"  Best specialization: {best_category} (potential={final_score:.4f})")
        
        # 显示前3个优势类别
        sorted_cats = sorted(category_analysis.items(), 
                            key=lambda x: x[1]['score'], 
                            reverse=True)[:3]
        for cat, stats in sorted_cats:
            if stats['score'] > 0:
                logger.info(f"    {cat}: {stats['c_k']}/{category_metadata[cat]} correct, "
                           f"Recall={stats['recall']:.4f}, Contrib={stats['contrib']:.4f}, "
                           f"Advantage={stats['advantage']:+.4f}, Score={stats['score']:.4f}")
    
    # Step 3: 检查是否有候选
    if not candidates:
        logger.warning("No valid candidates for differentiation")
        return None, None
    
    # Step 4: 基于分化潜力的 Softmax 采样
    potentials = np.array([c['split_potential'] for c in candidates])
    
    # 如果所有分化潜力都为0，随机选择
    if np.sum(potentials) == 0:
        logger.warning("All workflows have zero split potential, random selection")
        selected_idx = np.random.randint(len(candidates))
    else:
        # Softmax 采样
        potentials = potentials - np.max(potentials)  # 避免溢出
        exp_values = np.exp(potentials)
        probabilities = exp_values / np.sum(exp_values)
        
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        
        logger.info(f"Selection probabilities (top 3):")
        top_indices = np.argsort(probabilities)[::-1][:3]
        for idx in top_indices:
            logger.info(f"  Round {candidates[idx]['round']}: p={probabilities[idx]:.4f}, "
                       f"potential={candidates[idx]['split_potential']:.4f}")
    
    # Step 5: 返回选中的工作流和目标类别
    selected = candidates[selected_idx]
    
    logger.info("=" * 80)
    logger.info(f"SELECTED: Round {selected['round']} for specialization")
    logger.info(f"  Target Category: {selected['target_category']}")
    logger.info(f"  Split Potential: {selected['split_potential']:.4f}")
    logger.info(f"  Global Accuracy: {selected['acc_global']:.4f}")
    if selected['target_category']:
        target_stats = selected['category_analysis'][selected['target_category']]
        logger.info(f"  Target Stats: Recall={target_stats['recall']:.4f}, "
                   f"Contrib={target_stats['contrib']:.4f}, "
                   f"Advantage={target_stats['advantage']:+.4f}")
    logger.info("=" * 80)
    
    return selected['workflow'], selected['target_category']
```

#### 修改点2.4：更新 `_execute_single_differentiation()` 方法

**位置**：约第900-1000行

**当前代码**：
```python
selected_workflow = self._select_for_split(round_summaries)

if selected_workflow is None:
    logger.warning("No suitable workflow selected for differentiation")
    return None

source_round = selected_workflow.get('round', 0)
```

**修改为**：
```python
selected_workflow, target_category = self._select_for_split(round_summaries)

if selected_workflow is None:
    logger.warning("No suitable workflow selected for differentiation")
    return None

source_round = selected_workflow.get('round', 0)
original_score = selected_workflow.get('avg_score', 0.0)

logger.info(f"Selected workflow from round {source_round} for differentiation")
logger.info(f"  Score: {original_score:.4f}")
logger.info(f"  Target specialization category: {target_category if target_category else 'N/A'}")
```

**以及后续传递给分化处理器**：
```python
# 当前（约第940行）
target_category, example_problems = self.problem_classifier.select_target_category_for_differentiation(
    workflow_differentiation_history=self.workflow_manager.get_differentiation_history()
)

# 修改为（使用 _select_for_split 返回的 target_category）
if target_category is None:
    # 如果没有明确的目标类别，使用旧逻辑选择
    target_category, example_problems = self.problem_classifier.select_target_category_for_differentiation(
        workflow_differentiation_history=self.workflow_manager.get_differentiation_history()
    )
else:
    # 使用选定的目标类别获取示例
    example_problems = self.problem_classifier.get_category_examples(target_category, n=5)
```

---

### 📁 文件3：`scripts/problem_classifier.py`

#### 修改点3.1：添加获取类别示例的方法

**位置**：在类的末尾添加

**新增代码**：
```python
def get_category_examples(self, category: str, n: int = 5) -> List[Dict]:
    """
    获取指定类别的示例问题
    
    Args:
        category: 类别名称
        n: 返回的示例数量
        
    Returns:
        List[Dict]: 示例问题列表
    """
    classifications = self.load_classifications()
    
    # 找到属于该类别的所有问题ID
    category_problems = [
        problem_id for problem_id, cat in classifications.items() 
        if cat == category
    ]
    
    if not category_problems:
        logger.warning(f"No problems found in category: {category}")
        return []
    
    # 随机选择n个
    import random
    selected_ids = random.sample(category_problems, min(n, len(category_problems)))
    
    # 从验证集加载完整问题
    validation_file = f"data/datasets/{self.dataset.lower()}_validate.jsonl"
    
    if not os.path.exists(validation_file):
        logger.warning(f"Validation file not found: {validation_file}")
        return []
    
    examples = []
    with open(validation_file, 'r', encoding='utf-8') as f:
        for line in f:
            problem = json.loads(line)
            problem_id = str(problem.get('id') or problem.get('problem_id', ''))
            if problem_id in selected_ids:
                examples.append(problem)
                if len(examples) >= n:
                    break
    
    logger.info(f"Loaded {len(examples)} examples for category: {category}")
    return examples
```

---

## 修改总结

### 文件修改数量
- 3个文件需要修改
- 1个文件新增方法
- 1个文件核心方法重写

### 代码量估计
- 新增代码：~300行
- 修改代码：~50行
- 删除代码：~100行（旧的 `_select_for_split` 实现）

### 关键变更
1. ✅ **返回值变化**：`_select_for_split()` 从返回单个工作流改为返回 `(workflow, category)` 元组
2. ✅ **数据依赖**：需要评估系统记录问题类别标签
3. ✅ **新增方法**：2个辅助方法用于加载分类信息
4. ✅ **核心算法**：从势函数模型改为优势特征提取模型

### 向后兼容性
- ⚠️ **破坏性变更**：`_select_for_split()` 返回值类型改变
- ✅ **渐进式实现**：如果没有分类信息，回退到简单选择
- ✅ **日志完整**：详细记录每个决策过程

### 测试要点
1. 验证类别元数据正确加载
2. 验证工作流分类统计正确计算
3. 验证分化潜力计算公式正确
4. 验证 Softmax 采样正常工作
5. 验证目标类别正确传递给分化处理器

## 风险评估

### 高风险项
1. **数据依赖**：如果早期轮次缺少类别标签会导致回退
   - 缓解：实现回退机制
   
2. **返回值变化**：可能影响调用方
   - 缓解：仔细修改所有调用点

### 中风险项
1. **计算开销**：需要遍历所有类别
   - 缓解：只在分化触发时计算，频率不高

2. **类别不平衡**：某些类别问题很少
   - 缓解：算法自然处理（Recall 会很低）

### 低风险项
1. **日志量增加**：详细日志可能很长
   - 缓解：可以调整日志级别

## 实施建议

### 顺序
1. **第一步**：修改评估系统记录类别标签（foundation）
2. **第二步**：添加辅助方法（helper functions）
3. **第三步**：重写核心方法（core algorithm）
4. **第四步**：更新调用方传递类别（integration）
5. **第五步**：端到端测试（validation）

### 测试策略
1. **单元测试**：测试分化潜力计算公式
2. **集成测试**：测试完整的选择流程
3. **回归测试**：确保不影响其他功能
4. **案例验证**：用文档中的数值案例验证

### 回滚计划
如果新算法有问题：
1. 保留旧的 `_select_for_split` 实现为 `_select_for_split_old()`
2. 添加配置开关在新旧算法间切换
3. 对比新旧算法的选择结果

## 确认问题

在开始实施前，请确认：

1. ✅ 是否同意完全重写 `_select_for_split()` 方法？
2. ✅ 是否同意修改返回值类型为元组？
3. ✅ 是否同意在评估系统中记录类别标签？
4. ✅ 是否需要保留旧实现作为备份？
5. ✅ 是否需要添加配置开关？
6. ✅ 是否需要我提供更详细的某个部分的代码？

请确认以上修改方案，我将开始实施具体的代码修改。
