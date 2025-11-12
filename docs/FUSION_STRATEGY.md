# 融合策略详解 (Fusion Strategy)

## 1. 策略概述

融合策略识别并合并具有互补优势的多个工作流，创建能够综合不同专长的高性能工作流。

### 核心思想
```
专家工作流A + 专家工作流B + ... → LLM融合 → 综合工作流
```

### 类比
```
团队协作：数学专家 + 编程专家 + 推理专家 → 全能团队
```

## 2. 包络工作流 (Envelope Workflows)

### 2.1 概念定义

包络工作流是一组在不同问题子集上各有专长的工作流集合。

```
可视化示例：

问题空间分布：
  ┌─────────────────────────────┐
  │  ●  ●●      ●●    ● ●       │  ● = 问题点
  │   ●   ●● ●    ●●  ●  ●      │
  │ ●●      ●   ●●   ●●   ●     │
  └─────────────────────────────┘

Workflow A 的优势区域：
  ┌─────────────────────────────┐
  │ [✓✓✓]              │
  │ [✓✓✓]              │  擅长左上区域
  │ [✓✓✓]              │
  └─────────────────────────────┘

Workflow B 的优势区域：
  ┌─────────────────────────────┐
  │             [✓✓✓] │
  │             [✓✓✓] │  擅长右上区域
  │             [✓✓✓] │
  └─────────────────────────────┘

Workflow C 的优势区域：
  ┌─────────────────────────────┐
  │                             │
  │       [✓✓✓✓✓]│  擅长中下区域
  │       [✓✓✓✓✓]│
  └─────────────────────────────┘

结论：A、B、C 形成包络（各有专长，互补）
```

### 2.2 检测算法

```python
def find_envelope_workflows(max_count=3):
    """
    查找包络工作流
    
    算法：
    1. 计算每个工作流在每个问题上的表现
    2. 贪心选择：每次选择在未覆盖问题上表现最好的工作流
    3. 重复直到达到数量限制或覆盖率不再提升
    
    Returns:
        List[Workflow]: 包络工作流列表
    """
    workflows = get_all_workflows()
    envelope = []
    covered_problems = set()
    
    for _ in range(max_count):
        best_workflow = None
        best_new_coverage = 0
        
        for workflow in workflows:
            if workflow in envelope:
                continue
            
            # 计算该工作流能额外覆盖多少新问题
            new_problems = workflow.solved_problems - covered_problems
            
            if len(new_problems) > best_new_coverage:
                best_new_coverage = len(new_problems)
                best_workflow = workflow
        
        if best_workflow and best_new_coverage > 0:
            envelope.append(best_workflow)
            covered_problems.update(best_workflow.solved_problems)
        else:
            break  # 无法再增加覆盖
    
    # 验证是否真的互补
    if len(envelope) >= 2:
        # 计算重叠度
        overlap_ratio = calculate_overlap(envelope)
        if overlap_ratio > 0.8:  # 80%以上重叠，不算包络
            return []
    
    return envelope
```

## 3. 完整融合流程

```python
async def fuse_workflows(round):
    """
    工作流融合的完整流程
    
    Returns:
        score: 融合后工作流的评估分数，失败返回None
    """
    
    # ====== 第1步：触发条件检查 ======
    
    # 检查1：是否启用融合
    if not enable_fusion:
        return None
    
    # 检查2：是否达到起始轮次
    if round < fusion_start_round:
        print(f"未达到融合起始轮次 (需要{fusion_start_round}, 当前{round})")
        return None
    
    # 检查3：距离上次融合是否足够久
    if last_fusion_round > 0 and (round - last_fusion_round) < fusion_interval_rounds:
        print(f"距离上次融合太近 (需间隔{fusion_interval_rounds}轮)")
        return None
    
    # ====== 第2步：查找包络工作流 ======
    envelope_workflows = find_envelope_workflows(max_count=max_envelope_workflows)
    
    if len(envelope_workflows) < 2:
        print(f"包络工作流不足 (需要至少2个，找到{len(envelope_workflows)})")
        return None
    
    print(f"找到 {len(envelope_workflows)} 个包络工作流:")
    for wf in envelope_workflows:
        print(f"  - Round {wf.round}: 分数={wf.score:.4f}, "
              f"解决{len(wf.solved_problems)}个问题")
    
    # 计算融合前的基准分数
    min_score = min(wf.score for wf in envelope_workflows)
    avg_score = mean(wf.score for wf in envelope_workflows)
    
    print(f"融合前基准: 最低={min_score:.4f}, 平均={avg_score:.4f}")
    
    # ====== 第3步：检查是否已尝试过这个组合 ======
    fusion_signature = tuple(sorted(wf.round for wf in envelope_workflows))
    
    if check_fusion_attempted(fusion_signature):
        print(f"该融合组合已尝试过: {fusion_signature}")
        return None
    
    # ====== 第4步：加载工作流内容 ======
    workflow_contents = []
    for wf in envelope_workflows:
        content = {
            "round": wf.round,
            "score": wf.score,
            "solved_problems": list(wf.solved_problems),
            "graph": load_graph_code(wf.round),
            "prompt": load_prompt_code(wf.round)
        }
        workflow_contents.append(content)
    
    # ====== 第5步：构建融合提示词 ======
    fusion_prompt = f"""
你是一个工作流融合专家。请将以下 {len(envelope_workflows)} 个具有互补优势的工作流融合成一个综合工作流。

### 融合目标
创建一个能够综合各工作流优势的新工作流，在所有问题类型上都表现良好。

### 待融合的工作流

{format_workflows_for_fusion(workflow_contents)}

### 各工作流的专长分析

{analyze_workflow_specialties(envelope_workflows)}

### 融合策略指导

1. **识别各工作流的核心优势**
   - 分析每个工作流在哪些类型问题上表现最好
   - 提取其成功的关键设计模式

2. **设计融合架构**
   - 不是简单拼接，而是有机整合
   - 可以采用条件分支、策略选择等机制
   - 确保不同场景下启用合适的策略

3. **保持各工作流的核心能力**
   - 不要为了融合而丢失原有优势
   - 重点是"集大成"而非"折中"

4. **示例融合模式**
   
   模式1：条件路由
   ```python
   if problem_type == "mathematical":
       use_workflow_A_approach()
   elif problem_type == "code":
       use_workflow_B_approach()
   else:
       use_workflow_C_approach()
   ```
   
   模式2：并行评估
   ```python
   results = []
   results.append(workflow_A_solve(problem))
   results.append(workflow_B_solve(problem))
   final_answer = vote_or_merge(results)
   ```
   
   模式3：流水线整合
   ```python
   step1_result = workflow_A_preprocess(problem)
   step2_result = workflow_B_solve(step1_result)
   final_answer = workflow_C_postprocess(step2_result)
   ```

### 可用操作符
{operator_descriptions}

### 返回要求
1. modification: 详细的融合说明（如何整合各工作流优势）
2. graph: 融合后的图结构代码
3. prompt: 融合后的提示词代码
"""
    
    # ====== 第6步：调用 LLM 生成融合方案 ======
    response = await optimize_llm.call_with_format(
        fusion_prompt,
        format_schema=GraphOptimize
    )
    
    # ====== 第7步：保存融合工作流 ======
    new_round = round + 1
    new_directory = f"workspace/{dataset}/workflows/round_{new_round}"
    create_directory(new_directory)
    
    # 保存代码文件
    save_file(f"{new_directory}/graph.py", response['graph'])
    save_file(f"{new_directory}/prompt.py", response['prompt'])
    save_file(f"{new_directory}/__init__.py", "")
    
    # 保存经验记录（标记为融合类型）
    experience = {
        "father": [wf.round for wf in envelope_workflows],  # 多个父节点
        "before": avg_score,  # 融合前的平均分数
        "modification": f"[融合] {response['modification']}",
        "after": None,
        "operation": {
            "type": "fusion",
            "source_rounds": [wf.round for wf in envelope_workflows],
            "source_scores": [wf.score for wf in envelope_workflows]
        }
    }
    save_json(f"{new_directory}/experience.json", experience)
    
    # 保存融合元数据
    fusion_metadata = {
        "source_rounds": [wf.round for wf in envelope_workflows],
        "target_round": new_round,
        "fusion_signature": fusion_signature,
        "pre_fusion_scores": {wf.round: wf.score for wf in envelope_workflows},
        "timestamp": datetime.now().isoformat()
    }
    metadata_id = get_next_fusion_id()
    save_json(
        f"workspace/{dataset}/workflows/fusion_metadata_{metadata_id}.json",
        fusion_metadata
    )
    
    # ====== 第8步：评估融合工作流 ======
    new_graph = dynamic_import(f"{new_directory}/graph.py")
    
    result = await evaluate_workflow(new_graph, validation_data, rounds=5)
    fusion_score = result.avg_score
    
    print(f"融合工作流评估完成:")
    print(f"  融合分数: {fusion_score:.4f}")
    print(f"  vs 最低分: {fusion_score - min_score:+.4f}")
    print(f"  vs 平均分: {fusion_score - avg_score:+.4f}")
    
    # ====== 第9步：判断融合是否成功 ======
    threshold = min_score + fusion_score_threshold
    
    if fusion_score > threshold:
        print(f"✓ 融合成功！分数 {fusion_score:.4f} > 阈值 {threshold:.4f}")
        
        # 更新上次融合轮次
        last_fusion_round = round
        
        # 更新经验文件
        experience['after'] = fusion_score
        save_json(f"{new_directory}/experience.json", experience)
        
        # 标记为已采用
        fusion_metadata['adopted'] = True
        fusion_metadata['fusion_score'] = fusion_score
        save_json(
            f"workspace/{dataset}/workflows/fusion_metadata_{metadata_id}.json",
            fusion_metadata
        )
        
        return fusion_score
    else:
        print(f"✗ 融合未达到阈值 {fusion_score:.4f} <= {threshold:.4f}")
        
        # 虽然未达阈值，但仍然保留融合工作流
        # 因为它可能在后续优化中表现更好
        experience['after'] = fusion_score
        save_json(f"{new_directory}/experience.json", experience)
        
        fusion_metadata['adopted'] = False
        fusion_metadata['fusion_score'] = fusion_score
        save_json(
            f"workspace/{dataset}/workflows/fusion_metadata_{metadata_id}.json",
            fusion_metadata
        )
        
        return fusion_score
```

## 4. 关键组件详解

### 4.1 专长分析

```python
def analyze_workflow_specialties(workflows):
    """
    分析每个工作流的专长领域
    """
    analysis = []
    
    for wf in workflows:
        # 获取该工作流解决的问题
        solved = wf.solved_problems
        
        # 分类统计
        category_performance = defaultdict(lambda: {"solved": 0, "total": 0})
        
        for problem in all_problems:
            category = get_problem_category(problem.id)
            category_performance[category]["total"] += 1
            
            if problem.id in solved:
                category_performance[category]["solved"] += 1
        
        # 计算各类别准确率
        specialties = {}
        for category, stats in category_performance.items():
            if stats["total"] > 0:
                accuracy = stats["solved"] / stats["total"]
                specialties[category] = accuracy
        
        # 找出该工作流最擅长的类别
        best_categories = sorted(
            specialties.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:2]
        
        analysis.append({
            "round": wf.round,
            "overall_score": wf.score,
            "specialties": best_categories
        })
    
    # 格式化输出
    formatted = []
    for info in analysis:
        formatted.append(f"""
Workflow Round {info['round']}:
  整体准确率: {info['overall_score']:.2%}
  最擅长: {info['specialties'][0][0]} ({info['specialties'][0][1]:.2%})
  次擅长: {info['specialties'][1][0]} ({info['specialties'][1][1]:.2%})
""")
    
    return "\n".join(formatted)
```

### 4.2 重叠度计算

```python
def calculate_overlap(workflows):
    """
    计算工作流之间的重叠度
    
    Returns:
        float: 重叠度 [0, 1]，0=完全不重叠，1=完全重叠
    """
    if len(workflows) < 2:
        return 0.0
    
    # 计算所有工作流的并集和交集
    all_solved = set()
    common_solved = set(workflows[0].solved_problems)
    
    for wf in workflows:
        all_solved.update(wf.solved_problems)
        common_solved.intersection_update(wf.solved_problems)
    
    # 重叠度 = 交集 / 并集
    overlap = len(common_solved) / len(all_solved) if all_solved else 0
    
    return overlap
```

## 5. 融合策略的优势与局限

### 优势
✓ **突破性提升**：整合优势，可能实现大幅性能跃升  
✓ **综合能力**：在多个问题类型上都表现良好  
✓ **资源高效**：一个工作流替代多个专家工作流  
✓ **创新设计**：LLM可能创造出新的架构模式

### 局限
✗ **触发稀少**：需要多个互补工作流，机会较少  
✗ **复杂度高**：融合后的工作流可能过于复杂  
✗ **不确定性**：融合效果难以预测  
✗ **成本较高**：需要评估多个源工作流

## 6. 与其他策略的协同

### 优化 → 融合
```
Round 1-4: 优化创建多个好工作流
Round 5: 检测到包络，执行融合 → 性能跃升
```

### 分化 → 融合
```
Round 3: 分化 → 数学专家
Round 5: 分化 → 代码专家
Round 7: 融合 → 综合专家（数学+代码）
```

### 融合 → 优化
```
Round 7: 融合创建综合工作流
Round 8-10: 优化微调融合工作流
```

### 融合 → 分化
```
Round 7: 融合创建综合工作流
Round 9: 基于融合工作流再次分化
```

## 7. 实际效果示例

```
融合前的三个工作流:
  Workflow A (Round 5): 72% (擅长数学)
  Workflow B (Round 8): 70% (擅长代码)
  Workflow C (Round 9): 68% (擅长推理)
  
  平均分: 70%
  
↓ 融合

融合工作流 (Round 10):
  总体准确率: 81% (+11%)
  
  各类别表现:
  - 数学: 85% (继承A的优势)
  - 代码: 82% (继承B的优势)
  - 推理: 78% (继承C的优势)

价值：
1. 总体性能大幅提升 (+11%)
2. 在所有类别上都保持了高性能
3. 一个工作流替代三个专家工作流
```

## 8. 调优建议

| 参数 | 太低 | 推荐 | 太高 |
|------|------|------|------|
| 起始轮次 | <3 工作流不够成熟 | 5-7 | >10 错失机会 |
| 融合间隔 | <2 过于频繁 | 2-3 | >5 机会太少 |
| 包络数量 | 2 简单但效果有限 | 3 | >4 过于复杂 |
| 分数阈值 | 0 无选择性 | 0.02-0.05 | >0.1 要求过高 |

---

**核心价值**：融合策略通过整合互补优势，实现突破性的性能提升，是三种策略中潜在收益最大的。
