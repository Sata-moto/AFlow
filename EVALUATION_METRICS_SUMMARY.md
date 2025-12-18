# AFlow 各数据集评估指标总结

## 概览

当前 AFlow 系统对所有数据集都使用了**二元评分**（0.0 或 1.0），以统一"solved problem"的定义。但底层的评估逻辑各不相同。

---

## 数据集评估指标详情

### 1. **DROP** (Reading Comprehension with Discrete Operations)
- **主要方法**: LLM 语义评判
- **回退方法**: F1 Score（基于 token 级别的 precision 和 recall）
- **二元化阈值**: F1 >= 0.5 → 1.0，否则 0.0
- **计算逻辑**:
  ```python
  # F1 Score 计算（回退机制）
  precision = common_tokens / prediction_tokens
  recall = common_tokens / ground_truth_tokens
  f1 = 2 * precision * recall / (precision + recall)
  
  # 二元化
  binary_score = 1.0 if f1 >= 0.5 else 0.0
  ```
- **特点**: 
  - 使用 LLM (gpt-4o-mini) 进行语义判断
  - 支持多个可能的正确答案（用 `|` 分隔）
  - 文本归一化：去除冠词、标点、大小写
- **solved_threshold**: 0.5（默认）

---

### 2. **GSM8K** (Grade School Math)
- **主要方法**: LLM 语义评判
- **评分**: 二元（1.0 或 0.0）
- **计算逻辑**:
  ```python
  # 使用 LLM 判断答案是否数值等价
  score, explanation = await self.llm_judge_answer(
      question=input_text,
      ground_truth=ground_truth,
      prediction=output,
      task_description="solve the math problem and provide the correct numerical answer"
  )
  ```
- **特点**:
  - 专注于数值答案的等价性（"5" = "five" = "5.0"）
  - 忽略推理过程，只看最终答案
  - 使用父类的 `llm_judge_answer` 方法
- **solved_threshold**: 0.5（默认）

---

### 3. **MATH** (Competition Math)
- **主要方法**: LLM 语义评判
- **评分**: 二元（1.0 或 0.0）
- **计算逻辑**:
  ```python
  score, explanation = await self.llm_judge_answer(
      question=input_text,
      ground_truth=ground_truth,
      prediction=output,
      task_description="solve the advanced math problem and provide the correct answer"
  )
  ```
- **特点**:
  - 处理高级数学问题（代数、几何、数论等）
  - 可能包含复杂的数学表达式
  - 使用与 GSM8K 相同的评判机制
- **solved_threshold**: 0.5（默认）

---

### 4. **HotpotQA** (Multi-hop Question Answering)
- **主要方法**: LLM 语义评判
- **评分**: 二元（1.0 或 0.0）
- **计算逻辑**:
  ```python
  score, explanation = await self.llm_judge_answer(
      question=input_text,
      ground_truth=ground_truth,
      prediction=output,
      task_description="answer the multi-hop reasoning question based on the given context"
  )
  ```
- **特点**:
  - 需要多跳推理（跨多个文档）
  - 组合问题和上下文：`Context: {context}\nQuestion: {question}`
  - 评判时考虑语义等价性
- **solved_threshold**: 0.5（默认）

---

### 5. **HumanEval** (Python Code Generation)
- **主要方法**: 单元测试执行（Pass/Fail）
- **评分**: 二元（1.0 或 0.0）
- **计算逻辑**:
  ```python
  # 执行测试用例
  ret = self.check_solution(prediction, data["test"], data["entry_point"])
  
  # 二元评分
  score = 1.0 if ret[0] == self.PASS else 0.0
  ```
- **特点**:
  - **完全基于单元测试**，不是语义评判
  - 所有测试用例通过 → PASS (1.0)
  - 任何测试失败、超时、语法错误 → FAIL (0.0)
  - 执行超时：60秒
  - **这是真正的程序正确性验证，不是相似度**
- **solved_threshold**: 0.5（默认）
- **注意**: 这里的 1.0/0.0 不是相似度，而是功能正确性

---

### 6. **MBPP** (Mostly Basic Python Programming)
- **主要方法**: 单元测试执行（Pass/Fail）
- **评分**: 二元（1.0 或 0.0）
- **计算逻辑**:
  ```python
  # 执行测试用例
  ret = self.check_solution(prediction, data["test"], data["entry_point"])
  
  # 二元评分
  score = 1.0 if ret[0] == self.PASS else 0.0
  ```
- **特点**:
  - 与 HumanEval 完全相同的评估机制
  - 基于单元测试的功能正确性验证
  - 包含 3 个测试用例（通常）
  - 测试执行超时：5秒（每个测试）
- **solved_threshold**: 0.5（默认）

---

## 评分机制对比

### **LLM 语义评判** (DROP, GSM8K, MATH, HotpotQA)
| 数据集 | 底层指标 | 二元化方法 | LLM 模型 |
|--------|---------|-----------|---------|
| DROP | F1 Score (回退) | F1 >= 0.5 → 1.0 | gpt-4o-mini |
| GSM8K | 语义匹配 | 直接二元 | gpt-4o-mini |
| MATH | 语义匹配 | 直接二元 | gpt-4o-mini |
| HotpotQA | 语义匹配 | 直接二元 | gpt-4o-mini |

**评判提示词要点**:
- 数值等价：`"5" = "five" = "5.0"`
- 语义等价：`"Paris" = "paris"`, `"the French" = "French"`
- 忽略格式差异
- 要求返回 JSON: `{"correct": true/false, "explanation": "..."}`
- 如果 LLM 评判失败，回退到字符串匹配（仅 DROP 有 F1 回退）

---

### **单元测试执行** (HumanEval, MBPP)
| 数据集 | 评估方法 | 超时设置 | 返回值 |
|--------|---------|---------|--------|
| HumanEval | 运行测试用例 | 60秒 | PASS (1.0) / FAIL (0.0) |
| MBPP | 运行测试用例 | 5秒/测试 | PASS (1.0) / FAIL (0.0) |

**测试执行流程**:
1. 提取生成的代码
2. 与测试用例组合
3. 在隔离环境中执行
4. 检查是否有异常、超时、assertion 失败
5. 所有测试通过 → PASS，否则 FAIL

---

## 分化选择算法的关键影响

### 当前算法使用的指标
在 `_select_for_split()` 中计算分化潜力时：

```python
# 对于每个工作流
c_total = sum(category_stats.values())  # 总正确数
acc_global = c_total / total_problems   # 全局准确率

# 对于每个类别
c_k = category_stats.get(category, 0)  # 类别 k 的正确数
recall_k = c_k / n_k                    # 类别召回率

# 分化潜力
score_k = contrib_k * (recall_k - acc_global)
```

### 不同评分机制的影响

#### **二元评分的影响**
- ✅ **优点**: 统一了"solved"的定义，便于跨数据集比较
- ✅ **优点**: 避免了 F1 分数的歧义（什么是"部分正确"？）
- ⚠️ **缺点**: 丢失了细粒度信息（DROP 的 F1=0.8 和 F1=0.2 都是 0.0）

#### **LLM 评判的影响**
- ✅ **优点**: 更符合人类判断，处理语义等价
- ⚠️ **不确定性**: LLM 判断可能不稳定（同一答案多次评判结果可能不同）
- ⚠️ **成本**: 每个问题需要调用 LLM（虽然使用 gpt-4o-mini 较便宜）

#### **单元测试的影响**
- ✅ **优点**: 完全客观，可重复
- ✅ **优点**: 验证功能正确性，不是表面相似
- ⚠️ **严格性**: 可能因为小的格式问题（如缺少类型提示）导致 FAIL

---

## 对分化潜力计算的建议

### 当前实现已经合理
当前的分化选择算法使用 **二元评分** 是合理的：

1. **Recall_k** 是类别正确率（0到1之间）
2. **Acc_global** 是全局正确率（0到1之间）
3. **Recall_k - Acc_global** 是优势差（可正可负）

无论底层是 F1、LLM 评判还是单元测试，最终都转换为二元分数，因此算法逻辑一致。

### 潜在改进方向（未来）

如果需要更细粒度的分析：

1. **保留原始 F1 分数**（仅 DROP）:
   - 在 log.json 中额外记录 `f1_score` 字段
   - 分化潜力计算时可以使用 F1 而非二元分数
   - 公式变为：`Score_split(W) = max_k [Contrib_k * (F1_k - F1_global)]`

2. **LLM 置信度**:
   - 要求 LLM 返回置信度：`{"correct": true, "confidence": 0.95}`
   - 分化时优先选择高置信度的优势类别

3. **测试用例通过率**（代码数据集）:
   - 记录通过的测试数量：`3/5 tests passed`
   - 使用通过率作为分数：`score = 0.6`（而非二元 0.0）

---

## 总结

| 数据集 | 主要指标 | 二元化 | 特点 |
|--------|---------|--------|------|
| **DROP** | F1 Score (回退) | ✓ | 语义 + F1 回退 |
| **GSM8K** | LLM 语义 | ✓ | 数值等价 |
| **MATH** | LLM 语义 | ✓ | 高级数学 |
| **HotpotQA** | LLM 语义 | ✓ | 多跳推理 |
| **HumanEval** | 单元测试 | ✓ | 功能正确性 |
| **MBPP** | 单元测试 | ✓ | 功能正确性 |

**核心结论**:
- ✅ 所有数据集都使用二元评分（0.0 或 1.0）
- ✅ 分化选择算法基于二元分数是合理的
- ✅ 不同的底层评估机制（F1/LLM/测试）都适配到统一接口
- ⚠️ 如果需要更细粒度的分析，可以考虑保留原始分数作为额外信息

---

## 代码位置参考

- **BaseBenchmark**: `benchmarks/benchmark.py:20-440`
  - `llm_judge_answer()`: 第193-339行
  - `log_mismatch()`: 第139-188行
  - `save_results_to_csv()`: 第145-167行

- **DROP**: `benchmarks/drop.py:161-176` (F1 计算)
- **GSM8K**: `benchmarks/gsm8k.py:21-46` (LLM 评判)
- **MATH**: `benchmarks/math.py:25-51` (LLM 评判)
- **HotpotQA**: `benchmarks/hotpotqa.py:24-59` (LLM 评判)
- **HumanEval**: `benchmarks/humaneval.py:111-147` (单元测试)
- **MBPP**: `benchmarks/mbpp.py:93-121` (单元测试)
