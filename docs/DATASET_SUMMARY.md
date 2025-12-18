# AFlow 数据集汇总

## 数据集规模统计

| 数据集 | 验证集大小 | 测试集大小 | 总计 | 领域 |
|--------|-----------|-----------|------|------|
| **GSM8K** | 264 | 1,055 | 1,319 | 数学推理 |
| **MATH** | 119 | 486 | 605 | 高级数学 |
| **HumanEval** | 33 | 131 | 164 | 代码生成 |
| **MBPP** | 86 | 341 | 427 | 代码生成 |
| **DROP** | 200 | 800 | 1,000 | 阅读理解 |
| **HotpotQA** | 200 | 800 | 1,000 | 多跳推理 |
| **Mixed** | 87 | 114 | 201 | 混合任务 |

### 数据文件位置
```
data/datasets/
├── gsm8k_validate.jsonl      (264行)
├── gsm8k_test.jsonl           (1055行)
├── math_validate.jsonl        (119行)
├── math_test.jsonl            (486行)
├── humaneval_validate.jsonl   (33行)
├── humaneval_test.jsonl       (131行)
├── humaneval_public_test.jsonl (159行)
├── mbpp_validate.jsonl        (86行)
├── mbpp_test.jsonl            (341行)
├── mbpp_public_test.jsonl     (427行)
├── drop_validate.jsonl        (200行)
├── drop_test.jsonl            (800行)
├── hotpotqa_validate.jsonl    (200行)
├── hotpotqa_test.jsonl        (800行)
├── mixed_validate.jsonl       (87行)
└── mixed_test.jsonl           (114行)
```

---

## 各数据集详细信息

### 1. GSM8K (Grade School Math 8K)

**领域**: 小学数学应用题

**验证集**: 264题  
**测试集**: 1,055题

**数据格式**:
```json
{
  "question": "Natalia sold clips to 48 of her friends in April...",
  "answer": "106"
}
```

**评估方式**:
- **LLM评判**: 使用GPT-4o-mini进行语义评判
- **评判提示**: "solve the math problem and provide the correct numerical answer"
- **特点**: 
  - 不依赖精确匹配
  - 允许不同的表达方式（如"106"、"106个"、"一百零六"）
  - 评判模型理解数学语义

**问题ID格式**: `problem_{idx}` (从0开始)

**类别**: 支持问题分类（通过`problem_classifications.json`）

---

### 2. MATH (Competition Math)

**领域**: 高中/竞赛级数学

**验证集**: 119题  
**测试集**: 486题

**数据格式**:
```json
{
  "problem": "Solve for x: 3x + 7 = 22",
  "solution": "x = 5",
  "level": "Level 2",
  "type": "Algebra"
}
```

**评估方式**:
- **LLM评判**: 使用GPT-4o-mini进行语义评判
- **评判提示**: "solve the advanced math problem and provide the correct answer"
- **特点**:
  - 支持复杂数学表达式
  - 理解多种解题步骤
  - 容忍等价的数学表达（如`1/2`、`0.5`、`50%`）

**问题ID格式**: `problem_{idx}` (从0开始)

**难度级别**: Level 1-5

**数学类型**: Algebra, Geometry, Number Theory, Counting & Probability等

---

### 3. HumanEval

**领域**: Python代码生成

**验证集**: 33题  
**测试集**: 131题  
**公开测试集**: 159题

**数据格式**:
```json
{
  "task_id": "HumanEval/0",
  "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list...",
  "canonical_solution": "    for idx, elem in enumerate(numbers):\n        ...",
  "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    ...",
  "entry_point": "has_close_elements"
}
```

**评估方式**:
- **执行测试**: 运行测试用例验证代码正确性
- **超时控制**: 15秒超时限制
- **特殊处理**: 
  - `decode_cyclic`: 需要预先定义`encode_cyclic`
  - `decode_shift`: 需要预先定义`encode_shift`
  - `find_zero`: 需要预先定义`poly`
- **沙箱环境**: 
  - 允许导入: `math`, `hashlib`, `re`
  - 类型注解: `List`, `Dict`, `Tuple`, `Optional`, `Any`

**问题ID格式**: `task_id` (如`HumanEval/0`)

**评分**: Pass/Fail (0.0或1.0)

---

### 4. MBPP (Mostly Basic Programming Problems)

**领域**: Python基础编程

**验证集**: 86题  
**测试集**: 341题  
**公开测试集**: 427题

**数据格式**:
```json
{
  "task_id": 1,
  "text": "Write a function to find the minimum cost path...",
  "code": "def min_cost(cost, m, n): ...",
  "test_list": [
    "assert min_cost([[1, 2, 3], [4, 8, 2]], 2, 2) == 8",
    "assert min_cost([[1, 2, 3], [4, 8, 2]], 2, 1) == 4"
  ],
  "test_setup_code": "",
  "challenge_test_list": []
}
```

**评估方式**:
- **执行测试**: 运行`test_list`中的所有断言
- **超时控制**: 15秒超时限制
- **代码清洗**: 使用`sanitize`函数清理生成的代码
- **沙箱环境**: 与HumanEval相同

**问题ID格式**: `task_id` (数字ID)

**评分**: Pass/Fail (0.0或1.0)

**难度**: 以基础编程问题为主

---

### 5. DROP (Discrete Reasoning Over Paragraphs)

**领域**: 阅读理解（需要数值推理）

**验证集**: 200题  
**测试集**: 800题

**数据格式**:
```json
{
  "_id": "5a7613c15542994ccc9186bf",
  "passage": "The Redskins began their 2008 campaign...",
  "question": "How many points did the Redskins score in the first half?",
  "answer": "14"
}
```

**评估方式**:
- **LLM评判**: 使用GPT-4o-mini进行语义评判
- **评判提示**: "answer the reading comprehension question based on the given passage, focusing on numerical reasoning"
- **特殊处理**:
  - 多答案支持: 答案可能用`|`分隔（如`"14|fourteen"`）
  - 答案归一化: 去除冠词、标点、统一大小写
  - 上下文清洗: 去除敏感内容（如暴力、战争相关词汇）
- **特点**:
  - 需要从段落中提取信息
  - 经常涉及数值计算（加减、比较）
  - 答案通常是数字或短语

**问题ID格式**: `problem_{idx}` (从0开始，因为原始`_id`为MongoDB ID)

**敏感内容过滤**: 
```python
SENSITIVE_WORDS = [
    'holocaust', 'massacre', 'killed', 'extermination', 'murder', 
    'genocide', 'slaughter', 'execution', 'torture', 'violence',
    'death toll', 'casualties', 'ethnic cleansing', 'war crime',
    'atrocity', 'brutality', 'killing', 'mass murder', 'pogrom',
    'persecution', 'nazi', 'hitler', ...
]
```
- 在优化时，包含这些词的样本会被替换为`[REDACTED]`

---

### 6. HotpotQA

**领域**: 多跳问答推理

**验证集**: 200题  
**测试集**: 800题

**数据格式**:
```json
{
  "_id": "5a7ce444554299683c1c63c1",
  "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
  "answer": "Chief of Protocol",
  "context": [
    ["Kiss and Tell (1945 film)", ["Kiss and Tell is a 1945 American comedy film...", "..."]],
    ["Shirley Temple", ["Shirley Temple Black (April 23, 1928 – February 10, 2014) was...", "..."]]
  ],
  "supporting_facts": [["Kiss and Tell (1945 film)", 0], ["Shirley Temple", 0]],
  "type": "bridge"
}
```

**评估方式**:
- **LLM评判**: 使用GPT-4o-mini进行语义评判
- **评判提示**: "answer the multi-hop reasoning question based on the given context"
- **特点**:
  - 需要跨越多个文档进行推理
  - 理解隐含的逻辑链接
  - 支持多种推理类型（bridge, comparison等）

**问题ID格式**: `problem_{idx}` (从0开始，强制使用`_index`字段)

**推理类型**:
- **bridge**: 桥接推理（A→B→C）
- **comparison**: 比较推理（比较两个实体）

**上下文格式**: 
```python
input_text = f"Context: {context}\nQuestion: {question}"
```

---

### 7. Mixed

**领域**: 混合任务集

**验证集**: 87题  
**测试集**: 114题

**内容**: 包含上述数据集的混合样本，用于测试模型的通用性。

---

## 评估方式对比

| 数据集 | 评估方法 | 评分方式 | 关键特性 |
|--------|---------|---------|---------|
| **GSM8K** | LLM评判 | 0.0-1.0 (语义) | 数值答案，容忍格式差异 |
| **MATH** | LLM评判 | 0.0-1.0 (语义) | 复杂表达式，等价性判断 |
| **HumanEval** | 执行测试 | 0.0/1.0 (Pass/Fail) | 代码执行，超时控制 |
| **MBPP** | 执行测试 | 0.0/1.0 (Pass/Fail) | 断言测试，基础编程 |
| **DROP** | LLM评判 | 0.0-1.0 (语义) | 多答案支持，归一化 |
| **HotpotQA** | LLM评判 | 0.0-1.0 (语义) | 多跳推理，上下文理解 |

### LLM评判 vs 执行测试

**LLM评判** (GSM8K, MATH, DROP, HotpotQA):
- ✅ 容忍表达方式差异
- ✅ 理解语义等价
- ✅ 支持部分正确
- ⚠️ 依赖评判模型质量
- ⚠️ 评判成本较高

**执行测试** (HumanEval, MBPP):
- ✅ 绝对客观准确
- ✅ 快速高效
- ✅ 无额外成本
- ⚠️ 只能判断Pass/Fail
- ⚠️ 不支持部分正确

---

## 问题分类支持

### 支持问题分类的数据集

以下数据集支持问题分类（用于分化操作）：

1. **GSM8K**: `workspace/GSM8K/workflows/problem_classifications.json`
2. **MATH**: `workspace/MATH/workflows/problem_classifications.json`
3. **DROP**: `workspace/DROP/workflows/problem_classifications.json`
4. **HotpotQA**: `workspace/HotpotQA/workflows/problem_classifications.json`
5. **HumanEval**: `workspace/HumanEval/workflows/problem_classifications.json`
6. **MBPP**: `workspace/MBPP/workflows/problem_classifications.json`

### 分类文件格式

```json
{
  "problem_classifications": [
    {
      "problem_id": "problem_0",
      "category": "Arithmetic Operations"
    },
    {
      "problem_id": "problem_1",
      "category": "Word Problems"
    }
  ]
}
```

### 问题ID匹配规则

| 数据集 | 原始ID字段 | 标准化ID格式 | 说明 |
|--------|-----------|-------------|------|
| GSM8K | - | `problem_{idx}` | 使用`_index`生成 |
| MATH | - | `problem_{idx}` | 使用`_index`生成 |
| DROP | `_id` (MongoDB) | `problem_{idx}` | 强制使用`_index` |
| HotpotQA | `_id` (MongoDB) | `problem_{idx}` | 强制使用`_index` |
| HumanEval | `task_id` | `task_id` | 保持原样 |
| MBPP | `task_id` | `task_id` | 保持原样 |

**注意**: DROP和HotpotQA的原始数据使用MongoDB的`_id`字段（如`5a7613c15542994ccc9186bf`），但为了与分类文件匹配，在评估时强制使用`problem_{idx}`格式。

---

## 性能基准

### 验证集规模合理性

| 数据集 | 验证集比例 | 评估时间估计 | 是否合理 |
|--------|-----------|-------------|---------|
| GSM8K | 20% | ~5分钟 | ✅ 合理 |
| MATH | 20% | ~3分钟 | ✅ 合理 |
| HumanEval | 20% | ~1分钟 | ✅ 合理 |
| MBPP | 20% | ~2分钟 | ✅ 合理 |
| DROP | 20% | ~4分钟 | ✅ 合理 |
| HotpotQA | 20% | ~4分钟 | ✅ 合理 |

### 测试集规模

| 数据集 | 测试集大小 | 评估时间估计 | 适用场景 |
|--------|-----------|-------------|---------|
| GSM8K | 1,055 | ~20分钟 | 最终评估 |
| MATH | 486 | ~12分钟 | 最终评估 |
| HumanEval | 131 | ~3分钟 | 最终评估 |
| MBPP | 341 | ~8分钟 | 最终评估 |
| DROP | 800 | ~16分钟 | 最终评估 |
| HotpotQA | 800 | ~16分钟 | 最终评估 |

**注意**: 评估时间取决于并发数（`max_concurrent_tasks`）和LLM响应速度。

---

## 数据集选择建议

### 根据任务类型

- **数学推理**: GSM8K (基础), MATH (高级)
- **代码生成**: HumanEval (函数补全), MBPP (从头编写)
- **阅读理解**: DROP (数值推理), HotpotQA (多跳推理)
- **通用能力**: Mixed

### 根据优化目标

- **快速迭代**: 使用验证集较小的数据集 (HumanEval, MBPP, MATH)
- **稳定评估**: 使用验证集较大的数据集 (GSM8K, DROP, HotpotQA)
- **分化操作**: 确保有问题分类文件的数据集

### 根据计算资源

- **资源有限**: HumanEval (33), MBPP (86), MATH (119)
- **资源充足**: GSM8K (264), DROP (200), HotpotQA (200)

---

## 常见问题

### Q1: 为什么DROP和HotpotQA使用`problem_{idx}`而不是原始`_id`?

**原因**: 
- 原始数据使用MongoDB的`_id`字段（长字符串）
- 问题分类文件使用`problem_0`, `problem_1`等简洁格式
- 为了匹配分类文件，在评估时强制转换为`problem_{idx}`

**实现**:
```python
# HotpotQA/DROP benchmark代码
if "_index" in problem:
    problem_id = f"problem_{problem['_index']}"
elif "id" in problem:
    problem_id = problem["id"]
elif "_id" in problem:
    problem_id = problem["_id"]
```

### Q2: 为什么代码数据集（HumanEval/MBPP）不使用LLM评判?

**原因**:
- 代码正确性是绝对的（Pass/Fail）
- 执行测试比LLM评判更准确、更快
- 避免LLM幻觉导致的误判

### Q3: 如何处理DROP数据集的敏感内容?

**方案**:
1. **检测**: 检查question/context中是否包含敏感词
2. **过滤**: 在优化时跳过包含敏感词的样本
3. **清洗**: 将敏感词替换为`[REDACTED]`
4. **降级**: 如果过滤后样本不足，使用所有样本

**代码位置**: `scripts/optimizer_utils/data_utils.py` - `sanitize_text()`

### Q4: LLM评判的成本如何?

**估算** (基于GPT-4o-mini):
- **输入**: ~500 tokens (问题+答案+评判提示)
- **输出**: ~100 tokens (分数+解释)
- **成本**: ~$0.0001 per evaluation
- **GSM8K验证集**: 264题 × $0.0001 = ~$0.03
- **GSM8K测试集**: 1055题 × $0.0001 = ~$0.11

**优化建议**:
- 使用更便宜的模型（如GPT-3.5-turbo）
- 减少评判提示的长度
- 批量评估以提高效率

---

## 更新日志

### 2025-12-17
- 添加问题ID标准化说明
- 添加DROP/HotpotQA的ID匹配规则
- 添加敏感内容过滤说明
- 添加评估成本估算

### 2025-12-16
- 初始版本
- 统计所有数据集规模
- 整理评估方式

---

## 参考链接

- **GSM8K**: https://github.com/openai/grade-school-math
- **MATH**: https://github.com/hendrycks/math
- **HumanEval**: https://github.com/openai/human-eval
- **MBPP**: https://github.com/google-research/google-research/tree/master/mbpp
- **DROP**: https://allenai.org/data/drop
- **HotpotQA**: https://hotpotqa.github.io/
