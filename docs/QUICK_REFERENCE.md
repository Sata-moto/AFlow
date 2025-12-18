# 快速参考 (Quick Reference)

## 文档版本
- **创建日期**: 2025-12-15
- **适用版本**: AFlow Enhanced Optimizer
- **相关文档**: 完整文档集

---

## 1. 核心公式速查

### 1.1 停滞度计算

```python
# 停滞度 (Plateau)
t = len(performance_history)
k = sliding_window_k  # 默认: 3

if t < 2:
    plateau_t = 0.0
else:
    effective_k = min(k, t // 2)
    window = performance_history[-effective_k:]
    
    improvements = [window[i+1] - window[i] for i in range(len(window)-1)]
    avg_improvement = mean(improvements)
    
    plateau_t = (1 - avg_improvement / κ) * 100
    plateau_t = clip(plateau_t, 0, 100)

# κ (kappa): 敏感度, 默认 80.0
```

**直观理解**:
- `plateau_t ≈ 0`: 性能快速提升
- `plateau_t ≈ 50`: 性能缓慢提升
- `plateau_t ≈ 100`: 性能完全停滞

---

### 1.2 操作概率

```python
# 原始概率
p_opt_raw = (1 - α_s - α_m) · plateau_t
p_split_raw = α_s · plateau_t · exp(-η_s · N_s)
p_merge_raw = α_m · plateau_t · exp(-η_m · N_m)

# 归一化
total = p_opt_raw + p_split_raw + p_merge_raw
p_opt = p_opt_raw / total
p_split = p_split_raw / total
p_merge = p_merge_raw / total

# 概率采样
operation = random.choices(
    ['optimize', 'differentiate', 'fuse'],
    weights=[p_opt, p_split, p_merge]
)[0]
```

**参数**:
- `α_s`: 分化基础概率 (默认: 0.5)
- `α_m`: 融合基础概率 (默认: 0.6)
- `η_s`: 分化衰减因子 (默认: 0.03)
- `η_m`: 融合衰减因子 (默认: 0.03)
- `N_s`: 累计分化次数
- `N_m`: 累计融合次数

---

### 1.3 分化潜力

```python
# 全局性能
Acc_global = C_total / N

# 类别性能
for category k:
    Recall_k = C_k / N_k
    Contrib_k = C_k / N  # 绝对贡献度(防止小类别偏向)
    
    # 优势类别判定
    if Recall_k > Acc_global:
        Score_split(k) = Contrib_k × (Recall_k - Acc_global)
    else:
        Score_split(k) = 0

# 工作流分化潜力
Split_Potential(W) = max_k(Score_split(k))

# 选择策略（权衡潜力和准确率）
x = Split_Potential(W) / max(Split_Potential)  # 归一化
Adjusted_Score(W) = α · x + (1-α) · Acc_global
selected_workflow = argmax_W(Adjusted_Score(W))
target_category = argmax_k(Score_split(k) for selected_workflow)

# α: 潜力权重 (默认0.5)
#   - α=1.0: 只看潜力
#   - α=0.5: 平衡
#   - α=0.0: 只看准确率
```

**符号**:
- `C_total`: 总答对题目数
- `N`: 总题目数
- `C_k`: 类别k答对题目数
- `α`: 分化潜力权重 (默认: 0.5)
- `N_k`: 类别k总题目数

---

### 1.4 融合互补性

```python
# Pairwise互补性
C_i = {p: W_i correct on p}
C_j = {p: W_j correct on p}

Φ_pair(i, j) = |C_i ⊕ C_j|
             = |C_i \ C_j| + |C_j \ C_i|

# Triplet互补性
for each problem p:
    n_correct(p) = |{i: W_i correct on p}|

Φ_triple(i,j,k) = |{p: n_correct(p) = 1}| + 
                  |{p: n_correct(p) = 2}|

# 总融合潜力
Φ_merge(i,j,k) = β_p · (Φ_pair(i,j) + Φ_pair(j,k) + Φ_pair(i,k)) +
                 β_t · Φ_triple(i,j,k)

# 选择
best_triple = argmax_{i,j,k}(Φ_merge(i,j,k))
```

**参数**:
- `β_p`: Pairwise权重 (理论: 0.4, 代码: 0.4)
- `β_t`: Triplet权重 (理论: 0.3, 代码: 0.6 ⚠️)

---

## 2. 关键参数表

### 2.1 停滞检测

| 参数 | 符号 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `sliding_window_k` | k | 3 | 2-5 | 滑动窗口大小 |
| `stagnation_sensitivity_kappa` | κ | 80.0 | 50-100 | 停滞敏感度 |

**调参建议**:
- **k=2**: 对短期波动敏感
- **k=3**: 平衡 (推荐)
- **k=5**: 对长期趋势敏感

- **κ=50**: 更容易判定为停滞
- **κ=80**: 平衡 (推荐)
- **κ=100**: 更难判定为停滞

---

### 2.2 操作概率

| 参数 | 符号 | 理论值 | 代码默认 | 范围 | 说明 |
|------|------|--------|----------|------|------|
| `alpha_s` | α_s | 0.3 | 0.5 ⚠️ | 0.1-0.5 | 分化基础概率 |
| `alpha_m` | α_m | 0.4 | 0.6 ⚠️ | 0.1-0.6 | 融合基础概率 |
| `eta_s` | η_s | 0.1 | 0.03 ⚠️ | 0.01-0.2 | 分化衰减因子 |
| `eta_m` | η_m | 0.1 | 0.03 ⚠️ | 0.01-0.2 | 融合衰减因子 |

**调参建议**:

**保守策略** (少分化/融合):
```python
alpha_s = 0.2
alpha_m = 0.3
eta_s = 0.15
eta_m = 0.15
```

**激进策略** (多分化/融合):
```python
alpha_s = 0.5
alpha_m = 0.6
eta_s = 0.02
eta_m = 0.02
```

**平衡策略** (推荐):
```python
alpha_s = 0.3-0.4
alpha_m = 0.4-0.5
eta_s = 0.05-0.1
eta_m = 0.05-0.1
```

---

### 2.3 分化选择

| 参数 | 符号 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `alpha_split_potential` | α | 0.5 | 0.0-1.0 | 分化潜力权重 |

**调参建议**:
- **α=1.0**: 只看分化潜力 (可能选择"偏科"但性能差的)
- **α=0.5**: 平衡潜力和准确率 (推荐)
- **α=0.0**: 只看准确率 (退化为选最佳workflow)

**权衡公式**:
```
Adjusted_Score = α × (Potential/MaxPotential) + (1-α) × Accuracy
```

---

### 2.4 融合选择

| 参数 | 符号 | 理论值 | 代码默认 | 说明 |
|------|------|--------|----------|------|
| `beta_pair` | β_p | 0.4 | 0.4 ✅ | Pairwise权重 |
| `beta_triple` | β_t | 0.3 | 0.6 ⚠️ | Triplet权重 |
| `alpha_U` | α_U | - | 0.6 | 互补性权重 |
| `alpha_I` | α_I | - | 0.4 | 一致性权重 |
| `gamma_pair` | γ_p | - | 0.7 | Pair交集权重 |
| `gamma_triple` | γ_t | - | 0.3 | Triple交集权重 |

**调参建议**:

**强调差异** (差异大的workflows):
```python
beta_pair = 0.5
beta_triple = 0.3
```

**强调协同** (需要投票的场景):
```python
beta_pair = 0.3
beta_triple = 0.5
```

---

### 2.5 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_rounds` | 20 | 最大优化轮数 |
| `validation_rounds` | 5 | 验证轮数 |
| `sample` | -1 | 训练样本数 (-1=全部) |
| `max_envelope_workflows` | 3 | 包络线最大大小 |
| `fusion_score_threshold` | 0.0 | 融合最小分数提升 |
| `alpha_split_potential` | 0.5 | 分化潜力权重 |

---

## 3. 常用命令

### 3.1 运行优化

```bash
# 基本运行
python run.py --dataset MATH --max_rounds 20

# 自定义参数
python run.py \
  --dataset MATH \
  --max_rounds 30 \
  --sliding_window_k 3 \
  --alpha_s 0.3 \
  --alpha_m 0.4 \
  --eta_s 0.1 \
  --eta_m 0.1 \
  --beta_pair 0.4 \
  --beta_triple 0.3 \
  --alpha_split_potential 0.5

# 保守策略
python run.py \
  --dataset MATH \
  --alpha_s 0.2 \
  --alpha_m 0.3 \
  --eta_s 0.15 \
  --eta_m 0.15

# 激进策略
python run.py \
  --dataset MATH \
  --alpha_s 0.5 \
  --alpha_m 0.6 \
  --eta_s 0.02 \
  --eta_m 0.02
```

---

### 3.2 日志查询

```bash
# 查看停滞度
grep "Plateau" logs/AFlow.log

# 查看操作概率
grep "Operation Probabilities" logs/AFlow.log

# 查看选择的操作
grep "Selected operation" logs/AFlow.log

# 查看轮次分数
grep "Score for round" logs/AFlow.log

# 查看分化信息
grep "differentiation" logs/AFlow.log -i

# 查看融合信息
grep "fusion\|merge" logs/AFlow.log -i

# 查看包络线
grep "Envelope" logs/AFlow.log

# 查看互补性
grep "Complementarity\|Φ_merge" logs/AFlow.log
```

---

### 3.3 数据检查

```bash
# 查看results.json
cat workspace/MATH/workflows/results.json | jq '.'

# 查看特定轮次
cat workspace/MATH/workflows/results.json | \
  jq '.[] | select(.round == 5)'

# 查看所有轮次分数
cat workspace/MATH/workflows/results.json | \
  jq '.[] | {round: .round, score: .score, total: .total}'

# 查看log.json
cat workspace/MATH/workflows/round_5/log.json | jq '.'

# 统计类别分布
cat workspace/MATH/workflows/problem_classifications.json | \
  jq '.problem_classifications | group_by(.category) | 
      map({category: .[0].category, count: length})'

# 查看某轮的类别统计
cat workspace/MATH/workflows/round_5/log.json | \
  jq 'group_by(.category) | 
      map({category: .[0].category, 
           correct: [.[] | select(.score >= 0.5)] | length,
           total: length})'
```

---

### 3.4 调试命令

```bash
# 检查Python环境
python --version
pip list | grep -E "pydantic|numpy"

# 检查文件完整性
ls workspace/MATH/workflows/round_*/
find workspace/MATH/workflows/ -name "log.json"
find workspace/MATH/workflows/ -name "graph.py"

# 验证JSON格式
cat workspace/MATH/workflows/results.json | jq empty
cat workspace/MATH/workflows/problem_classifications.json | jq empty

# 清理workspace (谨慎!)
rm -rf workspace/MATH/workflows/round_*
# 保留template和problem_classifications.json

# 重新开始
python run.py --dataset MATH --initial_round 1
```

---

## 4. 典型问题诊断

### 4.1 操作概率全为0

**症状**:
```
Operation Probabilities (Round 4):
  optimize: 0.0000 (0.00%)
  differentiate: 0.0000 (0.00%)  
  fuse: 0.0000 (0.00%)
```

**原因**:
- `plateau_t = 0` (性能还在提升)
- `performance_history` 长度不足

**解决**:
- 继续运行,等待停滞
- 检查 `performance_history` 长度

**验证**:
```bash
grep "Plateau" logs/AFlow.log
grep "performance_history length" logs/AFlow.log
```

---

### 4.2 无优势类别

**症状**:
```
No advantageous specialization found
All categories have Recall <= Acc_global
```

**原因**:
- 所有类别表现都接近平均水平
- 没有明显的优势类别

**解决**:
- 这是正常的,不是bug
- 继续运行,等待分化机会
- 或者降低优势阈值

**验证**:
```bash
# 查看类别统计
cat workspace/MATH/workflows/round_5/log.json | \
  jq 'group_by(.category) | 
      map({category: .[0].category, 
           recall: ([.[] | select(.score >= 0.5)] | length) / length})'
```

---

### 4.3 融合失败

**症状**:
```
Insufficient workflows for fusion: found 2, need 3
```

**原因**:
- workflows数量不足3个
- 包络线太小

**解决**:
- 继续运行,积累更多workflows
- 降低 `max_envelope_workflows`

**验证**:
```bash
# 统计workflows数量
ls -d workspace/MATH/workflows/round_*/ | wc -l

# 查看包络线
grep "Envelope workflows" logs/AFlow.log
```

---

### 4.4 Category显示"unknown"

**症状**:
```json
{
  "problem_id": "problem_0",
  "category": "unknown",
  "score": 1.0
}
```

**原因**:
- `problem_classifications.json` 不存在
- Problem ID不匹配

**解决**:
1. ✅ 确保benchmarks已修改 (添加`_index`字段)
2. 第一次分化时自动生成分类文件
3. 或手动生成:
```bash
python scripts/problem_classifier.py --dataset MATH
```

**验证**:
```bash
# 检查分类文件
cat workspace/MATH/workflows/problem_classifications.json | \
  jq '.categories'

# 检查problem_id格式
cat workspace/MATH/workflows/round_5/log.json | \
  jq '.[0].problem_id'
# 应该是: "problem_0", "problem_1", ...
```

---

## 5. 性能优化技巧

### 5.1 加速评估

```bash
# 使用更少的样本
python run.py --dataset MATH --sample 50

# 并行评估 (如果支持)
python run.py --dataset MATH --parallel_workers 4
```

---

### 5.2 减少LLM调用

```bash
# 降低融合频率
python run.py --alpha_m 0.2 --eta_m 0.15

# 降低分化频率
python run.py --alpha_s 0.2 --eta_s 0.15

# 同时降低
python run.py --alpha_s 0.2 --alpha_m 0.2 --eta_s 0.15 --eta_m 0.15
```

---

### 5.3 缓存优化

```python
# 在enhanced_optimizer.py中添加
@functools.lru_cache(maxsize=128)
def _load_workflow_category_stats_cached(self, workflow_id):
    """缓存版本"""
    return self._load_workflow_category_stats(workflow_id)
```

---

## 6. 常见错误码

### 6.1 文件相关

| 错误信息 | 原因 | 解决方案 |
|---------|------|----------|
| `FileNotFoundError: log.json` | 轮次目录不存在 | 检查round路径 |
| `FileNotFoundError: problem_classifications.json` | 未生成分类文件 | 运行分化或手动生成 |
| `JSONDecodeError` | JSON格式错误 | 检查文件内容 |

---

### 6.2 计算相关

| 错误信息 | 原因 | 解决方案 |
|---------|------|----------|
| `ZeroDivisionError` | 除零错误 | 检查除法保护 |
| `IndexError: list index out of range` | 列表为空 | 检查长度保护 |
| `KeyError: 'category'` | 字段缺失 | 检查数据格式 |

---

### 6.3 逻辑相关

| 错误信息 | 原因 | 解决方案 |
|---------|------|----------|
| `No workflows to optimize` | 初始化失败 | 检查template/ |
| `Operation failed, retrying...` | LLM生成失败 | 检查LLM配置 |
| `Evaluation failed` | 评估错误 | 检查benchmark配置 |

---

## 7. 最佳实践

### 7.1 启动新实验

```bash
# 1. 清理旧数据
rm -rf workspace/MATH/workflows/round_*
rm workspace/MATH/workflows/results.json
rm workspace/MATH/workflows/processed_experience.json

# 保留
# - workspace/MATH/workflows/template/
# - workspace/MATH/workflows/problem_classifications.json (可选)

# 2. 配置参数
python run.py \
  --dataset MATH \
  --max_rounds 30 \
  --alpha_s 0.3 \
  --alpha_m 0.4 \
  --sliding_window_k 3

# 3. 监控日志
tail -f logs/AFlow.log
```

---

### 7.2 恢复中断的实验

```bash
# 查看最后一轮
cat workspace/MATH/workflows/results.json | jq '.[-1]'

# 从最后一轮+1继续
last_round=$(cat workspace/MATH/workflows/results.json | jq '.[-1].round')
next_round=$((last_round + 1))

python run.py \
  --dataset MATH \
  --initial_round $next_round \
  --max_rounds 30
```

---

### 7.3 性能分析

```bash
# 生成分数曲线
cat workspace/MATH/workflows/results.json | \
  jq -r '.[] | "\(.round),\(.score)"' > scores.csv

# 用Python绘图
python -c "
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('scores.csv', names=['round', 'score'])
plt.plot(df['round'], df['score'], marker='o')
plt.xlabel('Round')
plt.ylabel('Score')
plt.title('Optimization Progress')
plt.savefig('progress.png')
"

# 统计操作分布
grep "Selected operation" logs/AFlow.log | \
  awk '{print $NF}' | \
  sort | uniq -c

# 输出示例:
#   15 optimize
#    3 differentiate
#    2 fuse
```

---

## 8. 快速故障排除

### 8.1 5分钟检查清单

```bash
# 1. 检查环境
python --version  # Python 3.8+
pip show pydantic numpy  # 依赖安装

# 2. 检查文件
ls workspace/MATH/workflows/template/
ls workspace/MATH/workflows/round_*/

# 3. 检查日志
tail -100 logs/AFlow.log

# 4. 检查配置
cat config/config.yaml | grep -A5 llm

# 5. 检查数据
cat workspace/MATH/workflows/results.json | jq empty
```

---

### 8.2 常见问题快速修复

**问题**: 所有类别显示"unknown"
```bash
# 修复
python scripts/problem_classifier.py --dataset MATH

# 验证
cat workspace/MATH/workflows/problem_classifications.json | \
  jq '.categories'
```

**问题**: 融合失败 "found 2, need 3"
```bash
# 检查
ls -d workspace/MATH/workflows/round_*/ | wc -l

# 如果 < 3, 继续运行积累workflows
# 如果 >= 3, 检查包络线日志
grep "Envelope" logs/AFlow.log
```

**问题**: 概率全为0
```bash
# 检查停滞度
grep "Plateau" logs/AFlow.log

# 如果 plateau = 0, 继续运行等待停滞
# 如果 plateau > 0, 检查操作计数
grep "N_s\|N_m" logs/AFlow.log
```

---

## 9. 配置模板

### 9.1 快速启动

```yaml
# config/quick_start.yaml
dataset: MATH
max_rounds: 20
sliding_window_k: 3
stagnation_sensitivity_kappa: 80.0

# 平衡策略
alpha_s: 0.3
alpha_m: 0.4
eta_s: 0.1
eta_m: 0.1

beta_pair: 0.4
beta_triple: 0.3
```

```bash
python run.py --config config/quick_start.yaml
```

---

### 9.2 保守策略

```yaml
# config/conservative.yaml
dataset: MATH
max_rounds: 30

# 少分化/融合
alpha_s: 0.2
alpha_m: 0.3
eta_s: 0.15
eta_m: 0.15

# 更严格的停滞判定
stagnation_sensitivity_kappa: 100.0
sliding_window_k: 5
```

---

### 9.3 激进策略

```yaml
# config/aggressive.yaml
dataset: MATH
max_rounds: 30

# 多分化/融合
alpha_s: 0.5
alpha_m: 0.6
eta_s: 0.02
eta_m: 0.02

# 更敏感的停滞判定
stagnation_sensitivity_kappa: 60.0
sliding_window_k: 2
```

---

## 10. 相关文档索引

### 10.1 核心文档

- **[系统架构总览](SYSTEM_ARCHITECTURE.md)**: 完整系统设计
- **[优化操作详解](OPTIMIZE_OPERATION.md)**: OPTIMIZE操作
- **[分化操作详解](DIFFERENTIATION_OPERATION.md)**: DIFFERENTIATE操作
- **[融合操作详解](FUSION_OPERATION.md)**: FUSE操作
- **[代码一致性检查](CODE_CONSISTENCY_CHECK.md)**: 实现验证

---

### 10.2 按需查阅

**想要了解...**

- **整体流程**: [系统架构总览](SYSTEM_ARCHITECTURE.md) § 控制流程
- **停滞检测**: [系统架构总览](SYSTEM_ARCHITECTURE.md) § 停滞检测
- **操作选择**: [系统架构总览](SYSTEM_ARCHITECTURE.md) § 操作选择
- **优化细节**: [优化操作详解](OPTIMIZE_OPERATION.md)
- **分化算法**: [分化操作详解](DIFFERENTIATION_OPERATION.md) § 核心算法
- **融合算法**: [融合操作详解](FUSION_OPERATION.md) § 核心算法
- **参数不一致**: [代码一致性检查](CODE_CONSISTENCY_CHECK.md) § 超参数对照
- **Bug修复**: [代码一致性检查](CODE_CONSISTENCY_CHECK.md) § 潜在Bug

---

### 10.3 快速导航

```
问题诊断路径:
  操作概率为0 → 系统架构 § 停滞检测
  无优势类别 → 分化操作 § 分化潜力计算
  融合失败 → 融合操作 § 包络线选择
  Category unknown → 代码一致性 § 数据结构

算法细节路径:
  停滞度公式 → 快速参考 § 核心公式 → 系统架构 § 停滞检测
  分化公式 → 快速参考 § 核心公式 → 分化操作 § 核心算法
  融合公式 → 快速参考 § 核心公式 → 融合操作 § 核心算法

参数调优路径:
  查看当前值 → 快速参考 § 关键参数表
  理解含义 → 系统架构 / 操作详解
  检查一致性 → 代码一致性 § 超参数对照
  调整建议 → 快速参考 § 常用命令
```

---

## 11. 附录

### 11.1 符号表

| 符号 | 含义 | 取值范围 |
|------|------|----------|
| t | 当前轮次 | 1, 2, 3, ... |
| k | 滑动窗口大小 | 通常 2-5 |
| κ | 停滞敏感度 | 通常 50-100 |
| plateau_t | 停滞度 | 0-100 |
| α_s | 分化基础概率 | 0-1 |
| α_m | 融合基础概率 | 0-1 |
| η_s | 分化衰减因子 | 0-1 |
| η_m | 融合衰减因子 | 0-1 |
| N_s | 累计分化次数 | 0, 1, 2, ... |
| N_m | 累计融合次数 | 0, 1, 2, ... |
| β_p | Pairwise权重 | 0-1 |
| β_t | Triplet权重 | 0-1 |
| Φ_merge | 融合潜力 | >=0 |

---

### 11.2 缩写表

| 缩写 | 全称 | 含义 |
|------|------|------|
| W | Workflow | 工作流 |
| CoT | Chain-of-Thought | 思维链 |
| PoT | Program-of-Thought | 程序思维链 |
| LLM | Large Language Model | 大语言模型 |
| Acc | Accuracy | 准确率 |
| Contrib | Contribution | 贡献度 |
| Φ | Phi | 互补性/潜力 |
| ⊕ | XOR | 对称差 |

---

### 11.3 关键文件路径

```
workspace/DATASET/workflows/
├── template/                          # 初始workflow模板
│   ├── op_prompt.py
│   ├── operator.py
│   ├── operator_an.py
│   └── operator.json
├── round_1/                           # 第1轮 (初始workflow评估)
│   ├── __init__.py
│   ├── graph.py
│   ├── prompt.py
│   └── log.json
├── round_2/                           # 第2轮 (第一次优化)
│   ├── __init__.py
│   ├── graph.py
│   ├── prompt.py
│   └── log.json
├── ...
├── results.json                       # 所有轮次的性能汇总
├── processed_experience.json          # 优化经验
└── problem_classifications.json       # 问题分类 (首次分化时生成)
```

---

## 12. 联系与贡献

### 12.1 报告问题

```bash
# 收集诊断信息
python run.py --dataset MATH --debug > debug.log 2>&1

# 打包相关文件
tar -czf debug_package.tar.gz \
  debug.log \
  logs/AFlow.log \
  workspace/MATH/workflows/results.json \
  workspace/MATH/workflows/round_*/log.json
```

---

### 12.2 文档更新

本文档集包含:
1. **SYSTEM_ARCHITECTURE.md**: 系统架构总览
2. **OPTIMIZE_OPERATION.md**: 优化操作详解
3. **DIFFERENTIATION_OPERATION.md**: 分化操作详解
4. **FUSION_OPERATION.md**: 融合操作详解
5. **CODE_CONSISTENCY_CHECK.md**: 代码一致性检查
6. **QUICK_REFERENCE.md**: 快速参考 (本文档)

最后更新: 2025-12-15

---

**Happy Optimizing! 🚀**
