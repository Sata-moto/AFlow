# AFlow-Enhanced 工作流融合说明

本说明文档针对你在 AFlow 基础上实现的“工作流融合（Workflow Fusion）”版本，概述执行流程、关键函数、数据落盘与使用方式。该实现将融合过程直接嵌入优化循环（EnhancedOptimizer），并对原始独立融合脚本进行精简，仅保留必要组件。

## 目录
- 背景与目标
- 运行入口
- 执行流程（按调用顺序）
- 产物与目录结构
- 融合策略与阈值
- 与原版差异与精简
- 常见问题

---

## 背景与目标

- 通过“包络工作流”（最大化覆盖不同题目的若干高质量轮次），利用 LLM 生成一个更强的融合工作流；
- 将融合工作流直接作为下一轮 round_{n+1} 的标准产物，使用统一的评估与经验更新路径；
- 防止重复融合同一来源组合，记录融合元数据，支持可重复追踪。

## 运行入口

- 推荐使用入口：`run_enhanced.py`
  - 关键参数：
    - `--enable_fusion` 是否启用融合（默认 True）
    - `--max_envelope_workflows` 包络集最大工作流数（默认 3）
    - `--fusion_score_threshold` 融合得分需超过阈值（基于包络中最低分）（默认 0.0）

执行示例（可在 README 中的 Quick Start 基础上替换入口脚本）：

```bash
python run_enhanced.py --dataset MATH --enable_fusion true --max_envelope_workflows 3 --fusion_score_threshold 0.0
```

> 注意：`config/config2.yaml` 中需配置好使用的 LLM 模型与鉴权。

## 执行流程（按调用顺序）

以下为一次完整优化迭代内，融合相关路径的函数调用顺序与职责。文件路径见括号。

1) EnhancedOptimizer.optimize("Graph")（`scripts/enhanced_optimizer.py`）
   - 每轮开始时判断是否触发融合（`_should_attempt_fusion`）。

2) EnhancedOptimizer._should_attempt_fusion（`enhanced_optimizer.py`）
   - 条件：开启融合；当前轮次 ≥ 2；上一轮不是刚融合；可找到 `max_envelope_workflows` 个包络工作流；且该组合未融合过（`_check_fusion_already_attempted`）。

3) EnhancedOptimizer._attempt_fusion（`enhanced_optimizer.py`）
   - 记录 `last_fusion_round`；
   - 计算包络集最低分，用于阈值判断；
   - 调用 `_execute_fusion_async` 生成并保存融合候选为下一轮目录；
   - 按标准流程评估 `round_{next}`（`EvaluationUtils.evaluate_graph`）；
   - 依据 `fusion_score_threshold` 决定“达标”与否，并写入编号化 `fusion_metadata_{k}.json`。

4) EnhancedOptimizer._execute_fusion_async（`enhanced_optimizer.py`）
   - 从包络工作流读取 `prompt.py/graph.py`，提取 `class Workflow:` 片段与 solved_problems；
   - 读取 operator 描述；
   - 由 `WorkflowFusion.prompt_generator.create_fusion_prompt` 生成融合 Prompt（`scripts/workflow_fusion.py` + `scripts/prompts/fusion_prompt.py`）；
   - 调用 `fusion_llm` 产出 `<modification><graph><prompt>`（`WorkflowFusionResult` 模式）；
   - 写入下一轮目录：`workflows/round_{next}/graph.py`、`prompt.py`、`__init__.py`；
   - 生成融合专属 `experience.json`（father node=包络中得分最佳轮，before=其分数）与 `log.json`（记录来源轮与覆盖等元数据）。

5) EnhancedOptimizer._call_fusion_llm（`enhanced_optimizer.py`）
   - 首选 `XmlFormatter.from_model(WorkflowFusionResult)` 进行强约束解析；
   - 失败回退正则抽取；
   - 统一清理代码块标记，保证 `graph` 可被 `GraphUtils.write_graph_files` 正常处理。

6) EnhancedOptimizer._save_fused_workflow_direct（`enhanced_optimizer.py`）
   - 实际写入下一轮 round 目录及融合经验/日志文件。

7) EnhancedOptimizer._create_fusion_experience_file / _create_fusion_log_file（`enhanced_optimizer.py`）
   - 生成融合轮的 `experience.json` 与 `log.json`（仅元数据，执行日志由评估阶段写入 `results.json`）。

8) EnhancedOptimizer._save_fusion_metadata（`enhanced_optimizer.py`）
   - 将本次融合来源与分数、目标轮次、是否采用等信息写入 `fusion_metadata_{k}.json`。

9)（若未触发/失败）回退到常规 `_optimize_graph` 流程（基类 Optimizer）。

## 产物与目录结构

- `workspace/{dataset}/workflows/round_{n}/`
  - `graph.py` / `prompt.py` / `__init__.py`
  - `experience.json`：记录 father/before/after/succeed
  - `log.json`：融合轮含有 `fusion_metadata` 字段
- `workspace/{dataset}/workflows/results.json`
  - 记录每次验证的得分、成本、以及该次验证覆盖的 `solved_problems`（数组）
- `workspace/{dataset}/workflows/processed_experience.json`
  - 汇总所有父节点的成功/失败修改经验
- `workspace/{dataset}/workflows/fusion_metadata_{k}.json`
  - 全局的融合事件记录，避免重复融合组合

## 融合策略与阈值

- 包络选择：基于 `results.json` 中每轮的 union(solved_problems)，贪心选出至多 N 条工作流最大化覆盖；同等覆盖时偏向更高平均分。
- 采用判定：融合得分需大于（包络中最低分 + 阈值）；当前实现即便不满足阈值也会保留融合轮产物，用于后续进一步优化。

## 与原版差异与精简

- 不再创建 `round_fused` 与单一 `fusion_metadata.json`；统一写入下一轮 `round_{n+1}` 与编号化 `fusion_metadata_{k}.json`；
- `scripts/workflow_fusion.py` 已精简为“仅提供 LLM 与 Prompt 生成器”的轻量封装：
  - 保留 `WorkflowFusionResult` 与 `WorkflowFusion`（仅包含 `fusion_llm` 与 `prompt_generator`）。
  - 删除独立的执行/评估/落盘函数，避免与 `EnhancedOptimizer` 的逻辑重复；
- 关键流程集中在 `scripts/enhanced_optimizer.py`，统一复用 Graph/Data/Experience/Evaluation 工具链。

## 常见问题

- Q: 为什么我的融合轮没有被“采用”？
  - A: 采用仅用于阈值统计与元数据记录；融合轮仍会写入下一轮目录并参与后续优化/选择。

- Q: 如何避免重复融合同一组合？
  - A: 通过 `fusion_metadata_{k}.json` 检查来源轮集合签名，重复则跳过融合尝试。

- Q: 如何只跑常规模型优化，不启用融合？
  - A: 运行时设置 `--enable_fusion false` 即可。

---

如需进一步自定义融合 Prompt 或算子可用性，请修改：
- `scripts/prompts/fusion_prompt.py`
- `workspace/{dataset}/workflows/template/operator.json`
