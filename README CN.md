# AFlow：自动化智能体工作流生成
[![Arxiv](https://img.shields.io/badge/arXiv-AFlow-b31b1b)](https://arxiv.org/abs/2410.10762)
[![PR欢迎](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/FoundationAgents/AFlow/pulls)
> 如果您在使用或复现代码时遇到任何困难，请直接联系我（邮箱：didi4goooogle@gmail.com，微信：18831933368）。部分操作符在从MetaGPT迁移到本仓库的过程中可能存在bug。
AFlow是一个用于自动生成和优化智能体工作流的框架。它在代码表示的工作流空间中使用蒙特卡洛树搜索来寻找有效的工作流，用机器努力取代手动开发。我们的方法显示出在各种任务上超越手工设计工作流的潜力。
我们正在构建它以支持更多基准测试和开放式任务！如果您有任何问题，请提交issue或给我们发邮件！
<p align="center">
<a href=""><img src="assets/AFLOW-performance.jpg" alt="AFlow的性能" title="AFlow的性能<sub>1</sub>" width="80%"></a>
</p>
## 框架组件
- **节点（Node）**：LLM调用的基本单位。请参阅`metagpt_core/action_nodes/action_node.py`以获取控制LLM、温度、格式和提示的灵活接口。
- **操作符（Operator）**：预定义的节点组合，以提高搜索效率。封装了生成、格式化、审查、修订、集成、测试和程序员等常见操作。详情请参见`operator.py`。您可以参考此代码中的实现来自定义您自己的操作符。
- **工作流（Workflow）**：由边连接的LLM调用节点序列。可以表示为图、神经网络或代码，以表达各种执行结构。我们的实现请参见`workflow.py`。
- **优化器（Optimizer）**：在蒙特卡洛树搜索变体中使用LLM来探索和优化工作流。基于性能迭代地选择、扩展、评估和更新工作流。详情请参见`optimizer.py`。
- **评估器（Evaluator）**：评估工作流在给定任务上的性能。提供反馈以指导优化过程朝着更有效的工作流发展。详情请参见`evaluator.py`。
<p align="center">
<a href=""><img src="assets/AFLOW-method.jpg" alt="AFlow框架" title="AFlow框架 <sub>1</sub>" width="80%"></a>
</p>
## 数据集
### 实验数据集
我们在六个数据集（HumanEval、MBPP、GSM8K、MATH、HotpotQA、DROP）上进行了实验，并提供了它们的评估代码。数据可以在此[数据集](https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e)链接中找到，或者您可以使用`metagpt/ext/aflow/data/download_data.py`下载它们。
<p align="center">
<a href=""><img src="assets/AFLOW-experiment.jpg" alt="AFlow的性能" title="AFlow的性能<sub>1</sub>" width="80%"></a>
</p>
### 自定义数据集
对于自定义任务，您可以参考`benchmark`文件夹中的代码。继承`BaseBenchmark`类并实现`evaluate_problem`、`calculate_score`和`get_result_columns`来添加您的自定义数据集基准。然后，在`evaluator.py`和`optimizer.py`中添加您的基准名称，以为您的自定义数据集找到有效的工作流。
## 快速开始
1. 设置Python环境：
   ```bash
   # 创建并激活Python 3.9虚拟环境
   conda create -n <your_env_name> python=3.9
   # 安装依赖
   pip install -r requirements.txt
   ```
2. 配置优化参数：
   - 使用命令行参数或修改`run.py`中的默认参数：
     ```python
     --dataset              # （必需）数据集类型（HumanEval/MBPP/GSM8K/MATH/HotpotQA/DROP）
     --sample 4             # 写到 prompt 中的 log 数量
     --optimized_path PATH  # 优化结果保存路径
     --initial_round 1      # 初始轮次
     --max_rounds 20        # AFLOW的最大迭代轮次
     --check_convergence    # 是否启用提前停止
     --validation_rounds 5  # 验证集上计算某条 workflow 正确率时，计算几次取均值
     --if_force_download    # 如果设置为True，则强制下载数据集
     ```
3. 在`config/config2.yaml`中配置LLM参数（参考`config/config2.example.yaml`）
4. 在`run.py`和`operator.py`、`optimized_path/template/operator.json`中设置操作符。您可以参考我们的实现为特定数据集添加操作符
5. 首次使用时，通过在`run.py`中设置`download(["datasets"])`来下载数据集和初始轮次
6. （可选）按照[自定义数据集](#custom-datasets)部分添加您的自定义数据集和相应的评估函数
7. （可选）如果您想使用部分验证数据，可以在`evaluator.py`中设置`va_list`
8. 运行优化：
   ```bash
   # 使用默认参数
   python run.py --dataset MATH
   
   # 或使用自定义参数
   python run.py --dataset MATH --sample n --optimized_path xxx ...
   ```
## 复现论文中的结果
1. 我们在此[链接](https://drive.google.com/uc?export=download&id=1Sr5wjgKf3bN8OC7G6cO3ynzJqD4w6_Dv)中提供了从实验中获得的原始数据，包括每次迭代中生成的工作流和提示，以及它们在验证数据集上的轨迹。我们还提供了每个数据集的最优工作流以及测试数据集上的相应数据。您可以使用`data/download_data.py`下载这些数据。
2. 您可以通过使用`run.py`的不同`ExperimentConfig`直接复现我们的实验结果。
## 路线图
- 支持多种搜索算法
- 支持工作流中的多模型搜索
- 支持排行榜
- 支持更多基准测试
- 支持多模态任务
## 引用
如果您在研究中使用AFlow，请引用我们的论文：
```
@inproceedings{
   zhang2025aflow,
   title={{AF}low: Automating Agentic Workflow Generation},
   author={Jiayi Zhang and Jinyu Xiang and Zhaoyang Yu and Fengwei Teng and Xiong-Hui Chen and Jiaqi Chen and Mingchen Zhuge and Xin Cheng and Sirui Hong and Jinlin Wang and Bingnan Zheng and Bang Liu and Yuyu Luo and Chenglin Wu},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=z5uVAKwmjf}
}
```
