#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Workflow Testing Framework

用于测试和分析特定数据集特定轮次的工作流性能
可以追踪每一步的执行结果，帮助诊断性能问题
"""

import sys
import os
import asyncio
import json
import importlib.util
from typing import Dict, List, Any, Tuple
from pathlib import Path
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.logs import logger
from scripts.async_llm import create_llm_instance


class WorkflowTester:
    """工作流测试器 - 用于测试特定轮次的工作流"""
    
    def __init__(self, dataset: str, round_num: int, config_path: str = "config/config2.yaml"):
        """
        初始化测试器
        
        Args:
            dataset: 数据集名称，如 'DROP', 'MATH', 'MBPP'
            round_num: 轮次编号
            config_path: 配置文件路径
        """
        self.dataset = dataset.upper()
        self.round_num = round_num
        self.config_path = config_path
        
        # 工作流路径
        self.workflow_dir = project_root / "workspace" / self.dataset / "workflows" / f"round_{round_num}"
        
        if not self.workflow_dir.exists():
            raise FileNotFoundError(f"Workflow directory not found: {self.workflow_dir}")
        
        # 加载配置
        self.config = self._load_config()
        
        # 加载工作流
        self.workflow = None
        self.workflow_class = None
        
        # 执行记录
        self.execution_traces = []
        
        logger.info(f"Initialized WorkflowTester for {self.dataset} Round {self.round_num}")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config_file = project_root / self.config_path
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_workflow(self):
        """动态加载工作流模块"""
        graph_file = self.workflow_dir / "graph.py"
        
        if not graph_file.exists():
            raise FileNotFoundError(f"graph.py not found in {self.workflow_dir}")
        
        # 动态导入模块
        spec = importlib.util.spec_from_file_location(
            f"workflow_{self.dataset}_{self.round_num}",
            graph_file
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        # 获取 Workflow 类
        self.workflow_class = getattr(module, 'Workflow')
        
        # 准备 LLM 配置 - 使用第一个可用的模型
        available_models = list(self.config.get('models', {}).keys())
        if not available_models:
            raise ValueError("No models found in config file")
        
        # 使用第一个模型（通常是 gpt-4o-mini）
        default_model = available_models[0]
        model_config = self.config['models'][default_model]
        
        # 构建 LLMConfig 兼容的配置
        llm_config = {
            'model': default_model,
            'key': model_config.get('api_key'),  # 映射 api_key -> key
            'base_url': model_config.get('base_url', 'https://oneapi.deepwisdom.ai/v1'),
            'temperature': model_config.get('temperature', 1),
            'top_p': model_config.get('top_p', 1)
        }
        
        # 实例化工作流
        # DatasetType 是 Literal 类型，直接使用字符串
        self.workflow = self.workflow_class(
            name=f"{self.dataset}_round_{self.round_num}",
            llm_config=llm_config,
            dataset=self.dataset  # 直接使用字符串
        )
        
        logger.info(f"Loaded workflow from {graph_file}")
        return self.workflow
    
    async def test_single_problem(
        self, 
        problem: str, 
        ground_truth: Any = None,
        trace_execution: bool = True
    ) -> Dict[str, Any]:
        """
        测试单个问题
        
        Args:
            problem: 问题文本
            ground_truth: 正确答案（可选）
            trace_execution: 是否追踪执行过程
            
        Returns:
            包含答案、成本、执行轨迹的字典
        """
        if self.workflow is None:
            self.load_workflow()
        
        # 如果需要追踪执行，使用包装的工作流
        if trace_execution:
            answer, cost = await self._traced_execution(problem)
        else:
            answer, cost = await self.workflow(problem)
        
        result = {
            'problem': problem,
            'answer': answer,
            'cost': cost,
            'ground_truth': ground_truth,
        }
        
        if trace_execution:
            result['execution_trace'] = self.execution_traces[-1] if self.execution_traces else None
        
        return result
    
    async def _traced_execution(self, problem: str) -> Tuple[str, float]:
        """
        带追踪的执行，记录每一步的中间结果
        
        Args:
            problem: 问题文本
            
        Returns:
            (答案, 成本)
        """
        trace = {
            'problem': problem,
            'steps': [],
            'llm_calls': 0,
            'total_cost': 0.0
        }
        
        # 包装 LLM 以追踪调用
        original_llm = self.workflow.llm
        call_counter = {'count': 0}
        
        # 包装所有算子
        if hasattr(self.workflow, 'custom'):
            trace['steps'].append(self._wrap_operator(
                self.workflow.custom, 'Custom', trace, call_counter
            ))
        
        if hasattr(self.workflow, 'answer_generate'):
            trace['steps'].append(self._wrap_operator(
                self.workflow.answer_generate, 'AnswerGenerate', trace, call_counter
            ))
        
        if hasattr(self.workflow, 'sc_ensemble'):
            trace['steps'].append(self._wrap_operator(
                self.workflow.sc_ensemble, 'ScEnsemble', trace, call_counter
            ))
        
        # 执行工作流
        try:
            answer, cost = await self.workflow(problem)
            trace['final_answer'] = answer
            trace['total_cost'] = cost
            trace['llm_calls'] = call_counter['count']
            trace['success'] = True
        except Exception as e:
            trace['success'] = False
            trace['error'] = str(e)
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            answer = ""
            cost = 0.0
        
        self.execution_traces.append(trace)
        return answer, cost
    
    def _wrap_operator(self, operator, name: str, trace: Dict, counter: Dict):
        """包装算子以追踪调用"""
        # 这是一个简化的追踪，实际实现需要更复杂的包装
        # 由于算子是异步的，需要特殊处理
        return {'operator': name, 'wrapped': True}
    
    async def test_dataset(
        self, 
        problems: List[Dict[str, Any]], 
        limit: int = None,
        trace_execution: bool = False
    ) -> List[Dict[str, Any]]:
        """
        测试整个数据集
        
        Args:
            problems: 问题列表，每个问题是一个字典，包含 'problem' 和可选的 'answer'
            limit: 限制测试问题数量
            trace_execution: 是否追踪执行（会显著降低速度）
            
        Returns:
            测试结果列表
        """
        if limit:
            problems = problems[:limit]
        
        results = []
        total_cost = 0.0
        
        logger.info(f"Testing {len(problems)} problems...")
        
        for idx, problem_data in enumerate(problems, 1):
            logger.info(f"Testing problem {idx}/{len(problems)}")
            
            problem_text = self._extract_problem_text(problem_data)
            ground_truth = problem_data.get('answer') or problem_data.get('ground_truth')
            
            result = await self.test_single_problem(
                problem=problem_text,
                ground_truth=ground_truth,
                trace_execution=trace_execution
            )
            
            result['problem_id'] = idx
            results.append(result)
            total_cost += result['cost']
            
            logger.info(f"  Answer: {result['answer'][:100]}...")
            logger.info(f"  Cost: ${result['cost']:.6f}")
        
        logger.info(f"Total cost: ${total_cost:.6f}")
        logger.info(f"Average cost per problem: ${total_cost/len(problems):.6f}")
        
        return results
    
    def _extract_problem_text(self, problem_data: Dict) -> str:
        """从问题数据中提取问题文本"""
        if 'problem' in problem_data:
            return problem_data['problem']
        elif 'context' in problem_data:
            # DROP 数据集格式 - context 已经包含了 Passage 和 Question
            return problem_data['context']
        elif 'question' in problem_data:
            context = problem_data.get('context', '')
            question = problem_data['question']
            if context:
                return f"Context: {context}\nQuestion: {question}"
            return question
        elif 'prompt' in problem_data:
            return problem_data['prompt']
        else:
            return str(problem_data)
    
    def save_results(self, results: List[Dict], output_file: str):
        """保存测试结果"""
        output_path = project_root / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """分析测试结果"""
        analysis = {
            'total_problems': len(results),
            'total_cost': sum(r['cost'] for r in results),
            'average_cost': sum(r['cost'] for r in results) / len(results) if results else 0,
            'min_cost': min(r['cost'] for r in results) if results else 0,
            'max_cost': max(r['cost'] for r in results) if results else 0,
        }
        
        # 如果有执行追踪，分析步骤
        traced_results = [r for r in results if 'execution_trace' in r and r['execution_trace']]
        if traced_results:
            total_llm_calls = sum(
                r['execution_trace']['llm_calls'] 
                for r in traced_results 
                if 'llm_calls' in r['execution_trace']
            )
            analysis['average_llm_calls'] = total_llm_calls / len(traced_results)
        
        return analysis


def load_test_data(dataset: str, data_file: str = None, limit: int = 10) -> List[Dict]:
    """
    加载测试数据
    
    Args:
        dataset: 数据集名称
        data_file: 数据文件路径（可选）
        limit: 限制数量
        
    Returns:
        问题列表
    """
    dataset = dataset.upper()
    
    if data_file is None:
        # 使用默认验证集
        data_file = project_root / "data" / "datasets" / f"{dataset.lower()}_validate.jsonl"
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    problems = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            problems.append(json.loads(line))
    
    logger.info(f"Loaded {len(problems)} problems from {data_file}")
    return problems


async def main():
    """示例：测试 DROP 数据集第 8 轮工作流"""
    
    # 创建测试器
    tester = WorkflowTester(dataset='DROP', round_num=8)
    
    # 加载工作流
    tester.load_workflow()
    
    # 加载测试数据（限制 5 个问题用于快速测试）
    test_problems = load_test_data('DROP', limit=5)
    
    # 测试数据集（启用追踪以查看每一步）
    results = await tester.test_dataset(
        problems=test_problems,
        trace_execution=True  # 设为 False 可加快速度
    )
    
    # 分析结果
    analysis = tester.analyze_results(results)
    
    print("\n" + "="*80)
    print("TEST RESULTS ANALYSIS")
    print("="*80)
    print(f"Total problems tested: {analysis['total_problems']}")
    print(f"Total cost: ${analysis['total_cost']:.6f}")
    print(f"Average cost per problem: ${analysis['average_cost']:.6f}")
    print(f"Min cost: ${analysis['min_cost']:.6f}")
    print(f"Max cost: ${analysis['max_cost']:.6f}")
    
    if 'average_llm_calls' in analysis:
        print(f"Average LLM calls per problem: {analysis['average_llm_calls']:.1f}")
    
    # 保存结果
    tester.save_results(results, f"test/results_{tester.dataset}_round_{tester.round_num}.json")
    
    # 打印详细追踪（如果有）
    if results and 'execution_trace' in results[0]:
        print("\n" + "="*80)
        print("SAMPLE EXECUTION TRACE (First Problem)")
        print("="*80)
        trace = results[0]['execution_trace']
        print(json.dumps(trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
