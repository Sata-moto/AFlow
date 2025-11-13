#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detailed Workflow Step Tracer - Fixed Version

详细追踪工作流每一步的执行结果（修复版）
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test.workflow_tester import WorkflowTester, load_test_data
from scripts.logs import logger


class OperatorProxy:
    """算子代理 - 用于追踪算子调用"""
    
    def __init__(self, original_operator, operator_name, tracer):
        self.original = original_operator
        self.name = operator_name
        self.tracer = tracer
    
    async def __call__(self, *args, **kwargs):
        """拦截调用并追踪"""
        return await self.tracer._trace_call(
            self.name,
            self.original,
            *args,
            **kwargs
        )


class DetailedStepTracer:
    """详细的步骤追踪器 - 记录每个算子调用的输入输出"""
    
    def __init__(self, workflow):
        self.workflow = workflow
        self.step_logs = []
        self.step_counter = 0
        
        # 保存原始算子
        self._original_custom = None
        self._original_answer_gen = None
        self._original_ensemble = None
        
        # 包装所有算子
        self._wrap_operators()
    
    def _wrap_operators(self):
        """用代理对象包装工作流中的所有算子"""
        # 包装 Custom 算子
        if hasattr(self.workflow, 'custom'):
            self._original_custom = self.workflow.custom
            self.workflow.custom = OperatorProxy(
                self._original_custom, 
                'Custom', 
                self
            )
        
        # 包装 AnswerGenerate 算子
        if hasattr(self.workflow, 'answer_generate'):
            self._original_answer_gen = self.workflow.answer_generate
            self.workflow.answer_generate = OperatorProxy(
                self._original_answer_gen, 
                'AnswerGenerate',
                self
            )
        
        # 包装 ScEnsemble 算子
        if hasattr(self.workflow, 'sc_ensemble'):
            self._original_ensemble = self.workflow.sc_ensemble
            self.workflow.sc_ensemble = OperatorProxy(
                self._original_ensemble,
                'ScEnsemble',
                self
            )
    
    async def _trace_call(self, operator_name: str, original_operator, *args, **kwargs):
        """追踪一个算子调用"""
        self.step_counter += 1
        step_id = self.step_counter
        
        timestamp = datetime.now().isoformat()
        
        # 记录输入
        print(f"\n{'='*80}")
        print(f"STEP {step_id}: {operator_name}")
        print(f"{'='*80}")
        print(f"Input args: {[str(a)[:100] for a in args]}")
        print(f"Input kwargs keys: {list(kwargs.keys())}")
        
        # 调用原始算子
        try:
            result = await original_operator(*args, **kwargs)
            
            # 记录输出
            print(f"Output: {str(result)[:200]}...")
            
            # 保存日志
            log_entry = {
                'step_id': step_id,
                'operator': operator_name,
                'timestamp': timestamp,
                'input_args': self._serialize_args(args),
                'input_kwargs': self._serialize_kwargs(kwargs),
                'output': self._serialize_output(result),
                'success': True
            }
            
            self.step_logs.append(log_entry)
            
            return result
            
        except Exception as e:
            print(f"ERROR: {e}")
            
            log_entry = {
                'step_id': step_id,
                'operator': operator_name,
                'timestamp': timestamp,
                'input_args': self._serialize_args(args),
                'input_kwargs': self._serialize_kwargs(kwargs),
                'error': str(e),
                'success': False
            }
            
            self.step_logs.append(log_entry)
            raise
    
    def _serialize_args(self, args) -> List:
        """序列化位置参数"""
        serialized = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                serialized.append(arg)
            elif isinstance(arg, (list, dict)):
                serialized.append(arg)
            else:
                serialized.append(str(arg)[:500])
        return serialized
    
    def _serialize_kwargs(self, kwargs) -> Dict:
        """序列化关键字参数"""
        serialized = {}
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, list):
                serialized[key] = [
                    str(item)[:200] if not isinstance(item, (str, int, float)) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                serialized[key] = value
            else:
                serialized[key] = str(value)[:500]
        return serialized
    
    def _serialize_output(self, output) -> Any:
        """序列化输出"""
        if isinstance(output, (str, int, float, bool, type(None))):
            return output
        elif isinstance(output, dict):
            return {k: str(v)[:500] if not isinstance(v, (str, int, float, bool, type(None))) else v 
                    for k, v in output.items()}
        elif isinstance(output, list):
            return [str(item)[:500] if not isinstance(item, (str, int, float, bool, type(None))) else item 
                    for item in output]
        else:
            return str(output)[:500]
    
    def get_trace(self) -> Dict:
        """获取完整的追踪记录"""
        return {
            'total_steps': self.step_counter,
            'steps': self.step_logs
        }
    
    def reset(self):
        """重置追踪记录"""
        self.step_logs = []
        self.step_counter = 0


async def test_with_detailed_trace(dataset: str, round_num: int, num_problems: int = 5):
    """
    测试工作流并记录详细的执行步骤
    
    Args:
        dataset: 数据集名称
        round_num: 轮次编号
        num_problems: 测试问题数量
    """
    print(f"\n{'='*80}")
    print(f"DETAILED WORKFLOW TESTING: {dataset} Round {round_num}")
    print(f"{'='*80}\n")
    
    # 创建测试器
    tester = WorkflowTester(dataset=dataset, round_num=round_num)
    tester.load_workflow()
    
    # 创建追踪器
    tracer = DetailedStepTracer(tester.workflow)
    
    # 加载测试数据
    test_problems = load_test_data(dataset, limit=num_problems)
    
    all_results = []
    
    for idx, problem_data in enumerate(test_problems, 1):
        print(f"\n{'#'*80}")
        print(f"# TESTING PROBLEM {idx}/{len(test_problems)}")
        print(f"{'#'*80}\n")
        
        # 提取问题文本
        problem_text = tester._extract_problem_text(problem_data)
        print(f"Problem: {str(problem_data)[:500]}")
        
        # 重置追踪器
        tracer.reset()
        
        try:
            # 执行工作流
            answer, cost = await tester.workflow(problem_text)
            
            result = {
                'problem_id': idx,
                'problem': str(problem_data)[:500],
                'answer': answer,
                'cost': cost,
                'trace': tracer.get_trace()
            }
            
            print(f"\n{'='*80}")
            print(f"FINAL RESULT")
            print(f"{'='*80}")
            print(f"Answer: {answer}")
            print(f"Cost: ${cost:.6f}")
            print(f"Total steps: {tracer.step_counter}")
            
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"ERROR: {e}")
            print(f"{'!'*80}\n")
            result = {
                'problem_id': idx,
                'problem': str(problem_data)[:500],
                'error': str(e),
                'trace': tracer.get_trace()
            }
        
        all_results.append(result)
    
    # 保存详细结果
    output_file = project_root / f"test/detailed_trace_{dataset}_round_{round_num}.json"
    
    # 确保目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n\n{'='*80}")
        print(f"TESTING COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Results saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size} bytes")
    except Exception as e:
        print(f"\n\n{'='*80}")
        print(f"WARNING: Failed to save results")
        print(f"{'='*80}")
        print(f"✗ Error: {e}")
        print(f"  Attempted path: {output_file}")
        print(f"  Absolute path: {output_file.absolute()}")
    
    # 统计信息
    total_cost = sum(r.get('cost', 0) for r in all_results)
    avg_steps = sum(r['trace']['total_steps'] for r in all_results) / len(all_results)
    
    print(f"\nTotal cost: ${total_cost:.6f}")
    print(f"Average cost per problem: ${total_cost/len(all_results):.6f}")
    print(f"Average steps per problem: {avg_steps:.1f}")
    
    return all_results


async def compare_workflows(dataset: str, round_nums: List[int], num_problems: int = 5):
    """
    比较多个轮次的工作流
    
    Args:
        dataset: 数据集名称
        round_nums: 要比较的轮次列表
        num_problems: 测试问题数量
    """
    print(f"\n{'='*80}")
    print(f"COMPARING WORKFLOWS: {dataset} Rounds {round_nums}")
    print(f"{'='*80}\n")
    
    # 加载测试数据（所有轮次使用相同数据）
    test_problems = load_test_data(dataset, limit=num_problems)
    
    comparison_results = {}
    
    for round_num in round_nums:
        print(f"\n{'#'*80}")
        print(f"# TESTING ROUND {round_num}")
        print(f"{'#'*80}\n")
        
        try:
            # 创建测试器
            tester = WorkflowTester(dataset=dataset, round_num=round_num)
            tester.load_workflow()
            
            # 测试
            results = await tester.test_dataset(
                problems=test_problems,
                trace_execution=False  # 比较时不需要详细追踪
            )
            
            # 分析
            analysis = tester.analyze_results(results)
            
            comparison_results[f"round_{round_num}"] = {
                'analysis': analysis,
                'results': results
            }
            
            print(f"\nRound {round_num} Summary:")
            print(f"  Average cost: ${analysis['average_cost']:.6f}")
            print(f"  Total cost: ${analysis['total_cost']:.6f}")
            
        except Exception as e:
            print(f"Failed to test round {round_num}: {e}")
            comparison_results[f"round_{round_num}"] = {
                'error': str(e)
            }
    
    # 保存比较结果
    output_file = project_root / f"test/comparison_{dataset}_rounds_{'_'.join(map(str, round_nums))}.json"
    
    # 确保目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n\n{'='*80}")
        print(f"COMPARISON COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Results saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size} bytes")
    except Exception as e:
        print(f"\n\n{'='*80}")
        print(f"WARNING: Failed to save comparison results")
        print(f"{'='*80}")
        print(f"✗ Error: {e}")
        print(f"  Attempted path: {output_file}")
        print(f"  Absolute path: {output_file.absolute()}")
    
    # 比较摘要
    print(f"\n\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    for round_num in round_nums:
        key = f"round_{round_num}"
        if key in comparison_results and 'analysis' in comparison_results[key]:
            analysis = comparison_results[key]['analysis']
            print(f"Round {round_num}:")
            print(f"  Average cost: ${analysis['average_cost']:.6f}")
            print(f"  Total cost: ${analysis['total_cost']:.6f}")
        else:
            print(f"Round {round_num}: ERROR")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # 比较模式
        dataset = sys.argv[2] if len(sys.argv) > 2 else 'DROP'
        rounds = [int(r) for r in sys.argv[3:]] if len(sys.argv) > 3 else [7, 8]
        num_problems = 10
        asyncio.run(compare_workflows(dataset, rounds, num_problems))
    else:
        # 详细追踪模式
        dataset = sys.argv[1] if len(sys.argv) > 1 else 'DROP'
        round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        num_problems = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        asyncio.run(test_with_detailed_trace(dataset, round_num, num_problems))
