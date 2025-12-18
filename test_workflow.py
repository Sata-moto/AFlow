#!/usr/bin/env python3
"""
测试指定数据集的某一轮workflow在测试集上的性能

用法:
    python test_workflow.py --dataset GSM8K --round 5
    python test_workflow.py --dataset MATH --round 10 --llm_config config/config2.example.yaml
"""

import argparse
import asyncio
import os
import sys
import yaml
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from scripts.evaluator import Evaluator
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.logs import logger


async def test_workflow(dataset: str, round_num: int, llm_config_path: str = None):
    """
    测试指定轮次的workflow
    
    Args:
        dataset: 数据集名称 (如 GSM8K, MATH, DROP等)
        round_num: 轮次编号
        llm_config_path: LLM配置文件路径
    
    Returns:
        tuple: (score, avg_cost, total_cost)
    """
    logger.info("=" * 80)
    logger.info(f"Testing Workflow - Dataset: {dataset}, Round: {round_num}")
    logger.info("=" * 80)
    
    # 构建路径
    workspace_path = f"workspace/{dataset}"
    workflows_path = f"{workspace_path}/workflows"
    round_path = f"{workflows_path}/round_{round_num}"
    
    # 检查路径是否存在
    if not os.path.exists(round_path):
        logger.error(f"Round directory not found: {round_path}")
        logger.error(f"Available rounds:")
        if os.path.exists(workflows_path):
            rounds = [d for d in os.listdir(workflows_path) if d.startswith('round_')]
            for r in sorted(rounds):
                logger.error(f"  - {r}")
        return None, None, None
    
    # 加载LLM配置
    if llm_config_path is None:
        # 尝试默认配置路径
        config_candidates = [
            "config/config2.example.yaml",
            "config/config.yaml",
            "config.yaml"
        ]
        for candidate in config_candidates:
            if os.path.exists(candidate):
                llm_config_path = candidate
                break
        
        if llm_config_path is None:
            logger.error("No LLM config file found. Please specify with --llm_config")
            return None, None, None
    
    logger.info(f"Loading LLM config from: {llm_config_path}")
    
    from scripts.async_llm import LLMsConfig
    models_config = LLMsConfig.default()
    
    # 默认使用gpt-4o-mini作为执行模型
    exec_model_name = "gpt-4o-mini"
    llm_config = models_config.get(exec_model_name)
    
    logger.info(f"Using execution model: {exec_model_name}")
    
    # 加载workflow图
    logger.info(f"Loading workflow from round {round_num}...")
    graph_utils = GraphUtils(workspace_path)
    workflow_graph = graph_utils.load_graph(round_num, workflows_path)
    
    if workflow_graph is None:
        logger.error(f"Failed to load workflow from round {round_num}")
        return None, None, None
    
    logger.info(f"✓ Workflow loaded successfully")
    
    # 创建评估器
    evaluator = Evaluator(eval_path=round_path)
    
    # 在测试集上评估
    logger.info(f"Evaluating on {dataset} test set...")
    logger.info("This may take a while depending on dataset size...")
    
    try:
        score, avg_cost, total_cost, solved_problems = await evaluator.graph_evaluate(
            dataset=dataset,
            graph=workflow_graph,
            params={"dataset": dataset, "llm_config": llm_config},
            path=round_path,
            is_test=True  # 重要: 在测试集上评估
        )
        
        logger.info("=" * 80)
        logger.info("TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Dataset:        {dataset}")
        logger.info(f"Round:          {round_num}")
        logger.info(f"Test Score:     {score:.4f} ({score*100:.2f}%)")
        logger.info(f"Avg Cost:       ${avg_cost:.4f}")
        logger.info(f"Total Cost:     ${total_cost:.4f}")
        logger.info(f"Solved:         {len(solved_problems)} problems")
        logger.info("=" * 80)
        
        # 保存结果到文件
        result_file = f"{round_path}/test_results.txt"
        with open(result_file, 'w') as f:
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Round: {round_num}\n")
            f.write(f"Test Score: {score:.4f} ({score*100:.2f}%)\n")
            f.write(f"Avg Cost: ${avg_cost:.4f}\n")
            f.write(f"Total Cost: ${total_cost:.4f}\n")
            f.write(f"Solved Problems: {len(solved_problems)}\n")
            f.write(f"Problem IDs: {sorted(solved_problems)}\n")
        
        logger.info(f"Results saved to: {result_file}")
        
        return score, avg_cost, total_cost
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return None, None, None


def find_best_round(dataset: str):
    """
    找到指定数据集中性能最好的轮次
    
    Args:
        dataset: 数据集名称
    
    Returns:
        tuple: (best_round, best_score) 或 (None, None)
    """
    workspace_path = f"workspace/{dataset}"
    workflows_path = f"{workspace_path}/workflows"
    results_file = f"{workflows_path}/results.json"
    
    if not os.path.exists(results_file):
        logger.error(f"Results file not found: {results_file}")
        return None, None
    
    import json
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if not results:
        logger.error("No results found in results.json")
        return None, None
    
    # 找到最高分数的轮次
    best_result = max(results, key=lambda x: x.get('score', 0))
    best_round = best_result.get('round')
    best_score = best_result.get('score')
    
    logger.info(f"Best round found: Round {best_round} with score {best_score:.4f}")
    
    # 显示前5名
    logger.info("\nTop 5 rounds:")
    sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:5]
    for i, result in enumerate(sorted_results, 1):
        logger.info(f"  {i}. Round {result.get('round')}: {result.get('score', 0):.4f}")
    
    return best_round, best_score


async def main():
    parser = argparse.ArgumentParser(
        description='Test a specific workflow round on the test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific round
  python test_workflow.py --dataset GSM8K --round 5
  
  # Test best round automatically
  python test_workflow.py --dataset GSM8K --best
  
  # With custom LLM config
  python test_workflow.py --dataset MATH --round 10 --llm_config my_config.yaml
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., GSM8K, MATH, DROP)')
    parser.add_argument('--round', type=int, default=None,
                       help='Round number to test')
    parser.add_argument('--best', action='store_true',
                       help='Automatically find and test the best round')
    parser.add_argument('--llm_config', type=str, default=None,
                       help='Path to LLM config file (default: auto-detect)')
    
    args = parser.parse_args()
    
    # 确定要测试的轮次
    if args.best:
        logger.info(f"Finding best round for {args.dataset}...")
        round_num, validation_score = find_best_round(args.dataset)
        if round_num is None:
            logger.error("Could not find best round")
            sys.exit(1)
        logger.info(f"Testing best round: {round_num} (validation score: {validation_score:.4f})")
    elif args.round is not None:
        round_num = args.round
    else:
        logger.error("Please specify either --round or --best")
        sys.exit(1)
    
    # 执行测试
    score, avg_cost, total_cost = await test_workflow(
        dataset=args.dataset,
        round_num=round_num,
        llm_config_path=args.llm_config
    )
    
    if score is None:
        logger.error("Test failed")
        sys.exit(1)
    
    logger.info(f"\n✓ Test completed successfully!")
    logger.info(f"Final test score: {score:.4f} ({score*100:.2f}%)")


if __name__ == "__main__":
    asyncio.run(main())
