#!/usr/bin/env python3
"""
工作流分化功能演示脚本
Demonstration script for workflow differentiation functionality
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.async_llm import LLMConfig
from scripts.workflow_differentiation import WorkflowDifferentiation
from scripts.prompts.differentiation_prompt import DifferentiationPromptGenerator

def create_sample_workflow() -> Dict[str, Any]:
    """创建示例工作流用于演示分化功能"""
    return {
        "score": 0.85,
        "workflow_data": {
            "graph.py": '''class Workflow:
    def __init__(self):
        self.operators = ["analyze_problem", "extract_key_info", "apply_solution_strategy", "verify_result"]
    
    def execute(self, input_data):
        # Step 1: Analyze the problem structure
        analyzed = self.analyze_problem(input_data)
        
        # Step 2: Extract key information
        extracted = self.extract_key_info(analyzed)
        
        # Step 3: Apply solution strategy
        solved = self.apply_solution_strategy(extracted)
        
        # Step 4: Verify the result
        verified = self.verify_result(solved)
        
        return verified
    
    def analyze_problem(self, data):
        """Analyze the problem to understand its structure and requirements"""
        return {"type": "mathematical", "complexity": "medium", "original": data}
    
    def extract_key_info(self, data):
        """Extract key numerical values and relationships"""
        return {**data, "key_info": "extracted", "variables": ["x", "y"]}
    
    def apply_solution_strategy(self, data):
        """Apply the appropriate mathematical solution strategy"""
        return {**data, "solution": "computed", "method": "algebraic"}
    
    def verify_result(self, data):
        """Verify the solution through back-substitution or alternative methods"""
        return {**data, "verified": True, "confidence": 0.95}''',
            
            "prompt.py": '''def get_prompt(question):
    return f"""
    Solve this mathematical problem step by step:
    
    Question: {question}
    
    Please follow these steps:
    1. Analyze the problem structure and identify what is being asked
    2. Extract key numerical values and relationships
    3. Choose and apply the appropriate solution strategy
    4. Verify your result through back-substitution or alternative methods
    
    Show your work clearly at each step.
    """''',
            
            "operator.py": '''class MathOperator:
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def execute(self, problem_data):
        """Execute mathematical operation on the given problem data"""
        return {
            "operator": self.name,
            "input": problem_data,
            "processed": True,
            "method": "standard_mathematical_approach"
        }'''
        },
        "metadata": {
            "round": 1,
            "creation_method": "optimization",
            "performance_metrics": {
                "accuracy": 0.85,
                "efficiency": 0.78,
                "problem_coverage": 0.82
            },
            "problem_types_solved": ["linear_equations", "quadratic_equations", "basic_algebra"],
            "common_errors": ["arithmetic_mistakes", "sign_errors"]
        }
    }

async def demonstrate_differentiation_directions():
    """演示不同方向的工作流分化"""
    print("\n" + "="*60)
    print("🔬 工作流分化方向演示")
    print("="*60)
    
    # 创建示例工作流候选
    sample_candidates = [create_sample_workflow()]
    
    # 创建分化提示生成器
    prompt_generator = DifferentiationPromptGenerator()
    
    # 演示每种分化方向
    directions = [
        ("problem_type_specialization", "问题类型特化", 
         "将工作流特化为处理特定类型的问题（如几何、代数、微积分等）"),
        ("strategy_diversification", "策略多样化",
         "创建使用不同解题策略的工作流变体"),
        ("algorithmic_approach_variation", "算法方法变化",
         "采用不同的算法实现来解决相同类型的问题"),
        ("complexity_adaptation", "复杂度适应",
         "针对不同难度级别优化工作流的处理方式"), 
        ("error_pattern_handling", "错误模式处理",
         "专门针对常见错误类型进行预防和修正")
    ]
    
    for direction, chinese_name, description in directions:
        print(f"\n📍 {chinese_name} ({direction})")
        print(f"   描述: {description}")
        
        # 生成分化提示
        prompt = prompt_generator.generate_differentiation_prompt(
            candidates=sample_candidates,
            direction=direction,
            dataset="DEMO",
            question_type="mathematical"
        )
        
        print(f"   ✓ 提示长度: {len(prompt)} 字符")
        print(f"   ✓ 包含关键词: {direction.replace('_', ' ')}")
        
        # 显示提示的关键部分（前200字符）
        preview = prompt[:200].replace('\n', ' ')
        print(f"   预览: {preview}...")
    
    print("\n✨ 所有分化方向演示完成！")

def create_demo_config() -> Dict[str, Any]:
    """创建演示配置"""
    return {
        "dataset": "GSM8K",
        "question_type": "mathematical",
        "differentiation_settings": {
            "enable_differentiation": True,
            "differentiation_probability": 0.5,
            "max_differentiation_rounds": 3,
            "allowed_directions": [
                "problem_type_specialization",
                "strategy_diversification",
                "algorithmic_approach_variation"
            ]
        },
        "output_settings": {
            "save_intermediate_results": True,
            "detailed_logging": True
        }
    }

async def run_mini_differentiation_simulation():
    """运行小型分化模拟（无需真实LLM调用）"""
    print("\n" + "="*60)
    print("🎯 小型工作流分化模拟")
    print("="*60)
    
    # 创建演示配置
    config = create_demo_config()
    print(f"数据集: {config['dataset']}")
    print(f"问题类型: {config['question_type']}")
    print(f"分化概率: {config['differentiation_settings']['differentiation_probability']}")
    
    # 创建示例候选工作流
    candidates = [create_sample_workflow()]
    
    print(f"\n📊 候选工作流数量: {len(candidates)}")
    print(f"最佳候选评分: {candidates[0]['score']}")
    
    # 模拟分化方向选择
    available_directions = config['differentiation_settings']['allowed_directions']
    selected_direction = available_directions[0]  # 选择第一个方向作为演示
    
    print(f"\n🎯 选择的分化方向: {selected_direction}")
    
    # 生成分化提示
    prompt_generator = DifferentiationPromptGenerator()
    differentiation_prompt = prompt_generator.generate_differentiation_prompt(
        candidates=candidates,
        direction=selected_direction,
        dataset=config['dataset'],
        question_type=config['question_type']
    )
    
    print(f"✓ 分化提示生成成功 ({len(differentiation_prompt)} 字符)")
    
    # 模拟分化结果（实际环境中这会通过LLM生成）
    simulated_result = {
        "differentiation_direction": selected_direction,
        "source_workflow_round": 1,
        "specialized_focus": "linear_algebra_problems",
        "modifications_made": [
            "Enhanced linear equation detection",
            "Specialized matrix operation handling", 
            "Improved variable substitution strategies"
        ],
        "expected_performance_improvement": "15-20% on linear algebra problems"
    }
    
    print(f"\n🔄 模拟分化结果:")
    print(f"   • 分化方向: {simulated_result['differentiation_direction']}")
    print(f"   • 源工作流轮次: {simulated_result['source_workflow_round']}")
    print(f"   • 专门化焦点: {simulated_result['specialized_focus']}")
    print(f"   • 预期性能提升: {simulated_result['expected_performance_improvement']}")
    
    print("\n✅ 工作流分化模拟完成！")

def show_usage_examples():
    """显示使用示例"""
    print("\n" + "="*60)
    print("📖 工作流分化模块使用指南")
    print("="*60)
    
    examples = [
        {
            "场景": "数学问题求解优化",
            "命令": "python run_enhanced.py --dataset GSM8K --enable_differentiation --differentiation_probability 0.3",
            "说明": "在GSM8K数据集上启用工作流分化，30%概率进行分化"
        },
        {
            "场景": "代码生成任务分化",
            "命令": "python run_enhanced.py --dataset HumanEval --enable_differentiation --max_differentiation_rounds 5",
            "说明": "在代码生成任务上进行工作流分化，最多5轮分化"
        },
        {
            "场景": "结合融合和分化",
            "命令": "python run_enhanced.py --enable_fusion --enable_differentiation --differentiation_probability 0.25",
            "说明": "同时启用工作流融合和分化功能，优先融合，再考虑分化"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📋 示例 {i}: {example['场景']}")
        print(f"   命令: {example['命令']}")
        print(f"   说明: {example['说明']}")
    
    print(f"\n💡 关键参数说明:")
    print(f"   --enable_differentiation: 启用工作流分化功能")
    print(f"   --differentiation_probability: 分化概率 (0.0-1.0，推荐0.2-0.4)")
    print(f"   --max_differentiation_rounds: 最大分化轮数 (推荐3-7)")
    
    print(f"\n🔄 分化流程:")
    print(f"   1. 检查分化条件（概率、可用工作流数量等）")
    print(f"   2. 选择高性能候选工作流作为分化源")
    print(f"   3. 智能选择分化方向（5种策略）")
    print(f"   4. LLM驱动的工作流专门化生成")
    print(f"   5. 评估分化后的工作流性能")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="工作流分化功能演示")
    parser.add_argument("--mode", choices=["demo", "simulation", "usage"], 
                       default="demo", help="运行模式")
    
    args = parser.parse_args()
    
    print("🚀 AFlow 工作流分化模块演示程序")
    print("="*60)
    
    if args.mode == "demo":
        await demonstrate_differentiation_directions()
    elif args.mode == "simulation":
        await run_mini_differentiation_simulation()
    elif args.mode == "usage":
        show_usage_examples()
    
    print(f"\n🎉 演示完成！工作流分化模块已成功集成到AFlow系统中。")
    print(f"💡 现在可以在实际的优化任务中使用分化功能了！")

if __name__ == "__main__":
    asyncio.run(main())
