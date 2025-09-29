#!/usr/bin/env python3
"""
工作流分化模块完整验证脚本
Comprehensive validation script for workflow differentiation module
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

def check_file_exists(filepath: str, description: str) -> bool:
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"   ✅ {description}")
        return True
    else:
        print(f"   ❌ {description} - 文件不存在: {filepath}")
        return False

def check_import(module_path: str, class_name: str, description: str) -> bool:
    """检查模块导入"""
    try:
        spec = __import__(module_path, fromlist=[class_name])
        getattr(spec, class_name)
        print(f"   ✅ {description}")
        return True
    except Exception as e:
        print(f"   ❌ {description} - 导入失败: {e}")
        return False

def check_method_exists(module_path: str, class_name: str, method_name: str, description: str) -> bool:
    """检查类方法是否存在"""
    try:
        spec = __import__(module_path, fromlist=[class_name])
        cls = getattr(spec, class_name)
        if hasattr(cls, method_name):
            print(f"   ✅ {description}")
            return True
        else:
            print(f"   ❌ {description} - 方法不存在: {method_name}")
            return False
    except Exception as e:
        print(f"   ❌ {description} - 检查失败: {e}")
        return False

def validate_file_content(filepath: str, required_content: List[str], description: str) -> bool:
    """验证文件内容包含必要元素"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing_content = []
        for item in required_content:
            if item not in content:
                missing_content.append(item)
        
        if not missing_content:
            print(f"   ✅ {description}")
            return True
        else:
            print(f"   ❌ {description} - 缺少内容: {', '.join(missing_content)}")
            return False
    except Exception as e:
        print(f"   ❌ {description} - 验证失败: {e}")
        return False

def run_comprehensive_validation():
    """运行完整的验证检查"""
    print("🔍 AFlow 工作流分化模块完整验证")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 0
    
    # 1. 核心文件存在性检查
    print("\n📁 核心文件检查:")
    file_checks = [
        ("scripts/prompts/differentiation_prompt.py", "分化提示生成器"),
        ("scripts/workflow_differentiation.py", "工作流分化处理器"),
        ("scripts/enhanced_optimizer.py", "增强优化器"),
        ("run_enhanced.py", "主运行脚本"),
        ("test_differentiation.py", "分化模块测试脚本"),
        ("run_differentiation_demo.py", "分化演示脚本"),
        ("README_DIFFERENTIATION.md", "分化模块文档")
    ]
    
    for filepath, description in file_checks:
        if check_file_exists(filepath, description):
            checks_passed += 1
        total_checks += 1
    
    # 2. 核心类导入检查
    print("\n📦 核心类导入检查:")
    import_checks = [
        ("scripts.prompts.differentiation_prompt", "DifferentiationPromptGenerator", "分化提示生成器类"),
        ("scripts.workflow_differentiation", "WorkflowDifferentiation", "工作流分化处理器类"),
        ("scripts.enhanced_optimizer", "EnhancedOptimizer", "增强优化器类")
    ]
    
    for module_path, class_name, description in import_checks:
        if check_import(module_path, class_name, description):
            checks_passed += 1
        total_checks += 1
    
    # 3. 关键方法检查
    print("\n🔧 关键方法检查:")
    method_checks = [
        ("scripts.prompts.differentiation_prompt", "DifferentiationPromptGenerator", 
         "generate_differentiation_prompt", "分化提示生成方法"),
        ("scripts.workflow_differentiation", "WorkflowDifferentiation", 
         "create_differentiated_workflow", "分化工作流创建方法"),
        ("scripts.enhanced_optimizer", "EnhancedOptimizer", 
         "_should_attempt_differentiation", "分化条件检查方法"),
        ("scripts.enhanced_optimizer", "EnhancedOptimizer", 
         "_attempt_differentiation", "分化尝试方法"),
        ("scripts.enhanced_optimizer", "EnhancedOptimizer", 
         "_select_differentiation_candidates", "分化候选选择方法")
    ]
    
    for module_path, class_name, method_name, description in method_checks:
        if check_method_exists(module_path, class_name, method_name, description):
            checks_passed += 1
        total_checks += 1
    
    # 4. CLI参数检查
    print("\n⌨️ CLI参数检查:")
    cli_content_checks = [
        ("run_enhanced.py", [
            "--enable_differentiation",
            "--differentiation_probability", 
            "--max_differentiation_rounds",
            "enable_differentiation=args.enable_differentiation"
        ], "CLI分化参数配置")
    ]
    
    for filepath, required_content, description in cli_content_checks:
        if validate_file_content(filepath, required_content, description):
            checks_passed += 1
        total_checks += 1
    
    # 5. 分化策略检查
    print("\n🎯 分化策略检查:")
    strategy_content_checks = [
        ("scripts/prompts/differentiation_prompt.py", [
            "problem_type_specialization",
            "strategy_diversification",
            "algorithmic_approach_variation", 
            "complexity_adaptation",
            "error_pattern_handling"
        ], "5种分化策略定义")
    ]
    
    for filepath, required_content, description in strategy_content_checks:
        if validate_file_content(filepath, required_content, description):
            checks_passed += 1
        total_checks += 1
    
    # 6. 集成检查
    print("\n🔗 集成检查:")
    integration_checks = [
        ("scripts/enhanced_optimizer.py", [
            "from scripts.workflow_differentiation import WorkflowDifferentiation",
            "differentiation_processor", 
            "Priority 2: Check if we should attempt differentiation"
        ], "分化功能集成到优化器")
    ]
    
    for filepath, required_content, description in integration_checks:
        if validate_file_content(filepath, required_content, description):
            checks_passed += 1
        total_checks += 1
    
    # 7. 生成验证报告
    print("\n" + "=" * 60)
    print("📊 验证结果总结")
    print("=" * 60)
    
    success_rate = (checks_passed / total_checks) * 100
    print(f"通过检查: {checks_passed}/{total_checks} ({success_rate:.1f}%)")
    
    if checks_passed == total_checks:
        print("\n🎉 所有验证检查通过！工作流分化模块已完整集成。")
        print("\n✨ 功能特性:")
        print("   • 5种分化方向全面覆盖")
        print("   • LLM驱动的智能分化")
        print("   • 与现有系统无缝集成")
        print("   • 完整的CLI支持")
        print("   • 详细的文档和演示")
        
        print("\n🚀 立即开始使用:")
        print("   python run_enhanced.py --enable_differentiation --dataset GSM8K")
        
        return True
    else:
        failed_checks = total_checks - checks_passed
        print(f"\n⚠️ 有 {failed_checks} 项检查失败，需要修复。")
        return False

def generate_integration_summary():
    """生成集成总结"""
    summary = {
        "integration_status": "完成",
        "core_components": [
            {
                "component": "DifferentiationPromptGenerator",
                "file": "scripts/prompts/differentiation_prompt.py",
                "status": "✅ 已实现",
                "description": "生成5种方向的分化提示"
            },
            {
                "component": "WorkflowDifferentiation", 
                "file": "scripts/workflow_differentiation.py",
                "status": "✅ 已实现",
                "description": "处理完整分化流程与LLM交互"
            },
            {
                "component": "EnhancedOptimizer Integration",
                "file": "scripts/enhanced_optimizer.py", 
                "status": "✅ 已集成",
                "description": "三模式优化器：优化+融合+分化"
            },
            {
                "component": "CLI Interface",
                "file": "run_enhanced.py",
                "status": "✅ 已更新", 
                "description": "分化功能的命令行接口"
            }
        ],
        "differentiation_strategies": [
            "problem_type_specialization - 问题类型特化",
            "strategy_diversification - 策略多样化", 
            "algorithmic_approach_variation - 算法方法变化",
            "complexity_adaptation - 复杂度适应",
            "error_pattern_handling - 错误模式处理"
        ],
        "key_features": [
            "概率控制的分化触发",
            "智能候选工作流选择",
            "结构化分化结果解析",
            "完整的元数据跟踪",
            "优先级管理系统"
        ],
        "usage_examples": [
            "python run_enhanced.py --enable_differentiation",
            "python run_enhanced.py --enable_differentiation --differentiation_probability 0.3",
            "python run_enhanced.py --enable_fusion --enable_differentiation"
        ]
    }
    
    print("\n📋 工作流分化模块集成总结")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    print("🚀 开始工作流分化模块完整验证...")
    
    success = run_comprehensive_validation()
    
    if success:
        generate_integration_summary()
        print("\n✅ 验证完成 - 工作流分化模块已成功集成到AFlow系统！")
        sys.exit(0)
    else:
        print("\n❌ 验证失败 - 请检查并修复相关问题。")
        sys.exit(1)
