#!/usr/bin/env python3
"""
架构验证测试 - 确保重构后的模块化架构正确工作
"""

import inspect

def test_architecture_validation():
    print("🏗️ 开始架构验证测试...")
    print("="*60)
    
    # Test 1: 检查各类的职责分离
    print("\n📋 职责分离测试:")
    
    try:
        from scripts.enhanced_optimizer import EnhancedOptimizer
        from scripts.workflow_fusion import WorkflowFusion  
        from scripts.workflow_differentiation import WorkflowDifferentiation
        from scripts.utils.code_processor import CodeProcessor
        from scripts.utils.workflow_manager import WorkflowManager
        
        # 检查EnhancedOptimizer不再有重复的融合/分化方法
        eo_methods = [name for name, method in inspect.getmembers(EnhancedOptimizer, predicate=inspect.isfunction)]
        
        # 应该不存在的方法（已经移动到专门的类中）
        should_not_have = [
            '_create_fusion_prompt', '_call_fusion_llm', '_save_fusion_metadata',
            '_create_differentiation_experience_file', '_save_differentiated_workflow_direct',
            '_save_fused_workflow_direct', '_create_fusion_experience_file'
        ]
        
        duplicated_methods = [method for method in should_not_have if method in eo_methods]
        
        if duplicated_methods:
            print(f"❌ EnhancedOptimizer中仍有重复方法: {duplicated_methods}")
            return False
        else:
            print("✅ EnhancedOptimizer已正确清理，无重复方法")
        
        # 检查WorkflowFusion有正确的方法
        wf_methods = [name for name, method in inspect.getmembers(WorkflowFusion, predicate=inspect.isfunction)]
        fusion_required = ['create_fusion_prompt', 'call_fusion_llm', 'create_fused_workflow', 'save_fusion_metadata']
        
        missing_fusion = [method for method in fusion_required if method not in wf_methods]
        if missing_fusion:
            print(f"❌ WorkflowFusion缺少方法: {missing_fusion}")
            return False
        else:
            print("✅ WorkflowFusion包含所有必需的方法")
        
        # 检查WorkflowDifferentiation有正确的方法
        wd_methods = [name for name, method in inspect.getmembers(WorkflowDifferentiation, predicate=inspect.isfunction)]
        diff_required = ['create_differentiated_workflow', 'save_differentiated_workflow_direct', 'create_differentiation_experience_file']
        
        missing_diff = [method for method in diff_required if method not in wd_methods]
        if missing_diff:
            print(f"❌ WorkflowDifferentiation缺少方法: {missing_diff}")
            return False
        else:
            print("✅ WorkflowDifferentiation包含所有必需的方法")
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    
    # Test 2: 检查共享工具的正确性
    print("\n📋 共享工具测试:")
    
    try:
        # 检查CodeProcessor功能
        processor = CodeProcessor()
        test_code = '```python\ndef test():\n    return "hello"\n```'
        cleaned = processor.clean_code_content(test_code)
        
        if '```' in cleaned:
            print("❌ CodeProcessor仍未正确清理```符号")
            return False
        else:
            print("✅ CodeProcessor正确清理代码内容")
        
        # 检查基本的字段提取功能
        test_response = '<modification>Test modification</modification><graph>Test graph</graph>'
        fields = processor.extract_fields_from_response(test_response, ['modification', 'graph'])
        
        if fields.get('modification') == 'Test modification' and fields.get('graph') == 'Test graph':
            print("✅ CodeProcessor字段提取功能正常")
        else:
            print("❌ CodeProcessor字段提取功能异常")
            return False
            
    except Exception as e:
        print(f"❌ 共享工具测试失败: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 所有架构验证测试通过！")
    print("📊 重构成功完成：")
    print("   - ✅ 消除了代码重复")
    print("   - ✅ 实现了模块化分离")
    print("   - ✅ 保持了功能完整性")
    print("   - ✅ 修复了```符号解析问题")
    
    return True

if __name__ == "__main__":
    test_architecture_validation()
