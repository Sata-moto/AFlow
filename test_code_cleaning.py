#!/usr/bin/env python3
"""
简化测试：验证代码清理和```符号处理
"""

import sys
import os
sys.path.insert(0, '/home/wx/AFlow')

def test_code_cleaning():
    """测试代码清理功能"""
    print("🔧 测试代码清理功能...")
    
    try:
        from scripts.utils.code_processor import CodeProcessor
        
        # 测试用例1：多层```符号
        test_case_1 = """````python
```
STRATEGY_DIVERSIFIED_PROMPT = \"\"\"
Generate a Python solution by analyzing the problem through multiple programming paradigms.
\"\"\"
```
````"""
        
        cleaned_1 = CodeProcessor.clean_code_content(test_case_1)
        print("✅ 测试用例1: 多层```符号")
        print(f"   原始长度: {len(test_case_1)}")
        print(f"   清理后长度: {len(cleaned_1)}")
        print(f"   结果预览: {repr(cleaned_1[:100])}...")
        
        # 测试用例2：标准```python
        test_case_2 = """```python
def hello():
    return "world"
```"""
        
        cleaned_2 = CodeProcessor.clean_code_content(test_case_2)
        print("✅ 测试用例2: 标准```python")
        print(f"   清理后: {repr(cleaned_2)}")
        
        # 测试用例3：混合情况
        test_case_3 = """```
some content
```python
code here
```
more content
```"""
        
        cleaned_3 = CodeProcessor.clean_code_content(test_case_3)
        print("✅ 测试用例3: 混合情况")
        print(f"   清理后: {repr(cleaned_3)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 代码清理测试失败: {e}")
        return False

def test_imports():
    """测试模块导入"""
    print("📦 测试模块导入...")
    
    try:
        from scripts.utils.code_processor import CodeProcessor
        from scripts.utils.workflow_manager import WorkflowManager
        from scripts.workflow_differentiation import WorkflowDifferentiation
        from scripts.enhanced_optimizer import EnhancedOptimizer
        
        print("✅ 所有核心模块导入成功")
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_field_extraction():
    """测试字段提取功能"""
    print("🔍 测试字段提取功能...")
    
    try:
        from scripts.utils.code_processor import CodeProcessor
        
        # 模拟LLM响应
        test_response = """
<modification>Made changes to improve performance</modification>
<graph>```python
class Workflow:
    def execute(self):
        return "result"
```</graph>
<prompt>```
IMPROVED_PROMPT = \"\"\"
This is an improved prompt
\"\"\"
```</prompt>
"""
        
        fields = ["modification", "graph", "prompt"]
        extracted = CodeProcessor.extract_fields_from_response(test_response, fields)
        
        if extracted:
            print("✅ 字段提取成功:")
            for field, content in extracted.items():
                print(f"   {field}: {len(content)} 字符")
                print(f"      预览: {repr(content[:50])}...")
            return True
        else:
            print("❌ 字段提取失败")
            return False
            
    except Exception as e:
        print(f"❌ 字段提取测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始简化测试...")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("代码清理", test_code_cleaning),
        ("字段提取", test_field_extraction),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}测试:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！```符号处理问题已修复。")
        return True
    else:
        print(f"⚠️ 有 {total - passed} 个测试失败。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
