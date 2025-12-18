#!/usr/bin/env python3
"""
测试脚本：验证问题分类加载功能
"""
import json
import os

def test_classification_loading(dataset="DROP"):
    """测试分类文件加载"""
    root_path = f"workspace/{dataset}"
    classification_file = f"{root_path}/workflows/problem_classifications.json"
    
    print(f"测试数据集: {dataset}")
    print(f"分类文件路径: {classification_file}")
    print(f"文件存在: {os.path.exists(classification_file)}")
    print()
    
    if not os.path.exists(classification_file):
        print("❌ 文件不存在")
        return
    
    try:
        with open(classification_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ 文件加载成功")
        print(f"数据类型: {type(data)}")
        print(f"顶层键: {list(data.keys()) if isinstance(data, dict) else 'N/A (list)'}")
        print()
        
        # 解析问题分类映射
        problem_map = {}
        if isinstance(data, dict) and "problem_classifications" in data:
            classifications = data["problem_classifications"]
            for item in classifications:
                if isinstance(item, dict):
                    problem_id = str(item.get("problem_id", ""))
                    category = item.get("category", "unknown")
                    if problem_id:
                        problem_map[problem_id] = category
        
        print(f"✓ 问题总数: {len(problem_map)}")
        
        # 统计类别
        category_counts = {}
        for category in problem_map.values():
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"✓ 类别数量: {len(category_counts)}")
        print()
        print("类别分布:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count} 问题")
        print()
        
        # 显示前5个问题的分类
        print("前5个问题示例:")
        for i, (pid, cat) in enumerate(list(problem_map.items())[:5]):
            print(f"  问题 {pid}: {cat}")
        print()
        
        # 测试log.json格式
        log_path = f"{root_path}/workflows/round_1/log.json"
        if os.path.exists(log_path):
            print(f"✓ log.json 存在: {log_path}")
            with open(log_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            if isinstance(log_data, dict) and "execution_logs" in log_data:
                entries = log_data["execution_logs"]
                print(f"  格式: 带元数据的字典")
                print(f"  日志条目数: {len(entries)}")
            elif isinstance(log_data, list):
                entries = log_data
                print(f"  格式: 直接列表")
                print(f"  日志条目数: {len(entries)}")
            else:
                entries = []
                print(f"  格式: 未知")
            
            if entries:
                first_entry = entries[0]
                print(f"  第一条日志键: {list(first_entry.keys()) if isinstance(first_entry, dict) else 'N/A'}")
                if isinstance(first_entry, dict):
                    has_category = 'category' in first_entry
                    has_problem_id = 'problem_id' in first_entry
                    has_score = 'score' in first_entry
                    print(f"  包含 category: {has_category}")
                    print(f"  包含 problem_id: {has_problem_id}")
                    print(f"  包含 score: {has_score}")
                    
                    if has_category:
                        print(f"  示例 category: {first_entry['category']}")
                    if has_problem_id:
                        print(f"  示例 problem_id: {first_entry['problem_id']}")
            print()
        else:
            print(f"❌ log.json 不存在: {log_path}")
            print()
        
        print("✅ 测试通过")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "DROP"
    test_classification_loading(dataset)
