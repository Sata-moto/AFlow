#!/usr/bin/env python3
"""
测试分类文件加载和示例问题提取
"""
import json
import os

# 测试分类文件加载
classification_file = "/home/wx/AFlow/workspace/MBPP/workflows/problem_classifications.json"

print("=" * 80)
print("测试分类文件加载")
print("=" * 80)

with open(classification_file, 'r', encoding='utf-8') as f:
    classifications_data = json.load(f)

print(f"原始数据类型: {type(classifications_data)}")
print(f"原始数据键: {classifications_data.keys() if isinstance(classifications_data, dict) else 'N/A (list)'}")

# 处理不同格式
classifications = {}
if isinstance(classifications_data, dict):
    if 'problem_classifications' in classifications_data:
        print("检测到嵌套数组格式")
        for item in classifications_data['problem_classifications']:
            problem_id = str(item.get('problem_id', ''))
            category = item.get('category', 'unknown')
            if problem_id:
                classifications[problem_id] = category
    else:
        print("检测到字典格式")
        classifications = {str(k): v for k, v in classifications_data.items()}
elif isinstance(classifications_data, list):
    print("检测到纯数组格式")
    for item in classifications_data:
        problem_id = str(item.get('problem_id', ''))
        category = item.get('category', 'unknown')
        if problem_id:
            classifications[problem_id] = category

print(f"\n转换后的字典: {len(classifications)} 个问题")
print(f"示例 (前5个): {dict(list(classifications.items())[:5])}")

# 统计每个类别的问题数量
category_counts = {}
for cat in classifications.values():
    category_counts[cat] = category_counts.get(cat, 0) + 1

print(f"\n类别统计:")
for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {cat}: {count} 个问题")

# 测试查找特定类别的问题
target_category = "Mathematical Derivation"
category_problems = [
    problem_id for problem_id, cat in classifications.items() 
    if cat == target_category
]
print(f"\n类别 '{target_category}' 的问题: {len(category_problems)} 个")
print(f"问题ID列表: {category_problems[:5]}...")

# 测试加载示例问题
validation_file = "/home/wx/AFlow/data/datasets/mbpp_validate.jsonl"
print(f"\n从验证文件加载示例问题:")
print(f"验证文件: {validation_file}")

if os.path.exists(validation_file):
    import random
    selected_ids = random.sample(category_problems, min(5, len(category_problems)))
    print(f"随机选择的ID: {selected_ids}")
    
    example_problems = []
    with open(validation_file, 'r', encoding='utf-8') as f:
        for line in f:
            problem = json.loads(line)
            problem_id = str(problem.get('id') or problem.get('task_id') or problem.get('problem_id', ''))
            if problem_id in selected_ids:
                example_problems.append(problem)
                print(f"  找到问题 {problem_id}: {problem.get('prompt', problem.get('text', ''))[:80]}...")
                if len(example_problems) >= 5:
                    break
    
    print(f"\n成功加载 {len(example_problems)} 个示例问题")
else:
    print(f"验证文件不存在!")

print("=" * 80)
