#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试问题示例检索功能
"""

import json
import sys
sys.path.insert(0, '/home/wx/AFlow')

# 加载分类数据
print("=" * 80)
print("Testing Problem Example Retrieval")
print("=" * 80)

print("\n1. Loading classification data...")
with open('workspace/MIXED/workflows/problem_classifications.json', 'r') as f:
    classifications = json.load(f)
print(f"   ✓ Loaded {len(classifications['problem_classifications'])} classifications")

# 统计类别
from collections import Counter
categories = Counter(pc['category'] for pc in classifications['problem_classifications'])
print(f"   Categories: {dict(categories)}")

print("\n2. Loading validation data...")
validation_data = []
with open('data/datasets/mixed_validate.jsonl', 'r') as f:
    for line in f:
        validation_data.append(json.loads(line.strip()))
print(f"   ✓ Loaded {len(validation_data)} validation problems")

print("\n3. Testing get_problems_by_category logic...")

# 找到 Mathematical Derivation 类别的问题ID
category = 'Mathematical Derivation'
problem_ids = []
for pc in classifications['problem_classifications']:
    if pc['category'] == category:
        problem_ids.append(pc['problem_id'])

print(f"   Found {len(problem_ids)} problems in '{category}' category")

# 创建问题映射
problem_map = {}
for problem in validation_data:
    pid = problem.get('id') or problem.get('problem_id') or problem.get('index')
    if pid is not None:
        problem_map[pid] = problem
        problem_map[str(pid)] = problem
        try:
            if isinstance(pid, str):
                problem_map[int(pid)] = problem
        except (ValueError, TypeError):
            pass

print(f"   Built problem map with {len(validation_data)} unique problems")

# 提取示例
examples = []
limit = 3
for pid in problem_ids:
    problem = None
    if pid in problem_map:
        problem = problem_map[pid]
    elif str(pid) in problem_map:
        problem = problem_map[str(pid)]
    else:
        try:
            int_pid = int(pid)
            problem = problem_map.get(int_pid)
        except (ValueError, TypeError):
            pass
    
    if problem and problem not in examples:
        examples.append(problem)
        if len(examples) >= limit:
            break

print(f"\n4. Retrieved Examples:")
print(f"   Total: {len(examples)} examples")
for i, ex in enumerate(examples, 1):
    source = ex.get('source_dataset', 'unknown')
    pid = ex.get('id', 'unknown')
    # 获取问题内容预览
    content = ex.get('context') or ex.get('problem') or ex.get('prompt', '')
    preview = content[:80] + '...' if len(content) > 80 else content
    print(f"   {i}. [{source}] ID={pid}")
    print(f"      {preview}")

print("\n" + "=" * 80)
if len(examples) == limit:
    print("✓ Test PASSED: Successfully retrieved example problems")
else:
    print(f"✗ Test FAILED: Expected {limit} examples, got {len(examples)}")
print("=" * 80)
