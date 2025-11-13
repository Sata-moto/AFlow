#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修改后的 DROP prompt 格式
"""

import json

# 读取一个示例问题
with open('/home/wx/AFlow/data/datasets/drop_validate.jsonl', 'r') as f:
    sample = json.loads(f.readline())

print("="*80)
print("原始 context:")
print("="*80)
print(sample['context'][:300])
print("...")

print("\n" + "="*80)
print("新的 input_text 格式:")
print("="*80)

# 模拟新的格式
input_text = "IMPORTANT: Provide ONLY the final answer using digits/numbers (e.g., 5, not 'five'). Do not include explanations, reasoning, or extra text. Just give the direct numerical answer.\n\n" + sample['context']

print(input_text[:500])
print("...")

print("\n" + "="*80)
print("预期答案:")
print("="*80)
print(f"ref_text: {sample['ref_text']}")

print("\n" + "="*80)
print("提示改进说明:")
print("="*80)
print("✓ 要求放在最前面,避免干扰 Passage 和 Question")
print("✓ 明确要求使用数字 (e.g., 5, not 'five')")
print("✓ 强调只输出最终答案,不要解释")
print("✓ 这样可以减少模型输出 'One', 'Two' 等英文词汇导致的判错")
