#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建混合数据集以测试多任务场景下的工作流性能

混合数据集包含三种不同类型的任务：
- DROP: 阅读理解和数值推理
- MATH: 数学问题求解
- GSM8K: 小学数学应用题

这种混合模拟了现实世界中任务多样性的极端情况，
用于验证工作流生成框架在多样化任务上的泛化能力。
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def load_jsonl(filepath: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """保存为 JSONL 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples to {filepath}")


def add_dataset_label(data: List[Dict], dataset_name: str) -> List[Dict]:
    """为每条数据添加数据集标签"""
    for item in data:
        item['source_dataset'] = dataset_name
    return data


def create_mixed_dataset(
    datasets_dir: str = "data/datasets",
    output_dir: str = "data/datasets",
    validate_samples_per_dataset: int = 29,  # 86 / 3 ≈ 29
    test_samples_per_dataset: int = 38,      # 113 / 3 ≈ 38
    seed: int = 42
):
    """
    创建混合数据集
    
    Args:
        datasets_dir: 原始数据集目录
        output_dir: 输出目录
        validate_samples_per_dataset: 每个数据集在验证集中的样本数
        test_samples_per_dataset: 每个数据集在测试集中的样本数
        seed: 随机种子
    """
    random.seed(seed)
    datasets_dir = Path(datasets_dir)
    output_dir = Path(output_dir)
    
    # 数据集配置
    datasets = ['drop', 'math', 'gsm8k']
    
    print("="*60)
    print("Creating Mixed Dataset")
    print("="*60)
    print(f"Samples per dataset (validate): {validate_samples_per_dataset}")
    print(f"Samples per dataset (test): {test_samples_per_dataset}")
    print(f"Total samples (validate): {validate_samples_per_dataset * len(datasets)}")
    print(f"Total samples (test): {test_samples_per_dataset * len(datasets)}")
    print(f"Random seed: {seed}")
    print("="*60)
    
    # 创建验证集
    print("\n[1/2] Creating Validate Set...")
    validate_mixed = []
    
    for dataset in datasets:
        # 加载原始数据
        filepath = datasets_dir / f"{dataset}_validate.jsonl"
        data = load_jsonl(filepath)
        print(f"  {dataset.upper()}: loaded {len(data)} samples")
        
        # 随机采样
        if len(data) >= validate_samples_per_dataset:
            sampled = random.sample(data, validate_samples_per_dataset)
        else:
            print(f"  Warning: {dataset} has only {len(data)} samples, using all")
            sampled = data
        
        # 添加数据集标签
        sampled = add_dataset_label(sampled, dataset.upper())
        validate_mixed.extend(sampled)
        print(f"  {dataset.upper()}: sampled {len(sampled)} samples")
    
    # 打乱验证集
    random.shuffle(validate_mixed)
    
    # 保存验证集
    output_file = output_dir / "mixed_validate.jsonl"
    save_jsonl(validate_mixed, str(output_file))
    
    # 创建测试集
    print("\n[2/2] Creating Test Set...")
    test_mixed = []
    
    for dataset in datasets:
        # 加载原始数据
        filepath = datasets_dir / f"{dataset}_test.jsonl"
        data = load_jsonl(filepath)
        print(f"  {dataset.upper()}: loaded {len(data)} samples")
        
        # 随机采样
        if len(data) >= test_samples_per_dataset:
            sampled = random.sample(data, test_samples_per_dataset)
        else:
            print(f"  Warning: {dataset} has only {len(data)} samples, using all")
            sampled = data
        
        # 添加数据集标签
        sampled = add_dataset_label(sampled, dataset.upper())
        test_mixed.extend(sampled)
        print(f"  {dataset.upper()}: sampled {len(sampled)} samples")
    
    # 打乱测试集
    random.shuffle(test_mixed)
    
    # 保存测试集
    output_file = output_dir / "mixed_test.jsonl"
    save_jsonl(test_mixed, str(output_file))
    
    # 统计信息
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    for split, data in [("Validate", validate_mixed), ("Test", test_mixed)]:
        print(f"\n{split} Set:")
        print(f"  Total samples: {len(data)}")
        
        # 按数据集统计
        dataset_counts = {}
        for item in data:
            ds = item.get('source_dataset', 'Unknown')
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        
        for ds, count in sorted(dataset_counts.items()):
            print(f"  {ds}: {count} samples ({count/len(data)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("Mixed Dataset Created Successfully!")
    print("="*60)


if __name__ == "__main__":
    create_mixed_dataset()
