#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合数据集结果分析脚本（正确版本）

正确处理 DROP 数据集的 F1 分数计算
- DROP: F1 Score (0-1 浮点数)
- MATH: Exact Match (0 或 1)
- GSM8K: Exact Match (0 或 1)
"""

import json
import pandas as pd
import sys
from pathlib import Path

def analyze_mixed_results(csv_path: str, dataset_path: str):
    """
    分析混合数据集的详细结果
    
    Args:
        csv_path: CSV 结果文件路径
        dataset_path: 数据集 JSONL 文件路径
    """
    # 读取 CSV 结果
    df = pd.read_csv(csv_path)
    
    # 读取数据集以获取 source_dataset 信息
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    if len(df) != len(dataset):
        print(f"警告: CSV 行数 ({len(df)}) 与数据集样本数 ({len(dataset)}) 不匹配")
        return None
    
    # 添加 source_dataset 列
    df['source_dataset'] = [item['source_dataset'] for item in dataset]
    
    # 分析结果
    results = {
        'total_samples': len(df),
        'total_score': df['score'].sum(),
        'avg_score': df['score'].mean(),
        'datasets': {}
    }
    
    print("=" * 80)
    print("混合数据集详细分析报告")
    print("=" * 80)
    print(f"\nCSV 文件: {csv_path}")
    print(f"数据集文件: {dataset_path}")
    
    for ds_name in ['DROP', 'MATH', 'GSM8K']:
        subset = df[df['source_dataset'] == ds_name]
        if len(subset) == 0:
            continue
        
        count = len(subset)
        total_score = subset['score'].sum()
        avg_score = subset['score'].mean()
        
        dataset_info = {
            'count': count,
            'total_score': total_score,
            'avg_score': avg_score,
        }
        
        print(f"\n{'=' * 80}")
        print(f"【{ds_name}】")
        print(f"{'=' * 80}")
        print(f"样本数: {count}")
        
        if ds_name == 'DROP':
            # DROP 使用 F1 Score
            print(f"评分方式: F1 Score (0-1 浮点数)")
            print(f"平均 F1: {avg_score:.4f}")
            print(f"总 F1 累加: {total_score:.4f}")
            
            # 统计 F1 分布
            high_f1 = (subset['score'] >= 0.8).sum()
            mid_f1 = ((subset['score'] >= 0.5) & (subset['score'] < 0.8)).sum()
            low_f1 = (subset['score'] < 0.5).sum()
            
            print(f"\nF1 分布:")
            print(f"  高分 (≥0.8): {high_f1} ({high_f1/count*100:.1f}%)")
            print(f"  中分 (0.5-0.8): {mid_f1} ({mid_f1/count*100:.1f}%)")
            print(f"  低分 (<0.5): {low_f1} ({low_f1/count*100:.1f}%)")
            
            dataset_info.update({
                'type': 'F1 Score',
                'high_f1': high_f1,
                'mid_f1': mid_f1,
                'low_f1': low_f1,
            })
            
        else:
            # MATH 和 GSM8K 使用精确匹配
            solved = (subset['score'] >= 0.99).sum()
            accuracy = solved / count if count > 0 else 0
            
            print(f"评分方式: Exact Match (0 或 1)")
            print(f"正确题数: {solved} / {count}")
            print(f"准确率: {accuracy*100:.2f}%")
            print(f"贡献分数: {total_score:.4f}")
            
            dataset_info.update({
                'type': 'Exact Match',
                'solved': solved,
                'accuracy': accuracy,
            })
        
        # 成本统计
        avg_cost = subset['cost'].mean()
        total_cost = subset['cost'].sum()
        print(f"\n平均成本: ${avg_cost:.6f}")
        print(f"总成本: ${total_cost:.6f}")
        
        dataset_info.update({
            'avg_cost': avg_cost,
            'total_cost': total_cost,
        })
        
        results['datasets'][ds_name] = dataset_info
    
    # 总体统计
    print(f"\n{'=' * 80}")
    print(f"【总体统计】")
    print(f"{'=' * 80}")
    print(f"总样本数: {results['total_samples']}")
    print(f"总得分: {results['total_score']:.4f}")
    print(f"平均分: {results['avg_score']:.5f}")
    print(f"总成本: ${df['cost'].sum():.6f}")
    
    # 计算各数据集对总分的贡献
    print(f"\n各数据集对总分的贡献:")
    for ds_name in ['DROP', 'MATH', 'GSM8K']:
        if ds_name in results['datasets']:
            info = results['datasets'][ds_name]
            contribution = info['total_score'] / results['total_score'] * 100
            print(f"  {ds_name}: {info['total_score']:.4f} ({contribution:.1f}%)")
    
    print("=" * 80)
    
    return results


def compare_experiments(baseline_csv, diff_csv, full_csv, dataset_path):
    """
    比较三个实验的结果
    """
    print("\n\n" + "=" * 80)
    print("实验对比分析")
    print("=" * 80)
    
    experiments = {
        'Baseline (仅优化)': baseline_csv,
        'With Differentiation': diff_csv,
        'Full Framework': full_csv,
    }
    
    all_results = {}
    
    for name, csv_path in experiments.items():
        if not Path(csv_path).exists():
            print(f"\n跳过 {name}: 文件不存在")
            continue
        
        print(f"\n\n{'#' * 80}")
        print(f"# {name}")
        print(f"{'#' * 80}")
        
        results = analyze_mixed_results(csv_path, dataset_path)
        if results:
            all_results[name] = results
    
    # 对比表格
    if len(all_results) > 1:
        print("\n\n" + "=" * 80)
        print("实验对比汇总表")
        print("=" * 80)
        
        print(f"\n{'指标':<30} ", end='')
        for name in all_results.keys():
            print(f"{name:<25} ", end='')
        print()
        print("-" * 110)
        
        # 总体平均分
        print(f"{'总体平均分':<30} ", end='')
        for results in all_results.values():
            print(f"{results['avg_score']:.4f}                    ", end='')
        print()
        
        # 各数据集得分
        for ds_name in ['DROP', 'MATH', 'GSM8K']:
            if ds_name == 'DROP':
                print(f"{f'{ds_name} (平均F1)':<30} ", end='')
                for results in all_results.values():
                    if ds_name in results['datasets']:
                        score = results['datasets'][ds_name]['avg_score']
                        print(f"{score:.4f}                    ", end='')
                    else:
                        print(f"{'N/A':<25} ", end='')
                print()
            else:
                print(f"{f'{ds_name} (准确率)':<30} ", end='')
                for results in all_results.values():
                    if ds_name in results['datasets']:
                        acc = results['datasets'][ds_name]['accuracy']
                        print(f"{acc*100:.2f}%                   ", end='')
                    else:
                        print(f"{'N/A':<25} ", end='')
                print()
        
        print("=" * 110)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  单个实验分析:")
        print("    python analyze_mixed_results_correct.py <csv_path> <dataset_path>")
        print("  多实验对比:")
        print("    python analyze_mixed_results_correct.py compare <baseline_csv> <diff_csv> <full_csv> <dataset_path>")
        print("\n示例:")
        print("  python scripts/analyze_mixed_results_correct.py \\")
        print("    logs/MIXED/workflows_test/round_10/0.70363_20251104_171033.csv \\")
        print("    data/datasets/mixed_test.jsonl")
        sys.exit(1)
    
    if sys.argv[1] == 'compare':
        if len(sys.argv) < 6:
            print("对比模式需要4个CSV文件路径和1个数据集路径")
            sys.exit(1)
        compare_experiments(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        if len(sys.argv) < 3:
            print("需要提供 CSV 路径和数据集路径")
            sys.exit(1)
        analyze_mixed_results(sys.argv[1], sys.argv[2])
