#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIXED Dataset Benchmark

混合数据集包含三种不同类型的任务：
- DROP: 阅读理解和数值推理
- MATH: 数学问题求解
- GSM8K: 小学数学应用题

该 benchmark 能够根据样本的 source_dataset 字段
自动路由到对应的评估方法。
"""

import json
from typing import Dict, Any, Callable, List, Tuple
from collections import Counter

from benchmarks.drop import DROPBenchmark
from benchmarks.math import MATHBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class MixedBenchmark(BaseBenchmark):
    """
    混合数据集 Benchmark
    
    根据每个样本的 source_dataset 字段，自动路由到对应的子 benchmark 进行评估。
    支持的数据集：DROP、MATH、GSM8K
    """
    
    def __init__(self, name: str, file_path: str, log_path: str, solved_threshold: float = 0.5):
        super().__init__(name, file_path, log_path, solved_threshold)
        
        # 初始化子 benchmarks（使用相同的文件路径、日志路径和阈值）
        self.drop_benchmark = DROPBenchmark(
            name="DROP",
            file_path=file_path,
            log_path=log_path,
            solved_threshold=solved_threshold
        )
        
        self.math_benchmark = MATHBenchmark(
            name="MATH",
            file_path=file_path,
            log_path=log_path,
            solved_threshold=solved_threshold
        )
        
        self.gsm8k_benchmark = GSM8KBenchmark(
            name="GSM8K",
            file_path=file_path,
            log_path=log_path,
            solved_threshold=solved_threshold
        )
        
        logger.info(f"Initialized MixedBenchmark with sub-benchmarks: DROP, MATH, GSM8K (threshold: {solved_threshold})")
    
    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        """
        Route problem to appropriate benchmark based on source_dataset.
        """
        source = problem.get('source_dataset', 'DROP')
        
        benchmark_map = {
            'DROP': self.drop_benchmark,
            'MATH': self.math_benchmark,
            'GSM8K': self.gsm8k_benchmark
        }
        
        benchmark = benchmark_map.get(source, self.drop_benchmark)
        return await benchmark.evaluate_problem(problem, graph)
    
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        """
        This method won't be called directly in MixedBenchmark since evaluate_problem
        is overridden to route to sub-benchmarks. Provided for interface compliance.
        """
        # Default to MATH benchmark's scoring if called directly
        return self.math_benchmark.calculate_score(expected_output, prediction)
    
    def get_result_columns(self) -> List[str]:
        """
        Return column names for results CSV.
        """
        return ["inputs", "prediction", "expected_output", "score", "cost"]


# 导出类（不是实例）
MIXED = MixedBenchmark
