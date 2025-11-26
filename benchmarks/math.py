#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MATH Benchmark - 高级数学问题评测
使用 LLM 进行语义评判
"""

from typing import Callable, List, Tuple
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class MATHBenchmark(BaseBenchmark):
    """MATH 数据集 benchmark"""
    
    def __init__(self, name: str, file_path: str, log_path: str, solved_threshold: float = 0.5):
        super().__init__(name, file_path, log_path, solved_threshold)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        """评估单个问题"""
        input_text = problem.get("problem", problem.get("question", ""))
        ground_truth = problem.get("solution", problem.get("answer", ""))

        try:
            output, cost = await self._generate_output(graph, input_text)
            
            # 直接使用原始输出进行 LLM 评判
            score, explanation = await self.llm_judge_answer(
                question=input_text,
                ground_truth=ground_truth,
                prediction=output,
                task_description="solve the advanced math problem and provide the correct answer"
            )

            if score == 0:
                self.log_mismatch(input_text, ground_truth, output, explanation, score)

            return input_text, output, ground_truth, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            self.log_mismatch(input_text, ground_truth, str(e), f"Error: {e}", 0.0)
            return input_text, str(e), ground_truth, 0.0, 0.0

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        """
        使用 LLM 评判代替硬匹配
        注意：这个方法是为了兼容父类接口，实际评判在 evaluate_problem 中完成
        """
        return 0.0, "Use llm_judge_answer instead"
    
    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "ground_truth", "score", "cost"]
