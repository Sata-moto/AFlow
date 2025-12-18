import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from scripts.logs import logger
from scripts.utils.common import write_json_file
from scripts.async_llm import AsyncLLM


class BaseBenchmark(ABC):
    def __init__(self, name: str, file_path: str, log_path: str, solved_threshold: float = 0.5):
        self.name = name
        self.file_path = file_path
        self.log_path = log_path
        self.solved_threshold = solved_threshold  # Threshold for considering a problem as solved
        self.judge_llm = None  # Will be initialized lazily
        self.problem_classifications = self._load_problem_classifications()  # Load problem categories

    PASS = "PASS"
    FAIL = "FAIL"
    
    def _create_judge_llm(self):
        """创建用于评判的 LLM 实例（使用 gpt-4o-mini）"""
        if self.judge_llm is None:
            self.judge_llm = AsyncLLM("gpt-4o-mini")
        return self.judge_llm
    
    def _load_problem_classifications(self) -> dict:
        """
        加载问题分类信息
        
        Returns:
            dict: {problem_id: category}
        """
        # 尝试从 log_path 的父目录找 problem_classifications.json
        classification_file = None
        
        # 方式1: log_path/../../problem_classifications.json (workspace/DATASET/workflows/round_X -> workspace/DATASET/)
        candidate1 = os.path.join(os.path.dirname(os.path.dirname(self.log_path)), "problem_classifications.json")
        if os.path.exists(candidate1):
            classification_file = candidate1
        else:
            # 方式2: log_path/../problem_classifications.json (workspace/DATASET/workflows/)
            candidate2 = os.path.join(os.path.dirname(self.log_path), "problem_classifications.json")
            if os.path.exists(candidate2):
                classification_file = candidate2
            else:
                # 方式3: log_path/../../workflows/problem_classifications.json
                candidate3 = os.path.join(os.path.dirname(os.path.dirname(self.log_path)), "workflows", "problem_classifications.json")
                if os.path.exists(candidate3):
                    classification_file = candidate3
        
        if classification_file and os.path.exists(classification_file):
            try:
                with open(classification_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理新的文件格式：{"problem_classifications": [{"problem_id": ..., "category": ...}, ...]}
                if isinstance(data, dict) and "problem_classifications" in data:
                    classifications = data["problem_classifications"]
                    problem_map = {}
                    for item in classifications:
                        if isinstance(item, dict):
                            problem_id = str(item.get("problem_id", ""))
                            category = item.get("category", "unknown")
                            if problem_id:
                                problem_map[problem_id] = category
                    logger.debug(f"Loaded {len(problem_map)} problem classifications from {classification_file}")
                    return problem_map
                elif isinstance(data, list):
                    # 如果直接是数组
                    problem_map = {}
                    for item in data:
                        if isinstance(item, dict):
                            problem_id = str(item.get("problem_id", ""))
                            category = item.get("category", "unknown")
                            if problem_id:
                                problem_map[problem_id] = category
                    logger.debug(f"Loaded {len(problem_map)} problem classifications from {classification_file}")
                    return problem_map
                elif isinstance(data, dict):
                    # 旧格式：{problem_id: category}
                    problem_map = {str(k): v for k, v in data.items() 
                                 if not k.startswith("categories") and not k.startswith("category_descriptions")}
                    logger.debug(f"Loaded {len(problem_map)} problem classifications from {classification_file}")
                    return problem_map
                else:
                    logger.warning(f"Unexpected format in classification file: {type(data)}")
                    return {}
            except Exception as e:
                logger.warning(f"Failed to load problem classifications from {classification_file}: {e}")
                return {}
        else:
            logger.debug(f"No problem_classifications.json found (checked: {candidate1}, {candidate2}, {candidate3})")
            return {}
    
    def _get_problem_category(self, problem_id: Any) -> str:
        """
        获取问题的类别
        
        Args:
            problem_id: 问题ID
            
        Returns:
            str: 类别名称，如果未找到则返回 "unknown"
        """
        if not self.problem_classifications:
            return "unknown"
        
        # 尝试不同的ID格式
        str_id = str(problem_id)
        
        # 直接查找
        if str_id in self.problem_classifications:
            return self.problem_classifications[str_id]
        
        # 尝试整数格式
        try:
            int_id = int(problem_id)
            if int_id in self.problem_classifications:
                return self.problem_classifications[int_id]
            if str(int_id) in self.problem_classifications:
                return self.problem_classifications[str(int_id)]
        except (ValueError, TypeError):
            pass
        
        return "unknown"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            index = 0
            async for line in file:
                problem = json.loads(line)
                # 添加 _index 字段，用于生成统一的 problem_id 格式
                problem['_index'] = index
                data.append(problem)
                index += 1
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        t_cost = df["cost"].max()
        a_cost = t_cost / len(df) if len(df) > 0 else 0
        
        # Extract solved problems (problems with score >= solved_threshold)
        solved_problems = set()
        for i, result in enumerate(results):
            # Assume score is in the 4th position (index 3) based on get_result_columns
            if len(result) > 3 and result[3] >= self.solved_threshold:  # score >= threshold
                solved_problems.add(i)  # Use index as problem identifier
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Solved {len(solved_problems)} out of {len(results)} problems (threshold: {self.solved_threshold})")
        
        return avg_score, a_cost, t_cost, solved_problems

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        judge_explanation: str,
        score: float = 0.0,
        problem_id: Any = None,
        category: str = "unknown",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "score": score,
            "judge_explanation": judge_explanation,
            "problem_id": problem_id,
            "category": category,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    # Handle different log.json formats
                    if isinstance(data, dict):
                        # Differentiation/fusion format: {"differentiation_metadata": {...}, "execution_logs": [...]}
                        if "execution_logs" in data:
                            data["execution_logs"].append(log_data)
                            write_json_file(log_file, data, encoding="utf-8", indent=4)
                            return
                        else:
                            # Convert old dict format to list
                            data = []
                    elif not isinstance(data, list):
                        # Unexpected format, reset to empty list
                        data = []
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)
    
    async def llm_judge_answer(
        self, 
        question: str, 
        ground_truth: str, 
        prediction: str,
        task_description: str = "answer the question correctly"
    ) -> Tuple[float, str]:
        """
        使用 LLM 进行语义评判，替代硬匹配
        带重试机制以避免随机失败
        
        Args:
            question: 原始问题
            ground_truth: 正确答案（可能包含多个用 | 分隔的答案）
            prediction: 模型预测答案
            task_description: 任务描述（用于提示词）
            
        Returns:
            (score, explanation) - 分数（1.0 正确，0.0 错误）和评判理由
        """
        # 确保 judge_llm 已初始化
        if self.judge_llm is None:
            self._create_judge_llm()
        
        # 处理多个可能的正确答案
        acceptable_answers = [ans.strip() for ans in str(ground_truth).split("|") if ans.strip()]
        
        judge_prompt = f"""You are a answer verification judge. Your ONLY task is to check if the model's answer matches any of the acceptable answers.

**CRITICAL INSTRUCTIONS:**
1. DO NOT solve the question yourself
2. DO NOT think about what the correct answer should be
3. ONLY compare the model's answer with the provided acceptable answer(s)
4. The acceptable answers are **ALREADY CORRECT** - **just check if the model's answer matches them**

**Acceptable Answer(s) (Ground Truth):** 
{' OR '.join(acceptable_answers)}

**Model's Answer to Evaluate:** 
{prediction}

Question (**for context only**):
{question}

**Matching Rules:**
- Numerical equivalence: "5" = "five" = "5.0" = "5 years"
- Case insensitive: "Paris" = "paris" = "PARIS"
- Whitespace/punctuation: "twenty-two" = "twenty two" = "22"
- Semantic equivalence: "the French" = "French" = "France"
- If multiple acceptable answers exist (separated by OR), matching ANY ONE is correct
- Partial matches: If the model's answer CONTAINS the acceptable answer as a key component, it may be correct
- Extra explanation is OK: "The answer is 5" matches acceptable answer "5"

**Examples:**
- Acceptable: "66", Model: "66 yards" → CORRECT (contains the number)
- Acceptable: "2|3", Model: "2" → CORRECT (matches one option)
- Acceptable: "French", Model: "The French had more troops" → CORRECT (contains the answer)
- Acceptable: "22", Model: "30 yards" → WRONG (different number)

**Your Task:**
Compare the model's answer with the acceptable answer(s) using the rules above.
Respond ONLY with a JSON object:
{{"correct": true/false, "explanation": "<brief reason>"}}

Examples:
- {{"correct": true, "explanation": "Matches acceptable answer '66'"}}
- {{"correct": true, "explanation": "Contains acceptable answer 'French'"}}
- {{"correct": false, "explanation": "Model answered '30' but acceptable answer is '22'"}}
- {{"correct": false, "explanation": "Model answered 'combined forces' but acceptable answer is 'French'"}}
"""

        try:
            # 带重试的 LLM 调用（最多3次，每次间隔2秒）
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # 添加超时保护（judge 最多180秒）
                    response = await asyncio.wait_for(
                        self.judge_llm(judge_prompt),
                        timeout=180.0
                    )
                    response_text = response.strip()
                    
                    # 提取 JSON - 先尝试简单方式，再尝试复杂方式
                    result = None
                    
                    def fix_json_escapes(text):
                        """修复 JSON 中的非法转义字符"""
                        result_chars = []
                        i = 0
                        while i < len(text):
                            if text[i] == '\\' and i + 1 < len(text):
                                next_char = text[i + 1]
                                # 检查是否是合法的 JSON 转义
                                if next_char in '"\\\\/bfnrtu':
                                    # 保持原样
                                    result_chars.append('\\')
                                    result_chars.append(next_char)
                                    i += 2
                                else:
                                    # 非法转义，添加额外的反斜杠
                                    result_chars.append('\\\\')
                                    i += 1
                            else:
                                result_chars.append(text[i])
                                i += 1
                        return ''.join(result_chars)
                    
                    # 方式1：尝试直接解析整个响应
                    try:
                        result = json.loads(response_text)
                        logger.debug(f"JSON parsed successfully via direct parse")
                    except json.JSONDecodeError:
                        logger.debug(f"Direct parse failed, trying with escape fix")
                        
                        # 尝试修复非法的 JSON 转义字符
                        try:
                            fixed = fix_json_escapes(response_text)
                            result = json.loads(fixed)
                            logger.debug(f"JSON parsed successfully after escape fix")
                        except json.JSONDecodeError:
                            logger.debug(f"Escape fix failed, trying regex extraction")
                            
                            # 方式2：使用宽松的正则提取 JSON
                            json_match = re.search(r'\{.*?"correct".*?\}', response_text, re.DOTALL)
                            if json_match:
                                try:
                                    # 先尝试直接解析
                                    result = json.loads(json_match.group())
                                    logger.debug(f"JSON parsed successfully via regex match")
                                except json.JSONDecodeError:
                                    # 再尝试修复后解析
                                    try:
                                        fixed = fix_json_escapes(json_match.group())
                                        result = json.loads(fixed)
                                        logger.debug(f"JSON parsed successfully via regex + fix")
                                    except json.JSONDecodeError as e:
                                        logger.debug(f"All parsing methods failed: {e}")
                            else:
                                logger.debug(f"Regex did not match, response: {response_text[:100]}")
                    
                    if result and "correct" in result:
                        is_correct = result.get("correct", False)
                        explanation = result.get("explanation", "No explanation provided")
                        
                        # 二元评分：只有 1.0 或 0.0
                        score = 1.0 if is_correct else 0.0
                        
                        # 如果重试过，记录成功信息
                        if attempt > 0:
                            logger.info(f"LLM judge succeeded on attempt {attempt+1}/{max_retries}")
                        
                        return score, explanation
                    else:
                        # 如果无法解析 JSON，记录并重试
                        # 只在最后一次失败时才用 WARNING，之前用 DEBUG
                        if attempt == max_retries - 1:
                            logger.warning(f"Failed to parse LLM judge response after {max_retries} attempts: {response_text[:200]}")
                        else:
                            logger.debug(f"Failed to parse LLM judge response (attempt {attempt+1}/{max_retries}), will retry: {response_text[:100]}")
                        
                        last_error = f"JSON parse failed: {response_text[:100]}"
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)  # 等待2秒后重试
                            continue
                        else:
                            # 最后一次尝试也失败，使用 fallback
                            return self._fallback_score(ground_truth, prediction)
                
                except asyncio.TimeoutError:
                    if attempt == max_retries - 1:
                        logger.warning(f"LLM judge timed out after {max_retries} attempts")
                    else:
                        logger.debug(f"LLM judge timed out (attempt {attempt+1}/{max_retries}), will retry")
                    
                    last_error = "Timeout"
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    else:
                        return self._fallback_score(ground_truth, prediction)
                
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.warning(f"LLM judge error after {max_retries} attempts: {e}")
                    else:
                        logger.debug(f"LLM judge error (attempt {attempt+1}/{max_retries}): {e}, will retry")
                    
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    else:
                        return self._fallback_score(ground_truth, prediction)
        
        except Exception as e:
            # 外层异常捕获（不应该到这里）
            logger.error(f"Unexpected error in llm_judge_answer: {e}")
            return self._fallback_score(ground_truth, prediction)
    
    def _fallback_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        回退评分机制：简单的字符串匹配
        
        Returns:
            (score, explanation) - 二元评分和解释
        """
        gt_normalized = str(ground_truth).strip().lower()
        pred_normalized = str(prediction).strip().lower()
        
        # 检查是否有多个可能的答案
        acceptable_answers = [ans.strip().lower() for ans in gt_normalized.split("|") if ans.strip()]
        
        if pred_normalized in acceptable_answers or any(ans in pred_normalized for ans in acceptable_answers):
            return 1.0, "Fallback: String match"
        else:
            return 0.0, "Fallback: No match"

    @abstractmethod
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(self, data: List[dict], agent: Callable, max_concurrent_tasks: int = 30):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_evaluate(idx_problem):
            idx, problem = idx_problem
            # 如果problem没有id/idx字段,使用索引作为ID
            if 'id' not in problem and 'idx' not in problem:
                problem['_index'] = idx
            
            async with semaphore:
                try:
                    # 添加超时保护（每个问题最多10分钟）
                    return await asyncio.wait_for(
                        self.evaluate_problem(problem, agent),
                        timeout=600.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Problem evaluation timed out after 600s")
                    # 返回默认失败结果
                    columns = self.get_result_columns()
                    return tuple([problem.get("question", str(problem)[:100]), "TIMEOUT", "", 0.0, 0.0][:len(columns)])
                except Exception as e:
                    logger.error(f"Problem evaluation failed: {e}")
                    columns = self.get_result_columns()
                    return tuple([problem.get("question", str(problem)[:100]), f"ERROR: {e}", "", 0.0, 0.0][:len(columns)])

        tasks = [sem_evaluate((idx, problem)) for idx, problem in enumerate(data)]
        return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

    async def run_evaluation(self, agent: Callable, va_list: List[int], max_concurrent_tasks: int = 30):
        data = await self.load_data(va_list)
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost, solved_problems = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        return average_score, average_cost, total_cost, solved_problems
    

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = 30):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        logger.info(f"Avg Cost:{average_cost:.5f}")
        return average_score, average_cost, total_cost

