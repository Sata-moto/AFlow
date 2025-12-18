import re
import string
from collections import Counter
from typing import Callable, List, Tuple
import json

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger
from scripts.async_llm import AsyncLLM, LLMConfig


class DROPBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str, solved_threshold: float = 0.5):
        super().__init__(name, file_path, log_path, solved_threshold)
        
        # 初始化评判 LLM (使用 GPT-4o-mini)
        self.judge_llm = self._create_judge_llm()
    
    def _create_judge_llm(self):
        """创建用于评判的 LLM 实例"""
        try:
            # 尝试从配置文件加载
            from scripts.async_llm import LLMsConfig
            config = LLMsConfig.default()
            llm_config = config.get("gpt-4o-mini")
            return AsyncLLM(llm_config)
        except:
            # 如果配置加载失败，使用默认配置
            llm_config = LLMConfig({
                "model": "gpt-4o-mini",
                "temperature": 0,
                "key": None,  # 将使用环境变量
                "base_url": "https://api.openai.com/v1"
            })
            return AsyncLLM(llm_config)

    def extract_question_only(self, context: str) -> str:
        """
        从 context 中提取问题部分，去掉 Passage
        DROP 格式: "Passage: ... Question: ... Answer:"
        """
        # 查找 "Question:" 的位置
        question_start = context.find("Question:")
        if question_start != -1:
            # 从 "Question:" 开始到结尾（包括 "Answer:"）
            return context[question_start:].strip()
        # 如果找不到 "Question:"，返回原文本
        return context
    
    def normalize_answer(self, s: str) -> List[str]:
        """
        Normalize answers for evaluation.
        """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    async def llm_judge_answer(self, question: str, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        使用 LLM 进行语义评判，替代硬匹配
        
        Args:
            question: 原始问题（包含 context）
            ground_truth: 正确答案（可能包含多个用 | 分隔的答案）
            prediction: 模型预测答案
            
        Returns:
            (score, explanation) - 分数 0-1 和评判理由
        """
        # 处理多个可能的正确答案
        acceptable_answers = [ans.strip() for ans in ground_truth.split("|") if ans.strip()]
        
        judge_prompt = f"""You are an expert judge evaluating question-answering systems.

Question (with context):
{question}

Correct Answer(s): {' OR '.join(acceptable_answers)}
Model's Answer: {prediction}

Task: Determine if the model's answer is semantically correct compared to the acceptable answer(s).

Consider:
1. Numerical equivalence (e.g., "5" = "five" = "5.0")
2. Semantic equivalence (e.g., "the Federales" = "Federales")
3. Different phrasings of the same meaning
4. Ignore minor formatting differences

IMPORTANT: Score must be EITHER 1.0 (correct) OR 0.0 (incorrect). NO partial scores.

Respond ONLY with a JSON object in this exact format:
{{"score": <1.0 or 0.0>, "explanation": "<brief reason>"}}

Examples:
- Correct: {{"score": 1.0, "explanation": "Semantically correct"}}
- Correct (number): {{"score": 1.0, "explanation": "Numerically equivalent"}}
- Wrong: {{"score": 0.0, "explanation": "Incorrect answer"}}
"""

        try:
            response = await self.judge_llm(judge_prompt)
            response_text = response.strip()
            
            # 提取 JSON
            # 尝试找到 JSON 对象
            json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                score = float(result.get("score", 0.0))
                explanation = result.get("explanation", "No explanation provided")
                
                # 强制二元评分：只允许 1.0 或 0.0
                # 任何 >= 0.5 的分数视为正确，否则视为错误
                if score >= 0.5:
                    score = 1.0
                else:
                    score = 0.0
                
                return score, explanation
            else:
                # 如果无法解析 JSON，尝试回退到基础匹配
                logger.warning(f"Failed to parse LLM judge response: {response_text[:200]}")
                return self._fallback_score(ground_truth, prediction)
                
        except Exception as e:
            logger.error(f"LLM judge error: {e}")
            # 发生错误时回退到基础评分
            return self._fallback_score(ground_truth, prediction)
    
    def _fallback_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """回退评分机制 - 使用原有的 F1 匹配，但二元化"""
        answers = ground_truth.split("|")
        f1_scores = []
        
        for answer in answers:
            if answer.strip():
                f1_score, _ = self.calculate_score(answer.strip(), prediction)
                f1_scores.append(f1_score)
        
        max_score = max(f1_scores) if f1_scores else 0.0
        
        # 二元化：F1 >= 0.5 视为正确，否则错误
        binary_score = 1.0 if max_score >= 0.5 else 0.0
        
        return binary_score, f"Fallback F1 score: {max_score:.2f} -> {binary_score}"

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        Compute the F1 score between prediction and ground truth answers.
        """
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, prediction
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        original_context = problem["context"]
        expected_output = problem["ref_text"]
        
        # 获取problem_id: 优先使用id字段,否则使用_index生成problem_{idx}格式
        if "id" in problem:
            problem_id = problem["id"]
        elif "_index" in problem:
            problem_id = f"problem_{problem['_index']}"
        else:
            problem_id = "unknown"
        
        category = self._get_problem_category(problem_id)
        
        # 在问题前添加简单指令，要求直接回答（只用于工作流输入）
        instruction = "Please provide a clear, direct answer without showing your reasoning process.\n\n"
        input_with_instruction = instruction + original_context  # 传给工作流的输入
        
        try:
            output, cost = await self._generate_output(graph, input_with_instruction)
            
            # 提取只包含问题的部分（不含 Passage），避免 judge 自行思考
            question_only = self.extract_question_only(original_context)
            
            # 使用 LLM 进行语义评判
            uni_score, explanation = await self.llm_judge_answer(
                question=question_only,  # 只传入问题部分
                ground_truth=expected_output,
                prediction=output
            )
            
            # 所有问题都记录日志（包含类别信息）
            self.log_mismatch(
                original_context,  # 记录原始 context，不含指令
                expected_output, 
                output, 
                explanation,
                uni_score,
                problem_id,
                category
            )

            return original_context, output, expected_output, uni_score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            # 错误情况也记录日志
            self.log_mismatch(
                original_context,
                expected_output,
                str(e),
                f"Error: {e}",
                0.0,
                problem_id,
                category
            )
            return original_context, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
