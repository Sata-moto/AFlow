import datetime
import json
import os
import random

import numpy as np
import pandas as pd

from scripts.logs import logger
from scripts.utils.common import read_json_file, write_json_file


class DataUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.top_scores = []

    def load_results(self, path: str) -> list:
        result_path = os.path.join(path, "results.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return []
        return []

    # Select and return the top `sample` rounds with the highest scores from previous rounds.
    def get_top_rounds(self, sample: int, path=None, mode="Graph"):
        # Get scores from previous rounds in result.json
        self._load_scores(path, mode)
        unique_rounds = set()
        unique_top_scores = []

        # Special handling: prioritize including the first round (to avoid premature convergence)
        first_round = next((item for item in self.top_scores if item["round"] == 1), None)
        if first_round:
            unique_top_scores.append(first_round)
            unique_rounds.add(1)

        # Add other rounds in descending order of scores until the sample size is reached
        for item in self.top_scores:
            if item["round"] not in unique_rounds:
                unique_top_scores.append(item)
                unique_rounds.add(item["round"])

                if len(unique_top_scores) >= sample:
                    break

        return unique_top_scores

    def select_round(self, items):
        if not items:
            raise ValueError("Item list is empty.")

        # Sort items by score in descending order(including round 1)
        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)

        # Scale the scores by 100 (for easier computation)
        scores = [item["score"] * 100 for item in sorted_items]

        # Compute mixed probability distribution and select the appropriate round
        probabilities = self._compute_probabilities(scores)
        
        # logger.info writes logs to the ./log folder under the root directory
        logger.info(f"\nMixed probability distribution: {probabilities}")
        logger.info(f"\nSorted rounds: {sorted_items}")

        selected_index = np.random.choice(len(sorted_items), p=probabilities)
        logger.info(f"\nSelected index: {selected_index}, Selected item: {sorted_items[selected_index]}")

        return sorted_items[selected_index]

    def _compute_probabilities(self, scores, alpha=0.2, lambda_=0.3):
        scores = np.array(scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            raise ValueError("Score list is empty.")

        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

        max_score = np.max(scores)
        shifted_scores = scores - max_score
        exp_weights = np.exp(alpha * shifted_scores)

        sum_exp_weights = np.sum(exp_weights)
        if sum_exp_weights == 0:
            raise ValueError("Sum of exponential weights is 0, cannot normalize.")

        score_prob = exp_weights / sum_exp_weights

        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0):
            mixed_prob = mixed_prob / total_prob

        return mixed_prob

    def load_log(self, cur_round, path=None, mode: str = "Graph"):
        if mode == "Graph":
            log_dir = os.path.join(self.root_path, "workflows", f"round_{cur_round}", "log.json")
        else:
            log_dir = path

        # 检查文件是否存在
        if not os.path.exists(log_dir):
            return ""  # 如果文件不存在，返回空字符串
        logger.info(log_dir)
        data = read_json_file(log_dir, encoding="utf-8")

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        if not data:
            return ""

        # 只保留错误样本 (score != 1.0)
        error_samples = [item for item in data if item.get('score', 0) != 1.0]
        
        if not error_samples:
            return ""  # 如果没有错误样本，返回空字符串
        
        # 过滤掉触发内容过滤的样本，并清洗敏感内容
        # 策略：
        # 1. 完全丢弃包含 content_filter 错误的样本
        # 2. 对于包含敏感词的样本，清洗内容而不是丢弃
        
        # 扩展的敏感词列表
        sensitive_keywords = [
            # 暴力相关
            'holocaust', 'massacre', 'killed', 'extermination', 'murder', 'genocide', 
            'slaughter', 'execution', 'torture', 'violence', 'death toll', 'casualties',
            'ethnic cleansing', 'war crime', 'atrocity', 'brutality', 'killing', 'mass murder',
            'pogrom', 'persecution', 'slaughtered', 'murdered', 'executed', 'exterminated',
            # 死亡相关
            'death', 'dead', 'died', 'fatalities', 'mortality', 'corpse', 'bodies',
            # 战争相关
            'war', 'battle', 'combat', 'conflict', 'invasion', 'occupation', 'bombing',
            # 纳粹相关
            'nazi', 'hitler', 'gestapo', 'ss troops', 'third reich',
            # 种族相关
            'jews', 'jewish', 'roma', 'romani', 'ethnic', 'racial'
        ]
        
        def sanitize_text(text):
            """清洗文本，替换敏感词为占位符"""
            if not isinstance(text, str):
                return text
            
            sanitized = text
            for keyword in sensitive_keywords:
                # 不区分大小写地替换
                import re
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                sanitized = pattern.sub('[REDACTED]', sanitized)
            
            return sanitized
        
        filtered_samples = []
        for item in error_samples:
            error_msg = item.get('error') or ''
            model_output = item.get('model_output') or ''
            judge_explanation = item.get('judge_explanation') or ''
            
            # 检查是否包含 content_filter 错误
            has_content_filter = ('content_filter' in error_msg 
                                 or 'content_filter' in model_output 
                                 or 'content_filter' in judge_explanation)
            
            # 如果包含content_filter错误，完全丢弃
            if has_content_filter:
                logger.debug(f"Skipping sample due to content_filter error")
                continue
            
            # 对样本内容进行清洗
            cleaned_item = item.copy()
            if 'question' in cleaned_item:
                cleaned_item['question'] = sanitize_text(cleaned_item['question'])
            if 'context' in cleaned_item:
                cleaned_item['context'] = sanitize_text(cleaned_item['context'])
            if 'answer' in cleaned_item:
                cleaned_item['answer'] = sanitize_text(cleaned_item['answer'])
            if 'model_output' in cleaned_item:
                cleaned_item['model_output'] = sanitize_text(cleaned_item['model_output'])
            
            filtered_samples.append(cleaned_item)
        
        if not filtered_samples:
            logger.warning(f"Round {cur_round}: All error samples contain content_filter issues, skipping.")
            return ""  # 如果所有错误样本都触发过滤，返回空字符串
        
        # 从过滤后的错误样本中随机抽取
        sample_size = min(3, len(filtered_samples))
        random_samples = random.sample(filtered_samples, sample_size)

        log = ""
        for sample in random_samples:
            log += json.dumps(sample, indent=4, ensure_ascii=False) + "\n\n"

        return log

    def get_results_file_path(self, graph_path: str) -> str:
        return os.path.join(graph_path, "results.json")

    def create_result_data(self, round: int, score: float, avg_cost: float, total_cost: float, solved_problems: set = None) -> dict:
        now = datetime.datetime.now()
        # Convert set to list for JSON serialization
        solved_problems_list = list(solved_problems) if solved_problems else []
        return {
            "round": round, 
            "score": score, 
            "avg_cost": avg_cost, 
            "total_cost": total_cost, 
            "time": now,
            "solved_problems": solved_problems_list
        }

    def save_results(self, json_file_path: str, data: list):
        # Custom JSON serialization to keep solved_problems on one line
        import json
        
        # First convert data to JSON string with default formatting
        json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        
        # Post-process to make solved_problems arrays compact
        import re
        
        # Pattern to match solved_problems arrays that span multiple lines
        pattern = r'"solved_problems":\s*\[\s*\n(\s*\d+(?:,\s*\n\s*\d+)*)\s*\n\s*\]'
        
        def compact_array(match):
            # Extract the numbers and format them compactly
            numbers_text = match.group(1)
            # Extract all numbers
            numbers = re.findall(r'\d+', numbers_text)
            return f'"solved_problems": [{", ".join(numbers)}]'
        
        # Apply the compaction
        json_str = re.sub(pattern, compact_array, json_str)
        
        # Write to file
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        with open(json_file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

    def _load_scores(self, path=None, mode="Graph"):
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        result_file = os.path.join(rounds_dir, "results.json")
        self.top_scores = []

        data = read_json_file(result_file, encoding="utf-8")
        df = pd.DataFrame(data)

        scores_per_round = df.groupby("round")["score"].mean().to_dict()

        for round_number, average_score in scores_per_round.items():
            self.top_scores.append({"round": round_number, "score": average_score})

        self.top_scores.sort(key=lambda x: x["score"], reverse=True)

        return self.top_scores

    def find_envelope_workflows(self, max_workflows: int = 3, should_log: bool = False) -> list:
        """
        Find the minimum set of workflows (at most max_workflows) that solve the maximum number of problems
        using a greedy approach.
        
        Args:
            max_workflows: Maximum number of workflows in the envelope set
            should_log: Whether to log the selected envelope workflows

        Returns:
            List of workflow data with solved problems information
        """
        # Load current results and group by round
        results = self.load_results(os.path.join(self.root_path, "workflows"))
        if not results:
            return []
        
        # Group results by round and collect solved problems for each round
        rounds_data = {}
        for result in results:
            round_num = result["round"]
            if round_num not in rounds_data:
                rounds_data[round_num] = {
                    "round": round_num,
                    "scores": [],
                    "avg_costs": [],
                    "total_costs": [],
                    "solved_problems": set()
                }
            
            rounds_data[round_num]["scores"].append(result["score"])
            rounds_data[round_num]["avg_costs"].append(result["avg_cost"])
            rounds_data[round_num]["total_costs"].append(result["total_cost"])
            
            # Union solved problems across validation runs
            if "solved_problems" in result:
                rounds_data[round_num]["solved_problems"].update(result["solved_problems"])
        
        # Calculate average metrics for each round
        round_summaries = []
        for round_num, data in rounds_data.items():
            avg_score = np.mean(data["scores"]) if data["scores"] else 0.0
            avg_cost = np.mean(data["avg_costs"]) if data["avg_costs"] else 0.0
            total_cost = np.mean(data["total_costs"]) if data["total_costs"] else 0.0
            
            round_summaries.append({
                "round": round_num,
                "avg_score": avg_score,
                "avg_cost": avg_cost,
                "total_cost": total_cost,
                "solved_problems": data["solved_problems"]
            })
        
        # Greedy algorithm to find envelope workflows
        envelope_workflows = []
        covered_problems = set()
        available_workflows = round_summaries.copy()
        
        for _ in range(min(max_workflows, len(available_workflows))):
            if not available_workflows:
                break
                
            # Find workflow that covers the most uncovered problems
            best_workflow = None
            best_new_coverage = 0
            best_index = -1
            
            for i, workflow in enumerate(available_workflows):
                new_problems = workflow["solved_problems"] - covered_problems
                new_coverage = len(new_problems)
                
                # If tie, prefer workflow with higher score
                if (new_coverage > best_new_coverage or 
                    (new_coverage == best_new_coverage and 
                     (best_workflow is None or workflow["avg_score"] > best_workflow["avg_score"]))):
                    best_workflow = workflow
                    best_new_coverage = new_coverage
                    best_index = i
            
            # If no new coverage, break
            if best_new_coverage == 0:
                break
                
            # Add best workflow to envelope
            envelope_workflows.append(best_workflow)
            covered_problems.update(best_workflow["solved_problems"])
            available_workflows.pop(best_index)
        
        if should_log:
            logger.info(f"Found envelope workflows covering {len(covered_problems)} problems:")
            for workflow in envelope_workflows:
                logger.info(f"  Round {workflow['round']}: score={workflow['avg_score']:.4f}, problems={len(workflow['solved_problems'])}")
        
        return envelope_workflows
