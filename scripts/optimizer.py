# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph (updated with AsyncLLM integration)

import asyncio
import time
from typing import List, Literal, Dict

from pydantic import BaseModel, Field

from scripts.evaluator import DatasetType
from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.optimizer_utils.experience_utils import ExperienceUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds
        
        # Initialize attribute to track solved problems for current round
        self.current_round_solved_problems = set()

        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            test_n = 1  # validation datasets's execution number
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test())
            return None

        for opt_round in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 1

            while retry_count < max_retries:
                try:
                    score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)

            # Early stop when convergence is detected
            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                # Print average scores and standard deviations for each round
                self.convergence_utils.print_results()
                break

            time.sleep(5)

    async def _optimize_graph(self):
        validation_n = self.validation_rounds  # validation datasets's execution number
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)

        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            # Load graph using graph_utils
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=True)

        # Create a loop until the generated graph meets the check conditions
        while True:
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)

            # Select and return the top `sample` rounds with the highest scores from previous rounds.
            top_rounds = self.data_utils.get_top_rounds(self.sample)
            # Sort and process the scores of each round, and select the optimal one based on the probability distribution
            sample = self.data_utils.select_round(top_rounds)

            # Load the graph and prompt for the selected round
            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            # Remove unnecessary prefixes and extract the graph from the loaded data(see extract_solve_graph.sample)
            graph = self.graph_utils.extract_solve_graph(graph_load)

            # Read historical experience and process it into the corresponding dataset's processed_experience.json
            processed_experience = self.experience_utils.load_experience()
            # Integrate and format the processed_experience of selected round into a string suitable for LLM(see format_experience.sample)
            experience = self.experience_utils.format_experience(processed_experience, sample["round"])

            # Load operator descriptions for the current dataset (workspace/{dataset}/template/operator.json), self.operators is set as a hyperparameter in run.py and needs to correspond to the content in operator.json
            operator_description = self.graph_utils.load_operators_description(self.operators)
            # Extract three entries from the selected round's log.json (error cases), which will be passed to the LLM for optimization
            log_data = self.data_utils.load_log(sample["round"])

            # Create the graph optimization prompt(see graph_optimize_prompt.sample)
            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience, sample["score"], graph[0], prompt, operator_description, self.type, log_data
            )

            # Replace ActionNode with AsyncLLM and XmlFormatter
            try:
                # Create XmlFormatter based on GraphOptimize model
                graph_formatter = XmlFormatter.from_model(GraphOptimize)
                
                # Call the LLM with formatter
                response = await self.optimize_llm.call_with_format(
                    graph_optimize_prompt, 
                    graph_formatter
                )
                
                # If we reach here, response is properly formatted and validated
                logger.info(f"Graph optimization response received successfully")
            except FormatError as e:
                # Handle format validation errors
                logger.error(f"Format error in graph optimization: {str(e)}")
                # Try again with a fallback approach - direct call with post-processing
                raw_response = await self.optimize_llm(graph_optimize_prompt)
                
                # Try to extract fields using basic parsing
                response = self._extract_fields_from_response(raw_response)
                if not response:
                    logger.error("Failed to extract fields from raw response, retrying...")
                    continue

            # Check if the modification is different from previous modifications
            check = self.experience_utils.check_modification(
                processed_experience, response["modification"], sample["round"]
            )

            # If `check` is True, break the loop; otherwise, regenerate the graph
            if check:
                break

        # Save the graph as static graph file
        self.graph_utils.write_graph_files(directory, response, self.round + 1, self.dataset)

        # Save the experience(without the current round score)
        experience = self.experience_utils.create_experience_data(sample, response["modification"])

        # English ver: Load the dynamic graph for testing(Import the static graph string as a valid callable dynamic class)
        self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)

        logger.info(directory)

        # Evaluate the graph
        avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=False)

        # Update the current round score in the experience file
        self.experience_utils.update_experience(directory, experience, avg_score)

        # After evaluation, find envelope workflows for potential fusion
        logger.info("Finding envelope workflows for potential fusion...")
        envelope_workflows = self.data_utils.find_envelope_workflows(max_workflows=3)
        
        if len(envelope_workflows) >= 2:
            logger.info(f"Found {len(envelope_workflows)} envelope workflows. Fusion candidate detected.")
            # TODO: Implement workflow fusion logic here
            # For now, just log the envelope workflows information
            total_coverage = set()
            for workflow in envelope_workflows:
                total_coverage.update(workflow['solved_problems'])
            logger.info(f"Envelope workflows cover {len(total_coverage)} unique problems")
        else:
            logger.info("Insufficient workflows for fusion (need at least 2)")

        return avg_score

    def _extract_fields_from_response(self, response: str) -> Dict[str, str]:
        """
        Fallback method to extract fields from raw response text using basic parsing
        
        Args:
            response: Raw response text from LLM
            
        Returns:
            Dictionary with extracted fields or None if extraction fails
        """
        try:
            # Try to extract XML tags with regex
            import re
            
            # Initialize result dictionary with default values
            result = {
                "modification": "",
                "graph": "",
                "prompt": ""
            }
            
            # Extract each field with regex
            for field in result.keys():
                pattern = rf"<{field}>(.*?)</{field}>"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    result[field] = match.group(1).strip()
            
            # Verify we have at least some content
            if not any(result.values()):
                logger.error("No fields could be extracted from response")
                return None
            
            return result
        except Exception as e:
            logger.error(f"Error extracting fields from response: {str(e)}")
            return None

    async def test(self):
        # 自动选择表现最好的工作流轮次进行测试
        source_graph_path = f"{self.root_path}/workflows"
        source_results = self.data_utils.load_results(source_graph_path)
        
        if not source_results:
            print("No optimization results found. Please run optimization first.")
            return
        
        # 找到分数最高的轮次
        best_result = max(source_results, key=lambda x: x["score"])
        best_round = best_result["round"]
        print(f"Testing best workflow from round {best_round} with score {best_result['score']:.4f}")
        
        rounds = [best_round]  # 测试最佳轮次
        data = []

        # 测试结果保存路径
        test_graph_path = f"{self.root_path}/workflows_test"
        # 工作流实际所在路径（优化完成的工作流）
        
        json_file_path = self.data_utils.get_results_file_path(test_graph_path)
        data = self.data_utils.load_results(test_graph_path)

        for round in rounds:
            # 在测试路径创建目录（用于保存测试日志和结果）
            directory = self.graph_utils.create_round_directory(test_graph_path, round)
            # 从优化路径加载工作流（而不是从测试路径）
            self.graph = self.graph_utils.load_graph(round, source_graph_path)

            print(f"Running test evaluation for round {round}...")
            score, avg_cost, total_cost, solved_problems = await self.evaluation_utils.evaluate_graph_test(self, directory, is_test=True)

            print(f"Test results - Score: {score:.4f}, Avg Cost: {avg_cost:.4f}, Total Cost: {total_cost:.4f}")
            print(f"Solved {len(solved_problems)} problems in test set")

            new_data = self.data_utils.create_result_data(round, score, avg_cost, total_cost, solved_problems)
            data.append(new_data)

            self.data_utils.save_results(json_file_path, data)