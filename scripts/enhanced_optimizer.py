import asyncio
import os
import json
import time
from typing import List, Literal, Dict
import random
import os
import json
import time

from pydantic import BaseModel, Field

from scripts.evaluator import DatasetType
from scripts.optimizer import Optimizer, QuestionType, OptimizerType, GraphOptimize
from scripts.workflow_fusion import WorkflowFusion
from scripts.workflow_differentiation import WorkflowDifferentiation
from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.optimizer_utils.experience_utils import ExperienceUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger
from scripts.utils.code_processor import CodeProcessor
from scripts.utils.workflow_manager import WorkflowManager, FusionChecker


class EnhancedOptimizer(Optimizer):
    """
    Enhanced optimizer that integrates workflow fusion and differentiation into the optimization process
    """
    
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
        enable_fusion: bool = True,
        max_envelope_workflows: int = 3,
        fusion_score_threshold: float = 0.0,  # Minimum score improvement required for fusion
        fusion_start_round: int = 5,  # Round from which fusion is allowed
        fusion_interval_rounds: int = 2,  # Minimum rounds between fusion attempts
        enable_differentiation: bool = True,
        differentiation_probability: float = 0.3,  # Probability of differentiation vs regular optimization
        max_differentiation_rounds: int = 5,  # Maximum rounds with differentiation per optimization cycle
    ) -> None:
        # Initialize parent class
        super().__init__(
            dataset=dataset,
            question_type=question_type,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            operators=operators,
            sample=sample,
            check_convergence=check_convergence,
            optimized_path=optimized_path,
            initial_round=initial_round,
            max_rounds=max_rounds,
            validation_rounds=validation_rounds,
        )
        
        # Fusion-specific parameters
        self.enable_fusion = enable_fusion
        self.max_envelope_workflows = max_envelope_workflows
        self.fusion_score_threshold = fusion_score_threshold
        self.fusion_start_round = fusion_start_round  # Round from which fusion is allowed
        self.fusion_interval_rounds = fusion_interval_rounds  # Minimum rounds between fusion attempts
        
        # Differentiation-specific parameters
        self.enable_differentiation = enable_differentiation
        self.differentiation_probability = differentiation_probability  # Base probability
        self.max_differentiation_rounds = max_differentiation_rounds
        
        # Retry parameters for fusion and differentiation
        self.max_retries = 3  # Maximum number of retries for fusion and differentiation
        self.fusion_retry_count = 0  # Current retry count for fusion
        self.differentiation_retry_count = 0  # Current retry count for differentiation
        
        # Track differentiation attempts
        self.differentiation_rounds_used = 0
        self.workflow_differentiation_counts = {}  # Track how many times each workflow was differentiated
        
        # Track fusion state
        self.last_fusion_round = -1  # Initialize to -1 to indicate no fusion has occurred yet
        
        # Track fusion metadata counter for proper numbering
        self.fusion_metadata_counter = 0
        self._initialize_fusion_counter()
        
        # Initialize fusion processor
        if self.enable_fusion:
            self.fusion_processor = WorkflowFusion(
                dataset=self.dataset,
                question_type=self.type,
                opt_llm_config=self.optimize_llm_config,
                exec_llm_config=self.execute_llm_config,
                operators=self.operators,
                optimized_path=optimized_path,
                max_envelope_workflows=self.max_envelope_workflows,
                validation_rounds=self.validation_rounds,
            )
        
        # Initialize differentiation processor
        if self.enable_differentiation:
            self.differentiation_processor = WorkflowDifferentiation(
                dataset=self.dataset,
                question_type=self.type,
                opt_llm_config=self.optimize_llm_config,
                exec_llm_config=self.execute_llm_config,
                operators=self.operators,
                optimized_path=optimized_path,
                validation_rounds=self.validation_rounds,
            )
        
        # Initialize workflow management utilities
        self.workflow_manager = WorkflowManager(
            root_path=self.root_path,
            data_utils=self.data_utils,
            graph_utils=self.graph_utils
        )
        
        # Initialize fusion checker
        self.fusion_checker = FusionChecker(root_path=self.root_path)
    
    def _initialize_fusion_counter(self):
        """Initialize fusion metadata counter by checking existing files"""
        fusion_metadata_dir = f"{self.root_path}/workflows"
        counter = 0
        
        while True:
            metadata_file = os.path.join(fusion_metadata_dir, f"fusion_metadata_{counter + 1}.json")
            if os.path.exists(metadata_file):
                counter += 1
            else:
                break
        
        self.fusion_metadata_counter = counter
        logger.info(f"Initialized fusion counter at {self.fusion_metadata_counter}")
    
    def optimize(self, mode: OptimizerType = "Graph"):
        """Enhanced optimize method with integrated fusion and differentiation"""
        if mode == "Test":
            test_n = 1  # validation datasets's execution number
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test())
            return None

        while self.round < self.max_rounds:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 1
            score = None

            while retry_count < max_retries:
                try:
                    # Priority 1: Check if we should attempt fusion with retry
                    if self._should_attempt_fusion():
                        logger.info(f"Attempting workflow fusion instead of optimization for round {self.round + 1}")
                        score = self._attempt_with_retry(
                            lambda: loop.run_until_complete(self._attempt_fusion()),
                            "fusion", 3
                        )
                        if score is not None:
                            # Fusion successful, skip other strategies
                            break
                        else:
                            # Fusion failed after retries, try other strategies
                            logger.warning("Fusion failed after retries, trying other strategies")
                    
                    # Priority 2: Check if we should attempt differentiation with retry
                    if score is None and self._should_attempt_differentiation():
                        logger.info(f"Attempting workflow differentiation for round {self.round + 1}")
                        score = self._attempt_with_retry(
                            lambda: loop.run_until_complete(self._attempt_differentiation()),
                            "differentiation", 3
                        )
                        if score is not None:
                            # Differentiation successful, skip regular optimization
                            break
                        else:
                            # Differentiation failed after retries, fall back to regular optimization
                            logger.warning("Differentiation failed after retries, falling back to regular optimization")
                    
                    # Priority 3: Regular optimization process
                    if score is None:
                        score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.warning("Max retries reached. Moving to next round.")
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

    def _attempt_with_retry(self, operation, operation_name: str, max_retries: int = 3):
        """
        Attempt an operation with retry mechanism.
        
        Args:
            operation: Callable operation to retry
            operation_name: Name of the operation for logging
            max_retries: Maximum number of retry attempts
            
        Returns:
            Result of the operation if successful, None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                result = operation()
                if result is not None:
                    if attempt > 0:
                        logger.info(f"{operation_name.capitalize()} succeeded on attempt {attempt + 1}")
                    return result
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"{operation_name.capitalize()} failed (attempt {attempt + 1}/{max_retries}), retrying...")
                    else:
                        logger.warning(f"{operation_name.capitalize()} failed after {max_retries} attempts")
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"{operation_name.capitalize()} error on attempt {attempt + 1}/{max_retries}: {e}, retrying...")
                else:
                    logger.error(f"{operation_name.capitalize()} error after {max_retries} attempts: {e}")
        
        return None

    def _should_attempt_fusion(self) -> bool:
        """
        Determine if we should attempt fusion based on current conditions
        
        Returns:
            bool: True if fusion should be attempted
        """
        if not self.enable_fusion:
            return False
        
        # Don't attempt fusion before the specified start round
        if self.round < self.fusion_start_round:
            logger.info(f"Skipping fusion - not yet at start round {self.fusion_start_round} (current: {self.round})")
            return False
        
        # Don't attempt fusion if insufficient rounds have passed since last fusion
        if self.last_fusion_round != -1 and (self.round - self.last_fusion_round) < self.fusion_interval_rounds:
            rounds_since_last = self.round - self.last_fusion_round
            logger.info(f"Skipping fusion - insufficient interval (need {self.fusion_interval_rounds} rounds, only {rounds_since_last} have passed)")
            return False
        
        # Check if we have enough envelope workflows
        envelope_workflows = self.data_utils.find_envelope_workflows(self.max_envelope_workflows)
        if len(envelope_workflows) < self.max_envelope_workflows:
            logger.info(f"Insufficient workflows for fusion (found {len(envelope_workflows)}, need at least {self.max_envelope_workflows})")
            return False
        
        # Check if this specific fusion combination has been attempted before
        if self.fusion_checker.check_fusion_already_attempted(envelope_workflows):
            logger.info("Skipping fusion - this combination has been attempted before")
            return False
        
        logger.info(f"Fusion conditions met: {len(envelope_workflows)} envelope workflows available")
        return True
    
    def _should_attempt_differentiation(self) -> bool:
        """
        Determine if we should attempt differentiation based on current conditions.
        Uses dynamic probability that increases linearly with rounds from base_p to 2*base_p
        
        Returns:
            bool: True if differentiation should be attempted
        """
        if not self.enable_differentiation:
            return False
        
        # Don't attempt differentiation in the first round
        if self.round < 2:
            return False
        
        # Check if we've exceeded max differentiation rounds
        if self.differentiation_rounds_used >= self.max_differentiation_rounds:
            logger.info("Skipping differentiation - maximum differentiation rounds reached")
            return False
        
        # Calculate dynamic probability that increases with rounds
        base_probability = self.differentiation_probability  # Base probability (p)
        max_probability = 2 * base_probability  # Maximum probability (2*p)
        
        # Linear increase from base_p to 2*base_p over max_rounds
        # For round r: probability = base_p + (2*base_p - base_p) * min(r-2, max_rounds-2) / (max_rounds-2)
        # Simplified: probability = base_p * (1 + min(r-2, max_rounds-2) / (max_rounds-2))
        if self.max_rounds > 2:
            progress_ratio = min(self.round - 2, self.max_rounds - 2) / (self.max_rounds - 2)
            current_probability = base_probability * (1 + progress_ratio)
        else:
            current_probability = base_probability
        
        # Clamp to maximum probability
        current_probability = min(current_probability, max_probability)
        
        # Probabilistic decision for differentiation
        import random
        probability_check = random.random() <= current_probability
        
        logger.info(f"Differentiation probability check: {current_probability:.3f} (base: {base_probability:.3f}, round: {self.round})")
        
        if not probability_check:
            return False
        
        # Check if we have workflows to differentiate from
        workflows_data = self.data_utils.load_results(f"{self.root_path}/workflows")
        if len(workflows_data) < 2:
            logger.info("Insufficient workflows for differentiation (need at least 2)")
            return False
        
        logger.info(f"Differentiation conditions met: probability check passed, {len(workflows_data)} workflows available")
        return True
    
    async def _attempt_differentiation(self) -> float:
        """
        Attempt workflow differentiation with retry mechanism.
        
        Returns:
            float: Score of differentiated workflow if successful, None if failed after all retries
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Differentiation attempt {attempt + 1}/{self.max_retries}")
                result = await self._execute_single_differentiation()
                if result is not None:
                    logger.info(f"Differentiation successful on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Differentiation attempt {attempt + 1} failed, retrying...")
            except Exception as e:
                logger.error(f"Differentiation attempt {attempt + 1} failed with error: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying differentiation (attempt {attempt + 2}/{self.max_retries})...")
                    continue
        
        logger.error(f"All {self.max_retries} differentiation attempts failed")
        return None

    async def _execute_single_differentiation(self) -> float:
        """
        Attempt workflow differentiation for problem type specialization.
        
        Returns:
            float: Score of differentiated workflow if successful, None if failed
        """
        try:
            # Get workflow results and update differentiation counts
            workflow_results = self.data_utils.load_results(f"{self.root_path}/workflows")
            round_summaries = self.workflow_manager.get_round_summaries(workflow_results)
            self._update_differentiation_counts(round_summaries)
            
            # Find best candidate for differentiation
            candidates = self.differentiation_processor.analyze_differentiation_candidates(round_summaries)
            if not candidates:
                logger.warning("No suitable differentiation candidates found")
                return None
            
            # Select best candidate and update its differentiation count
            selected_candidate = candidates[0]["workflow"]
            source_round = selected_candidate['round']
            self.workflow_differentiation_counts[source_round] = self.workflow_differentiation_counts.get(source_round, 0) + 1
            
            logger.info(f"Selected workflow from round {source_round} for problem type specialization")
            logger.info(f"  Score: {selected_candidate.get('avg_score', 0):.4f}, Weight: {candidates[0].get('final_weight', 0):.4f}")
            
            # Load source workflow and create differentiated version
            source_workflow_content = await self.workflow_manager.load_workflow_content(source_round)
            operator_description = self.graph_utils.load_operators_description(self.operators)
            
            # Calculate target round
            next_round = self.round + 1
            
            differentiation_response = await self.differentiation_processor.create_differentiated_workflow(
                source_workflow=source_workflow_content,
                differentiation_direction="problem_type_specialization",
                operator_description=operator_description,
                target_round=next_round
            )
            
            if not differentiation_response:
                logger.error("Differentiation process failed")
                return None
            
            # Save differentiated workflow to next round
            next_round = self.round + 1
            success = self.differentiation_processor.save_differentiated_workflow_direct(
                differentiation_response, selected_candidate, "problem_type_specialization",
                next_round, self.root_path, self.graph_utils, self.experience_utils
            )
            
            if not success:
                logger.error("Failed to save differentiated workflow")
                return None
            
            # Evaluate the differentiated workflow
            graph_path = f"{self.root_path}/workflows"
            directory = f"{self.root_path}/workflows/round_{next_round}"
            
            self.graph = self.graph_utils.load_graph(next_round, graph_path)
            if self.graph is None:
                logger.error("Failed to load differentiated workflow")
                return None
            
            data = self.data_utils.load_results(graph_path)
            differentiation_score = await self.evaluation_utils.evaluate_graph(
                self, directory, self.validation_rounds, data, initial=False
            )
            
            # Update experience.json if needed
            experience_path = os.path.join(directory, "experience.json")
            if os.path.exists(experience_path):
                with open(experience_path, 'r', encoding='utf-8') as f:
                    experience_data = json.load(f)
                if experience_data.get("after") is None:
                    self.experience_utils.update_experience(directory, experience_data, differentiation_score)
            
            # Save metadata
            self.differentiation_processor.save_differentiation_metadata(
                source_workflow=selected_candidate,
                differentiated_workflow=differentiation_response,
                differentiation_direction="problem_type_specialization",
                target_round=next_round,
                differentiation_score=differentiation_score
            )
            
            # Only increment differentiation count if successful
            self.differentiation_rounds_used += 1
            
            logger.info(f"Problem type specialization completed with score: {differentiation_score:.4f}")
            return differentiation_score
        
        except Exception as e:
            logger.error(f"Error in single differentiation execution: {e}")
            return None

    def _update_differentiation_counts(self, round_summaries: List[Dict]) -> None:
        """
        Update differentiation count information for each workflow
        
        Args:
            round_summaries: Workflow round summary data
        """
        for summary in round_summaries:
            round_num = summary.get('round')
            if round_num is not None:
                # Inject differentiation count information into workflow data
                current_count = self.workflow_differentiation_counts.get(round_num, 0)
                summary['differentiation_count'] = current_count
                
                # Check if this is a differentiated workflow (can be determined through metadata files or other methods)
                summary['is_differentiated'] = self._check_if_differentiated_workflow(round_num)
    
    def _check_if_differentiated_workflow(self, round_num: int) -> bool:
        """
        Check if the workflow of specified round was created through differentiation
        
        Args:
            round_num: Round number
            
        Returns:
            bool: Whether this is a differentiated workflow
        """
        try:
            # Check if differentiation metadata file exists
            workflows_dir = f"{self.root_path}/workflows"
            metadata_file = f"differentiation_metadata_{round_num}.json"
            metadata_path = os.path.join(workflows_dir, metadata_file)
            
            if os.path.exists(metadata_path):
                return True
                
            # Check modification field in experience.json file
            experience_file = os.path.join(workflows_dir, f"round_{round_num}", "experience.json")
            if os.path.exists(experience_file):
                with open(experience_file, 'r', encoding='utf-8') as f:
                    experience_data = json.load(f)
                    modification = experience_data.get('modification', '').lower()
                    if 'differentiation' in modification or 'specialized' in modification:
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking if workflow {round_num} is differentiated: {e}")
            return False

    async def _attempt_fusion(self) -> float:
        """
        Attempt workflow fusion with retry mechanism.
        
        Returns:
            float: Score of fused workflow if successful, None if failed after all retries
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fusion attempt {attempt + 1}/{self.max_retries}")
                result = await self._execute_single_fusion()
                if result is not None:
                    logger.info(f"Fusion successful on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Fusion attempt {attempt + 1} failed, retrying...")
            except Exception as e:
                logger.error(f"Fusion attempt {attempt + 1} failed with error: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying fusion (attempt {attempt + 2}/{self.max_retries})...")
                    continue
        
        logger.error(f"All {self.max_retries} fusion attempts failed")
        return None

    async def _execute_single_fusion(self) -> float:
        """
        Execute a single fusion attempt.
        
        Returns:
            float: Score of fused workflow if successful, None if failed
        """
        try:
            # Get envelope workflows before fusion
            envelope_workflows = self.data_utils.find_envelope_workflows(self.max_envelope_workflows, True)
            min_envelope_score = min(w["avg_score"] for w in envelope_workflows)
            
            logger.info(f"Minimum envelope score: {min_envelope_score:.4f}")
            
            # Execute fusion process (creates workflow directly in next round directory)
            fusion_success = await self._execute_fusion_async()
            
            if not fusion_success:
                logger.error("Fusion process failed")
                return None
            
            # Now evaluate the adopted fusion workflow using standard evaluation process
            next_round = self.round + 1
            graph_path = f"{self.root_path}/workflows"
            directory = f"{self.root_path}/workflows/round_{next_round}"
            
            # Load the fused graph
            self.graph = self.graph_utils.load_graph(next_round, graph_path)
            if self.graph is None:
                logger.error("Failed to load adopted fused workflow")
                return None
            
            # Load data for evaluation context
            data = self.data_utils.load_results(graph_path)
            
            # Evaluate using standard evaluation process
            fusion_score = await self.evaluation_utils.evaluate_graph(
                self, directory, self.validation_rounds, data, initial=False
            )
            
            # The standard evaluation process should have updated the experience.json
            # But let's make sure by explicitly calling the update if needed
            experience_path = os.path.join(directory, "experience.json")
            if os.path.exists(experience_path):
                # Load the experience data to check if it was properly updated
                with open(experience_path, 'r', encoding='utf-8') as f:
                    experience_data = json.load(f)
                
                if experience_data.get("after") is None:
                    # If not updated by standard process, update it manually using standard method
                    self.experience_utils.update_experience(directory, experience_data, fusion_score)
            
            logger.info(f"Fused workflow score: {fusion_score:.4f}")
            
            # Check if fusion meets threshold
            if fusion_score > min_envelope_score + self.fusion_score_threshold:
                logger.info(f"Fusion successful! Score {fusion_score:.4f} > threshold {min_envelope_score + self.fusion_score_threshold:.4f}")
                
                # Record successful fusion round
                self.last_fusion_round = self.round
                
                # Save metadata for successful adoption
                self.fusion_processor.save_fusion_metadata(
                    envelope_workflows, fusion_score, self.round, self.fusion_metadata_counter + 1, adopted=True
                )
                self.fusion_metadata_counter += 1
                
                return fusion_score
            else:
                logger.info(f"Fusion score {fusion_score:.4f} below threshold {min_envelope_score + self.fusion_score_threshold:.4f}")
                # Even if below threshold, we keep the fusion workflow since it was already created
                # Record successful fusion round (even if below threshold, fusion was successful)
                self.last_fusion_round = self.round
                
                # Save metadata for tracking purposes
                self.fusion_processor.save_fusion_metadata(
                    envelope_workflows, fusion_score, self.round, self.fusion_metadata_counter + 1, adopted=True
                )
                self.fusion_metadata_counter += 1
                return fusion_score
        
        except Exception as e:
            logger.error(f"Error in single fusion execution: {e}")
            return None
    
    async def _execute_fusion_async(self) -> bool:
        """
        Execute the fusion process asynchronously
        
        Returns:
            bool: True if fusion successful
        """
        try:
            # Find envelope workflows
            envelope_workflows = self.data_utils.find_envelope_workflows(self.max_envelope_workflows)
            
            if len(envelope_workflows) < 2:
                return False
            
            # Load workflow contents for fusion
            workflow_contents = []
            workflows_path = f"{self.root_path}/workflows"
            
            for workflow in envelope_workflows:
                round_num = workflow["round"]
                
                # Load graph and prompt for this workflow
                prompt, graph_load = self.graph_utils.read_graph_files(round_num, workflows_path)
                graph = self.graph_utils.extract_solve_graph(graph_load)
                
                workflow_content = {
                    "round": round_num,
                    "score": workflow["avg_score"],
                    "solved_problems": workflow["solved_problems"],
                    "prompt": prompt,
                    "graph": graph[0] if graph else "",
                }
                workflow_contents.append(workflow_content)
            
            # Get operator descriptions
            operator_description = self.graph_utils.load_operators_description(self.operators)
            
            # Create fusion prompt and call LLM
            fusion_response = await self.fusion_processor.create_fused_workflow(
                envelope_workflows=envelope_workflows,
                workflow_contents=workflow_contents,
                operator_description=operator_description
            )
            
            if not fusion_response:
                return False
            
            # Save fused workflow directly to next round directory using fusion processor
            return self.fusion_processor.save_fused_workflow_direct(
                fusion_response, envelope_workflows, self.root_path, self.round, 
                self.graph_utils, self.experience_utils
            )
            
        except Exception as e:
            logger.error(f"Error executing fusion: {e}")
            return False
    
    async def _evaluate_fused_workflow(self) -> float:
        """
        Evaluate the fused workflow from the next round directory and return its score
        
        Returns:
            float: Average score of fused workflow, None if evaluation failed
        """
        try:
            # Load fused workflow from next round directory
            next_round = self.round + 1
            fusion_path = f"{self.root_path}/workflows"
            fused_graph = self.graph_utils.load_graph(next_round, fusion_path)
            
            if fused_graph is None:
                logger.error("Failed to load fused workflow from next round directory")
                return None
            
            # Create temporary evaluation directory
            eval_dir = self.graph_utils.create_round_directory(fusion_path, "fused_eval")
            
            # Set the graph for evaluation
            original_graph = self.graph
            self.graph = fused_graph
            
            # Load existing data for evaluation context
            data = self.data_utils.load_results(fusion_path)
            
            # Evaluate the fused graph
            avg_score = await self.evaluation_utils.evaluate_graph(
                self, eval_dir, self.validation_rounds, data, initial=False
            )
            
            # Restore original graph
            self.graph = original_graph
            
            return avg_score
            
        except Exception as e:
            logger.error(f"Error evaluating fused workflow: {e}")
            return None

    async def _optimize_graph(self):
        """Override parent method to remove the original fusion logic"""
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

        # Load the dynamic graph for testing(Import the static graph string as a valid callable dynamic class)
        self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)

        logger.info(directory)

        # Evaluate the graph
        avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=False)

        # Update the current round score in the experience file
        self.experience_utils.update_experience(directory, experience, avg_score)

        return avg_score
