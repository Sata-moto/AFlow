import asyncio
import os
import json
import time
from typing import List, Literal, Dict

from pydantic import BaseModel, Field

from scripts.evaluator import DatasetType
from scripts.optimizer import Optimizer, QuestionType, OptimizerType, GraphOptimize
from scripts.workflow_fusion import WorkflowFusion
from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.optimizer_utils.experience_utils import ExperienceUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger


class EnhancedOptimizer(Optimizer):
    """
    Enhanced optimizer that integrates workflow fusion into the optimization process
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
        
        # Track fusion attempts to prevent consecutive fusion rounds
        self.last_fusion_round = -1
        
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
    
    def _check_fusion_already_attempted(self, envelope_workflows: List[Dict]) -> bool:
        """
        Check if this specific fusion combination has been attempted before
        
        Args:
            envelope_workflows: List of workflows to be fused
            
        Returns:
            bool: True if this combination was already attempted
        """
        try:
            # Create a sorted signature of the fusion combination
            workflow_rounds = sorted([w["round"] for w in envelope_workflows])
            fusion_signature = tuple(workflow_rounds)
            
            # Check all existing fusion metadata files
            fusion_metadata_dir = f"{self.root_path}/workflows"
            counter = 1
            
            while True:
                metadata_file = os.path.join(fusion_metadata_dir, f"fusion_metadata_{counter}.json")
                if not os.path.exists(metadata_file):
                    break
                
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Extract source workflow rounds from metadata
                    source_rounds = sorted([w["round"] for w in metadata.get("source_workflows", [])])
                    existing_signature = tuple(source_rounds)
                    
                    # Check if signatures match
                    if fusion_signature == existing_signature:
                        logger.info(f"Fusion combination {workflow_rounds} already attempted (found in fusion_metadata_{counter}.json)")
                        return True
                        
                except Exception as e:
                    logger.error(f"Error reading fusion metadata {counter}: {e}")
                
                counter += 1
            
            logger.info(f"Fusion combination {workflow_rounds} not attempted before")
            return False
            
        except Exception as e:
            logger.error(f"Error checking fusion history: {e}")
            return False
    
    def optimize(self, mode: OptimizerType = "Graph"):
        """Enhanced optimize method with integrated fusion"""
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
                    # Check if we should attempt fusion instead of optimization
                    if self._should_attempt_fusion():
                        logger.info(f"Attempting workflow fusion instead of optimization for round {self.round + 1}")
                        score = loop.run_until_complete(self._attempt_fusion())
                        if score is not None:
                            # Fusion successful, skip regular optimization
                            break
                        else:
                            # Fusion failed, fall back to regular optimization
                            logger.info("Fusion failed, falling back to regular optimization")
                    
                    # Regular optimization process
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

    def _should_attempt_fusion(self) -> bool:
        """
        Determine if we should attempt fusion based on current conditions
        
        Returns:
            bool: True if fusion should be attempted
        """
        if not self.enable_fusion:
            return False
        
        # Don't attempt fusion in the first round
        if self.round < 2:
            return False
        
        # Don't attempt fusion if we just did fusion in the previous round
        if self.last_fusion_round == self.round - 1:
            logger.info("Skipping fusion - fusion was attempted in the previous round")
            return False
        
        # Check if we have enough envelope workflows
        envelope_workflows = self.data_utils.find_envelope_workflows(self.max_envelope_workflows)
        if len(envelope_workflows) < self.max_envelope_workflows:
            logger.info(f"Insufficient workflows for fusion (found {len(envelope_workflows)}, need at least {self.max_envelope_workflows})")
            return False
        
        # Check if this specific fusion combination has been attempted before
        if self._check_fusion_already_attempted(envelope_workflows):
            logger.info("Skipping fusion - this combination has been attempted before")
            return False
        
        logger.info(f"Fusion conditions met: {len(envelope_workflows)} envelope workflows available")
        return True
    
    async def _attempt_fusion(self) -> float:
        """
        Attempt workflow fusion and evaluate the result
        
        Returns:
            float: Score of fused workflow if successful, None if fusion failed
        """
        fusion_score = None
        try:
            # Record that we're attempting fusion this round
            self.last_fusion_round = self.round
            
            # Get envelope workflows before fusion
            envelope_workflows = self.data_utils.find_envelope_workflows(self.max_envelope_workflows, True)
            min_envelope_score = min(w["avg_score"] for w in envelope_workflows)
            
            logger.info(f"Minimum envelope score: {min_envelope_score:.4f}")
            
            # Execute fusion process
            fusion_success = await self._execute_fusion_async()
            
            if not fusion_success:
                logger.error("Fusion process failed")
                return None
            
            # Evaluate the fused workflow
            fusion_score = await self._evaluate_fused_workflow()
            
            if fusion_score is None:
                logger.error("Failed to evaluate fused workflow")
                return None
            
            logger.info(f"Fused workflow score: {fusion_score:.4f}")
            
            # Check if fusion meets threshold
            if fusion_score > min_envelope_score + self.fusion_score_threshold:
                logger.info(f"Fusion successful! Score {fusion_score:.4f} > threshold {min_envelope_score + self.fusion_score_threshold:.4f}")
                
                # Move fused workflow to next round directory
                success = self._adopt_fused_workflow(envelope_workflows)
                if success:
                    # Save metadata for successful adoption
                    self._save_fusion_metadata(envelope_workflows, fusion_score, adopted=True)
                    # Add fused workflow to processed_experience.json
                    self._add_fusion_to_experience(envelope_workflows, fusion_score)
                    return fusion_score
                else:
                    logger.error("Failed to adopt fused workflow")
                    return None
            else:
                logger.info(f"Fusion score {fusion_score:.4f} below threshold {min_envelope_score + self.fusion_score_threshold:.4f}, discarding fused workflow")
                # Still save metadata for tracking purposes even if not adopted
                self._save_fusion_metadata(envelope_workflows, fusion_score, adopted=False)
                return None
                
        except Exception as e:
            logger.error(f"Error in fusion attempt: {e}")
            return None
        finally:
            # Always clean up temporary fusion directories
            self._cleanup_fusion_directories()
    
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
            
            # Create fusion prompt
            fusion_prompt = self._create_fusion_prompt(workflow_contents, operator_description)
            
            # Call LLM for fusion
            fusion_response = await self._call_fusion_llm(fusion_prompt)
            
            if not fusion_response:
                return False
            
            # Save fused workflow
            return self._save_fused_workflow(fusion_response, envelope_workflows)
            
        except Exception as e:
            logger.error(f"Error executing fusion: {e}")
            return False
    
    def _create_fusion_prompt(self, workflow_contents: List[Dict], operator_description: str) -> str:
        """
        Create fusion prompt for LLM using optimizer pattern
        
        Args:
            workflow_contents: List of workflow data dictionaries
            operator_description: Description of available operators
            
        Returns:
            Generated fusion prompt
        """
        # Format workflows for prompt
        workflows_formatted = ""
        
        for i, workflow in enumerate(workflow_contents, 1):
            solved_problems = workflow["solved_problems"]
            if isinstance(solved_problems, list):
                solved_problems = set(solved_problems)
            
            workflows_formatted += f"""
    <workflow_{i}>
        <round>{workflow['round']}</round>
        <score>{workflow['score']:.4f}</score>
        <solved_problems>{len(solved_problems)} problems</solved_problems>
        <prompt>
{workflow['prompt']}
        </prompt>
        <graph>
{workflow['graph']}
        </graph>
    </workflow_{i}>
"""
        
        # Use fusion processor's prompt generator with optimizer pattern
        return self.fusion_processor.prompt_generator.create_fusion_prompt(
            dataset=self.dataset,
            type=self.type,
            workflows=workflows_formatted,
            operator_description=operator_description
        )
    
    async def _call_fusion_llm(self, fusion_prompt: str) -> Dict[str, str]:
        """
        Call LLM for workflow fusion
        
        Args:
            fusion_prompt: Prompt for fusion
            
        Returns:
            Dict containing fusion response or None if failed
        """
        try:
            # Create formatter for fusion result
            from scripts.workflow_fusion import WorkflowFusionResult
            fusion_formatter = XmlFormatter.from_model(WorkflowFusionResult)
            
            # Call LLM with formatter
            response = await self.fusion_processor.fusion_llm.call_with_format(
                fusion_prompt,
                fusion_formatter
            )
            
            # Clean the graph content even for properly formatted responses
            if "graph" in response and response["graph"]:
                response["graph"] = self._clean_code_content(response["graph"])
            
            logger.info("Workflow fusion LLM call successful")
            return response
            
        except FormatError as e:
            logger.error(f"Format error in workflow fusion: {str(e)}")
            # Try fallback approach
            raw_response = await self.fusion_processor.fusion_llm(fusion_prompt)
            response = self._extract_fusion_fields(raw_response)
            if response:
                return response
            else:
                logger.error("Failed to extract fields from fusion response")
                return None
                
        except Exception as e:
            logger.error(f"Error calling fusion LLM: {e}")
            return None
    
    def _extract_fusion_fields(self, response: str) -> Dict[str, str]:
        """
        Extract fields from raw fusion response using regex
        
        Args:
            response: Raw response text from LLM
            
        Returns:
            Dictionary with extracted fields or None if extraction fails
        """
        try:
            import re
            
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
                    content = match.group(1).strip()
                    
                    # Clean up code blocks (especially for graph field)
                    if field == "graph":
                        content = self._clean_code_content(content)
                    
                    result[field] = content
            
            # Verify we have at least some content
            if not any(result.values()):
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting fusion fields: {e}")
            return None
    
    def _clean_code_content(self, content: str) -> str:
        """
        Clean code content by removing markdown code block markers
        
        Args:
            content: Raw code content that may contain ```python markers
            
        Returns:
            Cleaned code content
        """
        import re
        
        # Remove ```python and ``` markers
        content = re.sub(r'^```python\s*\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n?```\s*$', '', content, flags=re.MULTILINE)
        
        # Remove any leading/trailing whitespace but preserve internal formatting
        content = content.strip()
        
        return content
    
    def _save_fused_workflow(self, fusion_response: Dict[str, str], envelope_workflows: List[Dict]) -> bool:
        """
        Save the fused workflow to appropriate location
        
        Args:
            fusion_response: Response from fusion LLM
            envelope_workflows: Original envelope workflows used for fusion
            
        Returns:
            bool: True if save successful
        """
        try:
            # Create fusion directory in workflows (not workflows_fusion)
            fusion_dir = f"{self.root_path}/workflows"
            os.makedirs(fusion_dir, exist_ok=True)
            
            # Create round directory for fused workflow
            fusion_round_dir = os.path.join(fusion_dir, "round_fused")
            os.makedirs(fusion_round_dir, exist_ok=True)
            
            # Save fused workflow files using graph_utils pattern
            self.graph_utils.write_graph_files(
                fusion_round_dir,
                fusion_response,
                "fused",  # Use "fused" as round identifier
                self.dataset
            )
            
            logger.info(f"Fused workflow saved to {fusion_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving fused workflow: {e}")
            return False
    
    async def _evaluate_fused_workflow(self) -> float:
        """
        Evaluate the fused workflow and return its score
        
        Returns:
            float: Average score of fused workflow, None if evaluation failed
        """
        try:
            # Load fused workflow
            fusion_path = f"{self.root_path}/workflows"
            fused_graph = self.graph_utils.load_graph("fused", fusion_path)
            
            if fused_graph is None:
                logger.error("Failed to load fused workflow")
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
    
    def _adopt_fused_workflow(self, envelope_workflows: List[Dict]) -> bool:
        """
        Adopt the fused workflow as the next round's workflow
        
        Args:
            envelope_workflows: Original envelope workflows used for fusion
            
        Returns:
            bool: True if adoption successful
        """
        try:
            import shutil
            
            # Source: fused workflow directory
            fusion_dir = f"{self.root_path}/workflows/round_fused"
            
            # Target: next round directory
            next_round = self.round + 1
            target_dir = f"{self.root_path}/workflows/round_{next_round}"
            
            # Ensure target directory exists
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy fused workflow files to next round
            if os.path.exists(os.path.join(fusion_dir, "graph.py")):
                shutil.copy2(
                    os.path.join(fusion_dir, "graph.py"),
                    os.path.join(target_dir, "graph.py")
                )
            
            if os.path.exists(os.path.join(fusion_dir, "prompt.py")):
                shutil.copy2(
                    os.path.join(fusion_dir, "prompt.py"),
                    os.path.join(target_dir, "prompt.py")
                )
            
            # Create __init__.py
            init_file = os.path.join(target_dir, "__init__.py")
            with open(init_file, 'w') as f:
                f.write("")
            
            # Update the current graph to the fused one
            self.graph = self.graph_utils.load_graph(next_round, f"{self.root_path}/workflows")
            
            logger.info(f"Successfully adopted fused workflow as round_{next_round}")
            return True
            
        except Exception as e:
            logger.error(f"Error adopting fused workflow: {e}")
            return False

    def _save_fusion_metadata(self, envelope_workflows: List[Dict], fusion_score: float, adopted: bool = True) -> None:
        """
        Save fusion metadata with proper numbering
        
        Args:
            envelope_workflows: Original envelope workflows used for fusion
            fusion_score: Score achieved by fused workflow
            adopted: Whether the fused workflow was adopted
        """
        try:
            # Increment counter for new metadata file
            self.fusion_metadata_counter += 1
            
            # Create fusion metadata
            fusion_metadata = {
                "fusion_id": self.fusion_metadata_counter,
                "fusion_timestamp": time.time(),
                "fusion_round": self.round,
                "source_workflows": [
                    {
                        "round": w["round"],
                        "score": w["avg_score"],
                        "solved_problems_count": len(w["solved_problems"])
                    }
                    for w in envelope_workflows
                ],
                "fusion_score": fusion_score,
                "adopted": adopted,
                "target_round": self.round + 1 if adopted else None
            }
            
            # Save fusion metadata to numbered file
            fusion_dir = f"{self.root_path}/workflows"
            metadata_path = os.path.join(fusion_dir, f"fusion_metadata_{self.fusion_metadata_counter}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(fusion_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Fusion metadata saved to fusion_metadata_{self.fusion_metadata_counter}.json")
            
        except Exception as e:
            logger.error(f"Error saving fusion metadata: {e}")
    
    def _add_fusion_to_experience(self, envelope_workflows: List[Dict], fusion_score: float) -> None:
        """
        Add fused workflow information to processed_experience.json
        
        Args:
            envelope_workflows: Original envelope workflows used for fusion
            fusion_score: Score achieved by fused workflow
        """
        try:
            # Create fusion modification description
            source_rounds = [w["round"] for w in envelope_workflows]
            source_scores = [f"{w['avg_score']:.4f}" for w in envelope_workflows]
            
            fusion_modification = f"Fused from rounds {source_rounds} (scores: {source_scores}). " \
                                f"Combined the best aspects of {len(envelope_workflows)} high-performing workflows " \
                                f"to achieve improved coverage and performance."
            
            # Load current processed experience
            experience_path = f"{self.root_path}/workflows/processed_experience.json"
            processed_experience = {}
            
            if os.path.exists(experience_path):
                with open(experience_path, 'r', encoding='utf-8') as f:
                    processed_experience = json.load(f)
            
            # Add entry for the new fused round
            next_round = self.round + 1
            processed_experience[str(next_round)] = {
                "score": fusion_score,
                "success": {},
                "failure": {},
                "fusion_info": {
                    "is_fusion": True,
                    "source_rounds": source_rounds,
                    "fusion_modification": fusion_modification
                }
            }
            
            # Save updated experience
            with open(experience_path, 'w', encoding='utf-8') as f:
                json.dump(processed_experience, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Added fusion workflow to processed_experience.json for round {next_round}")
            
        except Exception as e:
            logger.error(f"Error adding fusion to experience: {e}")
    
    def _cleanup_fusion_directories(self) -> None:
        """
        Clean up temporary fusion directories (round_fused and round_fused_eval)
        """
        try:
            import shutil
            
            fusion_dirs = [
                f"{self.root_path}/workflows/round_fused",
                f"{self.root_path}/workflows/round_fused_eval"
            ]
            
            for fusion_dir in fusion_dirs:
                if os.path.exists(fusion_dir):
                    shutil.rmtree(fusion_dir)
                    logger.info(f"Cleaned up fusion directory: {fusion_dir}")
            
        except Exception as e:
            logger.error(f"Error cleaning up fusion directories: {e}")

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
