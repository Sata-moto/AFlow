"""
Workflow differentiation module for creating specialized workflows.

This module handles the differentiation process, creating specialized versions
of existing workflows that focus on specific problem types, strategies, or approaches.
"""

import asyncio
import os
import json
import time
from typing import List, Dict, Optional

from pydantic import BaseModel, Field

from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger
from scripts.prompts.differentiation_prompt import DifferentiationPromptGenerator
from scripts.utils.code_processor import CodeProcessor


class WorkflowDifferentiationResult(BaseModel):
    """Schema for workflow differentiation LLM output."""
    modification: str = Field(default="", description="Description of the differentiation modifications")
    graph: str = Field(default="", description="Differentiated workflow graph code")
    prompt: str = Field(default="", description="Differentiated workflow prompt")


class WorkflowDifferentiation:
    """
    Handles workflow differentiation process to create specialized workflows.
    """
    
    def __init__(
        self,
        dataset: str,
        question_type: str,
        opt_llm_config,
        exec_llm_config,
        operators: List[str],
        optimized_path: str = "workspace",
        validation_rounds: int = 1,
    ):
        self.dataset = dataset
        self.question_type = question_type
        self.opt_llm_config = opt_llm_config
        self.exec_llm_config = exec_llm_config
        self.operators = operators
        self.optimized_path = optimized_path
        self.validation_rounds = validation_rounds
        
        # Initialize LLM for differentiation
        self.differentiation_llm = create_llm_instance(self.opt_llm_config)
        
        # Initialize prompt generator
        self.prompt_generator = DifferentiationPromptGenerator()
        
        # Initialize root path
        self.root_path = f"{optimized_path}/{dataset}"
        
    def get_available_directions(self) -> List[str]:
        """Get available differentiation directions."""
        return self.prompt_generator.get_available_directions()
    
    def analyze_differentiation_candidates(self, workflow_results: List[Dict]) -> List[Dict]:
        """
        Analyze workflows to identify good candidates for problem type specialization.
        Focus on avoiding over-differentiation and selecting diverse high-quality workflows.
        
        Args:
            workflow_results: List of workflow performance data
            
        Returns:
            List of candidates sorted by selection priority
        """
        candidates = []
        
        for workflow in workflow_results:
            # Skip already differentiated workflows
            if self._is_differentiated_workflow(workflow):
                continue
                
            base_score = workflow.get("avg_score", 0.0)
            differentiation_count = workflow.get("differentiation_count", 0)
            
            # Apply differentiation count penalty: 5% reduction per previous differentiation
            final_weight = base_score * (1.0 - differentiation_count * 0.05)
            
            # Only consider workflows with good performance and not over-differentiated
            if final_weight > 0.3 and differentiation_count < 3:
                candidates.append({
                    "workflow": workflow,
                    "final_weight": final_weight,
                    "differentiation_count": differentiation_count
                })
        
        # Sort by final weight (performance with differentiation penalty)
        candidates.sort(key=lambda x: x["final_weight"], reverse=True)
        
        return candidates[:5]
    
    def _is_differentiated_workflow(self, workflow: Dict) -> bool:
        """
        Check if workflow was created through differentiation.
        
        Args:
            workflow: Workflow data
            
        Returns:
            bool: True if this is a differentiated workflow
        """
        return any([
            workflow.get("is_differentiated", False),
            workflow.get("differentiation_source") is not None,
            "differentiation" in workflow.get("creation_type", "").lower(),
            "specialized" in workflow.get("description", "").lower()
        ])
    
    def select_differentiation_direction(self, workflow: Dict, **kwargs) -> str:
        """
        Select differentiation direction. Since we only support problem type specialization,
        always returns the same direction.
        
        Returns:
            str: Always returns "problem_type_specialization"
        """
        return "problem_type_specialization"
    
    async def create_differentiated_workflow(
        self,
        source_workflow: Dict,
        differentiation_direction: str,
        operator_description: str,
        target_round: int,
        performance_gaps: List[Dict] = None
    ) -> Optional[Dict[str, str]]:
        """
        Create a differentiated workflow from a source workflow.
        
        Args:
            source_workflow: Source workflow data
            differentiation_direction: Direction for specialization
            operator_description: Available operators description
            performance_gaps: Performance gaps to address
            
        Returns:
            Dict containing differentiation response or None if failed
        """
        try:
            # Create differentiation prompt
            diff_prompt = self.prompt_generator.create_differentiation_prompt(
                dataset=self.dataset,
                question_type=self.question_type,
                source_workflow=source_workflow,
                operator_description=operator_description,
                differentiation_direction=differentiation_direction,
                target_round=target_round,
                performance_gaps=performance_gaps
            )
            
            # Call LLM for differentiation
            response = await self._call_differentiation_llm(diff_prompt)
            
            if response:
                logger.info(f"Successfully created differentiated workflow in direction: {differentiation_direction}")
                return response
            else:
                logger.warning(f"Failed to create differentiated workflow for direction: {differentiation_direction}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating differentiated workflow: {e}")
            return None
    
    async def _call_differentiation_llm(self, differentiation_prompt: str) -> Optional[Dict[str, str]]:
        """
        Call LLM for workflow differentiation.
        
        Args:
            differentiation_prompt: Prompt for differentiation
            
        Returns:
            Dict containing differentiation response or None if failed
        """
        try:
            # Create formatter for differentiation result
            diff_formatter = XmlFormatter.from_model(WorkflowDifferentiationResult)
            
            # Call LLM with formatter
            response = await self.differentiation_llm.call_with_format(
                differentiation_prompt,
                diff_formatter
            )
            
            # Clean the graph content
            if "graph" in response and response["graph"]:
                response["graph"] = CodeProcessor.clean_code_content(response["graph"])
            
            # Clean the prompt content  
            if "prompt" in response and response["prompt"]:
                response["prompt"] = CodeProcessor.clean_prompt_content(response["prompt"])
            
            logger.info("Workflow differentiation LLM call successful")
            return response
            
        except FormatError as e:
            logger.warning(f"Format error in workflow differentiation: {str(e)}")
            # Try fallback approach
            raw_response = await self.differentiation_llm(differentiation_prompt)
            response = CodeProcessor.extract_fields_from_response(
                raw_response, 
                ["modification", "graph", "prompt"]
            )
            if response:
                logger.info("Fallback differentiation extraction successful")
                return response
            else:
                logger.error("Failed to extract fields from differentiation response")
                return None
                
        except Exception as e:
            logger.error(f"Error calling differentiation LLM: {e}")
            return None
    
    def save_differentiation_metadata(
        self,
        source_workflow: Dict,
        differentiated_workflow: Dict,
        differentiation_direction: str,
        target_round: int,
        differentiation_score: float = None
    ) -> None:
        """
        Save differentiation metadata for tracking and analysis.
        
        Args:
            source_workflow: Original workflow that was differentiated
            differentiated_workflow: Result of differentiation
            differentiation_direction: Direction used for differentiation
            target_round: Round number where differentiated workflow was saved
            differentiation_score: Score achieved by differentiated workflow
        """
        try:
            # Create differentiation metadata
            metadata = {
                "differentiation_timestamp": time.time(),
                "source_workflow": {
                    "round": source_workflow.get("round"),
                    "score": source_workflow.get("avg_score"),
                    "solved_problems_count": len(source_workflow.get("solved_problems", []))
                },
                "differentiation_direction": differentiation_direction,
                "target_round": target_round,
                "differentiation_score": differentiation_score,
                "modification_summary": differentiated_workflow.get("modification", "")[:500]  # Truncate for brevity
            }
            
            # Save metadata to workflows directory
            workflows_dir = f"{self.root_path}/workflows"
            metadata_file = f"differentiation_metadata_{target_round}.json"
            metadata_path = os.path.join(workflows_dir, metadata_file)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Differentiation metadata saved to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving differentiation metadata: {e}")
    
    def save_differentiated_workflow_direct(
        self,
        differentiation_response: Dict[str, str],
        source_workflow: Dict,
        differentiation_direction: str,
        target_round: int,
        root_path: str,
        graph_utils,
        experience_utils
    ) -> bool:
        """
        Save the differentiated workflow directly to the target round directory.
        
        Args:
            differentiation_response: Response from differentiation LLM
            source_workflow: Source workflow that was differentiated
            differentiation_direction: Direction used for differentiation
            target_round: Target round number
            root_path: Root path for workflows
            graph_utils: Graph utilities instance
            experience_utils: Experience utilities instance
            
        Returns:
            bool: True if save successful
        """
        try:
            # Create target round directory
            workflows_dir = f"{root_path}/workflows"
            target_dir = os.path.join(workflows_dir, f"round_{target_round}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Save differentiated workflow files
            graph_utils.write_graph_files(
                target_dir,
                differentiation_response,
                target_round,
                self.dataset
            )
            
            # Create __init__.py
            init_file = os.path.join(target_dir, "__init__.py")
            with open(init_file, 'w') as f:
                f.write("")
            
            # Create experience.json for differentiated workflow
            self.create_differentiation_experience_file(
                target_dir, source_workflow, differentiation_direction, 
                differentiation_response, experience_utils
            )
            
            # Create log.json for differentiated workflow
            self.create_differentiation_log_file(
                target_dir, source_workflow, differentiation_direction
            )
            
            logger.info(f"Differentiated workflow saved directly to round_{target_round}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving differentiated workflow directly: {e}")
            return False
    
    def create_differentiation_experience_file(
        self,
        target_dir: str,
        source_workflow: Dict,
        differentiation_direction: str,
        differentiation_response: Dict[str, str],
        experience_utils
    ) -> None:
        """
        Create experience.json file for differentiated workflow.
        
        Args:
            target_dir: Target directory for the differentiated round
            source_workflow: Source workflow that was differentiated
            differentiation_direction: Direction used for differentiation
            differentiation_response: Response from differentiation LLM
            experience_utils: Experience utilities instance
        """
        try:
            # Create differentiation modification description
            differentiation_modification = f"Workflow Differentiation: Specialized from round {source_workflow['round']} " \
                                         f"(score: {source_workflow['avg_score']:.4f}) in direction '{differentiation_direction}'. " \
                                         f"{differentiation_response.get('modification', 'Enhanced with targeted specialization.')[:200]}"
            
            # Create a mock sample object for the differentiation
            differentiation_sample = {
                "round": source_workflow["round"],  # Use source workflow as father node
                "score": source_workflow["avg_score"]  # Use source score as "before"
            }
            
            # Use standard experience creation method
            experience_data = experience_utils.create_experience_data(
                differentiation_sample, differentiation_modification
            )
            
            # Save experience.json
            experience_path = os.path.join(target_dir, "experience.json")
            with open(experience_path, 'w', encoding='utf-8') as f:
                json.dump(experience_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Created differentiation experience.json with father node {source_workflow['round']}")
            
        except Exception as e:
            logger.error(f"Error creating differentiation experience file: {e}")
    
    def create_differentiation_log_file(
        self,
        target_dir: str,
        source_workflow: Dict,
        differentiation_direction: str
    ) -> None:
        """
        Create log.json file for differentiated workflow.
        
        Args:
            target_dir: Target directory for the differentiated round
            source_workflow: Source workflow that was differentiated
            differentiation_direction: Direction used for differentiation
        """
        try:
            # Create log data for differentiation
            log_data = {
                "differentiation_metadata": {
                    "timestamp": time.time(),
                    "differentiation_type": "workflow_specialization",
                    "source_workflow": {
                        "round": source_workflow["round"],
                        "score": source_workflow["avg_score"],
                        "solved_problems": len(source_workflow.get("solved_problems", []))
                    },
                    "differentiation_direction": differentiation_direction,
                    "strategy": f"Specialized the workflow from round {source_workflow['round']} to focus on {differentiation_direction}"
                },
                # Initialize empty list for actual execution logs (will be populated during evaluation)
                "execution_logs": []
            }
            
            # Save log.json
            log_path = os.path.join(target_dir, "log.json")
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Created differentiation log.json with direction '{differentiation_direction}'")
            
        except Exception as e:
            logger.error(f"Error creating differentiation log file: {e}")
