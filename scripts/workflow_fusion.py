from typing import List, Dict, Optional
import os
import json
import time
from pydantic import BaseModel, Field
from scripts.async_llm import create_llm_instance
from scripts.prompts.fusion_prompt import FusionPromptGenerator
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger
from scripts.utils.code_processor import CodeProcessor


class WorkflowFusionResult(BaseModel):
    """Schema for workflow fusion LLM output."""
    modification: str = Field(default="", description="Description of the fusion modifications")
    graph: str = Field(default="", description="Fused workflow graph code")
    prompt: str = Field(default="", description="Fused workflow prompt")


class WorkflowFusion:
    def __init__(
        self,
        dataset: str,
        question_type: str,
        opt_llm_config,
        exec_llm_config,
        operators: List[str],
        optimized_path: str = "workspace",
        max_envelope_workflows: int = 3,
        validation_rounds: int = 1,
    ):
        self.dataset = dataset
        self.question_type = question_type
        self.opt_llm_config = opt_llm_config
        self.exec_llm_config = exec_llm_config
        self.operators = operators
        self.optimized_path = optimized_path
        self.max_envelope_workflows = max_envelope_workflows
        self.validation_rounds = validation_rounds

        # LLM and prompt generator used by EnhancedOptimizer
        self.fusion_llm = create_llm_instance(self.opt_llm_config)
        self.prompt_generator = FusionPromptGenerator()
        
        # Initialize root path
        self.root_path = f"{optimized_path}/{dataset}"
    
    def create_fusion_prompt(self, workflow_contents: List[Dict], operator_description: str) -> str:
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
        
        # Use prompt generator with optimizer pattern
        return self.prompt_generator.create_fusion_prompt(
            dataset=self.dataset,
            type=self.question_type,
            workflows=workflows_formatted,
            operator_description=operator_description
        )
    
    async def call_fusion_llm(self, fusion_prompt: str) -> Optional[Dict[str, str]]:
        """
        Call LLM for workflow fusion
        
        Args:
            fusion_prompt: Prompt for fusion
            
        Returns:
            Dict containing fusion response or None if failed
        """
        try:
            # Create formatter for fusion result
            fusion_formatter = XmlFormatter.from_model(WorkflowFusionResult)
            
            # Call LLM with formatter
            response = await self.fusion_llm.call_with_format(
                fusion_prompt,
                fusion_formatter
            )
            
            # Clean the graph content even for properly formatted responses
            if "graph" in response and response["graph"]:
                response["graph"] = CodeProcessor.clean_code_content(response["graph"])
            
            # Clean the prompt content
            if "prompt" in response and response["prompt"]:
                response["prompt"] = CodeProcessor.clean_prompt_content(response["prompt"])
            
            logger.info("Workflow fusion LLM call successful")
            return response
            
        except FormatError as e:
            logger.error(f"Format error in workflow fusion: {str(e)}")
            # Try fallback approach
            raw_response = await self.fusion_llm(fusion_prompt)
            response = CodeProcessor.extract_fields_from_response(
                raw_response,
                ["modification", "graph", "prompt"]
            )
            if response:
                logger.info("Fallback fusion extraction successful")
                return response
            else:
                logger.error("Failed to extract fields from fusion response")
                return None
                
        except Exception as e:
            logger.error(f"Error calling fusion LLM: {e}")
            return None
    
    async def create_fused_workflow(
        self,
        envelope_workflows: List[Dict],
        workflow_contents: List[Dict],
        operator_description: str
    ) -> Optional[Dict[str, str]]:
        """
        Create a fused workflow from envelope workflows.
        
        Args:
            envelope_workflows: Envelope workflows metadata
            workflow_contents: Loaded workflow contents (prompt, graph, etc.)
            operator_description: Available operators description
            
        Returns:
            Dict containing fusion response or None if failed
        """
        try:
            # Create fusion prompt
            fusion_prompt = self.create_fusion_prompt(workflow_contents, operator_description)
            
            # Call LLM for fusion
            response = await self.call_fusion_llm(fusion_prompt)
            
            if response:
                logger.info("Successfully created fused workflow")
                return response
            else:
                logger.warning("Failed to create fused workflow")
                return None
                
        except Exception as e:
            logger.error(f"Error creating fused workflow: {e}")
            return None
    
    def save_fusion_metadata(
        self, 
        envelope_workflows: List[Dict], 
        fusion_score: float, 
        fusion_round: int,
        fusion_id: int,
        adopted: bool = True
    ) -> None:
        """
        Save fusion metadata with proper numbering
        
        Args:
            envelope_workflows: Original envelope workflows used for fusion
            fusion_score: Score achieved by fused workflow
            fusion_round: Round when fusion was attempted
            fusion_id: Unique fusion identifier
            adopted: Whether the fused workflow was adopted
        """
        try:
            # Create fusion metadata
            fusion_metadata = {
                "fusion_id": fusion_id,
                "fusion_timestamp": time.time(),
                "fusion_round": fusion_round,
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
                "target_round": fusion_round + 1 if adopted else None
            }
            
            # Save fusion metadata to numbered file
            fusion_dir = f"{self.root_path}/workflows"
            metadata_path = os.path.join(fusion_dir, f"fusion_metadata_{fusion_id}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(fusion_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Fusion metadata saved to fusion_metadata_{fusion_id}.json")
            
        except Exception as e:
            logger.error(f"Error saving fusion metadata: {e}")
    
    def create_fusion_experience_file(
        self, 
        target_dir: str, 
        envelope_workflows: List[Dict], 
        fusion_round: int,
        experience_utils
    ) -> None:
        """
        Create experience.json file for fusion workflow following the standard format
        
        Args:
            target_dir: Target directory for the fusion round
            envelope_workflows: Source workflows that were fused
            fusion_round: The round number of this fusion
            experience_utils: Experience utilities instance
        """
        try:
            # Find the best source workflow to use as "father node"
            best_workflow = max(envelope_workflows, key=lambda w: w["avg_score"])
            
            # Create fusion modification description
            source_rounds = [w["round"] for w in envelope_workflows]
            source_scores = [f"{w['avg_score']:.4f}" for w in envelope_workflows]
            
            fusion_modification = f"Workflow Fusion: Combined high-performing workflows from rounds {source_rounds} " \
                                f"(scores: {source_scores}). Integrated the best aspects of {len(envelope_workflows)} " \
                                f"workflows to achieve improved coverage and performance."
            
            # Create a mock sample object for the fusion
            fusion_sample = {
                "round": best_workflow["round"],  # Use best workflow as father node
                "score": best_workflow["avg_score"]  # Use best score as "before"
            }
            
            # Use standard experience creation method
            experience_data = experience_utils.create_experience_data(fusion_sample, fusion_modification)
            
            # Save experience.json using standard method
            experience_path = os.path.join(target_dir, "experience.json")
            with open(experience_path, 'w', encoding='utf-8') as f:
                json.dump(experience_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Created fusion experience.json with father node {best_workflow['round']}")
            
        except Exception as e:
            logger.error(f"Error creating fusion experience file: {e}")
    
    def create_fusion_log_file(self, target_dir: str, envelope_workflows: List[Dict]) -> None:
        """
        Create log.json file for fusion workflow
        
        Args:
            target_dir: Target directory for the fusion round
            envelope_workflows: Source workflows that were fused
        """
        try:
            # Create a comprehensive log entry for fusion
            log_data = {
                "fusion_metadata": {
                    "timestamp": time.time(),
                    "fusion_type": "envelope_fusion",
                    "source_workflows": [
                        {
                            "round": w["round"],
                            "score": w["avg_score"],
                            "solved_problems": len(w["solved_problems"])
                        }
                        for w in envelope_workflows
                    ],
                    "total_unique_problems": len(set().union(*[set(w["solved_problems"]) for w in envelope_workflows])),
                    "fusion_strategy": "Combined best aspects of envelope workflows using LLM-guided fusion"
                },
                # Initialize empty list for actual execution logs (will be populated during evaluation)
                "execution_logs": []
            }
            
            # Save log.json
            log_path = os.path.join(target_dir, "log.json")
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Created fusion log.json with metadata for {len(envelope_workflows)} source workflows")
            
        except Exception as e:
            logger.error(f"Error creating fusion log file: {e}")
    
    def save_fused_workflow_direct(
        self, 
        fusion_response: Dict[str, str], 
        envelope_workflows: List[Dict], 
        root_path: str,
        current_round: int,
        graph_utils,
        experience_utils
    ) -> bool:
        """
        Save the fused workflow directly to the next round directory
        
        Args:
            fusion_response: Response from fusion LLM
            envelope_workflows: Original envelope workflows used for fusion
            root_path: Root path for workflows
            current_round: Current optimization round
            graph_utils: Graph utilities instance
            experience_utils: Experience utilities instance
            
        Returns:
            bool: True if save successful
        """
        try:
            # Calculate next round number
            next_round = current_round + 1
            
            # Create next round directory directly
            workflows_dir = f"{root_path}/workflows"
            target_dir = os.path.join(workflows_dir, f"round_{next_round}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Save fused workflow files using graph_utils pattern
            graph_utils.write_graph_files(
                target_dir,
                fusion_response,
                next_round,  # Use actual round number
                self.dataset
            )
            
            # Create __init__.py
            init_file = os.path.join(target_dir, "__init__.py")
            with open(init_file, 'w') as f:
                f.write("")
            
            # Create experience.json for fusion workflow
            self.create_fusion_experience_file(
                target_dir, envelope_workflows, next_round, experience_utils
            )
            
            # Create log.json for fusion workflow
            self.create_fusion_log_file(target_dir, envelope_workflows)
            
            logger.info(f"Fused workflow saved directly to round_{next_round}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving fused workflow directly: {e}")
            return False