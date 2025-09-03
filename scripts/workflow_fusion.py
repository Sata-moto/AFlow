# -*- coding: utf-8 -*-
# @Date    : 8/28/2025 20:00 PM
# @Author  : didi
# @Desc    : Workflow Fusion Module for AFlow

import asyncio
import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger
from scripts.prompts.fusion_prompt import FusionPromptGenerator
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils


class WorkflowFusionResult(BaseModel):
    """Model for workflow fusion result"""
    modification: str = Field(default="", description="Description of the fusion modifications")
    graph: str = Field(default="", description="Fused workflow graph code")
    prompt: str = Field(default="", description="Fused workflow prompt")


class WorkflowFusion:
    """
    Class to handle workflow fusion process
    """
    
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
        
        # Initialize LLM for fusion
        self.fusion_llm = create_llm_instance(self.opt_llm_config)
        
        # Initialize prompt generator
        self.prompt_generator = FusionPromptGenerator()
        
        # Initialize utility classes
        self.root_path = f"{optimized_path}/{dataset}"
        self.data_utils = DataUtils(self.root_path)
        self.graph_utils = GraphUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        
        # Initialize fused workflow storage
        self.fused_workflow = None
        
    def execute_fusion(self):
        """
        Execute the complete workflow fusion process
        """
        logger.info("Starting workflow fusion process...")
        
        # Step 1: Find envelope workflows
        envelope_workflows = self.data_utils.find_envelope_workflows(self.max_envelope_workflows)
        
        if len(envelope_workflows) < 2:
            logger.info(f"Insufficient workflows for fusion (found {len(envelope_workflows)}, need at least 2)")
            logger.info("Skipping fusion process.")
            return False
        
        logger.info(f"Found {len(envelope_workflows)} envelope workflows for fusion")
        
        # Step 2: Create fusion event loop and execute fusion
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            fusion_result = loop.run_until_complete(self._fuse_workflows(envelope_workflows))
            if fusion_result:
                logger.info("Workflow fusion completed successfully")
                return True
            else:
                logger.error("Workflow fusion failed")
                return False
        except Exception as e:
            logger.error(f"Error during workflow fusion: {e}")
            return False
        finally:
            loop.close()
    
    async def _fuse_workflows(self, envelope_workflows: List[Dict]) -> bool:
        """
        Fuse the envelope workflows into a single workflow
        
        Args:
            envelope_workflows: List of envelope workflow data
            
        Returns:
            bool: True if fusion successful, False otherwise
        """
        try:
            # Step 1: Load workflow contents for fusion
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
            
            # Step 2: Get operator descriptions
            operator_description = self.graph_utils.load_operators_description(self.operators)
            
            # Step 3: Create fusion prompt
            fusion_prompt = self._create_fusion_prompt(workflow_contents, operator_description)
            
            # Step 4: Call LLM for fusion
            fusion_response = await self._call_fusion_llm(fusion_prompt)
            
            if not fusion_response:
                return False
            
            # Step 5: Save fused workflow
            return self._save_fused_workflow(fusion_response, envelope_workflows)
            
        except Exception as e:
            logger.error(f"Error in workflow fusion: {e}")
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
        
        # Use prompt generator with optimizer pattern
        return self.prompt_generator.create_fusion_prompt(
            dataset=self.dataset,
            type=self.question_type,
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
            fusion_formatter = XmlFormatter.from_model(WorkflowFusionResult)
            
            # Call LLM with formatter
            response = await self.fusion_llm.call_with_format(
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
            raw_response = await self.fusion_llm(fusion_prompt)
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
            
            # Save fusion metadata
            fusion_metadata = {
                "fusion_timestamp": os.times(),
                "source_workflows": [
                    {
                        "round": w["round"],
                        "score": w["avg_score"],
                        "solved_problems_count": len(w["solved_problems"])
                    }
                    for w in envelope_workflows
                ],
                "modification": fusion_response["modification"]
            }
            
            # Save fusion metadata to workflows directory
            metadata_path = os.path.join(fusion_dir, "fusion_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(fusion_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            # Save fused workflow files using graph_utils pattern
            self.graph_utils.write_graph_files(
                fusion_round_dir,
                fusion_response,
                "fused",  # Use "fused" as round identifier
                self.dataset
            )
            
            # Store reference to fused workflow
            self.fused_workflow = {
                "directory": fusion_round_dir,
                "metadata": fusion_metadata,
                "response": fusion_response
            }
            
            logger.info(f"Fused workflow saved to {fusion_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving fused workflow: {e}")
            return False
    
    def test_fused_workflow(self):
        """
        Test the fused workflow on test dataset
        """
        if not self.fused_workflow:
            logger.error("No fused workflow available for testing")
            return
        
        logger.info("Testing fused workflow...")
        
        # Create event loop for testing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._test_fused_workflow_async())
        except Exception as e:
            logger.error(f"Error testing fused workflow: {e}")
        finally:
            loop.close()
    
    async def _test_fused_workflow_async(self):
        """
        Async method to test fused workflow
        """
        try:
            # Load fused workflow from correct path (workflows/round_fused)
            fusion_path = f"{self.root_path}/workflows"
            fused_graph = self.graph_utils.load_graph("fused", fusion_path)
            
            # Create test directory
            test_dir = os.path.join(fusion_path, "test_results")
            os.makedirs(test_dir, exist_ok=True)
            
            # Run evaluation on test set
            score, avg_cost, total_cost, solved_problems = await self.evaluation_utils.evaluate_graph_test(
                type('MockOptimizer', (), {
                    'dataset': self.dataset,
                    'graph': fused_graph,
                    'execute_llm_config': self.exec_llm_config
                })(),
                test_dir,
                is_test=True
            )
            
            # Save test results
            test_results = {
                "fused_workflow_score": score,
                "fused_workflow_avg_cost": avg_cost,
                "fused_workflow_total_cost": total_cost,
                "fused_workflow_solved_problems": len(solved_problems),
                "test_timestamp": str(os.times())
            }
            
            results_path = os.path.join(test_dir, "fused_test_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Fused workflow test completed:")
            logger.info(f"  Score: {score:.4f}")
            logger.info(f"  Avg Cost: {avg_cost:.4f}")
            logger.info(f"  Total Cost: {total_cost:.4f}")
            logger.info(f"  Solved Problems: {len(solved_problems)}")
            logger.info(f"  Results saved to: {results_path}")
            
        except Exception as e:
            logger.error(f"Error in fused workflow testing: {e}")
            import traceback
            traceback.print_exc()
