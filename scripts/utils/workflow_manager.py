"""
Workflow management utilities for the enhanced optimizer.

This module provides utilities for managing workflows, including round summaries,
metadata extraction, and workflow content loading.
"""

import os
import json
from typing import List, Dict, Set
from collections import defaultdict
from scripts.logs import logger


class WorkflowManager:
    """Utilities for managing workflow data and operations."""
    
    def __init__(self, root_path: str, data_utils, graph_utils):
        self.root_path = root_path
        self.data_utils = data_utils
        self.graph_utils = graph_utils
    
    def get_round_summaries(self, workflow_results: List[Dict]) -> List[Dict]:
        """
        Get round summaries from workflow results.
        
        Args:
            workflow_results: Raw workflow results
            
        Returns:
            List of round summaries with averaged metrics
        """
        # Group results by round
        rounds_data = defaultdict(lambda: {
            "scores": [],
            "avg_costs": [],
            "total_costs": [],
            "solved_problems": set()
        })
        
        for result in workflow_results:
            round_num = result.get("round", 0)
            rounds_data[round_num]["scores"].append(result.get("score", 0))
            rounds_data[round_num]["avg_costs"].append(result.get("avg_cost", 0))
            rounds_data[round_num]["total_costs"].append(result.get("total_cost", 0))
            
            # Add solved problems (handle both list and set formats)
            solved = result.get("solved_problems", [])
            if isinstance(solved, (list, set)):
                rounds_data[round_num]["solved_problems"].update(solved)
        
        # Calculate summaries
        round_summaries = []
        for round_num, data in rounds_data.items():
            if data["scores"]:  # Only include rounds with data
                summary = {
                    "round": round_num,
                    "avg_score": sum(data["scores"]) / len(data["scores"]),
                    "best_score": max(data["scores"]),
                    "worst_score": min(data["scores"]),
                    "avg_cost": sum(data["avg_costs"]) / len(data["avg_costs"]) if data["avg_costs"] else 0,
                    "total_cost": sum(data["total_costs"]) if data["total_costs"] else 0,
                    "solved_problems": data["solved_problems"],
                    "problems_count": len(data["solved_problems"])
                }
                round_summaries.append(summary)
        
        # Sort by round number
        round_summaries.sort(key=lambda x: x["round"])
        return round_summaries
    
    async def load_workflow_content(self, round_number: int) -> Dict:
        """
        Load workflow content (prompt and graph) for a given round.
        
        Args:
            round_number: Round number to load
            
        Returns:
            Dict containing workflow content
        """
        workflows_path = f"{self.root_path}/workflows"
        
        # Read graph and prompt files
        prompt, graph_load = self.graph_utils.read_graph_files(round_number, workflows_path)
        graph = self.graph_utils.extract_solve_graph(graph_load)
        
        # Get round summary for score and solved problems
        workflow_results = self.data_utils.load_results(workflows_path)
        round_summaries = self.get_round_summaries(workflow_results)
        
        round_summary = next((r for r in round_summaries if r["round"] == round_number), {})
        
        return {
            "round": round_number,
            "prompt": prompt,
            "graph": graph[0] if graph else "",
            "score": round_summary.get("avg_score", 0.0),
            "solved_problems": round_summary.get("solved_problems", set())
        }
    
    def save_workflow_files(
        self, 
        target_dir: str, 
        graph_content: str, 
        prompt_content: str
    ) -> bool:
        """
        Save workflow files to target directory.
        
        Args:
            target_dir: Target directory path
            graph_content: Graph code content
            prompt_content: Prompt content
            
        Returns:
            bool: True if save successful
        """
        try:
            # Save graph.py file
            graph_path = os.path.join(target_dir, "graph.py")
            with open(graph_path, 'w', encoding='utf-8') as f:
                f.write(graph_content)
            
            # Save prompt.py file  
            prompt_path = os.path.join(target_dir, "prompt.py")
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt_content)
                
            logger.info(f"Workflow files saved to {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving workflow files: {e}")
            return False
    
    def create_experience_file(
        self,
        target_dir: str,
        source_info: Dict,
        operation_type: str,
        additional_info: Dict = None
    ) -> None:
        """
        Create experience.json file with operation metadata.
        
        Args:
            target_dir: Target directory for the experience file
            source_info: Information about source workflows
            operation_type: Type of operation (fusion, differentiation, optimization)
            additional_info: Additional metadata specific to the operation
        """
        try:
            experience_data = {
                "operation_type": operation_type,
                "source_workflows": source_info.get("source_workflows", []),
                "timestamp": source_info.get("timestamp", ""),
                "parameters": additional_info or {}
            }
            
            if operation_type == "fusion":
                experience_data.update({
                    "envelope_workflows": source_info.get("envelope_workflows", []),
                    "fusion_strategy": additional_info.get("fusion_strategy", "envelope_combination")
                })
            elif operation_type == "differentiation":
                experience_data.update({
                    "source_workflow_round": source_info.get("source_round", 0),
                    "differentiation_direction": additional_info.get("direction", ""),
                    "specialization_focus": additional_info.get("focus", "")
                })
            
            experience_path = os.path.join(target_dir, "experience.json")
            with open(experience_path, 'w', encoding='utf-8') as f:
                json.dump(experience_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Experience file created in {target_dir}")
            
        except Exception as e:
            logger.error(f"Error creating experience file: {e}")
    
    def create_log_file(
        self,
        target_dir: str,
        operation_info: Dict
    ) -> None:
        """
        Create log.json file for the operation.
        
        Args:
            target_dir: Target directory for the log file
            operation_info: Information about the operation performed
        """
        try:
            import time
            
            log_data = {
                "timestamp": time.time(),
                "operation": operation_info.get("operation", "unknown"),
                "status": "completed",
                "details": operation_info.get("details", {}),
                "performance": operation_info.get("performance", {})
            }
            
            log_path = os.path.join(target_dir, "log.json")
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Log file created in {target_dir}")
            
        except Exception as e:
            logger.error(f"Error creating log file: {e}")
    
    def get_differentiation_history(self) -> Dict[str, Dict]:
        """
        Get differentiation history for all categories.
        
        Returns:
            Dict mapping category names to their differentiation info:
            {
                "category_name": {
                    "last_differentiation_round": int,
                    "differentiation_count": int
                }
            }
        """
        differentiation_history = {}
        
        try:
            workflows_path = f"{self.root_path}/workflows"
            
            # Check all round directories for differentiation metadata
            for round_dir in os.listdir(workflows_path):
                if not round_dir.startswith("round_"):
                    continue
                
                round_path = os.path.join(workflows_path, round_dir)
                if not os.path.isdir(round_path):
                    continue
                
                # Check for differentiation metadata
                experience_path = os.path.join(round_path, "experience.json")
                if os.path.exists(experience_path):
                    try:
                        with open(experience_path, 'r', encoding='utf-8') as f:
                            experience = json.load(f)
                        
                        # Check if this is a differentiated workflow
                        operation = experience.get("operation", {})
                        if operation.get("type") == "differentiation":
                            target_category = operation.get("target_category")
                            if target_category:
                                round_num = int(round_dir.split("_")[1])
                                
                                if target_category not in differentiation_history:
                                    differentiation_history[target_category] = {
                                        "last_differentiation_round": round_num,
                                        "differentiation_count": 1
                                    }
                                else:
                                    history = differentiation_history[target_category]
                                    history["last_differentiation_round"] = max(
                                        history["last_differentiation_round"], 
                                        round_num
                                    )
                                    history["differentiation_count"] += 1
                    
                    except Exception as e:
                        logger.warning(f"Error reading experience file {experience_path}: {e}")
                        continue
            
            return differentiation_history
            
        except Exception as e:
            logger.error(f"Error getting differentiation history: {e}")
            return {}


class FusionChecker:
    """Utilities for checking fusion-related conditions."""
    
    def __init__(self, root_path: str):
        self.root_path = root_path
    
    def check_fusion_already_attempted(self, envelope_workflows: List[Dict]) -> bool:
        """
        Check if this specific fusion combination has been attempted before.
        
        Args:
            envelope_workflows: List of workflows to be fused
            
        Returns:
            bool: True if this combination was already attempted
        """
        try:
            # Create signature for this fusion combination
            workflow_rounds = sorted([w.get("round", 0) for w in envelope_workflows])
            fusion_signature = f"rounds_{'-'.join(map(str, workflow_rounds))}"
            
            # Check existing fusion metadata files
            fusion_metadata_dir = f"{self.root_path}/workflows"
            
            for i in range(100):  # Check up to 100 fusion metadata files
                metadata_file = os.path.join(fusion_metadata_dir, f"fusion_metadata_{i}.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            
                        # Check if this combination matches
                        existing_rounds = metadata.get("source_workflow_rounds", [])
                        if sorted(existing_rounds) == workflow_rounds:
                            logger.info(f"Fusion combination {fusion_signature} already attempted")
                            return True
                            
                    except Exception as e:
                        logger.warning(f"Error reading fusion metadata {metadata_file}: {e}")
                        continue
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking fusion attempts: {e}")
            return False  # Default to allowing fusion if check fails
