"""
Differentiation prompt generator for workflow specialization.

This module generates prompts to guide LLMs in creating specialized workflows
that diverge from existing ones in specific directions (problem types, strategies, etc.).
"""

from typing import List, Dict


class DifferentiationPromptGenerator:
    """Generate prompts for workflow differentiation/specialization."""
    
    def __init__(self):
        self.differentiation_strategies = [
            "problem_type_specialization", 
        ]
    
    def generate_differentiation_prompt(
        self,
        candidates: List[Dict],
        direction: str,
        dataset: str,
        question_type: str
    ) -> str:
        """
        Generate a differentiation prompt based on candidate workflows and direction.
        
        Args:
            candidates: List of candidate workflows with scores and metadata
            direction: Differentiation direction to pursue
            dataset: Target dataset name
            question_type: Type of questions (math, code, qa)
            
        Returns:
            Generated differentiation prompt
        """
        # Select best candidate as source
        best_candidate = max(candidates, key=lambda x: x.get('score', 0))
        
        return self.create_differentiation_prompt(
            dataset=dataset,
            question_type=question_type,
            source_workflow=best_candidate.get('workflow_data', {}),
            operator_description="Available operators for workflow construction",
            differentiation_direction=direction,
            performance_gaps=None
        )
    
    def create_differentiation_prompt(
        self,
        dataset: str,
        question_type: str,
        source_workflow: Dict,
        operator_description: str,
        differentiation_direction: str,
        target_round: int,
        performance_gaps: List[Dict] = None
    ) -> str:
        """
        Create a differentiation prompt for workflow specialization.
        
        Args:
            dataset: Target dataset name
            question_type: Type of questions (math, code, qa)
            source_workflow: Source workflow data containing prompt, graph, score, solved_problems
            operator_description: Available operators description
            differentiation_direction: Direction for specialization
            performance_gaps: Performance gaps analysis for targeted improvement
            
        Returns:
            Generated differentiation prompt
        """
        
        # Extract source workflow information
        source_prompt = source_workflow.get("prompt", "")
        source_graph = source_workflow.get("graph", "")
        source_score = source_workflow.get("score", 0.0)
        solved_problems = source_workflow.get("solved_problems", set())
        
        # Generate direction-specific guidance
        direction_guidance = self._get_direction_guidance(
            differentiation_direction, question_type, performance_gaps
        )
        
        # Create base prompt template
        base_prompt = f"""You are an expert in {question_type} problem-solving workflow design. Your task is to create a SPECIALIZED version of an existing workflow by differentiating it in a specific direction.

## Current Workflow Analysis
**Dataset**: {dataset}
**Target Round**: {target_round}
**Current Score**: {source_score:.4f}
**Problems Solved**: {len(solved_problems)} problems
**Specialization Direction**: {differentiation_direction}

## Source Workflow
### Current Prompt:
```
{source_prompt}
```

### Current Graph:
```python
{source_graph}
```

## Specialization Objective
{direction_guidance}

## Available Operators
{operator_description}

## Differentiation Requirements
1. **Single Specialization Focus**: Create a workflow that excels in ONE specific area, not a general multi-strategy system
2. **NO Problem Classification**: Do NOT create workflows that classify problems and apply different strategies
3. **Deep Specialization**: Sacrifice generality for excellence in the target specialization
4. **Simple Structure**: Avoid complex branching logic - focus on streamlined, specialized processing
5. **Targeted Optimization**: Every component should contribute to the specialization goal

## IMPORTANT RESTRICTIONS
- ❌ Do NOT create problem type classifiers or conditional strategy selection
- ❌ Do NOT use multiple solution generation with different approaches  
- ❌ Do NOT create "if problem_type == X then do Y" logic
- ✅ DO create a single, focused approach optimized for the specialization
- ✅ DO use specialized prompts and operators for the target domain
- ✅ DO ensure every step contributes to the specialization goal

## Output Format
Provide your differentiated workflow in the following XML format:

<modification>
[Detailed description of how you differentiated the workflow, what specialization was introduced, and why this direction was chosen. Explain the key changes and expected benefits.]
</modification>

<graph>
[Complete Python class definition for the specialized workflow. MUST follow this exact import structure:

```python
from typing import Literal
import workspace.{dataset}.workflows.template.operator as operator
import workspace.{dataset}.workflows.round_{target_round}.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        # Initialize operators like: self.custom = operator.Custom(self.llm)
        
    async def __call__(self, problem: str, entry_point: str = None):
        # Your specialized workflow implementation
        pass
```

Ensure correct import paths and operator usage!]
</graph>

<prompt>
[Updated prompt that reflects the specialization. Should be clear about the workflow's specialized focus while maintaining effectiveness.]
</prompt>

Remember: The goal is to create a workflow that excels in the specified specialization direction."""

        # Format the template placeholders
        base_prompt = base_prompt.format(dataset=dataset, target_round=target_round)
        
        return base_prompt
    
    def _get_direction_guidance(
        self,
        direction: str,
        question_type: str,
        performance_gaps: List[Dict] = None
    ) -> str:
        """Generate direction-specific guidance for differentiation."""
        
        # Only handle problem_type_specialization since other directions are removed
        if direction == "problem_type_specialization":
            return self._get_problem_type_guidance(question_type, performance_gaps)
        else:
            return self._get_generic_guidance(direction, question_type)
    
    def _get_problem_type_guidance(self, question_type: str, performance_gaps: List[Dict] = None) -> str:
        """Generate guidance for problem type specialization."""
        
        # Provide specific, focused specialization directions instead of generic ones
        if question_type == "math":
            specializations = [
                "algebraic manipulation and equation solving",
                "geometric reasoning and spatial visualization", 
                "combinatorial counting and probability",
                "calculus and analysis techniques",
                "number theory and modular arithmetic"
            ]
        elif question_type == "code":
            specializations = [
                "string processing and text manipulation",
                "array and list manipulation algorithms",
                "recursive problem decomposition",
                "graph traversal and network algorithms",
                "dynamic programming optimization"
            ]
        elif question_type == "qa":
            specializations = [
                "multi-step logical reasoning chains",
                "factual information extraction and synthesis",
                "causal relationship analysis",
                "comparative and analytical reasoning",
                "evidence-based conclusion drawing"
            ]
        else:
            specializations = ["domain-specific pattern recognition"]
        
        # Choose one specific specialization rather than asking LLM to pick
        import random
        chosen_specialization = random.choice(specializations)
        
        guidance = f"""Create a workflow that SPECIALIZES EXCLUSIVELY in {chosen_specialization}.

**CRITICAL**: Do NOT create a general problem classifier or multi-strategy workflow. Instead:

1. **Single Focus**: The entire workflow should be optimized for {chosen_specialization} problems
2. **Targeted Approach**: Use specialized operators, prompts, and logic specifically for this domain
3. **Deep Specialization**: Sacrifice generality for excellence in this specific area
4. **Simple Structure**: Avoid complex branching or multiple strategies - focus on one specialized approach

**Example Specialization Approach**:
- Identify the core techniques needed for {chosen_specialization}
- Design operators that excel at these specific techniques
- Create prompts that guide the model to use domain-specific reasoning
- Structure the workflow to maximize performance on this narrow focus"""
        
        if performance_gaps:
            gap_info = "\n\n**Performance Gaps Analysis**:\n"
            for gap in performance_gaps[:3]:  # Limit to top 3 gaps
                gap_info += f"- {gap.get('category', 'Unknown')}: {gap.get('description', 'N/A')}\n"
            guidance += gap_info + f"\nFocus your {chosen_specialization} specialization on addressing these specific weaknesses."
        
        return guidance
    
    def _get_generic_guidance(self, direction: str, question_type: str) -> str:
        """Generate generic guidance for custom differentiation directions."""
        return f"Specialize the workflow in the '{direction}' direction for {question_type} problems. Introduce focused improvements and targeted capabilities while maintaining overall effectiveness."

    def get_available_directions(self) -> List[str]:
        """Get list of available differentiation directions."""
        return self.differentiation_strategies.copy()

    def analyze_performance_gaps(self, workflow_results: List[Dict], failed_problems: List[Dict] = None) -> List[Dict]:
        """
        Analyze performance gaps to guide differentiation.
        
        Args:
            workflow_results: Historical workflow performance data
            failed_problems: Problems that consistently fail
            
        Returns:
            List of performance gap analyses
        """
        gaps = []
        
        # Analyze score patterns
        if workflow_results:
            scores = [r.get("score", 0.0) for r in workflow_results]
            avg_score = sum(scores) / len(scores)
            
            if avg_score < 0.3:
                gaps.append({
                    "category": "low_overall_performance",
                    "description": f"Overall performance is low (avg: {avg_score:.3f}). Needs fundamental approach improvement."
                })
            elif avg_score < 0.7:
                gaps.append({
                    "category": "moderate_performance",
                    "description": f"Moderate performance (avg: {avg_score:.3f}). Could benefit from specialization."
                })
        
        # Analyze failed problems patterns
        if failed_problems:
            gaps.append({
                "category": "failure_patterns",
                "description": f"Consistent failures on {len(failed_problems)} problem types. Needs targeted improvements."
            })
        
        return gaps
