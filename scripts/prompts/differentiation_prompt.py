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
            "strategy_diversification", 
            "algorithmic_approach_variation",
            "complexity_adaptation",
            "error_pattern_handling"
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
1. **Maintain Core Functionality**: The new workflow should still be capable of solving {question_type} problems
2. **Introduce Specialization**: Focus on the specified direction while maintaining or improving performance
3. **Ensure Diversity**: Create meaningful differences from the source workflow, not just minor variations
4. **Optimize for Target**: The specialization should address specific weaknesses or enhance particular strengths

## Output Format
Provide your differentiated workflow in the following XML format:

<modification>
[Detailed description of how you differentiated the workflow, what specialization was introduced, and why this direction was chosen. Explain the key changes and expected benefits.]
</modification>

<graph>
[Complete Python class definition for the specialized workflow. Ensure it's syntactically correct and implements the specialization strategy.]
</graph>

<prompt>
[Updated prompt that reflects the specialization. Should be clear about the workflow's specialized focus while maintaining effectiveness.]
</prompt>

Remember: The goal is to create a workflow that excels in the specified specialization direction while maintaining overall competence."""

        return base_prompt
    
    def _get_direction_guidance(
        self,
        direction: str,
        question_type: str,
        performance_gaps: List[Dict] = None
    ) -> str:
        """Generate direction-specific guidance for differentiation."""
        
        guidance_templates = {
            "problem_type_specialization": self._get_problem_type_guidance(question_type, performance_gaps),
            "strategy_diversification": self._get_strategy_guidance(question_type),
            "algorithmic_approach_variation": self._get_algorithmic_guidance(question_type),
            "complexity_adaptation": self._get_complexity_guidance(question_type),
            "error_pattern_handling": self._get_error_handling_guidance(question_type, performance_gaps)
        }
        
        return guidance_templates.get(direction, self._get_generic_guidance(direction, question_type))
    
    def _get_problem_type_guidance(self, question_type: str, performance_gaps: List[Dict] = None) -> str:
        """Generate guidance for problem type specialization."""
        base_guidance = {
            "math": "Specialize in specific mathematical domains (algebra, geometry, calculus, combinatorics, etc.). Focus on problem patterns, theorem applications, and domain-specific reasoning strategies.",
            "code": "Specialize in specific programming paradigms (algorithmic, data structures, optimization, etc.). Focus on code patterns, efficiency considerations, and implementation strategies.",
            "qa": "Specialize in specific reasoning types (factual, inferential, analytical, etc.). Focus on information processing patterns and reasoning chain construction."
        }
        
        guidance = base_guidance.get(question_type, "Specialize in a specific aspect of the problem domain.")
        
        if performance_gaps:
            gap_info = "\n\n**Performance Gaps Analysis**:\n"
            for gap in performance_gaps[:3]:  # Limit to top 3 gaps
                gap_info += f"- {gap.get('category', 'Unknown')}: {gap.get('description', 'N/A')}\n"
            guidance += gap_info + "\nFocus your specialization on addressing these specific weaknesses."
        
        return guidance
    
    def _get_strategy_guidance(self, question_type: str) -> str:
        """Generate guidance for strategy diversification."""
        strategies = {
            "math": "Introduce alternative problem-solving strategies such as: visual/geometric approaches vs algebraic, constructive vs proof by contradiction, recursive vs iterative thinking, or pattern recognition vs first principles.",
            "code": "Diversify programming strategies such as: bottom-up vs top-down design, iterative vs recursive solutions, space-optimized vs time-optimized approaches, or functional vs imperative paradigms.",
            "qa": "Explore different reasoning strategies such as: deductive vs inductive reasoning, holistic vs analytical thinking, context-driven vs rule-based approaches, or multi-perspective analysis."
        }
        
        return strategies.get(question_type, "Introduce alternative problem-solving strategies and approaches.")
    
    def _get_algorithmic_guidance(self, question_type: str) -> str:
        """Generate guidance for algorithmic approach variation."""
        approaches = {
            "math": "Vary computational and logical approaches: introduce symbolic manipulation, numerical methods, graph-theoretic thinking, optimization techniques, or probabilistic reasoning.",
            "code": "Introduce different algorithmic paradigms: dynamic programming, greedy algorithms, divide-and-conquer, backtracking, or machine learning-based approaches.",
            "qa": "Implement varied information processing algorithms: semantic analysis, logical inference, evidence aggregation, or multi-step reasoning chains."
        }
        
        return approaches.get(question_type, "Introduce varied algorithmic approaches and computational methods.")
    
    def _get_complexity_guidance(self, question_type: str) -> str:
        """Generate guidance for complexity adaptation."""
        return f"Adapt the workflow to handle different complexity levels more effectively. For {question_type} problems, introduce multi-stage processing: simple cases with direct methods, medium cases with enhanced reasoning, complex cases with advanced techniques and validation."
    
    def _get_error_handling_guidance(self, question_type: str, performance_gaps: List[Dict] = None) -> str:
        """Generate guidance for error pattern handling."""
        guidance = f"Specialize in robust error detection and recovery for {question_type} problems. Introduce validation steps, alternative solution paths, and error pattern recognition."
        
        if performance_gaps:
            guidance += "\n\nBased on performance analysis, focus on handling these specific error patterns and failure modes."
        
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
