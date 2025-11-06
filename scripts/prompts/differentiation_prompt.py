from typing import List

WORKFLOW_DIFFERENTIATION_PROMPT = """You are an expert workflow designer tasked with differentiating an existing workflow to create a specialized variant for {type} problems.

**CRITICAL - What is Differentiation**:
Differentiation means converting ABSTRACT/GENERAL steps into CONCRETE/SPECIFIC steps for a particular problem domain.
- You MUST keep the same number of main steps as the parent workflow
- You MUST keep the same operators in the same order
- You ONLY change: prompt content to be more specific for the target domain

**What You ARE Doing**:
✅ Converting "analyze problem" → "analyze [specific problem type] characteristics"
✅ Converting "solve" → "apply [specific technique] to solve"
✅ Converting "verify" → "verify using [specific domain] validation rules"
✅ Making prompts more specific and detailed for the target domain

Use logical and control flow (IF-ELSE, loops) for a more enhanced graphical representation.
Ensure that all the prompts required by the current graph from prompt_custom are included. Exclude any other prompts.
Output the modified graph and all the necessary Prompts in prompt_custom (if needed).
The prompt you need to generate is only the one used in `prompt_custom.XXX` within Custom. Other methods already have built-in prompts and are prohibited from being generated. Only generate those needed for use in `prompt_custom`; please remove any unused prompts in prompt_custom.
The generated prompt must not contain any placeholders.
"""

WORKFLOW_DIFFERENTIATION_INPUT = """
You are differentiating an existing workflow to make it specialized for {specialization_focus}.

<differentiation_data>
    <dataset>{dataset}</dataset>
    <target_round>{target_round}</target_round>
    <specialization_direction>{direction}</specialization_direction>
    <specialization_focus>{specialization_focus}</specialization_focus>
    <parent_workflow>
        <score>{source_score}</score>
        <graph>{source_graph}</graph>
        <prompt>{source_prompt}</prompt>
    </parent_workflow>
    <operator_description>{operator_description}</operator_description>
</differentiation_data>

**Your Task**:
Convert the parent workflow's ABSTRACT steps into CONCRETE steps specialized for {specialization_focus}.

**Differentiation Rules (MUST FOLLOW)**:
1. **Keep Same Structure**: Use the EXACT SAME operators in the EXACT SAME order as parent
2. **Same Number of Steps**: If parent has N operators, differentiated workflow has N operators
3. **Only Change Prompts**: Make prompt content specific to {specialization_focus}
4. **No Structural Changes**: Do NOT add/remove operators or change workflow logic

**How to Convert Abstract → Concrete**:
- Parent: "analyze the problem"
  Child: "analyze {specialization_focus} characteristics: identify key variables, constraints, and target"
  
- Parent: "solve the problem" 
  Child: "apply {specialization_focus} techniques: [specific method 1], [specific method 2]"
  
- Parent: "verify the solution"
  Child: "verify using {specialization_focus} rules: check [specific constraint 1], [specific constraint 2]"

**Example Differentiation**:
If parent workflow is:
```python
analysis = await self.custom(input=problem, instruction=prompt_custom.ANALYSIS_PROMPT)
solution = await self.custom(input=analysis['response'], instruction=prompt_custom.SOLUTION_PROMPT)
```

Differentiated workflow should be:
```python
analysis = await self.custom(input=problem, instruction=prompt_custom.ALGEBRAIC_ANALYSIS_PROMPT)  # More specific name
solution = await self.custom(input=analysis['response'], instruction=prompt_custom.ALGEBRAIC_SOLUTION_PROMPT)  # More specific name
```

The prompts should contain {specialization_focus}-specific instructions, but the structure remains identical.

When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operator, prompt_custom, create_llm_instance, and CostManage, which have already been automatically imported.
**Under no circumstances should Graph output None for any field.**
Use custom methods to restrict your output format, rather than using code (outside of the code, the system will extract answers based on certain rules and score them).
You do not need to manually import prompt_custom or operator to use them; they are already included in the execution environment.
"""

WORKFLOW_DIFFERENTIATION_CUSTOM_USE = """\nHere's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>prompt_custom</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
# You can also concatenate previously generated string results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=prompt_custom.XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.

**CRITICAL REMINDER FOR DIFFERENTIATION**:
- Keep the SAME workflow structure as parent (same operators, same order)
- ONLY make prompts more specific for the target domain
- Do NOT add extra processing steps or complex branching
- Simple abstract-to-concrete conversion is enough
"""


class DifferentiationPromptGenerator:
    """Generator for workflow differentiation prompts following optimizer pattern"""
    
    def __init__(self):
        # Specialization focuses for different problem types
        self.math_specializations = [
            "algebraic manipulation and equation solving",
            "geometric reasoning and spatial visualization", 
            "combinatorial counting and probability",
            "calculus and analysis techniques",
            "number theory and modular arithmetic"
        ]
        
        self.code_specializations = [
            "string processing and text manipulation",
            "array and list manipulation algorithms",
            "recursive problem decomposition",
            "graph traversal and network algorithms",
            "dynamic programming optimization"
        ]
        
        self.qa_specializations = [
            "multi-step logical reasoning chains",
            "factual information extraction and synthesis",
            "causal relationship analysis",
            "comparative and analytical reasoning",
            "evidence-based conclusion drawing"
        ]
    
    def _get_specialization_focus(self, question_type: str, direction: str) -> str:
        """Get specific specialization focus based on problem type"""
        import random
        
        if question_type == "math":
            return random.choice(self.math_specializations)
        elif question_type == "code":
            return random.choice(self.code_specializations)
        elif question_type == "qa":
            return random.choice(self.qa_specializations)
        else:
            return "domain-specific pattern recognition"
    
    def create_differentiation_prompt(self,
                                    dataset: str,
                                    target_round: int,
                                    question_type: str,
                                    differentiation_direction: str,
                                    source_score: float,
                                    source_graph: str,
                                    source_prompt: str,
                                    operator_description: str,
                                    target_category: str = None,
                                    category_description: str = None,
                                    category_examples: List = None) -> str:
        """
        Create differentiation prompt following optimizer pattern
        
        Args:
            dataset: Dataset name
            target_round: Target round number
            question_type: Type of problem (math/code/qa)
            differentiation_direction: Direction of differentiation
            source_score: Parent workflow score
            source_graph: Parent workflow graph code
            source_prompt: Parent workflow prompt code
            operator_description: Available operators description
            target_category: 目标问题类别（用于定向分化）
            category_description: 目标类别的描述
            category_examples: 目标类别的示例问题（最多3个）
            
        Returns:
            Complete differentiation prompt string
        """
        # Determine specialization focus
        if target_category:
            # Use explicit category for directed differentiation
            specialization_focus = target_category
            
            # Add category examples section if provided
            category_section = f"\n<target_category>\n    <name>{target_category}</name>\n"
            if category_description:
                category_section += f"    <description>{category_description}</description>\n"
            if category_examples:
                category_section += "    <examples>\n"
                for idx, example in enumerate(category_examples[:3], 1):
                    # Extract problem text based on dataset structure
                    problem_text = (example.get('question') or 
                                  example.get('problem') or 
                                  example.get('context') or 
                                  example.get('prompt') or 
                                  str(example))
                    # Truncate if too long
                    if len(problem_text) > 500:
                        problem_text = problem_text[:500] + "..."
                    category_section += f"        <example_{idx}>{problem_text}</example_{idx}>\n"
                category_section += "    </examples>\n"
            category_section += "</target_category>\n"
        else:
            # Use random specialization for exploratory differentiation
            specialization_focus = self._get_specialization_focus(question_type, differentiation_direction)
            category_section = ""
        
        # Format input section
        differentiation_input = WORKFLOW_DIFFERENTIATION_INPUT.format(
            dataset=dataset,
            target_round=target_round,
            type=question_type,
            direction=differentiation_direction,
            specialization_focus=specialization_focus,
            source_score=source_score,
            source_graph=source_graph,
            source_prompt=source_prompt,
            operator_description=operator_description
        )
        
        # Insert category section if available
        if category_section:
            # Insert before </differentiation_data>
            differentiation_input = differentiation_input.replace(
                "</differentiation_data>",
                category_section + "</differentiation_data>"
            )
            
            # Add category-specific instructions
            category_instructions = f"""
**IMPORTANT - Targeted Differentiation**:
You are creating a workflow specialized SPECIFICALLY for problems in the category: "{target_category}"
{f'Description: {category_description}' if category_description else ''}

The example problems above show the exact type of problems this workflow should excel at.
Make sure your differentiated workflow is optimized for THIS SPECIFIC category of problems.
"""
            differentiation_input += category_instructions
        
        # Format system prompt
        differentiation_system = WORKFLOW_DIFFERENTIATION_PROMPT.format(type=question_type)
        
        # Combine all parts
        return differentiation_input + WORKFLOW_DIFFERENTIATION_CUSTOM_USE + differentiation_system
