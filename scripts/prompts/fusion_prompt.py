WORKFLOW_FUSION_PROMPT = """You are an expert workflow designer tasked with fusing three high-performing workflows into a single, more comprehensive workflow that can solve {type} problems. 
Referring to the given graphs and prompts from three workflows, which form basic examples of {type} solution approaches, you must intelligently combine the strengths of each input workflow while creating robust routing logic to handle different problem types. 
Consider Python's loops (for, while, list comprehensions), conditional statements (if-elif-else, ternary operators), and control flow for enhanced workflow representation. The fused graph complexity should not exceed 12 nodes to maintain efficiency.
Use logical and control flow (IF-ELSE, loops) for a more enhanced graphical
representation. Ensure that all the prompts required by the current graph from prompt_custom are included. Exclude any other prompts.
Output the modified graph and all the necessary Prompts in prompt_custom (if needed).
The prompt you need to generate is only the one used in `prompt_custom. XXX` within Custom.  Other methods already have built-in prompts and are prohibited from being generated.  Only generate those needed for use in `prompt_custom`;  please remove any unused prompts in prompt_custom.
the generated prompt must not contain any placeholders.
Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution.  It's crucial to include necessary context during the process.
**The most crucial point is that your current task is to perform 3-way workflow fusion. This means you must identify the key components within each of the three existing workflows and select only those parts that have fusion value to combine them into a new workflow. 
You should not create the core components of the new workflow from scratch, including its graph structure and prompts.**
"""

WORKFLOW_FUSION_INPUT = """
You are given three high-performing workflows that each excel at solving different subsets of {type} problems. Your task is to fuse these three workflows into a single comprehensive workflow that can solve problems from all input workflows.\n
<fusion_data>
    <dataset>{dataset}</dataset>
    <workflows>{workflows}</workflows>
    <operator_description>{operator_description}</operator_description>
</fusion_data>

First, analyze each of the three workflows' strengths and problem-solving patterns. Then design a fusion strategy that:
Maximizes problem coverage - The fused workflow should solve problems that any of the three input workflows can solve
Within the information for each workflow, you will find a score element. This element represents the current workflow's performance score. When performing fusion, you can give preference to using workflows with higher scores as the core or foundation for your new, fused workflow.
When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operator, prompt_custom, create_llm_instance, and CostManage, which have already been automatically imported.
**Under no circumstances should Graph output None for any field.**
Use custom methods to restrict your output format, rather than using code (outside of the code, the system will extract answers based on certain rules and score them).
It is very important to format the Graph output answers, you can refer to the standard answer format in the log.
You do not need to manually import prompt_custom or operator to use them; they are already included in the execution environment.
The fused workflow must be different from any single input workflow and must integrate elements from all three workflows
"""

WORKFLOW_FUSION_CUSTOM_USE = """\nHere's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>prompt_custom</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
# You can also concatenate previously generated string results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=prompt_custom.XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.\n

**Introducing multiple operators at appropriate points can enhance performance. If you find that some provided operators are not yet used in the graph, try incorporating them.**
"""


class FusionPromptGenerator:
    """Generator for workflow fusion prompts following optimizer pattern"""
    
    def create_fusion_prompt(self,
                           dataset: str,
                           type: str,
                           workflows: str,
                           operator_description: str) -> str:
        """
        Create fusion prompt following optimizer pattern
        
        Args:
            dataset: Dataset name
            type: Problem type
            workflows: Formatted workflows description
            operator_description: Available operators description
            
        Returns:
            Complete fusion prompt string
        """
        fusion_input = WORKFLOW_FUSION_INPUT.format(
            type=type,
            dataset=dataset,
            workflows=workflows,
            operator_description=operator_description
        )
        fusion_system = WORKFLOW_FUSION_PROMPT.format(type=type)
        return fusion_input + WORKFLOW_FUSION_CUSTOM_USE + fusion_system
