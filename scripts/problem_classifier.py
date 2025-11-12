#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Problem Classifier

对验证集中的问题进行分类，为定向分化提供基础数据
"""

import json
import os
from typing import List, Dict, Any, Tuple
from collections import Counter
import asyncio

from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger
from pydantic import BaseModel, Field


class SingleProblemClassification(BaseModel):
    """单个问题的分类结果"""
    category: str = Field(..., description="The category name for this problem")
    is_new_category: str = Field(..., description="'true' if this is a new category, 'false' if existing")
    category_description: str = Field(default="", description="Description of the category (required if new category, optional otherwise)")
    reasoning: str = Field(..., description="Brief reasoning for the classification")


class ProblemClassifier:
    """问题分类器 - 分析验证集并对问题进行分类"""
    
    def __init__(self, exec_llm_config, dataset: str, optimized_path: str = "workspace"):
        """
        初始化问题分类器
        
        Args:
            exec_llm_config: 执行模型配置
            dataset: 数据集名称
            optimized_path: 工作目录路径
        """
        self.exec_llm_config = exec_llm_config
        self.dataset = dataset
        self.root_path = os.path.join(optimized_path, dataset)
        self.classification_file = os.path.join(self.root_path, "workflows", "problem_classifications.json")
        
        # 分类缓存
        self._classifications = None
        self._category_stats = None
    
    async def analyze_and_classify_problems(self, validation_data: List[Dict], max_retries: int = 3) -> Dict:
        """
        逐个分析验证集问题并进行分类
        对每个问题：
        1. 检查是否属于已有类别
        2. 如果不属于，创建新类别
        
        Args:
            validation_data: 验证集数据
            max_retries: 每个问题的最大重试次数
            
        Returns:
            Dict: 分类分析结果
        """
        logger.info(f"Starting incremental problem classification for {len(validation_data)} problems")
        
        # 初始化分类结果
        categories = []  # 类别名称列表
        category_descriptions = {}  # 类别描述字典
        problem_classifications = []  # 问题分类列表
        
        llm_instance = create_llm_instance(self.exec_llm_config)
        
        # 逐个分类问题
        for idx, problem_data in enumerate(validation_data):
            logger.info(f"Classifying problem {idx + 1}/{len(validation_data)}")
            
            # 提取问题内容
            problem_text = self._extract_problem_text(problem_data)
            problem_id = self._extract_problem_id(problem_data, idx)
            
            # 使用重试机制进行分类
            classification_result = await self._classify_single_problem_with_retry(
                llm_instance,
                problem_text,
                problem_id,
                categories,
                category_descriptions,
                max_retries=max_retries
            )
            
            if classification_result:
                # 分类成功
                category = classification_result['category']
                is_new = classification_result['is_new']
                category_desc = classification_result.get('category_description', '')
                
                # 检查类别数量限制
                current_category_count = len(categories)
                
                # 如果是新类别，检查是否应该添加
                if is_new and category not in categories:
                    # 如果已经有 5 个或更多类别，拒绝创建新类别
                    if current_category_count >= 5:
                        logger.warning(f"  → Rejected new category '{category}': already have {current_category_count} categories (limit: 5)")
                        logger.warning(f"  → Forcing problem into most similar category instead")
                        # 将问题分配到现有类别中（这里简单地分配到第一个，实际可以更智能）
                        category = categories[0] if categories else "Other"
                        is_new = False
                        logger.info(f"  → Reassigned to existing category: {category}")
                    else:
                        # 允许创建新类别
                        categories.append(category)
                        if not category_desc:
                            category_desc = f"Problems related to {category}"
                        category_descriptions[category] = category_desc
                        logger.info(f"  → New category created: {category} (total: {len(categories)}/5)")
                        
                        # 如果达到 5 个类别，发出警告
                        if len(categories) == 5:
                            logger.warning("  ⚠️  Reached 5 categories - future problems will be forced into existing categories")
                        
                elif not is_new and category not in categories:
                    # 模型说不是新类别但类别不存在
                    if current_category_count >= 5:
                        # 已经达到限制，分配到现有类别
                        logger.warning(f"  → Category '{category}' doesn't exist and limit reached, using fallback")
                        category = categories[0] if categories else "Other"
                        logger.info(f"  → Reassigned to: {category}")
                    else:
                        # 还未达到限制，允许添加
                        categories.append(category)
                        if not category_desc:
                            category_desc = f"Problems related to {category}"
                        category_descriptions[category] = category_desc
                        logger.info(f"  → Category added (fallback): {category} (total: {len(categories)}/5)")
                else:
                    logger.info(f"  → Assigned to existing category: {category}")
                
                # 记录分类结果
                problem_classifications.append({
                    'problem_id': problem_id,
                    'category': category,
                    'reasoning': classification_result.get('reasoning', '')
                })
            else:
                # 所有重试都失败，分配到 "Other" 类别
                logger.warning(f"Failed to classify problem {idx + 1} after {max_retries} retries")
                if "Other" not in categories:
                    categories.append("Other")
                    category_descriptions["Other"] = "Problems that don't fit into other categories"
                problem_classifications.append({
                    'problem_id': problem_id,
                    'category': 'Other',
                    'reasoning': f'Classification failed after {max_retries} retries'
                })
        
        logger.info(f"Classification completed: {len(categories)} categories identified")
        for cat in categories:
            count = sum(1 for p in problem_classifications if p['category'] == cat)
            logger.info(f"  - {cat}: {count} problems")
        
        # 保存分类结果
        classification_result = {
            'categories': categories,
            'category_descriptions': category_descriptions,
            'problem_classifications': problem_classifications
        }
        self._save_classifications(classification_result)
        
        return classification_result
    
    async def _classify_single_problem_with_retry(
        self,
        llm_instance,
        problem_text: str,
        problem_id: str,
        categories: List[str],
        category_descriptions: Dict[str, str],
        max_retries: int = 3
    ) -> Dict:
        """
        使用重试机制对单个问题进行分类
        
        Args:
            llm_instance: LLM 实例
            problem_text: 问题文本
            problem_id: 问题 ID
            categories: 已有类别列表
            category_descriptions: 类别描述字典
            max_retries: 最大重试次数
            
        Returns:
            Dict: 分类结果，包含 category, is_new, category_description, reasoning
                  如果失败返回 None
        """
        import asyncio
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"  Retry attempt {attempt + 1}/{max_retries}")
                    # 添加指数退避延迟
                    await asyncio.sleep(2 ** attempt)
                
                # 构建单个问题的分类提示词
                prompt = self._create_single_classification_prompt(
                    problem_text, 
                    problem_id,
                    categories, 
                    category_descriptions
                )
                
                # 创建简单的分类响应模型
                formatter = XmlFormatter.from_model(SingleProblemClassification)
                response_dict = await llm_instance.call_with_format(prompt, formatter)
                
                # 解析响应
                category = response_dict['category'].strip()
                is_new = response_dict['is_new_category'].lower() == 'true'
                category_desc = response_dict.get('category_description', '').strip()
                reasoning = response_dict.get('reasoning', '')
                
                return {
                    'category': category,
                    'is_new': is_new,
                    'category_description': category_desc,
                    'reasoning': reasoning
                }
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"  Classification attempt {attempt + 1} failed: {error_msg}")
                
                # 如果是最后一次尝试，记录完整错误
                if attempt == max_retries - 1:
                    logger.error(f"  All {max_retries} classification attempts failed for problem {problem_id}", exc_info=True)
                    return None
                
                # 对于某些不可恢复的错误，直接失败不重试
                if "invalid" in error_msg.lower() and "api" not in error_msg.lower():
                    logger.error(f"  Non-retryable error encountered: {error_msg}")
                    return None
        
        return None
    
    def _extract_problem_text(self, problem_data: Dict) -> str:
        """从问题数据中提取问题文本"""
        if 'question' in problem_data:
            return problem_data['question']
        elif 'problem' in problem_data:
            return problem_data['problem']
        elif 'context' in problem_data:
            # DROP 数据集格式
            context = problem_data['context']
            question = problem_data.get('question', '')
            return f"Context: {context}\nQuestion: {question}"
        elif 'prompt' in problem_data:
            return problem_data['prompt']
        else:
            return str(problem_data)
    
    def _extract_problem_id(self, problem_data: Dict, index: int) -> str:
        """从问题数据中提取问题ID"""
        # 优先使用 task_id (MBPP, HumanEval 等)
        if 'task_id' in problem_data:
            return str(problem_data['task_id'])
        elif 'id' in problem_data:
            return str(problem_data['id'])
        elif 'problem_id' in problem_data:
            return str(problem_data['problem_id'])
        elif 'idx' in problem_data:
            return str(problem_data['idx'])
        else:
            return f"problem_{index}"
    
    def _create_single_classification_prompt(
        self, 
        problem_text: str, 
        problem_id: str,
        existing_categories: List[str], 
        category_descriptions: Dict[str, str]
    ) -> str:
        """
        创建单个问题分类的提示词
        
        Args:
            problem_text: 问题文本
            problem_id: 问题ID
            existing_categories: 已有的类别列表
            category_descriptions: 已有类别的描述
            
        Returns:
            str: 分类提示词
        """
        # 构建已有类别信息
        current_count = len(existing_categories)
        if existing_categories:
            categories_info = f"Existing categories ({current_count}/5 recommended):\n"
            for cat in existing_categories:
                desc = category_descriptions.get(cat, "No description")
                categories_info += f"- {cat}: {desc}\n"
            
            # 添加类别数量状态提示
            if current_count >= 5:
                categories_info += f"\n⚠️  LIMIT REACHED: Already have {current_count} categories. DO NOT create new categories unless absolutely impossible to fit into existing ones.\n"
            elif current_count == 4:
                categories_info += f"\n⚠️  NEAR LIMIT: Have {current_count}/5 categories. Be very selective about creating new categories.\n"
            elif current_count == 3:
                categories_info += f"\nℹ️  OPTIMAL RANGE: Have {current_count}/5 categories. Can add 1-2 more if truly necessary.\n"
            else:
                categories_info += f"\nℹ️  Have {current_count}/5 categories. Can add more, but try to keep it under 5 total.\n"
        else:
            categories_info = "No existing categories yet. This is the first problem to classify.\nTarget: Create 3-5 broad categories total for the entire dataset."
        
        # 添加类别数量提示
        category_count = len(existing_categories)
        if category_count == 0:
            count_guidance = "You are classifying the first problem. Create a broad category that can encompass similar solution approaches."
        elif category_count < 3:
            count_guidance = f"Current categories: {category_count}. We need 3-5 categories total. You may create new categories if this problem needs a fundamentally different approach."
        elif category_count < 5:
            count_guidance = f"Current categories: {category_count}. We are approaching the target of 3-5 categories. Only create a new category if absolutely necessary."
        else:
            count_guidance = f"Current categories: {category_count}. We have reached the target range (3-5 categories). STRONGLY prefer existing categories. Only create a new category if the problem is completely incompatible with all existing ones."
        
        prompt = f"""You are a problem classification expert. Your task is to classify problems based on the SOLUTION APPROACH/ALGORITHM TYPE required, NOT based on the specific problem content or application domain.

IMPORTANT: Focus on HOW to solve the problem, not WHAT the problem is about.

{categories_info}

**Category Count Status**: {count_guidance}

Problem ID: {problem_id}
Problem:
{problem_text}

Classification Dimensions (Examples):
- Algorithm Types: Dynamic Programming, Greedy Algorithm, Divide and Conquer, Backtracking, etc.
- Data Structures: Tree/Graph Problems, Array/String Manipulation, Stack/Queue Problems, etc.
- Problem Types: Search/Optimization, Counting/Combinatorics, Simulation, Pattern Matching, etc.
- Reasoning Types: Mathematical Reasoning, Logical Deduction, Rule Application, etc.

Instructions:
1. **TARGET**: Aim for 3-5 total categories that are broad and well-balanced
2. **KEY FOCUS**: What algorithm/strategy/approach is needed to solve this problem?
3. **REUSE WHEN POSSIBLE**: If this problem can fit into ANY existing category, choose that category (set is_new_category to 'false')
4. **CREATE STRATEGICALLY**: 
   - If we have < 3 categories: Feel free to create new ones for distinct solution approaches
   - If we have 3-5 categories: Only create new ones if truly necessary
   - If we have > 5 categories: AVOID creating new ones unless absolutely impossible to fit
5. When creating a new category:
   - Use BROAD names that can cover many problems:
     * ✓ "Dynamic Programming & Recursion" (covers DP, memoization, recursive solutions)
     * ✓ "Mathematical & Logical Reasoning" (covers math, logic, proofs)
     * ✓ "Data Structure Operations" (covers arrays, strings, trees, graphs)
     * ✓ "Search & Optimization" (covers greedy, search algorithms, optimization)
     * ✓ "Simulation & Implementation" (covers direct simulation, rule following)
   - Provide a description that explains the broad scope
6. Reasoning: Explain WHY this problem fits the selected category

Guidelines:
- Each category should represent a DISTINCT solution methodology
- Categories should be BALANCED in scope (not too narrow, not too broad)
- Good category examples (3-5 total for a dataset):
  * "Dynamic Programming & Recursion"
  * "Mathematical & Logical Reasoning"  
  * "Data Structure Operations"
  * "Search & Optimization Algorithms"
  * "String/Pattern Manipulation"
- Avoid: Too many specific categories OR one catch-all category

**Remember**: The goal is 3-5 well-defined, balanced categories. Think strategically about how to group similar solution approaches together!

Think carefully about the SOLUTION APPROACH and respond with your classification.
"""
        
        return prompt
    
    def _save_classifications(self, classification_result: Dict) -> None:
        """
        保存分类结果到文件
        
        Args:
            classification_result: 分类分析结果字典
        """
        os.makedirs(os.path.dirname(self.classification_file), exist_ok=True)
        with open(self.classification_file, 'w', encoding='utf-8') as f:
            json.dump(classification_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Classifications saved to {self.classification_file}")
    
    def load_classifications(self) -> Dict[str, Any]:
        """
        加载已保存的分类结果
        
        Returns:
            Dict: 分类数据
        """
        if self._classifications is not None:
            return self._classifications
        
        if not os.path.exists(self.classification_file):
            logger.warning(f"Classification file not found: {self.classification_file}")
            return None
        
        with open(self.classification_file, 'r', encoding='utf-8') as f:
            self._classifications = json.load(f)
        
        return self._classifications
    
    def get_category_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取每个类别的统计信息
        
        Returns:
            Dict: 每个类别的统计数据，包括问题数量和问题列表
        """
        if self._category_stats is not None:
            return self._category_stats
        
        classifications = self.load_classifications()
        if not classifications:
            return {}
        
        stats = {}
        for category in classifications['categories']:
            stats[category] = {
                'count': 0,
                'problems': [],
                'description': classifications['category_descriptions'].get(category, '')
            }
        
        for pc in classifications['problem_classifications']:
            category = pc['category']
            if category in stats:
                stats[category]['count'] += 1
                stats[category]['problems'].append({
                    'id': pc['problem_id'],
                    'reasoning': pc['reasoning']
                })
        
        self._category_stats = stats
        return stats
    
    def get_problems_by_category(self, category: str, validation_data: List[Dict], limit: int = 3) -> List[Dict]:
        """
        获取指定类别的问题示例
        
        Args:
            category: 类别名称
            validation_data: 完整的验证集数据
            limit: 返回的问题数量限制
            
        Returns:
            List[Dict]: 该类别的问题示例
        """
        classifications = self.load_classifications()
        if not classifications:
            logger.warning("No classifications found")
            return []
        
        # 找到该类别的所有问题ID（保持原始类型，可能是字符串或整数）
        problem_ids = []
        for pc in classifications['problem_classifications']:
            if pc['category'] == category:
                problem_ids.append(pc['problem_id'])
        
        logger.info(f"Found {len(problem_ids)} problems in category '{category}'")
        
        # 创建问题ID到问题数据的映射（支持字符串和整数ID）
        problem_map = {}
        for problem in validation_data:
            # 尝试多个可能的ID字段，优先使用 task_id
            pid = problem.get('task_id') or problem.get('id') or problem.get('problem_id') or problem.get('index')
            if pid is not None:
                # 存储多种形式以支持灵活匹配
                problem_map[pid] = problem
                problem_map[str(pid)] = problem
                # 如果是整数，也存储字符串形式
                try:
                    if isinstance(pid, str):
                        problem_map[int(pid)] = problem
                except (ValueError, TypeError):
                    pass
        
        logger.info(f"Built problem map with {len(validation_data)} problems")
        
        # 提取对应的问题数据
        examples = []
        for pid in problem_ids:
            # 尝试多种匹配方式
            problem = None
            if pid in problem_map:
                problem = problem_map[pid]
            elif str(pid) in problem_map:
                problem = problem_map[str(pid)]
            else:
                # 尝试整数形式
                try:
                    int_pid = int(pid)
                    problem = problem_map.get(int_pid)
                except (ValueError, TypeError):
                    pass
            
            if problem and problem not in examples:  # 避免重复
                examples.append(problem)
                if len(examples) >= limit:
                    break
        
        logger.info(f"Retrieved {len(examples)} example problems for category '{category}'")
        return examples
    
    def calculate_category_priority(
        self, 
        category: str, 
        last_differentiation_rounds: Dict[str, int],
        current_round: int
    ) -> float:
        """
        计算类别的分化优先级权重
        
        Args:
            category: 类别名称
            last_differentiation_rounds: 每个类别上次分化的轮次
            current_round: 当前轮次
            
        Returns:
            float: 优先级权重分数（越高越优先）
        """
        stats = self.get_category_statistics()
        if category not in stats:
            return 0.0
        
        # 1. 问题数量权重（归一化到0-1）
        max_count = max(s['count'] for s in stats.values())
        count_weight = stats[category]['count'] / max_count if max_count > 0 else 0
        
        # 2. 时间权重（距离上次分化的轮次数）
        last_round = last_differentiation_rounds.get(category, 0)
        rounds_since_last = current_round - last_round
        
        # 归一化时间权重：假设10轮为满分
        time_weight = min(rounds_since_last / 10.0, 1.0)
        
        # 3. 综合权重：问题数量占70%，时间占30%
        priority = 0.7 * count_weight + 0.3 * time_weight
        
        logger.debug(f"Category '{category}' priority: {priority:.4f} "
                    f"(count_weight={count_weight:.4f}, time_weight={time_weight:.4f}, "
                    f"problems={stats[category]['count']}, rounds_since_last={rounds_since_last})")
        
        return priority
    
    def select_differentiation_category(
        self, 
        last_differentiation_rounds: Dict[str, int],
        current_round: int
    ) -> Tuple[str, float, str]:
        """
        选择优先级最高的分化类别
        
        Args:
            last_differentiation_rounds: 每个类别上次分化的轮次
            current_round: 当前轮次
            
        Returns:
            Tuple[str, float, str]: (类别名称, 优先级分数, 类别描述)
        """
        stats = self.get_category_statistics()
        if not stats:
            logger.error("No category statistics available")
            return None, 0.0, ""
        
        # 计算所有类别的优先级
        priorities = {}
        for category in stats.keys():
            priority = self.calculate_category_priority(
                category, last_differentiation_rounds, current_round
            )
            priorities[category] = priority
        
        # 选择优先级最高的类别
        selected_category = max(priorities, key=priorities.get)
        selected_priority = priorities[selected_category]
        selected_description = stats[selected_category]['description']
        
        logger.info(f"Selected category for differentiation: '{selected_category}' "
                   f"(priority={selected_priority:.4f}, problems={stats[selected_category]['count']})")
        
        return selected_category, selected_priority, selected_description
    
    def select_target_category_for_differentiation(
        self,
        workflow_differentiation_history: Dict[str, Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        选择目标类别进行分化，并返回该类别的示例问题
        
        Args:
            workflow_differentiation_history: 分化历史，格式：
                {
                    "category_name": {
                        "last_differentiation_round": int,
                        "differentiation_count": int
                    }
                }
        
        Returns:
            Tuple[str, List[Dict]]: (目标类别名称, 该类别的示例问题列表)
        """
        # 构建 last_differentiation_rounds 字典
        last_differentiation_rounds = {}
        for category, history in workflow_differentiation_history.items():
            last_differentiation_rounds[category] = history.get("last_differentiation_round", 0)
        
        # 获取当前轮次（使用最大值作为当前轮次）
        current_round = max(last_differentiation_rounds.values()) + 1 if last_differentiation_rounds else 1
        
        # 选择分化类别
        selected_category, priority, description = self.select_differentiation_category(
            last_differentiation_rounds, current_round
        )
        
        if not selected_category:
            logger.warning("No suitable category found for differentiation")
            return None, []
        
        # 获取该类别的示例问题
        try:
            # 加载验证集数据（尝试小写和原始大小写）
            validate_file = f"data/datasets/{self.dataset.lower()}_validate.jsonl"
            if not os.path.exists(validate_file):
                # 尝试原始大小写
                validate_file = f"data/datasets/{self.dataset}_validate.jsonl"
                if not os.path.exists(validate_file):
                    logger.warning(f"Validation file not found: {validate_file}")
                    return selected_category, []
            
            validation_data = []
            with open(validate_file, 'r', encoding='utf-8') as f:
                for line in f:
                    validation_data.append(json.loads(line.strip()))
            
            # 获取该类别的示例问题
            example_problems = self.get_problems_by_category(
                selected_category, validation_data, limit=3
            )
            
            logger.info(f"Selected target category: {selected_category}")
            logger.info(f"Category description: {description}")
            logger.info(f"Retrieved {len(example_problems)} example problems")
            
            return selected_category, example_problems
            
        except Exception as e:
            logger.error(f"Error loading example problems: {e}")
            return selected_category, []
