import asyncio
import os
import json
import time
from typing import List, Literal, Dict
import random
import os
import json
import time
import numpy as np  # 用于概率采样

from pydantic import BaseModel, Field

from scripts.evaluator import DatasetType
from scripts.optimizer import Optimizer, QuestionType, OptimizerType, GraphOptimize
from scripts.workflow_fusion import WorkflowFusion
from scripts.workflow_differentiation import WorkflowDifferentiation
from scripts.problem_classifier import ProblemClassifier
from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.optimizer_utils.experience_utils import ExperienceUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger
from scripts.utils.code_processor import CodeProcessor
from scripts.utils.workflow_manager import WorkflowManager, FusionChecker


class EnhancedOptimizer(Optimizer):
    """
    Enhanced optimizer that integrates workflow fusion and differentiation into the optimization process
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
        enable_differentiation: bool = True,
        # === 新的理论化超参数 ===
        # Stagnation Detection Parameters
        sliding_window_k: int = 3,              # 滑动窗口大小 k
        stagnation_sensitivity_kappa: float = 80.0,  # 停滞敏感度 κ
        # Base Probabilities  
        alpha_s: float = 0.50,  # 分化基础概率 α_s
        alpha_m: float = 0.60,  # 融合基础概率 α_m
        # Decay Factors
        eta_s: float = 0.03,    # 分化衰减因子 η_s
        eta_m: float = 0.03,    # 融合衰减因子 η_m
        # === Fusion Selection Weights ===
        alpha_U: float = 0.6,   # 融合互补性权重 α_U
        alpha_I: float = 0.4,   # 融合一致性权重 α_I
        beta_triple: float = 0.6,   # 三路并集权重 β_triple
        beta_pair: float = 0.4,     # 两两并集权重 β_pair
        gamma_pair: float = 0.7,    # 两两交集权重 γ_pair
        gamma_triple: float = 0.3,  # 三路交集权重 γ_triple
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
        
        # Differentiation-specific parameters
        self.enable_differentiation = enable_differentiation
        
        # Track differentiation state
        self.differentiation_rounds_used = 0  # 已使用的分化轮次计数
        
        # === 理论化概率控制参数 ===
        # Stagnation detection
        self.sliding_window_k = sliding_window_k
        self.stagnation_sensitivity_kappa = stagnation_sensitivity_kappa
        
        # Base probabilities
        self.alpha_s = alpha_s  # 分化基础概率
        self.alpha_m = alpha_m  # 融合基础概率
        
        # Decay factors
        self.eta_s = eta_s  # 分化衰减因子
        self.eta_m = eta_m  # 融合衰减因子
        
        # Fusion selection weights
        self.alpha_U = alpha_U      # 融合互补性权重
        self.alpha_I = alpha_I      # 融合一致性权重
        self.beta_triple = beta_triple  # 三路并集权重
        self.beta_pair = beta_pair      # 两两并集权重
        self.gamma_pair = gamma_pair    # 两两交集权重
        self.gamma_triple = gamma_triple  # 三路交集权重
        
        # Global operation counters (N_s, N_m)
        self.N_s = 0  # 分化操作计数
        self.N_m = 0  # 融合操作计数
        
        # Performance history (H_R)
        self.performance_history = []  # 存储每轮的最佳性能
        
        # Retry parameters for fusion and differentiation
        self.max_retries = 3  # Maximum number of retries for fusion and differentiation
        self.fusion_retry_count = 0  # Current retry count for fusion
        self.differentiation_retry_count = 0  # Current retry count for differentiation
        
        # Track fusion state
        self.last_fusion_round = -1  # Initialize to -1 to indicate no fusion has occurred yet
        
        # Track fusion metadata counter for proper numbering
        self.fusion_metadata_counter = 0
        self._initialize_fusion_counter()
        
        # Track workflow differentiation counts (per source round)
        self.workflow_differentiation_counts = {}  # {round_num: count}
        
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
        
        # Initialize workflow management utilities (needed by differentiation processor)
        self.workflow_manager = WorkflowManager(
            root_path=self.root_path,
            data_utils=self.data_utils,
            graph_utils=self.graph_utils
        )
        
        # Initialize differentiation processor
        if self.enable_differentiation:
            self.differentiation_processor = WorkflowDifferentiation(
                dataset=self.dataset,
                question_type=self.type,
                opt_llm_config=self.optimize_llm_config,
                exec_llm_config=self.execute_llm_config,
                operators=self.operators,
                optimized_path=optimized_path,
                validation_rounds=self.validation_rounds,
                workflow_manager=self.workflow_manager  # Pass workflow_manager
            )
            
            # Initialize problem classifier for directed differentiation
            self.problem_classifier = ProblemClassifier(
                exec_llm_config=self.execute_llm_config,
                dataset=self.dataset,
                optimized_path=optimized_path
            )
            
            # Track last differentiation round for each category
            self.category_last_differentiation = {}
        
        # Initialize fusion checker
        self.fusion_checker = FusionChecker(root_path=self.root_path)
    
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
    
    def optimize(self, mode: OptimizerType = "Graph"):
        """Enhanced optimize method with integrated fusion and differentiation"""
        if mode == "Test":
            test_n = 1  # validation datasets's execution number
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test())
            return None

        # Step 1: Classify all problems in validation set before optimization starts
        if self.enable_differentiation:
            logger.info("=" * 80)
            logger.info("STEP 1: Classifying all problems in validation set")
            logger.info("=" * 80)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            classification_success = loop.run_until_complete(self._ensure_problem_classification())
            loop.close()
            
            if classification_success:
                logger.info("✓ Problem classification completed successfully")
                logger.info("=" * 80)
            else:
                logger.warning("Problem classification failed, differentiation will be disabled")
                self.enable_differentiation = False
        
        # Step 2: Start optimization loop
        while self.round < self.max_rounds:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 1
            score = None

            while retry_count < max_retries:
                try:
                    # === Phase 3: Sampling & Target Selection (理论化概率控制) ===
                    
                    # 计算动态概率分布
                    probs = self._calculate_operation_probabilities()
                    
                    # 基于概率采样操作
                    import numpy as np
                    operation = np.random.choice(
                        ['optimize', 'differentiate', 'fuse'],
                        p=[probs['optimize'], probs['differentiate'], probs['fuse']]
                    )
                    
                    logger.info(f"=" * 80)
                    logger.info(f"ROUND {self.round}: Selected operation = {operation.upper()}")
                    logger.info(f"=" * 80)
                    
                    # 执行选定的操作
                    if operation == 'differentiate':
                        # 分化操作
                        if self.enable_differentiation:
                            logger.info(f"Executing DIFFERENTIATE operation for round {self.round + 1}")
                            score = self._attempt_with_retry(
                                lambda: loop.run_until_complete(self._attempt_differentiation()),
                                "differentiation", 3
                            )
                            if score is not None:
                                self.N_s += 1  # 增加分化计数
                                logger.info(f"✓ Differentiation successful! Updated N_s = {self.N_s}")
                            else:
                                logger.warning("Differentiation failed, falling back to optimization")
                                score = loop.run_until_complete(self._optimize_graph())
                        else:
                            logger.warning("Differentiation disabled, falling back to optimization")
                            score = loop.run_until_complete(self._optimize_graph())
                    
                    elif operation == 'fuse':
                        # 融合操作
                        if self.enable_fusion:
                            logger.info(f"Executing FUSE operation for round {self.round + 1}")
                            # 检查融合前置条件
                            if self._check_fusion_preconditions():
                                score = self._attempt_with_retry(
                                    lambda: loop.run_until_complete(self._attempt_fusion()),
                                    "fusion", 3
                                )
                                if score is not None:
                                    self.N_m += 1  # 增加融合计数
                                    logger.info(f"✓ Fusion successful! Updated N_m = {self.N_m}")
                                else:
                                    logger.warning("Fusion failed, falling back to optimization")
                                    score = loop.run_until_complete(self._optimize_graph())
                            else:
                                logger.warning("Fusion preconditions not met, falling back to optimization")
                                score = loop.run_until_complete(self._optimize_graph())
                        else:
                            logger.warning("Fusion disabled, falling back to optimization")
                            score = loop.run_until_complete(self._optimize_graph())
                    
                    else:  # operation == 'optimize'
                        # 优化操作
                        logger.info(f"Executing OPTIMIZE operation for round {self.round + 1}")
                        score = loop.run_until_complete(self._optimize_graph())
                    
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.warning("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")
            
            # 更新性能历史记录（用于停滞检测）
            if score is not None:
                self.performance_history.append(score)
                logger.info(f"Updated performance history (length={len(self.performance_history)})")

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

    def _attempt_with_retry(self, operation, operation_name: str, max_retries: int = 3):
        """
        Attempt an operation with retry mechanism.
        
        Args:
            operation: Callable operation to retry
            operation_name: Name of the operation for logging
            max_retries: Maximum number of retry attempts
            
        Returns:
            Result of the operation if successful, None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                result = operation()
                if result is not None:
                    if attempt > 0:
                        logger.info(f"{operation_name.capitalize()} succeeded on attempt {attempt + 1}")
                    return result
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"{operation_name.capitalize()} failed (attempt {attempt + 1}/{max_retries}), retrying...")
                    else:
                        logger.warning(f"{operation_name.capitalize()} failed after {max_retries} attempts")
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"{operation_name.capitalize()} error on attempt {attempt + 1}/{max_retries}: {e}, retrying...")
                else:
                    logger.error(f"{operation_name.capitalize()} error after {max_retries} attempts: {e}", exc_info=True)
        
        return None

    def _calculate_plateau(self) -> float:
        """
        Phase 1: 计算当前停滞度 (Stagnation Detection)
        
        使用滑动窗口比较最近 k 轮和之前 k 轮的性能
        通过 Logistic 映射将性能差异转换为停滞度指标
        
        公式:
            Δ_t = max(H_R[t-k+1:t]) - max(H_R[t-2k+1:t-k])
            plateau_t = 1 / (1 + exp(κ * Δ_t))
        
        Returns:
            float: 停滞度 [0, 1]
                - 接近 0: 性能快速提升（不停滞）
                - 接近 0.5: 性能平稳（轻度停滞）
                - 接近 1: 性能下降（严重停滞）
        """
        import math
        
        k = self.sliding_window_k
        kappa = self.stagnation_sensitivity_kappa
        t = len(self.performance_history)
        
        # Warm-up phase: 不足 2k 轮时，停滞度为 0
        if t < 2 * k:
            logger.debug(f"Warm-up phase: round {t} < {2*k}, plateau = 0.0")
            return 0.0
        
        # 计算最近 k 轮的最大性能
        R_recent = max(self.performance_history[t - k : t])
        
        # 计算之前 k 轮的最大性能
        R_prev = max(self.performance_history[t - 2*k : t - k])
        
        # 计算性能差异 Δ_t
        delta_t = R_recent - R_prev
        
        # Logistic 映射: plateau_t = 1 / (1 + exp(κ * Δ_t))
        try:
            plateau_t = 1.0 / (1.0 + math.exp(kappa * delta_t))
        except OverflowError:
            # 处理极端情况
            plateau_t = 1.0 if delta_t < 0 else 0.0
        
        logger.info(f"Stagnation Detection: R_recent={R_recent:.4f}, R_prev={R_prev:.4f}, "
                   f"Δ_t={delta_t:.4f}, plateau_t={plateau_t:.4f}")
        
        return plateau_t

    def _calculate_operation_probabilities(self) -> dict:
        """
        Phase 2: 计算三种操作的动态概率分布
        
        基于停滞度和操作历史，计算优化、分化、融合三种操作的概率
        
        公式:
            p_split = α_s · plateau_t · exp(-η_s · N_s)
            p_fuse = α_m · plateau_t · exp(-η_m · N_m)
            p_opt = max(0, 1 - p_split - p_fuse)
            π_t = Normalize({p_opt, p_split, p_fuse})
        
        Returns:
            dict: {
                'optimize': float,      # 优化概率
                'differentiate': float, # 分化概率
                'fuse': float,         # 融合概率
                'plateau': float,      # 停滞度（用于日志）
                'raw_probs': dict      # 归一化前的原始概率（用于调试）
            }
        """
        import math
        
        # Phase 1: 计算停滞度
        plateau_t = self._calculate_plateau()
        
        # Phase 2: 计算原始概率（未归一化）
        # p_split = α_s · plateau_t · exp(-η_s · N_s)
        p_split_raw = self.alpha_s * plateau_t * math.exp(-self.eta_s * self.N_s)
        
        # p_fuse = α_m · plateau_t · exp(-η_m · N_m)  
        p_fuse_raw = self.alpha_m * plateau_t * math.exp(-self.eta_m * self.N_m)
        
        # p_opt = max(0, 1 - p_split - p_fuse)
        p_opt_raw = max(0.0, 1.0 - p_split_raw - p_fuse_raw)
        
        # Phase 3: 归一化概率
        total = p_opt_raw + p_split_raw + p_fuse_raw
        
        if total > 0:
            p_opt = p_opt_raw / total
            p_split = p_split_raw / total
            p_fuse = p_fuse_raw / total
        else:
            # 极端情况：全部为 0，默认使用优化
            p_opt = 1.0
            p_split = 0.0
            p_fuse = 0.0
        
        # 记录详细日志
        logger.info(f"Operation Probabilities (Round {self.round}):")
        logger.info(f"  Plateau: {plateau_t:.4f}")
        logger.info(f"  Counters: N_s={self.N_s}, N_m={self.N_m}")
        logger.info(f"  Raw probs: p_opt={p_opt_raw:.4f}, p_split={p_split_raw:.4f}, p_fuse={p_fuse_raw:.4f}")
        logger.info(f"  Normalized: p_opt={p_opt:.4f}, p_split={p_split:.4f}, p_fuse={p_fuse:.4f}")
        
        return {
            'optimize': p_opt,
            'differentiate': p_split,
            'fuse': p_fuse,
            'plateau': plateau_t,
            'raw_probs': {
                'optimize': p_opt_raw,
                'differentiate': p_split_raw,
                'fuse': p_fuse_raw
            }
        }
    
    def _select_for_split(self, workflow_results: List[Dict]) -> Dict:
        """
        理论化分化目标选择 (Algorithm 2: SelectForSplit)
        
        基于势函数 Φ_split 选择最适合分化的工作流：
        Φ_split(i) = L_i · [λ₁(1-R_i) + λ₂D_i] · exp(-η_s·s_i)
        
        其中：
        - L_i: 局部停滞度（该工作流在近期是否停滞）
        - R_i: 归一化性能得分（性能越低越适合改进）
        - D_i: 多样性得分（与其他工作流的差异度）
        - s_i: 该工作流已被分化的次数
        - λ₁, λ₂: 平衡参数
        
        Args:
            workflow_results: 所有工作流的结果列表
            
        Returns:
            Dict: 选中的工作流信息
        """
        logger.info("=" * 80)
        logger.info("SelectForSplit: Calculating split potential for all workflows")
        logger.info("=" * 80)
        
        # 超参数
        lambda_1 = 0.6  # 性能改进权重
        lambda_2 = 0.4  # 多样性权重
        
        if not workflow_results or len(workflow_results) == 0:
            logger.warning("No workflows available for split selection")
            return None
        
        # 计算全局统计信息
        all_scores = [w.get('avg_score', 0.0) for w in workflow_results]
        max_score = max(all_scores) if all_scores else 1.0
        min_score = min(all_scores) if all_scores else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        candidates = []
        
        for workflow in workflow_results:
            round_num = workflow.get('round', 0)
            score = workflow.get('avg_score', 0.0)
            
            # 归一化性能得分 R_i ∈ [0, 1]
            R_i = (score - min_score) / score_range if score_range > 0 else 0.5
            
            # 计算局部停滞度 L_i
            # 检查该工作流在近 k 轮是否有提升
            L_i = self._calculate_local_plateau(workflow, workflow_results)
            
            # 计算多样性 D_i
            # 与其他高性能工作流的差异度
            D_i = self._calculate_diversity(workflow, workflow_results)
            
            # 获取该工作流被分化的次数
            s_i = self.workflow_differentiation_counts.get(round_num, 0)
            
            # 计算势函数 Φ_split
            improvement_potential = lambda_1 * (1 - R_i) + lambda_2 * D_i
            usage_penalty = np.exp(-self.eta_s * s_i)
            phi_split = L_i * improvement_potential * usage_penalty
            
            candidates.append({
                'workflow': workflow,
                'round': round_num,
                'score': score,
                'R_i': R_i,
                'L_i': L_i,
                'D_i': D_i,
                's_i': s_i,
                'phi_split': phi_split
            })
            
            logger.info(f"  Round {round_num}: score={score:.4f}, R_i={R_i:.4f}, L_i={L_i:.4f}, "
                       f"D_i={D_i:.4f}, s_i={s_i}, Φ_split={phi_split:.4f}")
        
        # Softmax 采样
        phi_values = np.array([c['phi_split'] for c in candidates])
        
        # 避免数值溢出
        phi_values = phi_values - np.max(phi_values)
        exp_values = np.exp(phi_values)
        probabilities = exp_values / np.sum(exp_values)
        
        # 采样选择
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        selected = candidates[selected_idx]
        
        logger.info("=" * 80)
        logger.info(f"Selected workflow from Round {selected['round']} for differentiation")
        logger.info(f"  Score: {selected['score']:.4f}, Φ_split: {selected['phi_split']:.4f}")
        logger.info(f"  Selection probability: {probabilities[selected_idx]:.4f}")
        logger.info("=" * 80)
        
        return selected['workflow']
    
    def _calculate_local_plateau(self, workflow: Dict, all_workflows: List[Dict]) -> float:
        """
        计算局部停滞度：该工作流在近期是否停滞
        
        通过比较该工作流与后续轮次中类似工作流的性能变化
        
        Returns:
            float: 停滞度 ∈ [0, 1]，越高表示越停滞
        """
        current_round = workflow.get('round', 0)
        current_score = workflow.get('avg_score', 0.0)
        
        # 查找后续轮次的得分
        future_scores = []
        for w in all_workflows:
            if w.get('round', 0) > current_round:
                future_scores.append(w.get('avg_score', 0.0))
        
        if not future_scores:
            # 最新轮次，默认中等停滞度
            return 0.5
        
        # 计算平均后续得分
        avg_future = np.mean(future_scores)
        delta = avg_future - current_score
        
        # 使用 logistic 映射
        kappa_local = 40.0  # 局部灵敏度
        try:
            L_i = 1.0 / (1.0 + np.exp(kappa_local * delta))
        except OverflowError:
            L_i = 0.0 if delta > 0 else 1.0
        
        return L_i
    
    def _calculate_diversity(self, workflow: Dict, all_workflows: List[Dict]) -> float:
        """
        计算多样性：该工作流与其他高性能工作流的差异度
        
        通过比较问题覆盖集合的差异
        
        Returns:
            float: 多样性 ∈ [0, 1]，越高表示越独特
        """
        # 简化实现：使用轮次差异作为代理
        # 更完整的实现需要比较工作流的操作符序列或问题覆盖集
        
        current_round = workflow.get('round', 0)
        current_score = workflow.get('avg_score', 0.0)
        
        # 找到高性能工作流（top 30%）
        all_scores = sorted([w.get('avg_score', 0.0) for w in all_workflows], reverse=True)
        if len(all_scores) > 0:
            threshold = all_scores[int(len(all_scores) * 0.3)] if len(all_scores) > 3 else all_scores[-1]
        else:
            threshold = 0.0
        
        high_perf_workflows = [w for w in all_workflows if w.get('avg_score', 0.0) >= threshold]
        
        if not high_perf_workflows or current_score < threshold:
            # 不在高性能集合中，认为很独特
            return 0.8
        
        # 计算与高性能工作流的平均轮次差异（归一化）
        round_diffs = [abs(current_round - w.get('round', 0)) for w in high_perf_workflows]
        avg_diff = np.mean(round_diffs) if round_diffs else 0.0
        max_diff = max(round_diffs) if round_diffs else 1.0
        
        diversity = avg_diff / max_diff if max_diff > 0 else 0.5
        
        return diversity
    
    def _select_for_fuse(self, workflow_results: List[Dict]) -> tuple:
        """
        理论化融合目标选择 (Algorithm 3: SelectForFuse) - 三路版本
        
        分两步选择最佳融合三元组：
        1. 筛选候选集：选择 Top-6 高覆盖率工作流
        2. 三路评分：综合考虑两两互补性/一致性和三路指标
        
        评分公式：
            # 两两指标（平均）
            Φ_U^pair = avg(|C_i ∪ C_j|, |C_j ∪ C_k|, |C_i ∪ C_k|)
            Φ_I^pair = avg(|C_i ∩ C_j|, |C_j ∩ C_k|, |C_i ∩ C_k|)
            
            # 三路指标
            Φ_U^triple = |C_i ∪ C_j ∪ C_k|
            Φ_I^triple = |C_i ∩ C_j ∩ C_k|
            
            # 综合互补性和一致性
            Φ_U = β_triple·Φ_U^triple + β_pair·Φ_U^pair  (β_triple=0.6, β_pair=0.4)
            Φ_I = γ_pair·Φ_I^pair + γ_triple·Φ_I^triple  (γ_pair=0.7, γ_triple=0.3)
            
            # 最终评分
            Φ_merge = (α_U·Φ_U + α_I·Φ_I) · Penalty(i,j,k)  (α_U=0.6, α_I=0.4)
        
        Args:
            workflow_results: 所有工作流的结果列表
            
        Returns:
            tuple: (workflow_1, workflow_2, workflow_3) 选中的融合三元组
        """
        logger.info("=" * 80)
        logger.info("SelectForFuse: Finding optimal workflow triple for 3-way fusion")
        logger.info("=" * 80)
        
        # 使用实例超参数
        alpha_U = self.alpha_U
        alpha_I = self.alpha_I
        beta_triple = self.beta_triple
        beta_pair = self.beta_pair
        gamma_pair = self.gamma_pair
        gamma_triple = self.gamma_triple
        top_M = min(6, len(workflow_results))  # 候选集大小（与伪代码一致）
        
        if len(workflow_results) < 3:
            logger.warning("Insufficient workflows for 3-way fusion (need at least 3)")
            return None, None, None
        
        # Step 1: 筛选 Top-6 高覆盖率候选
        sorted_workflows = sorted(workflow_results, 
                                  key=lambda w: w.get('avg_score', 0.0), 
                                  reverse=True)
        candidates = sorted_workflows[:top_M]
        
        logger.info(f"Step 1: Selected top {len(candidates)} candidates by coverage:")
        for c in candidates:
            solved_count = len(c.get('solved_problems', []))
            logger.info(f"  Round {c.get('round', 0)}: score={c.get('avg_score', 0.0):.4f}, "
                       f"solved={solved_count} problems")
        
        # Step 2: 三路评分
        best_phi = -float('inf')
        best_triple = (None, None, None)
        best_details = None
        
        total_combinations = len(candidates) * (len(candidates) - 1) * (len(candidates) - 2) // 6
        logger.info(f"Step 2: Evaluating {total_combinations} triple combinations...")
        logger.info(f"Hyperparameters: α_U={alpha_U}, α_I={alpha_I}, "
                   f"β_triple={beta_triple}, β_pair={beta_pair}, "
                   f"γ_pair={gamma_pair}, γ_triple={gamma_triple}")
        
        for i, w_i in enumerate(candidates):
            for j, w_j in enumerate(candidates):
                for k, w_k in enumerate(candidates):
                    # 避免重复和自配对，保证 i < j < k
                    if i >= j or j >= k:
                        continue
                    
                    round_i = w_i.get('round', 0)
                    round_j = w_j.get('round', 0)
                    round_k = w_k.get('round', 0)
                    score_i = w_i.get('avg_score', 0.0)
                    score_j = w_j.get('avg_score', 0.0)
                    score_k = w_k.get('avg_score', 0.0)
                    
                    # 获取每个工作流成功解决的问题集合
                    solved_i = set(w_i.get('solved_problems', []))
                    solved_j = set(w_j.get('solved_problems', []))
                    solved_k = set(w_k.get('solved_problems', []))
                    
                    # === 两两互补性（Pairwise Complementarity）===
                    # 计算三对两两并集
                    U_ij = solved_i | solved_j
                    U_jk = solved_j | solved_k
                    U_ik = solved_i | solved_k
                    
                    phi_U_pair = (len(U_ij) + len(U_jk) + len(U_ik)) / 3.0
                    
                    # === 两两一致性（Pairwise Consensus）===
                    # 计算三对两两交集
                    I_ij = solved_i & solved_j
                    I_jk = solved_j & solved_k
                    I_ik = solved_i & solved_k
                    
                    phi_I_pair = (len(I_ij) + len(I_jk) + len(I_ik)) / 3.0
                    
                    # === 三路指标（Triple-wise Metrics）===
                    # 三路并集（总覆盖）
                    U_triple = solved_i | solved_j | solved_k
                    phi_U_triple = len(U_triple)
                    
                    # 三路交集（强一致性）
                    I_triple = solved_i & solved_j & solved_k
                    phi_I_triple = len(I_triple)
                    
                    # === 综合互补性和一致性（使用超参数）===
                    phi_U = beta_triple * phi_U_triple + beta_pair * phi_U_pair
                    phi_I = gamma_pair * phi_I_pair + gamma_triple * phi_I_triple
                    
                    # 检查是否已融合过
                    temp_workflows = [
                        {"round": round_i, "avg_score": score_i},
                        {"round": round_j, "avg_score": score_j},
                        {"round": round_k, "avg_score": score_k}
                    ]
                    if self.fusion_checker.check_fusion_already_attempted(temp_workflows):
                        penalty = 0.1  # 大幅惩罚
                    else:
                        penalty = 1.0
                    
                    # 计算融合势函数（与伪代码一致：α_U=0.6, α_I=0.4）
                    phi_merge = (alpha_U * phi_U + alpha_I * phi_I) * penalty
                    
                    # 详细日志
                    logger.info(f"  Triple ({round_i}, {round_j}, {round_k}):")
                    logger.info(f"    Individual solved: |C_i|={len(solved_i)}, "
                               f"|C_j|={len(solved_j)}, |C_k|={len(solved_k)}")
                    logger.info(f"    Pairwise unions: |U_ij|={len(U_ij)}, "
                               f"|U_jk|={len(U_jk)}, |U_ik|={len(U_ik)}")
                    logger.info(f"    Pairwise intersections: |I_ij|={len(I_ij)}, "
                               f"|I_jk|={len(I_jk)}, |I_ik|={len(I_ik)}")
                    logger.info(f"    Triple metrics: |U_triple|={phi_U_triple}, "
                               f"|I_triple|={phi_I_triple}")
                    logger.info(f"    Φ_U^pair={phi_U_pair:.2f}, Φ_I^pair={phi_I_pair:.2f}")
                    logger.info(f"    Φ_U^triple={phi_U_triple}, Φ_I^triple={phi_I_triple}")
                    logger.info(f"    Combined: Φ_U={phi_U:.2f}, Φ_I={phi_I:.2f}")
                    logger.info(f"    Penalty={penalty:.2f}, Φ_merge={phi_merge:.2f}")
                    logger.info(f"    Penalty={penalty:.2f}, Φ_merge={phi_merge:.2f}")
                    
                    if phi_merge > best_phi:
                        best_phi = phi_merge
                        best_triple = (w_i, w_j, w_k)
                        # 保存详细信息用于最终输出
                        best_details = {
                            'rounds': (round_i, round_j, round_k),
                            'scores': (score_i, score_j, score_k),
                            'solved_counts': (len(solved_i), len(solved_j), len(solved_k)),
                            'pairwise_unions': (len(U_ij), len(U_jk), len(U_ik)),
                            'pairwise_intersections': (len(I_ij), len(I_jk), len(I_ik)),
                            'triple_union': phi_U_triple,
                            'triple_intersection': phi_I_triple,
                            'phi_U_pair': phi_U_pair,
                            'phi_I_pair': phi_I_pair,
                            'phi_U': phi_U,
                            'phi_I': phi_I,
                            'penalty': penalty,
                            'phi_merge': phi_merge
                        }
        
        if best_triple[0] is None:
            logger.warning("No valid fusion triple found")
            return None, None, None
        
        # 输出最佳三元组的详细信息
        logger.info("=" * 80)
        logger.info("SELECTED FUSION TRIPLE - Detailed Information:")
        logger.info("=" * 80)
        logger.info(f"Rounds: ({best_details['rounds'][0]}, "
                   f"{best_details['rounds'][1]}, {best_details['rounds'][2]})")
        logger.info(f"Scores: ({best_details['scores'][0]:.4f}, "
                   f"{best_details['scores'][1]:.4f}, {best_details['scores'][2]:.4f})")
        logger.info(f"")
        logger.info(f"Individual coverage:")
        logger.info(f"  W_i (Round {best_details['rounds'][0]}): {best_details['solved_counts'][0]} problems")
        logger.info(f"  W_j (Round {best_details['rounds'][1]}): {best_details['solved_counts'][1]} problems")
        logger.info(f"  W_k (Round {best_details['rounds'][2]}): {best_details['solved_counts'][2]} problems")
        logger.info(f"")
        logger.info(f"Pairwise complementarity (unions):")
        logger.info(f"  |C_i ∪ C_j| = {best_details['pairwise_unions'][0]} problems")
        logger.info(f"  |C_j ∪ C_k| = {best_details['pairwise_unions'][1]} problems")
        logger.info(f"  |C_i ∪ C_k| = {best_details['pairwise_unions'][2]} problems")
        logger.info(f"  Average: Φ_U^pair = {best_details['phi_U_pair']:.2f}")
        logger.info(f"")
        logger.info(f"Pairwise consensus (intersections):")
        logger.info(f"  |C_i ∩ C_j| = {best_details['pairwise_intersections'][0]} problems")
        logger.info(f"  |C_j ∩ C_k| = {best_details['pairwise_intersections'][1]} problems")
        logger.info(f"  |C_i ∩ C_k| = {best_details['pairwise_intersections'][2]} problems")
        logger.info(f"  Average: Φ_I^pair = {best_details['phi_I_pair']:.2f}")
        logger.info(f"")
        logger.info(f"Triple-wise metrics:")
        logger.info(f"  Total coverage: |C_i ∪ C_j ∪ C_k| = {best_details['triple_union']} problems")
        logger.info(f"  Strong consensus: |C_i ∩ C_j ∩ C_k| = {best_details['triple_intersection']} problems")
        logger.info(f"")
        logger.info(f"Combined scores (Hyperparameters: β_triple={beta_triple}, β_pair={beta_pair}, "
                   f"γ_pair={gamma_pair}, γ_triple={gamma_triple}):")
        logger.info(f"  Φ_U = {beta_triple}×{best_details['triple_union']} + "
                   f"{beta_pair}×{best_details['phi_U_pair']:.2f} = {best_details['phi_U']:.2f}")
        logger.info(f"  Φ_I = {gamma_pair}×{best_details['phi_I_pair']:.2f} + "
                   f"{gamma_triple}×{best_details['triple_intersection']} = {best_details['phi_I']:.2f}")
        logger.info(f"")
        logger.info(f"Final score (α_U={alpha_U}, α_I={alpha_I}):")
        logger.info(f"  Φ_merge = ({alpha_U}×{best_details['phi_U']:.2f} + "
                   f"{alpha_I}×{best_details['phi_I']:.2f}) × {best_details['penalty']:.2f}")
        logger.info(f"  Φ_merge = {best_details['phi_merge']:.2f}")
        logger.info("=" * 80)
        
        return best_triple

    def _check_fusion_preconditions(self) -> bool:
        """
        检查融合操作的前置条件（不包括概率决策，仅检查必要条件）
        
        Returns:
            bool: True if preconditions are met
        """
        # 检查是否有足够的包络工作流
        envelope_workflows = self.data_utils.find_envelope_workflows(self.max_envelope_workflows)
        if len(envelope_workflows) < self.max_envelope_workflows:
            logger.info(f"Insufficient workflows for fusion (found {len(envelope_workflows)}, need at least {self.max_envelope_workflows})")
            return False
        
        # 检查此融合组合是否已尝试过
        if self.fusion_checker.check_fusion_already_attempted(envelope_workflows):
            logger.info("Skipping fusion - this combination has been attempted before")
            return False
        
        logger.info(f"Fusion preconditions met: {len(envelope_workflows)} envelope workflows available")
        return True
    
    def _should_attempt_fusion(self) -> bool:
        """
        Determine if we should attempt fusion based on current conditions
        
        Returns:
            bool: True if fusion should be attempted
        """
        if not self.enable_fusion:
            return False
        
        # Don't attempt fusion before the specified start round
        if self.round < self.fusion_start_round:
            logger.info(f"Skipping fusion - not yet at start round {self.fusion_start_round} (current: {self.round})")
            return False
        
        # Don't attempt fusion if insufficient rounds have passed since last fusion
        if self.last_fusion_round != -1 and (self.round - self.last_fusion_round) < self.fusion_interval_rounds:
            rounds_since_last = self.round - self.last_fusion_round
            logger.info(f"Skipping fusion - insufficient interval (need {self.fusion_interval_rounds} rounds, only {rounds_since_last} have passed)")
            return False
        
        # Check if we have enough envelope workflows
        envelope_workflows = self.data_utils.find_envelope_workflows(self.max_envelope_workflows)
        if len(envelope_workflows) < self.max_envelope_workflows:
            logger.info(f"Insufficient workflows for fusion (found {len(envelope_workflows)}, need at least {self.max_envelope_workflows})")
            return False
        
        # Check if this specific fusion combination has been attempted before
        if self.fusion_checker.check_fusion_already_attempted(envelope_workflows):
            logger.info("Skipping fusion - this combination has been attempted before")
            return False
        
        logger.info(f"Fusion conditions met: {len(envelope_workflows)} envelope workflows available")
        return True
    
    async def _ensure_problem_classification(self) -> bool:
        """
        确保问题分类已完成，如果未完成则执行分类
        从 round_1 的 log.json 中读取问题数据进行分类
        
        Returns:
            bool: 分类是否可用
        """
        if not self.enable_differentiation:
            return False
        
        # Check if classification already exists
        classifications = self.problem_classifier.load_classifications()
        if classifications:
            logger.info("Problem classifications already exist, skipping classification")
            return True
        
        # Load validation data directly from validation file
        try:
            logger.info("No existing problem classification found, starting classification process...")
            logger.info("Loading validation data from validation file...")
            
            # Construct validation file path
            validation_file = f"data/datasets/{self.dataset.lower()}_validate.jsonl"
            
            if not os.path.exists(validation_file):
                logger.warning(f"Validation file not found: {validation_file}")
                return False
            
            # Load validation data
            validation_data = []
            with open(validation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    validation_data.append(json.loads(line))
            
            if not validation_data:
                logger.warning("Validation file is empty")
                return False
            
            logger.info(f"Loaded {len(validation_data)} problems from validation set")
            
            # Perform classification
            await self.problem_classifier.analyze_and_classify_problems(validation_data)
            
            # Log statistics
            stats = self.problem_classifier.get_category_statistics()
            logger.info(f"Classification complete: {len(stats)} categories identified")
            for category, info in stats.items():
                logger.info(f"  - {category}: {info['count']} problems")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to classify problems: {e}", exc_info=True)
            return False
    
    # ========== 旧方法已删除：_should_attempt_differentiation() ==========
    # 旧的基于固定概率和规则的分化判断已被新的理论化概率控制替代
    # 现在使用 _calculate_operation_probabilities() 进行概率采样
    
    async def _attempt_differentiation(self) -> float:
        """
        Attempt workflow differentiation with retry mechanism.
        
        Returns:
            float: Score of differentiated workflow if successful, None if failed after all retries
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Differentiation attempt {attempt + 1}/{self.max_retries}")
                result = await self._execute_single_differentiation()
                if result is not None:
                    logger.info(f"Differentiation successful on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Differentiation attempt {attempt + 1} failed, retrying...")
            except Exception as e:
                logger.error(f"Differentiation attempt {attempt + 1} failed with error: {e}", exc_info=True)
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying differentiation (attempt {attempt + 2}/{self.max_retries})...")
                    continue
        
        logger.error(f"All {self.max_retries} differentiation attempts failed")
        return None

    async def _execute_single_differentiation(self) -> float:
        """
        Attempt workflow differentiation for problem type specialization.
        
        Returns:
            float: Score of differentiated workflow if successful, None if failed
        """
        try:
            # Get workflow results
            workflow_results = self.data_utils.load_results(f"{self.root_path}/workflows")
            round_summaries = self.workflow_manager.get_round_summaries(workflow_results)
            self._update_differentiation_counts(round_summaries)
            
            # === 使用理论化选择方法 (Algorithm 2: SelectForSplit) ===
            selected_workflow = self._select_for_split(round_summaries)
            
            if selected_workflow is None:
                logger.warning("No suitable workflow selected for differentiation")
                return None
            
            source_round = selected_workflow.get('round', 0)
            original_score = selected_workflow.get('avg_score', 0.0)
            
            # Update differentiation count
            self.workflow_differentiation_counts[source_round] = self.workflow_differentiation_counts.get(source_round, 0) + 1
            
            logger.info(f"Selected workflow from round {source_round} for problem type specialization")
            logger.info(f"  Score: {original_score:.4f}")
            
            # Load source workflow and create differentiated version
            source_workflow_content = await self.workflow_manager.load_workflow_content(source_round)
            operator_description = self.graph_utils.load_operators_description(self.operators)
            
            # Calculate target round
            next_round = self.round + 1
            
            # Select target category for differentiation
            target_category, example_problems = self.problem_classifier.select_target_category_for_differentiation(
                workflow_differentiation_history=self.workflow_manager.get_differentiation_history()
            )
            
            if not target_category:
                logger.warning("No suitable target category found for differentiation")
                return None
            
            logger.info(f"Selected target category for differentiation: {target_category}")
            logger.info(f"Providing {len(example_problems)} example problems for this category")
            
            differentiation_response = await self.differentiation_processor.create_differentiated_workflow(
                source_workflow=source_workflow_content,
                differentiation_direction="problem_type_specialization",
                operator_description=operator_description,
                target_round=next_round,
                target_category=target_category,
                category_examples=example_problems
            )
            
            if not differentiation_response:
                logger.error("Differentiation process failed")
                return None
            
            # Save differentiated workflow to next round
            next_round = self.round + 1
            success = self.differentiation_processor.save_differentiated_workflow_direct(
                differentiation_response, selected_workflow, "problem_type_specialization",
                next_round, self.root_path, self.graph_utils, self.experience_utils,
                target_category=target_category
            )
            
            if not success:
                logger.error("Failed to save differentiated workflow")
                return None
            
            # Evaluate the differentiated workflow
            graph_path = f"{self.root_path}/workflows"
            directory = f"{self.root_path}/workflows/round_{next_round}"
            
            self.graph = self.graph_utils.load_graph(next_round, graph_path)
            if self.graph is None:
                logger.error("Failed to load differentiated workflow")
                return None
            
            data = self.data_utils.load_results(graph_path)
            differentiation_score = await self.evaluation_utils.evaluate_graph(
                self, directory, self.validation_rounds, data, initial=False
            )
            
            # Update experience.json if needed
            experience_path = os.path.join(directory, "experience.json")
            if os.path.exists(experience_path):
                with open(experience_path, 'r', encoding='utf-8') as f:
                    experience_data = json.load(f)
                if experience_data.get("after") is None:
                    self.experience_utils.update_experience(directory, experience_data, differentiation_score)
            
            # Save metadata
            self.differentiation_processor.save_differentiation_metadata(
                source_workflow=selected_workflow,
                differentiated_workflow=differentiation_response,
                differentiation_direction="problem_type_specialization",
                target_round=next_round,
                differentiation_score=differentiation_score
            )
            
            # Only increment differentiation count if successful
            self.differentiation_rounds_used += 1
            
            logger.info(f"Problem type specialization completed with score: {differentiation_score:.4f}")
            return differentiation_score
        
        except Exception as e:
            logger.error(f"Error in single differentiation execution: {e}", exc_info=True)
            return None

    def _update_differentiation_counts(self, round_summaries: List[Dict]) -> None:
        """
        Update differentiation count information for each workflow
        
        Args:
            round_summaries: Workflow round summary data
        """
        for summary in round_summaries:
            round_num = summary.get('round')
            if round_num is not None:
                # Inject differentiation count information into workflow data
                current_count = self.workflow_differentiation_counts.get(round_num, 0)
                summary['differentiation_count'] = current_count
                
                # Check if this is a differentiated workflow (can be determined through metadata files or other methods)
                summary['is_differentiated'] = self._check_if_differentiated_workflow(round_num)
    
    def _check_if_differentiated_workflow(self, round_num: int) -> bool:
        """
        Check if the workflow of specified round was created through differentiation
        
        Args:
            round_num: Round number
            
        Returns:
            bool: Whether this is a differentiated workflow
        """
        try:
            # Check if differentiation metadata file exists
            workflows_dir = f"{self.root_path}/workflows"
            metadata_file = f"differentiation_metadata_{round_num}.json"
            metadata_path = os.path.join(workflows_dir, metadata_file)
            
            if os.path.exists(metadata_path):
                return True
                
            # Check modification field in experience.json file
            experience_file = os.path.join(workflows_dir, f"round_{round_num}", "experience.json")
            if os.path.exists(experience_file):
                with open(experience_file, 'r', encoding='utf-8') as f:
                    experience_data = json.load(f)
                    modification = experience_data.get('modification', '').lower()
                    if 'differentiation' in modification or 'specialized' in modification:
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking if workflow {round_num} is differentiated: {e}")
            return False

    async def _attempt_fusion(self) -> float:
        """
        Attempt workflow fusion with retry mechanism.
        
        Returns:
            float: Score of fused workflow if successful, None if failed after all retries
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fusion attempt {attempt + 1}/{self.max_retries}")
                result = await self._execute_single_fusion()
                if result is not None:
                    logger.info(f"Fusion successful on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Fusion attempt {attempt + 1} failed, retrying...")
            except Exception as e:
                logger.error(f"Fusion attempt {attempt + 1} failed with error: {e}", exc_info=True)
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying fusion (attempt {attempt + 2}/{self.max_retries})...")
                    continue
        
        logger.error(f"All {self.max_retries} fusion attempts failed")
        return None

    async def _execute_single_fusion(self) -> float:
        """
        Execute a single fusion attempt.
        
        Returns:
            float: Score of fused workflow if successful, None if failed
        """
        try:
            # === 使用理论化选择方法 (Algorithm 3: SelectForFuse - 三路版本) ===
            workflow_results = self.data_utils.load_results(f"{self.root_path}/workflows")
            round_summaries = self.workflow_manager.get_round_summaries(workflow_results)
            
            fusion_triple = self._select_for_fuse(round_summaries)
            
            if fusion_triple[0] is None or fusion_triple[1] is None or fusion_triple[2] is None:
                logger.warning("No suitable workflow triple selected for 3-way fusion")
                return None
            
            w1, w2, w3 = fusion_triple
            
            # 计算基准分数：三个工作流中的最小值
            min_envelope_score = min(
                w1.get('avg_score', 0.0), 
                w2.get('avg_score', 0.0),
                w3.get('avg_score', 0.0)
            )
            
            logger.info(f"Minimum score in fusion triple: {min_envelope_score:.4f}")
            logger.info(f"  W1 (Round {w1.get('round', 0)}): {w1.get('avg_score', 0.0):.4f}")
            logger.info(f"  W2 (Round {w2.get('round', 0)}): {w2.get('avg_score', 0.0):.4f}")
            logger.info(f"  W3 (Round {w3.get('round', 0)}): {w3.get('avg_score', 0.0):.4f}")
            
            # Execute fusion process (creates workflow directly in next round directory)
            fusion_success = await self._execute_fusion_async()
            
            if not fusion_success:
                logger.error("Fusion process failed")
                return None
            
            # Now evaluate the adopted fusion workflow using standard evaluation process
            next_round = self.round + 1
            graph_path = f"{self.root_path}/workflows"
            directory = f"{self.root_path}/workflows/round_{next_round}"
            
            # Load the fused graph
            self.graph = self.graph_utils.load_graph(next_round, graph_path)
            if self.graph is None:
                logger.error("Failed to load adopted fused workflow")
                return None
            
            # Load data for evaluation context
            data = self.data_utils.load_results(graph_path)
            
            # Evaluate using standard evaluation process
            fusion_score = await self.evaluation_utils.evaluate_graph(
                self, directory, self.validation_rounds, data, initial=False
            )
            
            # The standard evaluation process should have updated the experience.json
            # But let's make sure by explicitly calling the update if needed
            experience_path = os.path.join(directory, "experience.json")
            if os.path.exists(experience_path):
                # Load the experience data to check if it was properly updated
                with open(experience_path, 'r', encoding='utf-8') as f:
                    experience_data = json.load(f)
                
                if experience_data.get("after") is None:
                    # If not updated by standard process, update it manually using standard method
                    self.experience_utils.update_experience(directory, experience_data, fusion_score)
            
            logger.info(f"Fused workflow score: {fusion_score:.4f}")
            
            # 将融合三元组转换为兼容格式
            fusion_triple_list = [w1, w2, w3]
            
            # Check if fusion meets threshold
            if fusion_score > min_envelope_score + self.fusion_score_threshold:
                logger.info(f"3-way fusion successful! Score {fusion_score:.4f} > threshold {min_envelope_score + self.fusion_score_threshold:.4f}")
                
                # Record successful fusion round
                self.last_fusion_round = self.round
                
                # Save metadata for successful adoption
                self.fusion_processor.save_fusion_metadata(
                    fusion_triple_list, fusion_score, self.round, self.fusion_metadata_counter + 1, adopted=True
                )
                self.fusion_metadata_counter += 1
                
                return fusion_score
            else:
                logger.info(f"Fusion score {fusion_score:.4f} below threshold {min_envelope_score + self.fusion_score_threshold:.4f}")
                # Even if below threshold, we keep the fusion workflow since it was already created
                # Record successful fusion round (even if below threshold, fusion was successful)
                self.last_fusion_round = self.round
                
                # Save metadata for tracking purposes
                self.fusion_processor.save_fusion_metadata(
                    fusion_triple_list, fusion_score, self.round, self.fusion_metadata_counter + 1, adopted=True
                )
                self.fusion_metadata_counter += 1
                return fusion_score
        
        except Exception as e:
            logger.error(f"Error in single fusion execution: {e}", exc_info=True)
            return None
    
    async def _execute_fusion_async(self) -> bool:
        """
        Execute the fusion process asynchronously
        
        Returns:
            bool: True if fusion successful
        """
        try:
            # Find envelope workflows (need exactly 3 for 3-way fusion)
            envelope_workflows = self.data_utils.find_envelope_workflows(self.max_envelope_workflows)
            
            if len(envelope_workflows) < self.max_envelope_workflows:
                logger.warning(f"Insufficient workflows for 3-way fusion: found {len(envelope_workflows)}, need {self.max_envelope_workflows}")
                return False
            
            source_rounds = [w['round'] for w in envelope_workflows]
            source_scores = [f"{w['avg_score']:.4f}" for w in envelope_workflows]
            logger.info(f"Starting 3-way fusion with workflows from rounds: {source_rounds}")
            logger.info(f"Source workflow scores: {source_scores}")
            
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
            
            # Create fusion prompt and call LLM
            logger.info("Calling LLM to perform 3-way workflow fusion...")
            fusion_response = await self.fusion_processor.create_fused_workflow(
                envelope_workflows=envelope_workflows,
                workflow_contents=workflow_contents,
                operator_description=operator_description
            )
            
            if not fusion_response:
                logger.error("3-way fusion LLM call failed")
                return False
            
            logger.info("3-way fusion LLM call successful, saving fused workflow...")
            # Save fused workflow directly to next round directory using fusion processor
            success = self.fusion_processor.save_fused_workflow_direct(
                fusion_response, envelope_workflows, self.root_path, self.round, 
                self.graph_utils, self.experience_utils
            )
            
            if success:
                logger.info(f"✓ 3-way fusion completed: merged rounds {source_rounds} into round {self.round + 1}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing fusion: {e}", exc_info=True)
            return False
    
    async def _evaluate_fused_workflow(self) -> float:
        """
        Evaluate the fused workflow from the next round directory and return its score
        
        Returns:
            float: Average score of fused workflow, None if evaluation failed
        """
        try:
            # Load fused workflow from next round directory
            next_round = self.round + 1
            fusion_path = f"{self.root_path}/workflows"
            fused_graph = self.graph_utils.load_graph(next_round, fusion_path)
            
            if fused_graph is None:
                logger.error("Failed to load fused workflow from next round directory")
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
            logger.error(f"Error evaluating fused workflow: {e}", exc_info=True)
            return None

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
