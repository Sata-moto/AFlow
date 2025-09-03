# -*- coding: utf-8 -*-
# @Date    : 8/28/2025 20:00 PM
# @Author  : didi
# @Desc    : AFlow with Workflow Fusion Enhancement

import argparse
import os
import glob
import shutil
from typing import Dict, List

from data.download_data import download
from scripts.optimizer import Optimizer
from scripts.async_llm import LLMsConfig
from scripts.workflow_fusion import WorkflowFusion

class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators


EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "DROP": ExperimentConfig(
        dataset="DROP",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
    "HotpotQA": ExperimentConfig(
        dataset="HotpotQA",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "MBPP": ExperimentConfig(
        dataset="MBPP",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "LiveCodeBench": ExperimentConfig(
        dataset="LiveCodeBench",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
}


def clear_optimization_records(dataset: str, optimized_path: str = "workspace"):
    """
    Clear previous optimization records for the specified dataset.
    
    Args:
        dataset: Dataset name (e.g., "MATH", "MBPP")
        optimized_path: Base path where optimization records are stored
    """
    dataset_path = os.path.join(optimized_path, dataset)
    workflows_path = os.path.join(dataset_path, "workflows")
    
    if not os.path.exists(workflows_path):
        print(f"No existing records found for dataset {dataset}")
        return
    
    print(f"Clearing previous optimization records for dataset {dataset}...")
    
    # 1. Remove round_2 and later directories
    for round_dir in glob.glob(os.path.join(workflows_path, "round_*")):
        round_name = os.path.basename(round_dir)
        if round_name.startswith("round_"):
            try:
                round_num = int(round_name.split("_")[1])
                if round_num >= 2:
                    print(f"  Removing {round_dir}")
                    shutil.rmtree(round_dir)
            except (ValueError, IndexError):
                continue
    
    # 2. Clear processed_experience.json
    processed_exp_path = os.path.join(workflows_path, "processed_experience.json")
    if os.path.exists(processed_exp_path):
        print(f"  Clearing {processed_exp_path}")
        with open(processed_exp_path, 'w', encoding='utf-8') as f:
            f.write("")
    
    # 3. Clear results.json
    results_path = os.path.join(workflows_path, "results.json")
    if os.path.exists(results_path):
        print(f"  Clearing {results_path}")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("")
    
    # 4. Remove CSV files and clear log.json in round_1
    round_1_path = os.path.join(workflows_path, "round_1")
    if os.path.exists(round_1_path):
        # Remove CSV files (score files like 0.45378_20250821_170128.csv)
        for csv_file in glob.glob(os.path.join(round_1_path, "*.csv")):
            print(f"  Removing {csv_file}")
            os.remove(csv_file)
        
        # Clear log.json
        log_path = os.path.join(round_1_path, "log.json")
        if os.path.exists(log_path):
            print(f"  Clearing {log_path}")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("")
    
    print(f"Successfully cleared optimization records for dataset {dataset}")


def parse_args():
    parser = argparse.ArgumentParser(description="AFlow Optimizer with Workflow Fusion")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help="Dataset type",
    )
    parser.add_argument("--sample", type=int, default=3, help="Sample count")
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="workspace",
        help="Optimized result save path",
    )
    parser.add_argument("--initial_round", type=int, default=1, help="Initial round")
    parser.add_argument("--max_rounds", type=int, default=20, help="Max iteration rounds")
    parser.add_argument("--check_convergence", type=bool, default=True, help="Whether to enable early stop")
    parser.add_argument("--validation_rounds", type=int, default=1, help="Validation rounds")
    parser.add_argument(
        "--clear_previous_records",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to clear previous optimization records. Ignored when initial_round != 1.",
    )
    parser.add_argument(
        "--if_force_download",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether enforce dataset download.",
    )
    parser.add_argument(
        "--opt_model_name",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Specifies the name of the model used for optimization tasks.",
    )
    parser.add_argument(
        "--exec_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Specifies the name of the model used for execution tasks.",
    )
    parser.add_argument(
        "--enable_fusion",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to enable workflow fusion after optimization.",
    )
    parser.add_argument(
        "--max_envelope_workflows",
        type=int,
        default=3,
        help="Maximum number of workflows in envelope set for fusion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = EXPERIMENT_CONFIGS[args.dataset]

    models_config = LLMsConfig.default()
    opt_llm_config = models_config.get(args.opt_model_name)
    if opt_llm_config is None:
        raise ValueError(
            f"The optimization model '{args.opt_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --opt_model_name flag. "
        )

    exec_llm_config = models_config.get(args.exec_model_name)
    if exec_llm_config is None:
        raise ValueError(
            f"The execution model '{args.exec_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --exec_model_name flag. "
        )

    download(["datasets"], force_download=args.if_force_download)

    # Clear previous optimization records if requested and initial_round is 1
    if args.clear_previous_records and args.initial_round == 1:
        clear_optimization_records(config.dataset, args.optimized_path)

    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        check_convergence=args.check_convergence,
        operators=config.operators,
        optimized_path=args.optimized_path,
        sample=args.sample,
        initial_round=args.initial_round,
        max_rounds=args.max_rounds,
        validation_rounds=args.validation_rounds,
    )

    print("\n" + "="*50)
    print("Phase 1: Standard Workflow Optimization")
    print("="*50)
    
    # Phase 1: Standard optimization
    # optimizer.optimize("Graph")

    # Phase 2: Workflow Fusion (if enabled)
    if args.enable_fusion:
        print("\n" + "="*50)
        print("Phase 2: Workflow Fusion")
        print("="*50)
        
        fusion_processor = WorkflowFusion(
            dataset=config.dataset,
            question_type=config.question_type,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            operators=config.operators,
            optimized_path=args.optimized_path,
            max_envelope_workflows=args.max_envelope_workflows,
            validation_rounds=args.validation_rounds,
        )
        
        # Execute fusion process
        fusion_processor.execute_fusion()

    print("\n" + "="*50)
    print("Phase 3: Testing Phase")
    print("="*50)

    # Phase 3: Test original best workflow
    print("Testing original best workflow...")
    # optimizer.optimize("Test")
    
    # Phase 4: Test fused workflow (if fusion was performed)
    if args.enable_fusion:
        print("Testing fused workflow...")
        fusion_processor.test_fused_workflow()

    print("\n" + "="*50)
    print("Workflow optimization and fusion completed!")
    print("="*50)
