import argparse
import os
import glob
import shutil
from typing import Dict, List

from data.download_data import download
from scripts.enhanced_optimizer import EnhancedOptimizer
from scripts.async_llm import LLMsConfig

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
    "MIXED": ExperimentConfig(
        dataset="MIXED",
        question_type="mixed",  # 混合数据集包含 DROP (qa), MATH (math), GSM8K (math)
        operators=["Custom", "AnswerGenerate", "ScEnsemble", "Programmer"],
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
    
    # 1. Remove round_2 and later directories, including round_fused
    for round_dir in glob.glob(os.path.join(workflows_path, "round_*")):
        round_name = os.path.basename(round_dir)
        if round_name.startswith("round_"):
            # Handle special case of round_fused
            if round_name == "round_fused":
                print(f"  Removing {round_dir}")
                shutil.rmtree(round_dir)
                continue
                
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
    
    # 4. Remove fusion and differentiation metadata files
    # Remove old single fusion_metadata.json
    fusion_metadata_path = os.path.join(workflows_path, "fusion_metadata.json")
    if os.path.exists(fusion_metadata_path):
        print(f"  Removing {fusion_metadata_path}")
        os.remove(fusion_metadata_path)
    
    # Remove all numbered fusion metadata files (fusion_metadata_1.json, fusion_metadata_2.json, ...)
    for fusion_file in glob.glob(os.path.join(workflows_path, "fusion_metadata_*.json")):
        print(f"  Removing {fusion_file}")
        os.remove(fusion_file)
    
    # Remove all numbered differentiation metadata files (differentiation_metadata_1.json, ...)
    for diff_file in glob.glob(os.path.join(workflows_path, "differentiation_metadata_*.json")):
        print(f"  Removing {diff_file}")
        os.remove(diff_file)
    
    # 5. Remove CSV files and clear log.json in round_1
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
    parser = argparse.ArgumentParser(description="Enhanced AFlow Optimizer with Integrated Workflow Fusion and Differentiation")
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
    parser.add_argument(
        "--check_convergence",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to enable early stop"
    )
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
        default="claude-sonnet-4-20250514",
        help="Specifies the name of the model used for optimization tasks.",
    )
    parser.add_argument(
        "--exec_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Specifies the name of the model used for execution tasks.",
    )
    # Fusion-specific parameters
    parser.add_argument(
        "--enable_fusion",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to enable workflow fusion during optimization.",
    )
    parser.add_argument(
        "--max_envelope_workflows",
        type=int,
        default=3,
        help="Maximum number of workflows in envelope set for fusion.",
    )
    parser.add_argument(
        "--fusion_score_threshold",
        type=float,
        default=0.0,
        help="Minimum score improvement required for fusion adoption.",
    )
    # Differentiation-specific parameters
    parser.add_argument(
        "--enable_differentiation",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to enable workflow differentiation during optimization.",
    )
    # Theoretical probability control parameters (new)
    parser.add_argument(
        "--sliding_window_k",
        type=int,
        default=2,
        help="Sliding window size for stagnation detection.",
    )
    parser.add_argument(
        "--stagnation_sensitivity_kappa",
        type=float,
        default=30.0,
        help="Sensitivity parameter kappa for stagnation detection.",
    )
    parser.add_argument(
        "--alpha_s",
        type=float,
        default=0.60,
        help="Base probability for differentiation (α_s).",
    )
    parser.add_argument(
        "--alpha_m",
        type=float,
        default=0.50,
        help="Base probability for fusion (α_m).",
    )
    parser.add_argument(
        "--eta_s",
        type=float,
        default=0.1,
        help="Decay factor for differentiation (η_s).",
    )
    parser.add_argument(
        "--eta_m",
        type=float,
        default=0.1,
        help="Decay factor for fusion (η_m).",
    )
    # Fusion selection weights
    parser.add_argument(
        "--alpha_U",
        type=float,
        default=0.6,
        help="Fusion complementarity weight (α_U).",
    )
    parser.add_argument(
        "--alpha_I",
        type=float,
        default=0.4,
        help="Fusion consensus weight (α_I).",
    )
    parser.add_argument(
        "--beta_triple",
        type=float,
        default=0.6,
        help="Triple-wise union weight (β_triple).",
    )
    parser.add_argument(
        "--beta_pair",
        type=float,
        default=0.4,
        help="Pairwise union weight (β_pair).",
    )
    parser.add_argument(
        "--gamma_pair",
        type=float,
        default=0.7,
        help="Pairwise intersection weight (γ_pair).",
    )
    parser.add_argument(
        "--gamma_triple",
        type=float,
        default=0.3,
        help="Triple-wise intersection weight (γ_triple).",
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

    optimizer = EnhancedOptimizer(
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
        enable_fusion=args.enable_fusion,
        max_envelope_workflows=args.max_envelope_workflows,
        fusion_score_threshold=args.fusion_score_threshold,
        enable_differentiation=args.enable_differentiation,
        # Theoretical probability control parameters
        sliding_window_k=args.sliding_window_k,
        stagnation_sensitivity_kappa=args.stagnation_sensitivity_kappa,
        alpha_s=args.alpha_s,
        alpha_m=args.alpha_m,
        eta_s=args.eta_s,
        eta_m=args.eta_m,
        # Fusion selection weights
        alpha_U=args.alpha_U,
        alpha_I=args.alpha_I,
        beta_triple=args.beta_triple,
        beta_pair=args.beta_pair,
        gamma_pair=args.gamma_pair,
        gamma_triple=args.gamma_triple,
    )

    print("\n" + "="*50)
    print("Enhanced AFlow with Theoretical Probability Control")
    print("="*50)
    print(f"Dataset: {config.dataset}")
    print(f"Fusion enabled: {args.enable_fusion}")
    if args.enable_fusion:
        print(f"  Max envelope workflows: {args.max_envelope_workflows}")
        print(f"  Fusion score threshold: {args.fusion_score_threshold}")
    print(f"Differentiation enabled: {args.enable_differentiation}")
    print("\nTheoretical Probability Control Parameters:")
    print(f"  Sliding window k: {args.sliding_window_k}")
    print(f"  Stagnation sensitivity κ: {args.stagnation_sensitivity_kappa}")
    print(f"  Base probability α_s (differentiation): {args.alpha_s}")
    print(f"  Base probability α_m (fusion): {args.alpha_m}")
    print(f"  Decay factor η_s (differentiation): {args.eta_s}")
    print(f"  Decay factor η_m (fusion): {args.eta_m}")
    print("="*50)
    
    # Run enhanced optimization with integrated fusion
    optimizer.optimize("Graph")

    print("\n" + "="*50)
    print("Testing Phase")
    print("="*50)

    # Test the best workflow
    optimizer.optimize("Test")

    print("\n" + "="*50)
    print("Enhanced optimization with theoretical probability control completed!")
    print("="*50)
