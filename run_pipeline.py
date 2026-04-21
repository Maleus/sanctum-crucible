"""Sanctum Crucible - Full Pipeline Runner

Runs the complete experiment pipeline:
1. Baseline PAIR attack
2. Data preparation + QLoRA fine-tuning
3. Fine-tuned PAIR attack + comparison

Usage:
    python run_pipeline.py                    # Run all phases
    python run_pipeline.py --phase baseline   # Run only baseline
    python run_pipeline.py --phase finetune   # Run only fine-tuning
    python run_pipeline.py --phase evaluate   # Run only evaluation
    python run_pipeline.py --phase compare    # Run only comparison
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log"),
    ],
)
logger = logging.getLogger("crucible.pipeline")


def phase_baseline():
    """Phase 1: Run baseline PAIR attack."""
    from crucible.data import load_harmbench
    from crucible.pair import PAIROrchestrator
    from crucible.utils.serving import start_all_servers, stop_all_servers

    logger.info("=" * 60)
    logger.info("PHASE 1: Baseline PAIR Attack")
    logger.info("=" * 60)

    servers = start_all_servers("configs/models.yaml")

    try:
        behaviors = load_harmbench()
        orchestrator = PAIROrchestrator(config_dir="configs")
        summary = orchestrator.run(behaviors, run_name="baseline")

        logger.info(f"Baseline ASR: {summary['overall_asr']:.2%}")
        return summary
    finally:
        stop_all_servers(servers)


def phase_finetune():
    """Phase 2: Prepare data and fine-tune the attacker."""
    from crucible.finetune.prepare import prepare_dataset
    from crucible.finetune.train import run_finetune

    logger.info("=" * 60)
    logger.info("PHASE 2: Data Preparation & Fine-tuning")
    logger.info("=" * 60)

    # Prepare dataset
    result = prepare_dataset("configs/finetune.yaml")
    logger.info(
        f"Dataset: {len(result['train'])} train, "
        f"{len(result['validation'])} val examples"
    )

    # Fine-tune
    output_dir = run_finetune("configs/finetune.yaml")
    logger.info(f"Fine-tuned model saved to: {output_dir}")

    return output_dir


def phase_finetune_multiturn():
    """Phase 2b: Prepare multi-turn data and fine-tune the attacker."""
    from crucible.finetune.prepare_multiturn import prepare_multiturn_dataset
    from crucible.finetune.train import run_finetune

    logger.info("=" * 60)
    logger.info("PHASE 2b: Multi-turn Data Preparation & Fine-tuning")
    logger.info("=" * 60)

    config_path = "configs/finetune_multiturn.yaml"

    result = prepare_multiturn_dataset(config_path)
    if not result:
        logger.error("Multi-turn dataset preparation failed — no data produced")
        return None

    logger.info(
        f"Multi-turn dataset: {len(result['train'])} train, "
        f"{len(result['validation'])} val examples "
        f"({len(result['conversations'])} conversations)"
    )

    output_dir = run_finetune(config_path)
    logger.info(f"Multi-turn fine-tuned model saved to: {output_dir}")

    return output_dir


def _run_finetuned_evaluation(checkpoint_dir: str, run_name: str) -> dict:
    """Run PAIR attack with a fine-tuned attacker checkpoint."""
    import yaml
    from crucible.data import load_harmbench
    from crucible.pair import PAIROrchestrator
    from crucible.utils.serving import start_all_servers, stop_all_servers

    with open("configs/models.yaml") as f:
        config = yaml.safe_load(f)

    config["attacker"]["hf_repo"] = checkpoint_dir
    config["attacker"]["quantization"] = "none"

    finetuned_config = f"configs/models_{run_name}.yaml"
    with open(finetuned_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    servers = start_all_servers(finetuned_config)

    try:
        behaviors = load_harmbench()
        orchestrator = PAIROrchestrator(config_dir="configs")
        orchestrator.models_config = config
        summary = orchestrator.run(behaviors, run_name=run_name)

        logger.info(f"{run_name} ASR: {summary['overall_asr']:.2%}")
        return summary
    finally:
        stop_all_servers(servers)


def phase_evaluate():
    """Phase 3: Run fine-tuned PAIR attack (single-turn model)."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Fine-tuned PAIR Attack (single-turn)")
    logger.info("=" * 60)

    return _run_finetuned_evaluation(
        checkpoint_dir="checkpoints/attacker-finetuned",
        run_name="finetuned",
    )


def phase_evaluate_multiturn():
    """Phase 3b: Run fine-tuned PAIR attack (multi-turn model)."""
    logger.info("=" * 60)
    logger.info("PHASE 3b: Fine-tuned PAIR Attack (multi-turn)")
    logger.info("=" * 60)

    return _run_finetuned_evaluation(
        checkpoint_dir="checkpoints/attacker-finetuned-multiturn",
        run_name="finetuned-multiturn",
    )


def phase_compare():
    """Compare baseline vs single-turn vs multi-turn fine-tuned results."""
    from crucible.evaluate.metrics import compare_runs, print_comparison

    logger.info("=" * 60)
    logger.info("COMPARISON: Baseline vs Fine-tuned variants")
    logger.info("=" * 60)

    comparisons = {}

    # Baseline vs single-turn
    if Path("results/pair_logs/finetuned").exists():
        logger.info("--- Baseline vs Single-turn Fine-tuned ---")
        comp_st = compare_runs(
            baseline_dir="results/pair_logs/baseline",
            finetuned_dir="results/pair_logs/finetuned",
        )
        print_comparison(comp_st)
        comparisons["single_turn"] = comp_st

    # Baseline vs multi-turn
    if Path("results/pair_logs/finetuned-multiturn").exists():
        logger.info("--- Baseline vs Multi-turn Fine-tuned ---")
        comp_mt = compare_runs(
            baseline_dir="results/pair_logs/baseline",
            finetuned_dir="results/pair_logs/finetuned-multiturn",
        )
        print_comparison(comp_mt)
        comparisons["multi_turn"] = comp_mt

    if not comparisons:
        logger.error("No fine-tuned results found to compare")
        return None

    with open("results/comparison.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    logger.info("Comparison saved to results/comparison.json")
    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Sanctum Crucible Pipeline")
    parser.add_argument(
        "--phase",
        choices=[
            "baseline",
            "finetune", "finetune-multiturn",
            "evaluate", "evaluate-multiturn",
            "compare",
            "all", "all-multiturn",
        ],
        default="all",
        help="Which phase to run (default: all)",
    )
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    if args.phase in ("all", "all-multiturn", "baseline"):
        phase_baseline()

    if args.phase in ("all", "finetune"):
        phase_finetune()

    if args.phase in ("all-multiturn", "finetune-multiturn"):
        phase_finetune_multiturn()

    if args.phase in ("all", "evaluate"):
        phase_evaluate()

    if args.phase in ("all-multiturn", "evaluate-multiturn"):
        phase_evaluate_multiturn()

    if args.phase in ("all", "all-multiturn", "compare"):
        phase_compare()


if __name__ == "__main__":
    main()
