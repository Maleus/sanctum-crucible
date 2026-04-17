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


def phase_evaluate():
    """Phase 3: Run fine-tuned PAIR attack."""
    import yaml
    from crucible.data import load_harmbench
    from crucible.pair import PAIROrchestrator
    from crucible.utils.serving import start_all_servers, stop_all_servers

    logger.info("=" * 60)
    logger.info("PHASE 3: Fine-tuned PAIR Attack")
    logger.info("=" * 60)

    # Create config with fine-tuned attacker
    with open("configs/models.yaml") as f:
        config = yaml.safe_load(f)

    config["attacker"]["hf_repo"] = "checkpoints/attacker-finetuned"
    config["attacker"]["quantization"] = "none"

    finetuned_config = "configs/models_finetuned.yaml"
    with open(finetuned_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    servers = start_all_servers(finetuned_config)

    try:
        behaviors = load_harmbench()
        orchestrator = PAIROrchestrator(config_dir="configs")
        orchestrator.models_config = config
        summary = orchestrator.run(behaviors, run_name="finetuned")

        logger.info(f"Fine-tuned ASR: {summary['overall_asr']:.2%}")
        return summary
    finally:
        stop_all_servers(servers)


def phase_compare():
    """Compare baseline and fine-tuned results."""
    from crucible.evaluate.metrics import compare_runs, print_comparison

    logger.info("=" * 60)
    logger.info("COMPARISON: Baseline vs Fine-tuned")
    logger.info("=" * 60)

    comparison = compare_runs(
        baseline_dir="results/pair_logs/baseline",
        finetuned_dir="results/pair_logs/finetuned",
    )

    print_comparison(comparison)

    with open("results/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info("Comparison saved to results/comparison.json")
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Sanctum Crucible Pipeline")
    parser.add_argument(
        "--phase",
        choices=["baseline", "finetune", "evaluate", "compare", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    if args.phase in ("all", "baseline"):
        phase_baseline()

    if args.phase in ("all", "finetune"):
        phase_finetune()

    if args.phase in ("all", "evaluate"):
        phase_evaluate()

    if args.phase in ("all", "compare"):
        phase_compare()


if __name__ == "__main__":
    main()
