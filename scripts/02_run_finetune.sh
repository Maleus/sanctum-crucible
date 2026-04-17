#!/bin/bash
# Phase 2: Prepare data and fine-tune the attacker model
# Combines PAIR logs + HarmBench + AdvBench + Garak -> QLoRA fine-tune

set -euo pipefail

echo "============================================"
echo "  Phase 2: Data Preparation & Fine-tuning"
echo "============================================"

echo "[1/2] Preparing training dataset..."
python -c "
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/data_prep.log'),
    ]
)

from crucible.finetune.prepare import prepare_dataset

result = prepare_dataset('configs/finetune.yaml')
print(f'Training examples: {len(result[\"train\"])}')
print(f'Validation examples: {len(result[\"validation\"])}')
print(f'Train file: {result[\"train_path\"]}')
print(f'Val file: {result[\"val_path\"]}')
"

echo "[2/2] Running QLoRA fine-tuning..."
python -c "
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/finetune.log'),
    ]
)

from crucible.finetune.train import run_finetune

output_dir = run_finetune('configs/finetune.yaml')
print(f'Fine-tuned model saved to: {output_dir}')
"

echo ""
echo "============================================"
echo "  Fine-tuning complete!"
echo "  Checkpoint: checkpoints/attacker-finetuned/"
echo "  Next: bash scripts/03_run_evaluation.sh"
echo "============================================"
