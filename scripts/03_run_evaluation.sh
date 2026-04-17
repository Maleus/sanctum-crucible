#!/bin/bash
# Phase 3: Re-run PAIR with fine-tuned attacker and compare results
# Uses the QLoRA-finetuned 8x7b against the same 8x22b target

set -euo pipefail

echo "============================================"
echo "  Phase 3: Fine-tuned Evaluation & Comparison"
echo "============================================"

# Update attacker model path in config for the fine-tuned version
echo "[1/4] Configuring fine-tuned attacker..."
python -c "
import yaml

with open('configs/models.yaml') as f:
    config = yaml.safe_load(f)

# Point attacker to the fine-tuned checkpoint with LoRA merged
config['attacker']['hf_repo'] = 'checkpoints/attacker-finetuned'
config['attacker']['quantization'] = 'none'  # Already quantized via QLoRA

with open('configs/models_finetuned.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('Created configs/models_finetuned.yaml')
"

# Start servers with fine-tuned attacker
echo "[2/4] Starting model servers (fine-tuned attacker)..."
python -c "
from crucible.utils.serving import start_all_servers
servers = start_all_servers('configs/models_finetuned.yaml')
print('All servers ready.')
import json
status = {name: {'port': s.port, 'pid': s.process.pid} for name, s in servers.items()}
with open('logs/server_pids_finetuned.json', 'w') as f:
    json.dump(status, f)
"

echo "[3/4] Running PAIR attack with fine-tuned attacker..."
python -c "
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/finetuned_run.log'),
    ]
)

from crucible.data import load_harmbench
from crucible.pair import PAIROrchestrator

behaviors = load_harmbench()

orchestrator = PAIROrchestrator(config_dir='configs')
# Override config dir for fine-tuned models
orchestrator.models_config = __import__('yaml').safe_load(
    open('configs/models_finetuned.yaml')
)
orchestrator._init_models()

summary = orchestrator.run(behaviors, run_name='finetuned')

import json
print()
print('Fine-tuned Results:')
print(json.dumps(summary, indent=2))
"

# Stop servers
python -c "
import json, signal, os
with open('logs/server_pids_finetuned.json') as f:
    pids = json.load(f)
for name, info in pids.items():
    try:
        os.kill(info['pid'], signal.SIGTERM)
    except ProcessLookupError:
        pass
"

echo "[4/4] Generating comparison report..."
python -c "
from crucible.evaluate.metrics import compare_runs, print_comparison
import json

comparison = compare_runs(
    baseline_dir='results/pair_logs/baseline',
    finetuned_dir='results/pair_logs/finetuned',
)

print_comparison(comparison)

# Save full comparison
with open('results/comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print()
print('Full comparison saved to results/comparison.json')
"

echo ""
echo "============================================"
echo "  Evaluation complete!"
echo "  Results: results/comparison.json"
echo "============================================"
