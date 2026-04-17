#!/bin/bash
# Phase 1: Run baseline PAIR attack (pre-fine-tune)
# Attacker (8x7b) attacks Target (8x22b) and logs all conversations

set -euo pipefail

echo "============================================"
echo "  Phase 1: Baseline PAIR Attack"
echo "============================================"

# Start vLLM servers
echo "[1/3] Starting model servers..."
python -c "
from crucible.utils.serving import start_all_servers
servers = start_all_servers('configs/models.yaml')
print('All servers ready.')
# Keep reference for the pipeline
import json
status = {name: {'port': s.port, 'pid': s.process.pid} for name, s in servers.items()}
with open('logs/server_pids.json', 'w') as f:
    json.dump(status, f)
"

echo "[2/3] Running PAIR attack loop..."
python -c "
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/baseline_run.log'),
    ]
)

from crucible.data import load_harmbench
from crucible.pair import PAIROrchestrator

# Load behaviors
behaviors = load_harmbench()

# Run baseline attack
orchestrator = PAIROrchestrator(config_dir='configs')
summary = orchestrator.run(behaviors, run_name='baseline')

# Print results
import json
print()
print('Baseline Results:')
print(json.dumps(summary, indent=2))
"

echo "[3/3] Stopping servers..."
python -c "
import json, signal, os
with open('logs/server_pids.json') as f:
    pids = json.load(f)
for name, info in pids.items():
    try:
        os.kill(info['pid'], signal.SIGTERM)
        print(f'Stopped {name} (PID {info[\"pid\"]})')
    except ProcessLookupError:
        print(f'{name} already stopped')
"

echo ""
echo "============================================"
echo "  Baseline complete!"
echo "  Results: results/pair_logs/baseline/"
echo "  Next: bash scripts/02_run_finetune.sh"
echo "============================================"
