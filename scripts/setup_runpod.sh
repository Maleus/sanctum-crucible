#!/bin/bash
# RunPod environment setup for Sanctum Crucible
# Run this once after spinning up the pod

set -euo pipefail

echo "============================================"
echo "  Sanctum Crucible - RunPod Setup"
echo "============================================"

# Update system
echo "[1/6] Updating system packages..."
apt-get update -qq && apt-get install -y -qq git htop nvtop tmux > /dev/null 2>&1

# Install Python dependencies
echo "[2/6] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Install vLLM (separate since it can be finicky with CUDA versions)
echo "[3/6] Installing vLLM..."
pip install -q vllm

# Install Garak
echo "[4/6] Installing Garak..."
pip install -q garak

# Create directories
echo "[5/6] Creating directory structure..."
mkdir -p data/{harmbench,advbench,garak,finetune}
mkdir -p results/pair_logs/{baseline,finetuned}
mkdir -p checkpoints
mkdir -p logs
mkdir -p models

# Verify GPU access
echo "[6/6] Verifying GPU access..."
python -c "
import torch
gpu_count = torch.cuda.device_count()
print(f'  GPUs detected: {gpu_count}')
for i in range(gpu_count):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Next: bash scripts/download_models.sh"
echo "============================================"
