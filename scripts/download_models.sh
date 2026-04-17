#!/bin/bash
# Download all models needed for the experiment
# Run this after setup_runpod.sh

set -euo pipefail

echo "============================================"
echo "  Sanctum Crucible - Model Download"
echo "============================================"
echo ""
echo "This will download ~200GB+ of model weights."
echo "Ensure you have sufficient disk space."
echo ""

# Check for HuggingFace token (needed for Llama Guard)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Llama Guard 3 requires authentication."
    echo "Set it with: export HF_TOKEN=your_token_here"
    echo "Or run: huggingface-cli login"
    echo ""
fi

download_model() {
    local name=$1
    local repo=$2
    echo "[Downloading] $name ($repo)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '$repo',
    local_dir='models/$name',
    local_dir_use_symlinks=False,
)
print('  Done: $name')
" 2>&1 | tail -1
}

# Download models in sequence (disk I/O bound anyway)
echo "[1/4] Attacker: dolphin-2.5-mixtral-8x7b"
download_model "attacker-8x7b" "cognitivecomputations/dolphin-2.5-mixtral-8x7b"

echo "[2/4] Target: dolphin-2.5-mixtral-8x22b"
download_model "target-8x22b" "cognitivecomputations/dolphin-2.5-mixtral-8x22b"

echo "[3/4] Judge: HarmBench classifier"
download_model "harmbench-cls" "cais/HarmBench-Llama-2-13b-cls"

echo "[4/4] Judge: Llama Guard 3 8B"
download_model "llamaguard-3-8b" "meta-llama/Llama-Guard-3-8B"

echo ""
echo "============================================"
echo "  All models downloaded!"
echo "  Next: bash scripts/01_run_baseline.sh"
echo "============================================"

# Show disk usage
echo ""
echo "Disk usage:"
du -sh models/*
