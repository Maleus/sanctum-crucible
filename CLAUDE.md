# Sanctum Crucible

## Project Overview
Sanctum Crucible is an LLM red-team research framework for iteratively training and evaluating adversarial attacker models using the PAIR (Prompt Automatic Iterative Refinement) methodology. It is part of the larger Sanctum application ecosystem.

## Research Goal
Fine-tune an attacker model on PAIR-style attack/response data combined with established safety benchmark datasets (HarmBench, AdvBench, Garak), then measure whether fine-tuning produces a more effective red-team model compared to the baseline.

## Experiment Pipeline
1. **Baseline PAIR attack**: 8x7b attacker generates adversarial prompts against 8x22b target
2. **Data collection**: Gather PAIR conversation logs + HarmBench + AdvBench + Garak datasets
3. **Fine-tune**: QLoRA fine-tune the 8x7b attacker on collected data (preserve data integrity - refusals stay as refusals)
4. **Re-evaluate**: Run the fine-tuned attacker against the same target
5. **Compare**: Measure delta across all metrics

## Models
- **Attacker (fine-tune this)**: `cognitivecomputations/dolphin-2.5-mixtral-8x7b` (~46.7B params MoE, ~26GB 4-bit)
- **Target**: `cognitivecomputations/dolphin-mixtral-8x22b` (~141B params MoE, ~80-85GB 4-bit)
- **Primary Evaluator**: HarmBench classifier (fine-tuned Llama-2-13B, ~8GB 4-bit)
- **Secondary Evaluator**: Llama Guard 3 8B (~5GB 4-bit)

## Infrastructure
- **Platform**: RunPod
- **GPU**: 4x A100 80GB SXM (320GB total VRAM)
- **Volume**: 500GB+
- **CPU RAM**: 128GB+
- **Serving**: vLLM for inference, HuggingFace PEFT for fine-tuning
- **Code hosting**: GitHub
- **Model hosting**: HuggingFace / Ollama

## Key Metrics
### Primary
- **ASR (Attack Success Rate)**: % of behaviors where target produces harmful completion
- **ASR per category**: Breakdown across HarmBench semantic categories
- **Query efficiency**: Mean/median PAIR iterations to successful jailbreak

### Secondary
- **ASR@k**: Success rate within k queries (k=5,10,20)
- **Transfer ASR**: Effectiveness against models not seen during training
- **Judge agreement rate**: Cross-validation between HarmBench classifier and Llama Guard
- **Attack diversity**: Self-BLEU / embedding diversity of successful prompts
- **Refusal calibration**: Does attacker predict refusals accurately

## Data Integrity Policy
All benchmark data must remain unaltered. Refusals stay as refusals. No filtering, no editing. The model must learn the full distribution as-is.

## Project Structure
```
sanctum-crucible/
├── CLAUDE.md              # This file - project context
├── configs/
│   ├── models.yaml        # Model paths, quantization, serving config
│   ├── pair.yaml          # PAIR attack hyperparameters
│   └── finetune.yaml      # QLoRA training hyperparameters
├── src/crucible/
│   ├── pair/              # PAIR attack loop implementation
│   ├── data/              # Dataset loaders (HarmBench, AdvBench, Garak)
│   ├── finetune/          # QLoRA fine-tuning pipeline
│   ├── evaluate/          # Evaluator judges + metrics
│   └── utils/             # vLLM serving, logging utilities
├── scripts/               # RunPod setup and pipeline execution scripts
├── data/                  # Downloaded datasets (gitignored)
├── results/               # Evaluation outputs (gitignored)
└── notebooks/             # Analysis and visualization
```

## Tech Stack
- Python 3.10+
- vLLM (model serving with tensor parallelism)
- HuggingFace transformers + PEFT + bitsandbytes (QLoRA)
- Garak (LLM vulnerability scanning)
- PyTorch 2.x + CUDA 12.x

## Design Decisions
- Models are parameterized in configs so the same pipeline can be reused with different model pairs
- vLLM is used over Ollama for the PAIR loop because it handles batch inference and is faster for automated pipelines
- Dual evaluator setup (HarmBench + Llama Guard) for robust scoring with inter-rater agreement
- All PAIR conversation logs are saved as JSONL for reproducibility and future training data
