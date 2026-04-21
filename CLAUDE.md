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

---

## Experiment: Multi-turn Dataset Transformation

### Background
The first fine-tune round (single-turn) produced a model that behaves as a stateless prompt optimizer: it over-prioritizes the most recent input, treats instructions as content to rewrite, and loses state across turns. Root cause: `prepare.py` flattened each PAIR iteration into an independent `(user, assistant)` pair, stripping all cross-turn context.

### What changed
`prepare_multiturn.py` transforms the same PAIR logs into structured multi-turn conversations:
- Each behavior's full PAIR iteration sequence becomes ONE multi-turn conversation
- Every turn has explicit `INSTRUCTION:` / `TARGET PROMPT:` delimiters
- The model's response is the next adversarial prompt verbatim (no JSON, no analysis)
- All original adversarial content is preserved exactly — only structure is added

### Pipeline phases
```
python run_pipeline.py --phase all-multiturn
```
This runs: baseline → multiturn finetune → multiturn evaluate → compare

Individual phases:
```
python run_pipeline.py --phase baseline              # Phase 1: PAIR attack with base model
python run_pipeline.py --phase finetune-multiturn     # Phase 2b: multi-turn data prep + QLoRA
python run_pipeline.py --phase evaluate-multiturn     # Phase 3b: PAIR attack with multi-turn model
python run_pipeline.py --phase compare                # Phase 4: compare all available results
```

### Configs
- `configs/finetune.yaml` — original single-turn training config
- `configs/finetune_multiturn.yaml` — multi-turn variant (smaller batch, longer seq_len, separate checkpoint dir)

### Checkpoints
- `checkpoints/attacker-finetuned/` — single-turn fine-tuned model
- `checkpoints/attacker-finetuned-multiturn/` — multi-turn fine-tuned model

### What to measure
Compare baseline, single-turn, and multi-turn models across:
1. ASR (overall and per-category)
2. ASR@k (k=5,10,20)
3. Mean iterations to success
4. Judge agreement rate
5. Attack diversity (Self-BLEU)

---

## RunPod Experiment Instructions

These instructions are for the Claude instance executing the experiment on RunPod. Follow them sequentially. Do not skip validation steps.

### Prerequisites
- RunPod pod: 4x A100 80GB SXM, 500GB+ volume, 128GB+ RAM
- HuggingFace token set: `export HF_TOKEN=<token>` (needed for Llama Guard)
- This repo cloned to the pod

### Phase 0: Environment Setup
```bash
cd /workspace/sanctum-crucible    # or wherever the repo is cloned
bash scripts/setup_runpod.sh
bash scripts/download_models.sh
```

**Validate before proceeding:**
- `nvidia-smi` shows 4x A100 GPUs
- `python -c "import torch; print(torch.cuda.device_count())"` prints `4`
- `ls models/` shows all 4 model directories
- `pip install -e .` completes without error (installs crucible package)
- `python -c "from crucible.finetune.prepare_multiturn import generate_sample_batch; print('OK')"` prints `OK`

### Phase 1: Baseline PAIR Attack
```bash
python run_pipeline.py --phase baseline
```

**Runtime estimate:** 4-12 hours depending on behavior count and max_iterations.

**Validate before proceeding:**
- `results/pair_logs/baseline/summary.json` exists and contains `overall_asr`
- At least 50 `.jsonl` files in `results/pair_logs/baseline/`
- Spot-check 3-5 JSONL files: each should have multiple iterations with `attacker_prompt`, `target_response`, `harmbench_score` fields
- Log file `logs/baseline_run.log` shows no ERROR-level entries (warnings are OK)

**If something fails:**
- vLLM OOM: reduce `gpu_memory_utilization` in `configs/models.yaml` from 0.85 to 0.80
- Model download timeout: re-run `bash scripts/download_models.sh` — it's idempotent
- Server health check timeout: increase timeout in `src/crucible/utils/serving.py` or check `logs/vllm_*.log`

### Phase 2a: Single-turn Fine-tune (original method)
```bash
python run_pipeline.py --phase finetune
```

**Runtime estimate:** 1-3 hours for 3 epochs.

**Validate:**
- `data/finetune/train.jsonl` and `val.jsonl` exist and are non-empty
- `checkpoints/attacker-finetuned/` contains `adapter_model.safetensors` and `tokenizer.json`
- Training loss decreasing in `logs/finetune.log`

### Phase 2b: Multi-turn Fine-tune (new method)
```bash
python run_pipeline.py --phase finetune-multiturn
```

**Runtime estimate:** 1-4 hours (longer sequences = slower per step, but fewer total examples).

**Validate:**
- `data/finetune/multiturn_train.jsonl` and `multiturn_val.jsonl` exist
- Spot-check the JSONL: each entry's `text` field should contain multiple `<|im_start|>user` / `<|im_start|>assistant` turns (not just one pair)
- `checkpoints/attacker-finetuned-multiturn/` contains `adapter_model.safetensors`
- Training loss curve in logs — final loss should be in a similar range as Phase 2a

**If the multi-turn dataset is too small (<50 conversations):**
This means most baseline behaviors only ran 1 iteration (immediate success or single failure). Options:
1. Re-run baseline with `early_stopping: false` in `configs/pair.yaml` to force all 20 iterations
2. Lower `min_turns` from 2 to 1 in the `prepare_multiturn_dataset` call (but this defeats the purpose)
3. Increase `max_behaviors` or use AdvBench/Garak behaviors too for more data

### Phase 3a: Evaluate Single-turn Model
```bash
python run_pipeline.py --phase evaluate
```

**Validate:**
- `results/pair_logs/finetuned/summary.json` exists
- Same structure as baseline summary

### Phase 3b: Evaluate Multi-turn Model
```bash
python run_pipeline.py --phase evaluate-multiturn
```

**Validate:**
- `results/pair_logs/finetuned-multiturn/summary.json` exists

### Phase 4: Compare All Results
```bash
python run_pipeline.py --phase compare
```

**Output:** `results/comparison.json` with `single_turn` and `multi_turn` comparison blocks, each containing baseline vs. fine-tuned deltas.

**What to look for in results:**
- Multi-turn model should show improved ASR@5 and ASR@10 (better early convergence)
- Multi-turn model should show lower mean iterations to success (more efficient refinement)
- If multi-turn ASR is lower than single-turn but iterations-to-success is lower, the model is more efficient but may need more training data or epochs
- Check judge agreement — if it drops, the model may be finding adversarial prompts that exploit judge disagreement

### Saving Results
After all phases complete:
```bash
# Archive results
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz results/ logs/ data/finetune/

# Push any code changes (if you fixed bugs during the run)
git add -A && git commit -m "Experiment results and runtime fixes" && git push
```

### Troubleshooting Quick Reference
| Symptom | Likely cause | Fix |
|---|---|---|
| vLLM OOM on startup | Too many models loaded | Start models sequentially or reduce `gpu_memory_utilization` |
| Training OOM | Sequence too long | Reduce `max_seq_length` in finetune config or reduce batch size |
| Empty multi-turn dataset | Baseline had few multi-iteration behaviors | Set `early_stopping: false` and re-run baseline |
| Judge scores all 0.0 | HarmBench classifier not loaded correctly | Check `logs/vllm_harmbench*.log` for errors |
| Attacker outputs raw text instead of JSON | Expected for multi-turn model (it's trained without JSON wrapping) | This is correct behavior for the multi-turn variant |
| `ModuleNotFoundError: crucible` | Package not installed | Run `pip install -e .` from repo root |
