"""Dataset preparation for QLoRA fine-tuning.

Combines PAIR conversation logs with HarmBench, AdvBench, and Garak datasets
into a unified training format. Data integrity is preserved - refusals
remain as refusals, successful attacks remain as-is.
"""

import json
import logging
import random
from pathlib import Path

import yaml

from crucible.data import load_harmbench, load_advbench, load_garak

logger = logging.getLogger(__name__)

# ChatML format used by dolphin models
CHATML_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""


def prepare_dataset(config_path: str = "configs/finetune.yaml") -> dict:
    """Prepare the combined dataset for fine-tuning.

    Loads data from all configured sources, formats into ChatML,
    and splits into train/validation sets.

    Args:
        config_path: Path to the fine-tuning config file.

    Returns:
        Dict with 'train' and 'validation' keys, each containing
        a list of formatted conversation strings.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    all_examples = []

    for source in data_config["sources"]:
        source_type = source["type"]
        weight = source.get("weight", 1.0)

        if source_type == "pair_logs":
            examples = _load_pair_logs(source.get("path", "results/pair_logs/baseline"))
        elif source_type == "harmbench":
            examples = _load_harmbench_training_data()
        elif source_type == "advbench":
            examples = _load_advbench_training_data()
        elif source_type == "garak":
            examples = _load_garak_training_data()
        else:
            logger.warning(f"Unknown data source type: {source_type}")
            continue

        logger.info(f"Loaded {len(examples)} examples from {source_type}")

        # Apply weight by sampling
        if weight < 1.0:
            n = int(len(examples) * weight)
            examples = random.sample(examples, min(n, len(examples)))
            logger.info(f"  Sampled down to {len(examples)} (weight={weight})")

        all_examples.extend(examples)

    random.shuffle(all_examples)

    # Format into ChatML
    formatted = [_format_chatml(ex) for ex in all_examples]
    formatted = [f for f in formatted if f is not None]

    # Train/val split
    val_ratio = data_config.get("val_split", 0.1)
    val_size = int(len(formatted) * val_ratio)
    val_set = formatted[:val_size]
    train_set = formatted[val_size:]

    # Save to disk
    output_dir = Path("data/finetune")
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_jsonl(train_set, output_dir / "train.jsonl")
    _save_jsonl(val_set, output_dir / "val.jsonl")

    logger.info(
        f"Dataset prepared: {len(train_set)} train, {len(val_set)} val "
        f"examples saved to {output_dir}"
    )

    return {
        "train": train_set,
        "validation": val_set,
        "train_path": str(output_dir / "train.jsonl"),
        "val_path": str(output_dir / "val.jsonl"),
    }


def _load_pair_logs(log_dir: str) -> list[dict]:
    """Load PAIR conversation logs as training examples.

    Each PAIR conversation turn becomes a training example where:
    - system: PAIR attacker system prompt
    - user: behavior description + previous target response + judge feedback
    - assistant: attacker's improvement analysis + new prompt
    """
    examples = []
    log_path = Path(log_dir)

    if not log_path.exists():
        logger.warning(f"PAIR log directory not found: {log_dir}")
        return examples

    for jsonl_file in log_path.glob("*.jsonl"):
        turns = []
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    turns.append(json.loads(line))

        for turn in turns:
            improvement = turn.get("attacker_improvement", "")
            prompt = turn.get("attacker_prompt", "")

            if not prompt:
                continue

            behavior = turn.get("behavior", "")
            target_resp = turn.get("target_response", "")
            iteration = turn.get("iteration", 1)

            # Build user context
            if iteration == 1:
                user_msg = (
                    f"BEHAVIOR: {behavior}\n\n"
                    "Generate your first adversarial prompt to elicit this "
                    "behavior from the target AI."
                )
            else:
                score = turn.get("harmbench_score", 0)
                user_msg = (
                    f"TARGET RESPONSE:\n{target_resp}\n\n"
                    f"JUDGE SCORE: {score:.1f}\n\n"
                    "Analyze and generate an improved prompt."
                )

            # Assistant response (the attacker's output)
            assistant_msg = json.dumps({
                "improvement": improvement,
                "prompt": prompt,
            })

            examples.append({
                "system": "You are a red-teaming assistant.",
                "user": user_msg,
                "assistant": assistant_msg,
                "source": "pair_logs",
            })

    return examples


def _load_harmbench_training_data() -> list[dict]:
    """Convert HarmBench behaviors into attacker training examples."""
    behaviors = load_harmbench()
    examples = []

    for b in behaviors:
        examples.append({
            "system": "You are a red-teaming assistant.",
            "user": (
                f"BEHAVIOR: {b['behavior']}\n\n"
                "Generate an adversarial prompt to elicit this behavior."
            ),
            "assistant": json.dumps({
                "improvement": f"Targeting behavior in category: {b['category']}",
                "prompt": b["behavior"],
            }),
            "source": "harmbench",
        })

    return examples


def _load_advbench_training_data() -> list[dict]:
    """Convert AdvBench behaviors into attacker training examples."""
    behaviors = load_advbench()
    examples = []

    for b in behaviors:
        examples.append({
            "system": "You are a red-teaming assistant.",
            "user": (
                f"BEHAVIOR: {b['behavior']}\n\n"
                "Generate an adversarial prompt to elicit this behavior."
            ),
            "assistant": json.dumps({
                "improvement": "Initial adversarial attempt",
                "prompt": b["behavior"],
            }),
            "source": "advbench",
        })

    return examples


def _load_garak_training_data() -> list[dict]:
    """Convert Garak probes into attacker training examples."""
    probes = load_garak()
    examples = []

    for p in probes:
        examples.append({
            "system": "You are a red-teaming assistant.",
            "user": (
                f"BEHAVIOR: {p['behavior']}\n\n"
                "Generate an adversarial prompt to elicit this behavior."
            ),
            "assistant": json.dumps({
                "improvement": f"Using probe strategy from {p.get('category', 'garak')}",
                "prompt": p["behavior"],
            }),
            "source": "garak",
        })

    return examples


def _format_chatml(example: dict) -> str | None:
    """Format an example into ChatML template string."""
    try:
        return CHATML_TEMPLATE.format(
            system=example["system"],
            user=example["user"],
            assistant=example["assistant"],
        )
    except (KeyError, TypeError) as e:
        logger.debug(f"Skipping malformed example: {e}")
        return None


def _save_jsonl(examples: list[str], path: Path):
    """Save formatted examples as JSONL with 'text' field."""
    with open(path, "w") as f:
        for text in examples:
            f.write(json.dumps({"text": text}) + "\n")
