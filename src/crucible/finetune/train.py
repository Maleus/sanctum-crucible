"""QLoRA fine-tuning script for the attacker model.

Fine-tunes the attacker (dolphin-mixtral-8x7b) using QLoRA on the
prepared dataset of PAIR logs + benchmark data.
"""

import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)


def run_finetune(config_path: str = "configs/finetune.yaml"):
    """Run QLoRA fine-tuning on the attacker model.

    Args:
        config_path: Path to the fine-tuning config file.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    base_model = model_cfg["base_model"]
    output_dir = model_cfg["output_dir"]

    logger.info(f"Starting QLoRA fine-tune of {base_model}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    # Load model
    logger.info("Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # Load dataset
    data_dir = Path("data/finetune")
    train_path = str(data_dir / "train.jsonl")
    val_path = str(data_dir / "val.jsonl")

    if not Path(train_path).exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run `python -m crucible.finetune.prepare` first."
        )

    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
    )

    logger.info(
        f"Dataset: {len(dataset['train'])} train, "
        f"{len(dataset['validation'])} validation examples"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        max_seq_length=train_cfg["max_seq_length"],
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Fine-tuning complete.")
    return output_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_finetune()
