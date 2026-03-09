"""
Training script for function calling fine-tuning with QLoRA.

Usage:
    python src/train.py --model_id Qwen/Qwen2.5-3B-Instruct --train_samples 10000
"""

import argparse
import torch
import random
import numpy as np
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_utils import load_and_parse_dataset, prepare_datasets, tokenize_dataset


# LoRA target modules per model family
LORA_TARGETS = {
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
}


def get_lora_targets(model_id):
    """Auto-detect LoRA target modules based on model name."""
    model_lower = model_id.lower()
    if "phi" in model_lower:
        return LORA_TARGETS["phi"]
    elif "llama" in model_lower:
        return LORA_TARGETS["llama"]
    else:
        return LORA_TARGETS["qwen"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)
    
    # Auto-detect precision
    use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8 if torch.cuda.is_available() else False
    use_fp16 = not use_bf16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Precision: {'bf16' if use_bf16 else 'fp16'}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Load and prepare data
    parsed_data = load_and_parse_dataset()
    train_dataset, _, _ = prepare_datasets(
        parsed_data, tokenizer,
        max_train=args.train_samples,
        max_test=args.test_samples,
        seed=args.seed,
    )
    train_dataset_tok = tokenize_dataset(train_dataset, tokenizer, max_length=args.max_seq_length)
    
    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\nLoading {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_targets = get_lora_targets(args.model_id)
    print(f"LoRA targets: {lora_targets}")
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # Custom data collator
    def custom_collator(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(x["input_ids"]) + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tok,
        data_collator=custom_collator,
    )

    print(f"\nStarting training...")
    result = trainer.train()
    
    print(f"\nTraining complete!")
    print(f"  Time: {result.metrics['train_runtime']:.0f}s")
    print(f"  Final loss: {result.metrics['train_loss']:.4f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Model saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLMs for function calling")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./output/qwen25_3b")
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
