"""Fine-tune DNABERT-2 (zhihan1996/DNABERT-2-117M) with LoRA on a GUE subset.

Run from project root:
    python -m src.train_dnabert2 --subset human_tf_0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import load_gue_subset, tokenize_dataset

MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # some models return (logits, hidden_states, ...)
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "mcc": matthews_corrcoef(labels, preds),
    }


def find_lora_targets(model) -> list[str]:
    """DNABERT-2's custom BERT uses a fused QKV module called `Wqkv`.
    Standard BERT uses separate `query`/`key`/`value`. Try fused first."""
    leaf_names = {n.split(".")[-1] for n, _ in model.named_modules()}
    if "Wqkv" in leaf_names:
        # Fused QKV; also adapt the FFN output projection ("dense" inside BertOutput).
        return ["Wqkv", "dense"]
    if {"query", "value"} <= leaf_names:
        return ["query", "value"]
    raise RuntimeError(
        "Could not auto-detect LoRA target modules. "
        f"Sample of leaf module names: {sorted(leaf_names)[:30]}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="human_tf_0",
                   help="GUE subset name, e.g. human_tf_0 ... human_tf_4, mouse_0 ...")
    p.add_argument("--output_dir", default=None,
                   help="Defaults to outputs/dnabert2_lora_<subset>")
    p.add_argument("--max_length", type=int, default=128,
                   help="DNABERT-2 BPE compresses ~3-6x; 128 covers ~500 bp easily.")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run on 256 train / 64 eval to verify the pipeline.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir or f"outputs/dnabert2_lora_{args.subset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer + data
    print(f"[data] Loading {args.subset} from HF + DNABERT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ds = load_gue_subset(args.subset)
    tokenized = tokenize_dataset(ds, tokenizer, max_length=args.max_length)
    print(f"[data] Splits: { {k: len(v) for k, v in tokenized.items()} }")

    # GUE on HF may name its eval split either 'validation' or 'dev'.
    eval_split = "validation" if "validation" in tokenized else "dev"
    test_split = "test" if "test" in tokenized else None

    if args.smoke:
        tokenized["train"] = tokenized["train"].select(range(min(256, len(tokenized["train"]))))
        tokenized[eval_split] = tokenized[eval_split].select(range(min(64, len(tokenized[eval_split]))))
        if test_split:
            tokenized[test_split] = tokenized[test_split].select(range(min(64, len(tokenized[test_split]))))
        print("[data] SMOKE mode: subsetted to tiny splits")

    # ---- Base model
    print("[model] Loading DNABERT-2 base + classification head")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, trust_remote_code=True
    )

    # ---- LoRA wrap
    target_modules = find_lora_targets(model)
    print(f"[lora] target_modules = {target_modules}")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---- Trainer
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    print(f"[train] cuda={torch.cuda.is_available()} bf16={use_bf16} fp16={use_fp16}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        seed=args.seed,
        save_total_limit=2,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("[train] starting")
    trainer.train()

    if test_split:
        print(f"[eval] running on {test_split} split")
        test_metrics = trainer.evaluate(tokenized[test_split], metric_key_prefix="test")
        print(f"[eval] test metrics: {test_metrics}")

    print(f"[save] writing LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
