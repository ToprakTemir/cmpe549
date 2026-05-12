"""Fine-tune DNABERT-2 (zhihan1996/DNABERT-2-117M) with LoRA on a GUE subset.

Run from project root:
    python -m src.train_dnabert2 --subset human_tf_0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Silence the fork-after-tokenizer warning and disable HF download chatter
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from src.data import load_gue_subset, tokenize_dataset

MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
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
        return ["Wqkv", "dense"]
    if {"query", "value"} <= leaf_names:
        return ["query", "value"]
    raise RuntimeError(
        "Could not auto-detect LoRA target modules. "
        f"Sample of leaf module names: {sorted(leaf_names)[:30]}"
    )


def _patch_dnabert2_disable_triton() -> None:
    """DNABERT-2's `flash_attn_triton.py` uses `tl.dot(q, k, trans_b=True)`, an
    API that was removed in Triton 2.2+. Mac (no Triton) falls back to pytorch
    attention automatically; Colab (Triton 3.x) tries to compile and fails.

    The model already has a pytorch-attention fallback that activates whenever
    the `from .flash_attn_triton import ...` line raises ImportError. We patch
    the cached bert_layers.py to force that path."""
    import glob
    import sys

    bases = [
        os.path.expanduser("~/.cache/huggingface/modules"),
        "/root/.cache/huggingface/modules",
    ]
    candidates: list[str] = []
    for base in bases:
        candidates.extend(glob.glob(
            f"{base}/transformers_modules/zhihan1996/DNABERT-2-117M/**/bert_layers.py",
            recursive=True,
        ))

    OLD = "from .flash_attn_triton import flash_attn_qkvpacked_func"
    NEW = "raise ImportError('Triton flash-attn disabled (DNABERT-2 uses removed trans_b API)')"

    for p in candidates:
        with open(p) as f:
            text = f.read()
        if NEW in text:
            continue  # already patched
        if OLD in text:
            with open(p, "w") as f:
                f.write(text.replace(OLD, NEW))
            print(f"[patch] disabled Triton flash-attn in {p}")

    # Drop any cached import so the next get_class_from_dynamic_module re-reads from disk
    for name in list(sys.modules):
        if "DNABERT-2-117M" in name:
            del sys.modules[name]


def load_dnabert2_classifier(num_labels: int = 2):
    """Load DNABERT-2 with a sequence-classification head using the *custom*
    `bert_layers.BertForSequenceClassification` defined in the model's remote
    code. The model's `auto_map` in config.json doesn't expose this class to
    `AutoModelForSequenceClassification`, so we load it directly."""
    # AutoConfig.from_pretrained triggers the download of all auto_map'd .py
    # files (configuration_bert.py, bert_layers.py, flash_attn_triton.py,
    # bert_padding.py). After they land on disk we patch and *then* load.
    config = AutoConfig.from_pretrained(
        MODEL_NAME, num_labels=num_labels, trust_remote_code=True
    )
    _patch_dnabert2_disable_triton()
    cls = get_class_from_dynamic_module(
        "bert_layers.BertForSequenceClassification",
        pretrained_model_name_or_path=MODEL_NAME,
    )
    return cls.from_pretrained(MODEL_NAME, config=config, trust_remote_code=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="human_tf_0",
                   help="GUE subset name, e.g. human_tf_0 ... human_tf_4, mouse_0 ...")
    p.add_argument("--output_dir", default=None,
                   help="Defaults to outputs/dnabert2_lora_<subset>")
    p.add_argument("--max_length", type=int, default=128)
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
    model = load_dnabert2_classifier(num_labels=2)

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
        # MPS doesn't support multi-worker dataloading cleanly; CUDA is fine with >0
        dataloader_num_workers=0 if not torch.cuda.is_available() else 2,
        dataloader_pin_memory=torch.cuda.is_available(),
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