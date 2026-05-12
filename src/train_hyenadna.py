"""Fine-tune HyenaDNA (LongSafari/hyenadna-small-32k-seqlen-hf) with LoRA on a
GUE TFBS subset.

Architecture differences from DNABERT-2 worth noting:
  - Single-nucleotide tokenization (A/C/G/T/N -> one token each). A 500 bp
    sequence becomes ~510 tokens, vs ~80-120 BPE tokens for DNABERT-2.
  - Hyena long-convolutions instead of attention; no triton dependency on
    the HF-compatible build.
  - The HF auto_map exposes only AutoModel and AutoModelForCausalLM, not
    AutoModelForSequenceClassification. We add a small classification head
    on top of AutoModel ourselves (same trick avoids the DNABERT-2 trap).

Run from project root:
    python -m src.train_hyenadna --subset human_tf_0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Quiet the fork-after-tokenizer warning before HF imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from src.data import load_gue_subset, tokenize_dataset

MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"


# --------------------------------------------------------------------------- #
# Classifier wrapping AutoModel (HyenaDNA's auto_map doesn't expose SeqCls).  #
# --------------------------------------------------------------------------- #
class HyenaDNAForSequenceClassification(nn.Module):
    """Mean-pooled (mask-aware) classification head over HyenaDNA hidden states."""

    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config.num_labels = num_labels
        self.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden = getattr(self.config, "hidden_size", None) or getattr(self.config, "d_model")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        # HyenaDNA uses long convolutions, not attention, so its forward()
        # doesn't accept attention_mask. We still use the mask for mask-aware
        # pooling on the output below.
        outputs = self.backbone(input_ids=input_ids)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        # Mask-aware mean pool (HyenaDNA outputs per-token hidden states)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


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
    """HyenaDNA layers expose `in_proj`/`out_proj` in the mixer and `fc1`/`fc2`
    in the MLP. We adapt all four. The classifier head stays full-rank via
    `modules_to_save` so it can learn from scratch."""
    leaf = {n.split(".")[-1] for n, _ in model.named_modules()}
    cands = ["in_proj", "out_proj", "fc1", "fc2"]
    found = [c for c in cands if c in leaf]
    if not found:
        raise RuntimeError(
            "No HyenaDNA Linear targets found. Sample leaf names: "
            f"{sorted(leaf)[:40]}"
        )
    return found


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="human_tf_0")
    p.add_argument("--output_dir", default=None,
                   help="Defaults to outputs/hyenadna_lora_<subset>")
    p.add_argument("--max_length", type=int, default=512,
                   help="One token per nucleotide; 500 bp GUE seqs fit in 512.")
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

    output_dir = Path(args.output_dir or f"outputs/hyenadna_lora_{args.subset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer + data
    print(f"[data] Loading {args.subset} + HyenaDNA tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # HyenaDNA's CharacterTokenizer occasionally ships without a pad token set
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        print(f"[data] set pad_token = {tokenizer.pad_token!r}")

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

    # ---- Model
    print("[model] Loading HyenaDNA backbone + custom classification head")
    model = HyenaDNAForSequenceClassification(MODEL_NAME, num_labels=2)

    # ---- LoRA
    target_modules = find_lora_targets(model)
    print(f"[lora] target_modules = {target_modules}")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=["classifier"],   # train the head fully, no LoRA
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