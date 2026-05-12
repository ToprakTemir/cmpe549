"""Phase 3 - Integrated Gradients attribution for fine-tuned DNABERT-2 and HyenaDNA.

For each model:
  1. Load base + LoRA adapter, merge LoRA into the base for clean inference.
  2. Run inference over the GUE test split; keep label==1 AND pred==1 AND prob>=`min_prob`
     (correctly-classified high-confidence positives — what we want motifs from).
  3. Run Captum's `LayerIntegratedGradients` on the input-id embedding layer,
     target=positive class, n_steps interpolation steps.
  4. Reduce (B, L, D) embedding-attribution to (L,) per-token by summing across D.
  5. Map token-level attributions to per-nucleotide:
       - DNABERT-2: distribute each BPE token's score uniformly across its
         nucleotide span (from the tokenizer's offset_mapping).
       - HyenaDNA: 1-token-per-nucleotide, strip special tokens to align.
  6. Save one .npz per model with the format TF-MoDISco wants:
       one_hot:      (N, L, 4)  uint8   observed sequence
       attributions: (N, L, 4)  float32 IG score on the observed base, 0 elsewhere
       sequences:    (N,)       <U500   raw DNA strings
       labels/preds/probs: (N,) metadata

Run from project root:
    python -m src.attribute --model dnabert2 --adapter outputs/dnabert2_lora_human_tf_0
    python -m src.attribute --model hyenadna --adapter outputs/hyenadna_lora_human_tf_0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from src.data import load_gue_subset
from src.train_dnabert2 import (
    MODEL_NAME as DNABERT2_NAME,
    _patch_dnabert2_disable_triton,
)
from src.train_hyenadna import (
    MODEL_NAME as HYENA_NAME,
    HyenaDNAForSequenceClassification,
)

NUCLEOTIDE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


# --------------------------------------------------------------------------- #
# Sequence encoding helpers                                                   #
# --------------------------------------------------------------------------- #
def one_hot_encode(seq: str) -> np.ndarray:
    oh = np.zeros((len(seq), 4), dtype=np.uint8)
    for i, nt in enumerate(seq.upper()):
        idx = NUCLEOTIDE_INDEX.get(nt)
        if idx is not None:
            oh[i, idx] = 1
    return oh


# --------------------------------------------------------------------------- #
# Model + tokenizer loading                                                   #
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(kind: str, adapter_path: str, device: torch.device):
    """Return (model in eval mode on device, tokenizer, max_length)."""
    if kind == "dnabert2":
        config = AutoConfig.from_pretrained(DNABERT2_NAME, num_labels=2, trust_remote_code=True)
        _patch_dnabert2_disable_triton()
        cls = get_class_from_dynamic_module(
            "bert_layers.BertForSequenceClassification",
            pretrained_model_name_or_path=DNABERT2_NAME,
        )
        base = cls.from_pretrained(DNABERT2_NAME, config=config, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(DNABERT2_NAME, trust_remote_code=True)
        max_length = 128
    elif kind == "hyenadna":
        base = HyenaDNAForSequenceClassification(HYENA_NAME, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(HYENA_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        max_length = 512
    else:
        raise ValueError(f"Unknown model kind: {kind!r}")

    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()  # fold LoRA into base weights
    model.to(device).eval()
    return model, tokenizer, max_length


def find_embedding_layer(model: nn.Module) -> nn.Module:
    """First non-positional nn.Embedding under the model. Robust across both
    architectures (DNABERT-2 has `bert.embeddings.word_embeddings`; HyenaDNA-hf
    has `backbone.backbone.embeddings.word_embeddings` or similar)."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding) and "position" not in name.lower():
            print(f"[ig] embedding layer = {name}  weight shape={tuple(mod.weight.shape)}")
            return mod
    raise RuntimeError("Could not locate a non-positional nn.Embedding in model.")


# --------------------------------------------------------------------------- #
# Inference: find correctly-classified high-confidence positives              #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def select_positives(
    model,
    tokenizer,
    sequences,
    labels,
    device,
    max_length: int,
    min_prob: float,
    batch_size: int = 32,
):
    """Return indices of test rows where label==1 AND pred==1 AND P(pos)>=min_prob,
    plus arrays of all preds/probs for the full set (useful diagnostics)."""
    all_preds = np.zeros(len(sequences), dtype=np.int64)
    all_probs = np.zeros(len(sequences), dtype=np.float32)

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start:start + batch_size]
        enc = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        # Drop attention_mask for callers whose forward doesn't accept it; both
        # of our wrappers do accept it (HyenaDNA's just ignores it internally).
        out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out[0]
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(np.int64)
        all_probs[start:start + len(batch_seqs)] = probs
        all_preds[start:start + len(batch_seqs)] = preds

    labels_arr = np.asarray(labels, dtype=np.int64)
    keep_mask = (labels_arr == 1) & (all_preds == 1) & (all_probs >= min_prob)
    keep_idx = np.where(keep_mask)[0]

    print(f"[select] test set size = {len(sequences)}")
    print(f"[select] positives (label==1)             = {int((labels_arr == 1).sum())}")
    print(f"[select] correctly classified positives   = {int(((labels_arr == 1) & (all_preds == 1)).sum())}")
    print(f"[select] high-conf (prob>={min_prob})     = {len(keep_idx)}")
    return keep_idx, all_preds, all_probs


# --------------------------------------------------------------------------- #
# Token-level -> nucleotide-level mapping                                     #
# --------------------------------------------------------------------------- #
def bpe_to_nucleotide(seq: str, tokenizer, token_attrs: np.ndarray) -> np.ndarray:
    """Distribute per-BPE-token IG attribution uniformly across each token's
    nucleotide span. Special tokens (CLS/SEP) have offset (0,0) and are skipped."""
    enc = tokenizer(seq, add_special_tokens=True, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    nt_attrs = np.zeros(len(seq), dtype=np.float32)
    for tok_idx, (s, e) in enumerate(offsets):
        if e <= s or tok_idx >= len(token_attrs):
            continue
        nt_attrs[s:e] = token_attrs[tok_idx] / (e - s)
    return nt_attrs


def hyenadna_to_nucleotide(seq: str, tokenizer, token_attrs: np.ndarray) -> np.ndarray:
    """HyenaDNA uses one token per nucleotide. Strip special tokens to align 1:1."""
    enc = tokenizer(seq, add_special_tokens=True)
    ids = enc["input_ids"]
    special_ids = set(tokenizer.all_special_ids)
    nt_attrs = np.zeros(len(seq), dtype=np.float32)
    nt_idx = 0
    for tok_idx, tok_id in enumerate(ids):
        if tok_id in special_ids or tok_idx >= len(token_attrs):
            continue
        if nt_idx < len(seq):
            nt_attrs[nt_idx] = token_attrs[tok_idx]
            nt_idx += 1
    return nt_attrs


# --------------------------------------------------------------------------- #
# Integrated Gradients                                                        #
# --------------------------------------------------------------------------- #
def run_ig(
    model,
    tokenizer,
    sequences,
    kind: str,
    device,
    max_length: int,
    n_steps: int,
    internal_batch_size: int,
):
    """Run LayerIntegratedGradients on the embedding layer for each sequence;
    return a list[np.ndarray] of per-nucleotide attributions (one per sequence)."""
    embedding_layer = find_embedding_layer(model)

    def forward_fn(input_ids):
        out = model(input_ids=input_ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits

    lig = LayerIntegratedGradients(forward_fn, embedding_layer)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    mapping_fn = bpe_to_nucleotide if kind == "dnabert2" else hyenadna_to_nucleotide
    nt_attrs_list = []
    deltas = []

    for seq in tqdm(sequences, desc=f"IG ({kind})"):
        enc = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        baseline_ids = torch.full_like(input_ids, pad_id)

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            target=1,  # positive class
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
        )
        # attributions shape: (1, L, D) -> sum across embedding dim -> (L,)
        token_attrs = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        nt_attrs = mapping_fn(seq, tokenizer, token_attrs)
        nt_attrs_list.append(nt_attrs)
        deltas.append(float(delta.detach().cpu().item()))

    print(f"[ig] mean |convergence delta| = {np.mean(np.abs(deltas)):.4f}  "
          f"(should be small; <0.1 is healthy)")
    return nt_attrs_list


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["dnabert2", "hyenadna"])
    p.add_argument("--adapter", required=True, help="Path to the LoRA adapter folder")
    p.add_argument("--subset", default="human_tf_0")
    p.add_argument("--output", default=None,
                   help="Defaults to outputs/attributions_<model>_<subset>.npz")
    p.add_argument("--split", default="test", choices=["test", "dev", "validation"])
    p.add_argument("--min_prob", type=float, default=0.7,
                   help="Keep predicted-positive sequences with P(class=1) >= this.")
    p.add_argument("--n_max", type=int, default=300,
                   help="Cap on number of sequences to attribute (per model).")
    p.add_argument("--n_steps", type=int, default=50,
                   help="IG interpolation steps (more = more stable, slower).")
    p.add_argument("--internal_batch_size", type=int, default=8,
                   help="Captum's internal batch size for IG steps.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device = {device}")

    output_path = Path(args.output or f"outputs/attributions_{args.model}_{args.subset}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Model
    print(f"[model] loading {args.model} from adapter {args.adapter}")
    model, tokenizer, max_length = load_model_and_tokenizer(args.model, args.adapter, device)

    # ---- Data: load the raw GUE test split (we need raw sequences for IG + one-hot)
    print(f"[data] loading {args.subset} / {args.split}")
    ds = load_gue_subset(args.subset)
    split_name = args.split if args.split in ds else ("dev" if "dev" in ds else "test")
    sequences = list(ds[split_name]["sequence"])
    labels = list(ds[split_name]["label"])

    # ---- Filter to correctly-classified high-confidence positives
    keep_idx, preds_all, probs_all = select_positives(
        model, tokenizer, sequences, labels, device, max_length, args.min_prob
    )
    if len(keep_idx) == 0:
        raise RuntimeError("No high-confidence correctly-classified positives. "
                           "Lower --min_prob or re-check the trained model.")
    if len(keep_idx) > args.n_max:
        # Take the top-n_max by confidence so MoDISco sees the cleanest examples
        order = np.argsort(-probs_all[keep_idx])[:args.n_max]
        keep_idx = keep_idx[order]
    keep_idx = np.sort(keep_idx)
    print(f"[select] keeping {len(keep_idx)} sequences for attribution")

    kept_seqs = [sequences[i] for i in keep_idx]
    kept_labels = np.asarray([labels[i] for i in keep_idx], dtype=np.int64)
    kept_preds = preds_all[keep_idx]
    kept_probs = probs_all[keep_idx]

    # ---- IG
    nt_attrs_list = run_ig(
        model, tokenizer, kept_seqs, args.model, device,
        max_length=max_length, n_steps=args.n_steps,
        internal_batch_size=args.internal_batch_size,
    )

    # ---- Pack into TF-MoDISco shape (N, L, 4)
    # Pad to the longest observed sequence length (GUE TF subsets are usually
    # fixed-length per dataset, but we don't hard-code that)
    L = max(len(s) for s in kept_seqs)
    N = len(kept_seqs)
    one_hot_arr = np.zeros((N, L, 4), dtype=np.uint8)
    attr_arr = np.zeros((N, L, 4), dtype=np.float32)

    for i, (seq, nt_attrs) in enumerate(zip(kept_seqs, nt_attrs_list)):
        oh = one_hot_encode(seq)
        one_hot_arr[i, :len(seq)] = oh
        # Place per-nucleotide attribution on the observed base; zeros elsewhere
        attr_arr[i, :len(seq)] = nt_attrs[:, None] * oh.astype(np.float32)

    # ---- Save
    np.savez_compressed(
        output_path,
        one_hot=one_hot_arr,
        attributions=attr_arr,
        sequences=np.array(kept_seqs),
        labels=kept_labels,
        preds=kept_preds,
        probs=kept_probs,
    )
    print(f"[save] wrote {output_path}")
    print(f"[save]   one_hot      = {one_hot_arr.shape}  {one_hot_arr.dtype}")
    print(f"[save]   attributions = {attr_arr.shape}  {attr_arr.dtype}")


if __name__ == "__main__":
    main()