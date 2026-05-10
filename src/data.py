"""Data loading + tokenization for GUE TFBS subsets via the HuggingFace mirror.

The mirror lives at `leannmlindsey/GUE` and exposes ~37 named subsets, including
`human_tf_0` through `human_tf_4` and `mouse_0` through `mouse_4`. Each subset
has columns `sequence` (raw ACGT string, ~70-500 bp) and `label` (0/1).
"""

from __future__ import annotations

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

GUE_REPO = "leannmlindsey/GUE"


def load_gue_subset(subset_name: str = "human_tf_0") -> DatasetDict:
    """Load a GUE subset. Returns a DatasetDict with train/validation(or dev)/test."""
    ds = load_dataset(GUE_REPO, subset_name)

    expected = {"sequence", "label"}
    actual = set(ds[next(iter(ds))].column_names)
    missing = expected - actual
    if missing:
        raise ValueError(
            f"Subset {subset_name!r} missing columns {missing}; got {actual}. "
            f"Splits available: {list(ds.keys())}"
        )
    return ds


def tokenize_dataset(
    ds: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> DatasetDict:
    """BPE-tokenize the `sequence` column. Drops `sequence`, keeps `label`."""

    def _tok(batch):
        return tokenizer(
            batch["sequence"],
            truncation=True,
            padding=False,  # dynamic padding via DataCollatorWithPadding
            max_length=max_length,
            return_token_type_ids=False,
        )

    return ds.map(_tok, batched=True, remove_columns=["sequence"])


if __name__ == "__main__":
    # Quick smoke test: python -m src.data
    from transformers import AutoTokenizer

    ds = load_gue_subset("human_tf_0")
    print("Splits:", {k: len(v) for k, v in ds.items()})
    print("Sample:", ds[next(iter(ds))][0])

    tok = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    tds = tokenize_dataset(ds, tok)
    sample_lens = [len(tds["train"][i]["input_ids"]) for i in range(5)]
    print("First 5 token lengths:", sample_lens)
