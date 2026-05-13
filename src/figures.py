"""Figure generation for the CMPE 549 final presentation.

Reads attribution .npz files + TOMTOM .tsv files + cluster logos, emits a set
of presentation-ready PNGs into outputs/figures/.

Run from project root (with the attribution .npz files in outputs/ and the
simple_* folders alongside):

    python -m src.figures --output_dir outputs/figures

Or one figure at a time:

    python -m src.figures --only heatmap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
})

DNABERT2_NPZ = "outputs/attributions_dnabert2_human_tf_0.npz"
HYENADNA_NPZ = "outputs/attributions_hyenadna_human_tf_0.npz"
TOMTOM_DNABERT2 = "outputs/simple_dnabert2_human_tf_0/tomtom.tsv"
TOMTOM_HYENADNA = "outputs/simple_hyenadna_human_tf_0/tomtom.tsv"

# Curated JASPAR2022 ID -> TF symbol (only confident mappings; others kept
# as the raw MA ID so we don't mislabel anything).
JASPAR_TF = {
    "MA0079.5":  "SP2",
    "MA0146.2":  "ZFX",
    "MA0162.4":  "EGR1",
    "MA0471.2":  "E2F6",
    "MA0516.3":  "SP2",
    "MA0528.2":  "ZNF263",
    "MA0597.2":  "THAP1",
    "MA0685.2":  "SP4",
    "MA0740.2":  "KLF14",
    "MA0742.2":  "KLF12",
    "MA0753.2":  "ZNF410",
    "MA0830.2":  "TCF4",
    "MA1102.2":  "CTCFL",
    "MA1122.1":  "TFCP2",
    "MA1511.2":  "KLF11",
    "MA1513.1":  "KLF15",
    "MA1522.1":  "MAZ",
    "MA1630.2":  "ZNF263",
    "MA1650.1":  "ZNF148",
    "MA1713.1":  "ZNF341",
    "MA1721.1":  "ZNF135",
    "MA1723.1":  "ZNF263",
    "MA1959.1":  "ZBTB14",
    "MA1961.1":  "PATZ1",
    "MA1986.1":  "KLF13",
}

# Hand-entered metrics so figures don't depend on regenerating training logs.
IG_CONVERGENCE_DELTA = {
    "DNABERT-2": 2.50,
    "HyenaDNA":  0.0001,
}

TRAINING_METRICS = [
    # (model, metric, value)
    ("DNABERT-2", "test_acc", 0.845),
    ("DNABERT-2", "test_mcc", 0.695),
    ("HyenaDNA",  "test_acc", 0.773),
]

COLOR_DNABERT2 = "#2E86AB"
COLOR_HYENADNA = "#E07A5F"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_npz(path: str):
    d = np.load(path)
    return {
        "sequences":    np.array([str(s) for s in d["sequences"]]),
        "one_hot":      d["one_hot"],
        "attributions": d["attributions"],
        "probs":        d["probs"],
    }


def per_position_attr(attr_4: np.ndarray) -> np.ndarray:
    """Collapse the (N, L, 4) attribution to (N, L) by summing across the
    nucleotide axis. Only the observed base is non-zero, so the sum equals the
    IG score on the observed nucleotide at each position."""
    return attr_4.sum(axis=-1)


def find_best_shared_positive(data_a, data_b):
    seqs_b_index = {s: j for j, s in enumerate(data_b["sequences"])}
    candidates = []
    for i, s in enumerate(data_a["sequences"]):
        j = seqs_b_index.get(s)
        if j is None:
            continue
        combined = float(data_a["probs"][i] + data_b["probs"][j])
        candidates.append((combined, i, j))
    if not candidates:
        raise RuntimeError("No shared sequences between the two attribution sets.")
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2], len(candidates)


def parse_tomtom_tsv(path: str, top_n: int = 5):
    by_query: dict[str, list[dict]] = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 or line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            row = {
                "query":            parts[0],
                "target":           parts[1],
                "q_value":          float(parts[5]),
                "query_consensus":  parts[7],
                "target_consensus": parts[8],
                "orientation":      parts[9],
            }
            by_query.setdefault(row["query"], []).append(row)
    for q in by_query:
        by_query[q].sort(key=lambda r: r["q_value"])
        by_query[q] = by_query[q][:top_n]
    return by_query


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def make_heatmap(out_path: Path):
    a = load_npz(DNABERT2_NPZ)
    b = load_npz(HYENADNA_NPZ)
    i, j, n_shared = find_best_shared_positive(a, b)

    seq = str(a["sequences"][i])
    L = len(seq)
    attr_a = per_position_attr(a["attributions"][i])[:L]
    attr_b = per_position_attr(b["attributions"][j])[:L]

    # Normalize each model's row to its own |max| (sign preserved).
    norm_a = attr_a / (np.abs(attr_a).max() + 1e-12)
    norm_b = attr_b / (np.abs(attr_b).max() + 1e-12)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.55]},
    )

    for ax, norm, raw, name, color in [
        (axes[0], norm_a, attr_a, "DNABERT-2", COLOR_DNABERT2),
        (axes[1], norm_b, attr_b, "HyenaDNA",  COLOR_HYENADNA),
    ]:
        im = ax.imshow(norm[np.newaxis], aspect="auto", cmap="RdBu_r",
                       vmin=-1, vmax=1)
        ax.set_yticks([0])
        ax.set_yticklabels([name], fontsize=11)
        ax.set_xlim(-0.5, L - 0.5)
        peak = int(np.argmax(np.abs(raw)))
        ax.axvline(peak, color="black", lw=1.2, ls="--", alpha=0.9)
        ax.text(peak + 0.7, 0, f"peak@{peak}", va="center", fontsize=8,
                color="black")

    # Sequence row
    ax = axes[2]
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(0, 1)
    for k, nt in enumerate(seq):
        ax.text(k, 0.5, nt, ha="center", va="center", fontsize=7,
                family="monospace")
    ax.set_yticks([])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(
        f"Position in 101 bp sequence    "
        f"P(positive): DNABERT-2 = {a['probs'][i]:.3f},  HyenaDNA = {b['probs'][j]:.3f}"
    )

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01,
                        label="IG attribution (per-model normalized)")
    fig.suptitle(
        f"Per-nucleotide IG attribution on a shared positive  "
        f"({n_shared} sequences correctly classified by both models)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def make_convergence_delta_bar(out_path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    models = list(IG_CONVERGENCE_DELTA.keys())
    deltas = [IG_CONVERGENCE_DELTA[m] for m in models]
    colors = [COLOR_DNABERT2, COLOR_HYENADNA]
    bars = ax.bar(models, deltas, color=colors, width=0.55)
    ax.axhline(0.1, color="gray", linestyle="--", alpha=0.8,
               label="δ < 0.1 (healthy threshold)")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylabel(r"Mean $|$IG convergence $\delta|$")
    ax.set_title("IG attribution faithfulness across architectures")
    for b, d in zip(bars, deltas):
        if d > 0.01:
            label = f"{d:.2f}"
        else:
            label = f"{d:.1e}"
        ax.text(b.get_x() + b.get_width() / 2, d * 1.4, label,
                ha="center", va="bottom", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def make_peak_position_hist(out_path: Path):
    a = load_npz(DNABERT2_NPZ)
    b = load_npz(HYENADNA_NPZ)

    peaks_a = np.argmax(np.abs(per_position_attr(a["attributions"])), axis=1)
    peaks_b = np.argmax(np.abs(per_position_attr(b["attributions"])), axis=1)
    L = a["attributions"].shape[1]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.arange(0, L + 1, 4)
    ax.hist(peaks_a, bins=bins, alpha=0.65, color=COLOR_DNABERT2, label="DNABERT-2")
    ax.hist(peaks_b, bins=bins, alpha=0.65, color=COLOR_HYENADNA, label="HyenaDNA")
    ax.set_xlabel("Position of attribution peak (per sequence)")
    ax.set_ylabel("# sequences (out of 300)")
    ax.set_title("Where each model concentrates its attribution along the 101 bp window")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def make_attribution_distribution(out_path: Path):
    a = load_npz(DNABERT2_NPZ)
    b = load_npz(HYENADNA_NPZ)

    flat_a = np.abs(per_position_attr(a["attributions"])).flatten()
    flat_b = np.abs(per_position_attr(b["attributions"])).flatten()
    flat_a = flat_a[flat_a > 0]
    flat_b = flat_b[flat_b > 0]
    # Normalize each to its own max so we compare the *shape* of the distribution
    flat_a_n = flat_a / flat_a.max()
    flat_b_n = flat_b / flat_b.max()

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.linspace(0, 1, 40)
    ax.hist(flat_a_n, bins=bins, alpha=0.55, color=COLOR_DNABERT2,
            label="DNABERT-2", density=True)
    ax.hist(flat_b_n, bins=bins, alpha=0.55, color=COLOR_HYENADNA,
            label="HyenaDNA", density=True)
    ax.set_yscale("log")
    ax.set_xlabel("|attribution| / max per model")
    ax.set_ylabel("Density (log)")
    ax.set_title("Distribution of per-position attribution magnitudes")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def make_tomtom_table(out_path: Path):
    a = parse_tomtom_tsv(TOMTOM_DNABERT2, top_n=5)
    b = parse_tomtom_tsv(TOMTOM_HYENADNA, top_n=5)
    rows_a = a.get("consensus_all_peaks", [])
    rows_b = b.get("consensus_all_peaks", [])

    def fmt(r):
        if r is None:
            return ("—", "—")
        tf = JASPAR_TF.get(r["target"], r["target"])
        return (f"{tf}   (q = {r['q_value']:.1e})", r["target_consensus"])

    table_data = [["Rank",
                   "DNABERT-2 match", "DNABERT-2 consensus",
                   "HyenaDNA match",  "HyenaDNA consensus"]]
    for k in range(5):
        ra = rows_a[k] if k < len(rows_a) else None
        rb = rows_b[k] if k < len(rows_b) else None
        ta, ca = fmt(ra)
        tb, cb = fmt(rb)
        table_data.append([str(k + 1), ta, ca, tb, cb])

    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.axis("off")
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center", colColours=["#f0f0f0"] * 5)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    # Bold header
    for k in range(5):
        tbl[(0, k)].set_text_props(weight="bold")
    # Highlight TF names that appear in both columns
    tf_in_a = {r["target"] for r in rows_a}
    tf_in_b = {r["target"] for r in rows_b}
    shared = tf_in_a & tf_in_b
    for r_idx in range(1, 6):
        for col_idx, rows in [(1, rows_a), (3, rows_b)]:
            k = r_idx - 1
            if k < len(rows) and rows[k]["target"] in shared:
                tbl[(r_idx, col_idx)].set_facecolor("#fff5cc")

    fig.suptitle(
        "Top-5 JASPAR2022 CORE vertebrates matches  (TOMTOM, q < 0.05, "
        "query = consensus_all_peaks)\n"
        "Highlighted cells share JASPAR target IDs across both models",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #
ALL_FIGURES = {
    "heatmap":               ("attribution_heatmap.png",       make_heatmap),
    "convergence":           ("convergence_delta.png",         make_convergence_delta_bar),
    "peaks":                 ("peak_position_histogram.png",   make_peak_position_hist),
    "attr_dist":             ("attribution_distribution.png",  make_attribution_distribution),
    "tomtom":                ("tomtom_top5_table.png",         make_tomtom_table),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/figures",
                   help="Directory to write PNGs into.")
    p.add_argument("--only", choices=list(ALL_FIGURES.keys()) + ["all"], default="all",
                   help="Generate one named figure, or 'all' (default).")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(ALL_FIGURES.keys()) if args.only == "all" else [args.only]
    for name in names:
        fname, fn = ALL_FIGURES[name]
        try:
            fn(output_dir / fname)
        except FileNotFoundError as e:
            print(f"[skip] {name}: missing input file ({e})")
        except Exception as e:
            print(f"[error] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
