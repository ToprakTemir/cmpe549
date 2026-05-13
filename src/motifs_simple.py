"""Phase 4 fallback - simple peak-based motif extraction.

Used when TF-MoDISco fails to cluster (low N, noisy attributions). The basic
idea: for each correctly-classified positive sequence, find the window with the
highest summed |attribution|, extract the underlying nucleotides, then either
average them all into a single consensus PPM or cluster them into K groups.

This is less principled than MoDISco -- no FDR thresholds, no leiden, no
trimming -- but it produces sequence logos + a MEME file when MoDISco won't.

Run from project root:
    python -m src.motifs_simple \
        --attributions outputs/attributions_dnabert2_human_tf_0.npz \
        --output outputs/simple_dnabert2_human_tf_0 \
        --window_size 15 --n_clusters 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from sklearn.cluster import KMeans


def load_attributions(path: str):
    data = np.load(path)
    one_hot = data["one_hot"].astype(np.float32)
    attributions = data["attributions"].astype(np.float32)
    sequences = data["sequences"] if "sequences" in data else None
    return one_hot, attributions, sequences


def find_top_window_per_sequence(one_hot, attributions, window_size: int,
                                 center_mode: str = "peak"):
    """For each sequence, extract a window of `window_size` bases.

    Two centering modes:
      - "peak" (default): find the single position with the highest |attribution|
        and center the window on it. Best when the model concentrates attribution
        on one nucleotide (which is what IG often does — sparse attribution).
      - "window_sum": find the contiguous window with the largest summed
        |attribution|. Best when the model spreads attribution across the
        motif evenly. Picks the window whose total signal is highest, but
        can place the peak at the edges.
    """
    N, L, _ = one_hot.shape
    W = window_size
    if W > L:
        raise ValueError(f"window_size {W} > sequence length {L}")
    half = W // 2

    # per-position |attribution| collapsed across the 4 channels
    pos_score = np.abs(attributions).sum(axis=-1)            # (N, L)

    if center_mode == "peak":
        # Single-position argmax, then center the window on it.
        peak_pos = np.argmax(pos_score, axis=1)              # (N,)
        peak_starts = np.clip(peak_pos - half, 0, L - W)
        peak_scores = pos_score[np.arange(N), peak_pos]
    elif center_mode == "window_sum":
        # Rolling-sum argmax (original behavior).
        cumsum = np.concatenate([np.zeros((N, 1)), np.cumsum(pos_score, axis=1)], axis=1)
        window_sums = cumsum[:, W:] - cumsum[:, :-W]          # (N, L - W + 1)
        peak_starts = np.argmax(window_sums, axis=1)
        peak_scores = window_sums[np.arange(N), peak_starts]
    else:
        raise ValueError(f"Unknown center_mode: {center_mode!r}")

    windows_oh = np.stack([
        one_hot[i, peak_starts[i]:peak_starts[i] + W]
        for i in range(N)
    ]).astype(np.float32)
    return windows_oh, peak_starts, peak_scores


def consensus_ppm(windows_oh, pseudocount: float = 0.01) -> np.ndarray:
    """Average a stack of one-hot windows into a position-probability matrix."""
    pfm = windows_oh.sum(axis=0).astype(np.float64) + pseudocount
    return pfm / pfm.sum(axis=1, keepdims=True)


def cluster_windows(windows_oh, n_clusters: int, seed: int = 42):
    """K-means cluster windows on their flattened one-hot representation.
    Returns array of cluster labels and the list of unique labels (sorted)."""
    if n_clusters <= 1:
        return np.zeros(len(windows_oh), dtype=np.int64), [0]
    flat = windows_oh.reshape(len(windows_oh), -1)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(flat)
    return labels, sorted(set(labels.tolist()))


def plot_logo(ppm: np.ndarray, title: str, output_path: Path) -> None:
    """Information-content sequence logo (bits)."""
    df = pd.DataFrame(ppm, columns=list("ACGT"))
    bg = 0.25
    ic = (df * np.log2(df / bg + 1e-9)).sum(axis=1).clip(lower=0)
    ic_logo = df.multiply(ic, axis=0)

    fig, ax = plt.subplots(figsize=(max(6.0, len(df) * 0.4), 2.5))
    logomaker.Logo(ic_logo, ax=ax, color_scheme="classic")
    ax.set_ylabel("bits")
    ax.set_ylim(0, 2)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_meme(motifs, output_path: Path) -> None:
    with open(output_path, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("Background letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for name, ppm, nsites in motifs:
            f.write(f"MOTIF {name}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {ppm.shape[0]} "
                    f"nsites= {nsites}\n")
            for row in ppm:
                f.write("  " + "  ".join(f"{x:.6f}" for x in row) + "\n")
            f.write("\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attributions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window_size", type=int, default=21,
                   help="Width of the motif window to extract. 21 gives ~10bp "
                        "on either side of the peak, enough context for most TFBSs.")
    p.add_argument("--center_mode", default="peak",
                   choices=["peak", "window_sum"],
                   help="'peak' (default): center window on the single-position "
                        "|attribution| argmax. 'window_sum': pick window with "
                        "max summed |attribution| (less centered on the peak).")
    p.add_argument("--n_clusters", type=int, default=3,
                   help="K-means clusters. Use 1 for a single consensus PPM.")
    p.add_argument("--min_cluster_size", type=int, default=10,
                   help="Skip clusters smaller than this when emitting motifs.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    one_hot, attributions, _ = load_attributions(args.attributions)
    print(f"[load] one_hot      = {one_hot.shape}  attributions = {attributions.shape}")

    windows_oh, peak_starts, peak_scores = find_top_window_per_sequence(
        one_hot, attributions, args.window_size, center_mode=args.center_mode
    )
    print(f"[peaks] {len(windows_oh)} top windows of width {args.window_size} "
          f"extracted ({args.center_mode} centering)")
    print(f"[peaks] mean peak |attr| = {peak_scores.mean():.4f}  "
          f"median peak start = {int(np.median(peak_starts))} / {one_hot.shape[1]}")

    # Always emit a single consensus PPM (k=1 equivalent) for reference
    consensus = consensus_ppm(windows_oh)
    plot_logo(consensus, "consensus_all_peaks",
              output_dir / "consensus_all_peaks.png")
    print(f"[save] consensus logo -> {output_dir / 'consensus_all_peaks.png'}  "
          f"width={consensus.shape[0]}  nsites={len(windows_oh)}")

    motifs = [("consensus_all_peaks", consensus, len(windows_oh))]

    if args.n_clusters > 1:
        labels, unique = cluster_windows(windows_oh, args.n_clusters, seed=args.seed)
        # Order clusters by size (largest first) so MOTIF 1, 2, 3... are descending
        sizes = [(c, int((labels == c).sum())) for c in unique]
        sizes.sort(key=lambda x: -x[1])
        for rank, (c, n) in enumerate(sizes):
            if n < args.min_cluster_size:
                print(f"[skip] cluster {c}: size {n} < min_cluster_size {args.min_cluster_size}")
                continue
            members = windows_oh[labels == c]
            ppm = consensus_ppm(members)
            name = f"cluster_{rank:02d}"
            plot_logo(ppm, f"{name} (n={n})", output_dir / f"{name}.png")
            print(f"[save] logo -> {output_dir / (name + '.png')}  "
                  f"width={ppm.shape[0]}  nsites={n}")
            motifs.append((name, ppm, n))

    meme_path = output_dir / "motifs.meme"
    write_meme(motifs, meme_path)
    print(f"[save] MEME -> {meme_path}  ({len(motifs)} motifs)")
    print(f"[next] Upload {meme_path} to https://meme-suite.org/meme/tools/tomtom")
    print(f"[next]   Database: JASPAR (non-redundant) CORE vertebrates")


if __name__ == "__main__":
    main()
