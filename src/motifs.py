"""Phase 4 - TF-MoDISco motif extraction from IG attributions.

Loads the .npz produced by src/attribute.py, clusters high-attribution windows
into motifs via TF-MoDISco-lite, then exports:
  - HDF5 motif file (the canonical MoDISco output)
  - Sequence logos as PNG per motif (via logomaker)
  - One MEME-format file with all PWMs (input to TOMTOM web)

Run from project root:
    python -m src.motifs \
        --attributions outputs/attributions_dnabert2_human_tf_0.npz \
        --output outputs/modisco_dnabert2_human_tf_0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

import modiscolite
from modiscolite import tfmodisco


def load_attributions(path: str):
    data = np.load(path)
    one_hot = data["one_hot"].astype(np.float32)
    attributions = data["attributions"].astype(np.float32)
    return one_hot, attributions


def run_modisco(one_hot, attributions, sliding_window_size: int, flank_size: int,
                target_seqlet_fdr: float = 0.2,
                min_metacluster_size: int = 30,
                max_seqlets_per_metacluster: int = 2000,
                final_min_cluster_size: int = 20,
                min_num_to_trim_to: int = 30,
                subcluster_perplexity: float = 50.0,
                n_leiden_runs: int = 2):
    """Cluster high-attribution windows into pos/neg patterns."""
    print(f"[modisco] sequences = {one_hot.shape[0]}, length = {one_hot.shape[1]}")
    print(f"[modisco] window={sliding_window_size}  flank={flank_size}  "
          f"fdr={target_seqlet_fdr}  min_metacluster_size={min_metacluster_size}  "
          f"final_min_cluster_size={final_min_cluster_size}  "
          f"min_num_to_trim_to={min_num_to_trim_to}  "
          f"subcluster_perplexity={subcluster_perplexity}  "
          f"n_leiden_runs={n_leiden_runs}")
    pos_patterns, neg_patterns = tfmodisco.TFMoDISco(
        hypothetical_contribs=attributions,
        one_hot=one_hot,
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
        target_seqlet_fdr=target_seqlet_fdr,
        min_metacluster_size=min_metacluster_size,
        max_seqlets_per_metacluster=max_seqlets_per_metacluster,
        final_min_cluster_size=final_min_cluster_size,
        min_num_to_trim_to=min_num_to_trim_to,
        subcluster_perplexity=subcluster_perplexity,
        n_leiden_runs=n_leiden_runs,
        verbose=True,
    )
    n_pos = len(pos_patterns) if pos_patterns else 0
    n_neg = len(neg_patterns) if neg_patterns else 0
    print(f"[modisco] positive patterns = {n_pos}, negative patterns = {n_neg}")
    if pos_patterns:
        for i, p in enumerate(pos_patterns):
            n_seqlets = len(p.seqlets) if hasattr(p, "seqlets") and p.seqlets is not None else "?"
            print(f"[modisco]   pos_pattern_{i:02d}: seqlets = {n_seqlets}")
    return pos_patterns, neg_patterns


def pattern_to_ppm(pattern, pseudocount: float = 0.01) -> np.ndarray:
    """Build a position-probability matrix from a MoDISco Pattern's supporting
    seqlets. Falls back to the pattern's own `sequence` attribute if seqlets
    aren't directly accessible."""
    pfm = None
    if hasattr(pattern, "seqlets") and pattern.seqlets is not None:
        try:
            seqlets_oh = np.stack([np.asarray(s.sequence) for s in pattern.seqlets])
            pfm = seqlets_oh.sum(axis=0).astype(np.float64)
        except Exception:
            pfm = None
    if pfm is None:
        for attr in ("sequence", "ppm", "pwm", "trimmed_sequence"):
            if hasattr(pattern, attr):
                pfm = np.asarray(getattr(pattern, attr), dtype=np.float64)
                break
    if pfm is None:
        raise ValueError(f"Could not extract PFM from pattern of type {type(pattern).__name__}; "
                         f"attrs: {[a for a in dir(pattern) if not a.startswith('_')]}")
    pfm = pfm + pseudocount
    ppm = pfm / pfm.sum(axis=1, keepdims=True)
    return ppm


def plot_logo(ppm: np.ndarray, title: str, output_path: Path) -> None:
    """Render a sequence logo in information-content (bits) units."""
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


def write_meme(motifs: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    """Write a MEME-format file ready for TOMTOM upload."""
    with open(output_path, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("Background letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for name, ppm in motifs:
            f.write(f"MOTIF {name}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {ppm.shape[0]} nsites= 20\n")
            for row in ppm:
                f.write("  " + "  ".join(f"{x:.6f}" for x in row) + "\n")
            f.write("\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attributions", required=True, help="Path to attributions_*.npz")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--sliding_window_size", type=int, default=20,
                   help="MoDISco core window. 20-21 matches the paper; smaller "
                        "than 15 fragments motifs, larger than 25 merges them.")
    p.add_argument("--flank_size", type=int, default=10)
    p.add_argument("--target_seqlet_fdr", type=float, default=0.2,
                   help="FDR for declaring a seqlet significant. 0.2 is modisco "
                        "default; 0.05 is strict and may yield zero patterns at "
                        "N<=300 with noisy IG.")
    p.add_argument("--min_metacluster_size", type=int, default=30,
                   help="Modisco default is 100; lowered for our N=300 regime.")
    p.add_argument("--final_min_cluster_size", type=int, default=20,
                   help="Min seqlets per final sub-cluster. Modisco default 20; "
                        "drop to 8-10 when total seqlets <100.")
    p.add_argument("--min_num_to_trim_to", type=int, default=30,
                   help="Min seqlets needed to compute a trimmed pattern. "
                        "Modisco default 30; drop to 8-10 with few seqlets.")
    p.add_argument("--subcluster_perplexity", type=float, default=50.0,
                   help="tSNE perplexity for sub-clustering. Modisco default 50 "
                        "but tSNE needs N > 3*perplexity. With ~50 seqlets, use ~10-15.")
    p.add_argument("--n_leiden_runs", type=int, default=2,
                   help="More runs => more chance leiden finds a good partition.")
    p.add_argument("--top_k", type=int, default=10,
                   help="Number of top positive patterns to emit as logos + MEME entries")
    p.add_argument("--max_seqlets_per_metacluster", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    one_hot, attributions = load_attributions(args.attributions)
    print(f"[load] one_hot      = {one_hot.shape}  attributions = {attributions.shape}")

    pos_patterns, neg_patterns = run_modisco(
        one_hot, attributions,
        sliding_window_size=args.sliding_window_size,
        flank_size=args.flank_size,
        target_seqlet_fdr=args.target_seqlet_fdr,
        min_metacluster_size=args.min_metacluster_size,
        max_seqlets_per_metacluster=args.max_seqlets_per_metacluster,
        final_min_cluster_size=args.final_min_cluster_size,
        min_num_to_trim_to=args.min_num_to_trim_to,
        subcluster_perplexity=args.subcluster_perplexity,
        n_leiden_runs=args.n_leiden_runs,
    )

    # Save HDF5 (best-effort; modisco-lite's save API varies across versions)
    h5_path = output_dir / "modisco_results.h5"
    try:
        from modiscolite import io as modisco_io  # noqa: E501
        try:
            # modisco-lite >= 2.2: window_size is required
            modisco_io.save_hdf5(str(h5_path), pos_patterns, neg_patterns,
                                 args.sliding_window_size)
        except TypeError:
            # older modisco-lite: 3-arg signature
            modisco_io.save_hdf5(str(h5_path), pos_patterns, neg_patterns)
        print(f"[save] HDF5 -> {h5_path}")
    except Exception as e:
        print(f"[warn] HDF5 save failed ({e}); patterns still extracted in-memory")

    if not pos_patterns:
        print("[motifs] No positive patterns found.")
        print("[motifs] At N<=100 seqlets, the binding constraints are usually")
        print("[motifs] the *sub-cluster* thresholds, not metacluster size. Try:")
        print("[motifs]   --final_min_cluster_size 8 --min_num_to_trim_to 8 \\")
        print("[motifs]     --subcluster_perplexity 15 --target_seqlet_fdr 0.3")
        print("[motifs] If that still yields 0: try --n_leiden_runs 6")
        print("[motifs] Last resort: re-run attribute.py with --n_steps 200")
        return

    motifs: list[tuple[str, np.ndarray]] = []
    for i, pattern in enumerate(pos_patterns[: args.top_k]):
        try:
            ppm = pattern_to_ppm(pattern)
        except Exception as e:
            print(f"[skip] pattern {i}: {e}")
            continue
        name = f"pos_pattern_{i:02d}"
        motifs.append((name, ppm))
        plot_logo(ppm, name, output_dir / f"{name}.png")
        print(f"[save] logo -> {output_dir / (name + '.png')}  width={ppm.shape[0]}")

    meme_path = output_dir / "motifs.meme"
    write_meme(motifs, meme_path)
    print(f"[save] MEME -> {meme_path}")
    print(f"[next] Upload {meme_path} to https://meme-suite.org/meme/tools/tomtom")
    print(f"[next]   Database: JASPAR (non-redundant) CORE vertebrates")
    print(f"[next]   Significance threshold: q-value < 0.05")


if __name__ == "__main__":
    main()