# CMPE 549 Final Presentation — Slide Draft

> Format: 15 slides for ~15 min (≈60–80 s each, with Q&A at the end).
> Each slide has (1) title, (2) bullet-level content to project, (3) speaker
> notes in italics, (4) a `[Visual]` placeholder for the figure/image that
> belongs there. Figures live under `outputs/figures/` after `python -m
> src.figures`; cluster logos are under `outputs/simple_*/`.

---

## 1. Title slide

# Architecture-Specific Interpretability in DNA Foundation Models

### Comparing motif discovery via Integrated Gradients across DNABERT-2 and HyenaDNA

**Toprak [Surname]** — CMPE 549, [Date]

*Speaker note: 15 seconds. Read the title, name, course. Don't linger.*

---

## 2. Motivation & research question

- DNA foundation models (DNABERT-2, HyenaDNA, Caduceus, Nucleotide Transformer)
  set new state-of-the-art on regulatory genomics tasks like TF binding site
  classification.
- They are increasingly used in research and clinical settings — but we still
  don't know whether they learn **biologically meaningful** patterns or
  statistical shortcuts.
- Interpretability is the bridge between performance and trust.

**Research question.** Given the *same* binary classification task, do
DNA foundation models with *different* architectures learn the *same*
biological features — and how robust is the answer to the choice of
attribution method?

*Speaker note: This sets up the entire talk. Two stress points: "different
architectures" (the comparison) and "robust to attribution method" (the
methods caveat that pays off later).*

---

## 3. Models compared

| Model | Architecture | Tokenization | Base params | Pretraining |
|---|---|---|---|---|
| **DNABERT-2** | Transformer (BERT-style) | BPE (~4k vocab) | 117 M | Multi-species genomes |
| **HyenaDNA**  | Hyena (long convolution) | Single-nucleotide | 3.4 M | Human reference genome |

- ~35× parameter disparity, fundamentally different inductive biases.
- A third model, **Caduceus** (Mamba state-space), was originally part of the
  comparison but was **dropped within the 2-day project window**: its CUDA
  kernels are not currently compatible with Captum's gradient hooks, and
  resolving that would require attribution-framework work out of scope here.

`[Visual: simple 2-panel cartoon — transformer self-attention vs Hyena
long-convolution operator. Optional but helps lay audiences.]`

*Speaker note: Emphasize that these are real architectural alternatives, not
incremental variants. DNABERT-2 has full quadratic attention; HyenaDNA has
sub-quadratic implicit-conv kernels. If they recover the same biology that's
non-trivial.*

---

## 4. Task & dataset

- **Task.** Binary classification: does this 101 bp sequence contain a TFBS
  for one particular transcription factor?
- **Data.** GUE benchmark (Zhou et al., 2024), `human_tf_0` subset:
  ~32 K train, 1 K dev, 1 K test, balanced positive/negative.
- **TF identity is blind during training** — revealed only later by matching
  our extracted motifs against JASPAR.
- Both models fine-tuned with LoRA adapters (r = 8, α = 16, dropout = 0.05,
  3 epochs) — keeps base weights frozen and reduces compute by ~3 orders of
  magnitude.

`[Visual: small inset showing one positive sequence and one negative
sequence, labeled. Or a per-split count bar chart.]`

*Speaker note: Stress the blinding — the model never sees the TF name, and
neither do we until TOMTOM. Makes the motif discovery a real test, not a
sanity check.*

---

## 5. Pipeline overview

```
GUE human_tf_0
      │
      ▼
LoRA fine-tune ──── DNABERT-2 (test acc 0.845, MCC 0.695)
               ──── HyenaDNA  (test acc 0.773)
      │
      ▼
Filter to high-confidence correct positives (p ≥ 0.7)  →  300 sequences/model
      │
      ▼
Integrated Gradients (Captum LayerIG, 200 steps, PAD baseline)
      │
      ▼
Per-nucleotide attribution map
      │
      ▼
Peak-window motif extraction (21 bp centered on per-seq peak; k-means k=3)
      │
      ▼
TOMTOM motif matching vs JASPAR2022 CORE vertebrates
```

*Speaker note: Walk through it once, top to bottom. ~30 seconds. The
attentive listener will notice I'm using "peak-window" instead of the
expected TF-MoDISco — flag that we'll explain in two slides.*

---

## 6. Fine-tuning results

| Metric | DNABERT-2 | HyenaDNA |
|---|---|---|
| Test accuracy | **0.845** | 0.773 |
| Test MCC | 0.695 | — |
| Trainable parameters (LoRA) | ~150 K | ~25 K |
| Approx. train time (T4 GPU) | ~25 min | ~8 min |

- Both models clearly learn the task — neither is at chance.
- DNABERT-2 is ~7 absolute points stronger on accuracy, consistent with its
  35× larger parameter count.
- HyenaDNA's competitiveness with 35× fewer params is itself a story —
  parameter efficiency matters in genomics where data is limited.

*Speaker note: Don't dwell on the accuracy gap; the interpretability question
makes sense at any reasonable accuracy.*

---

## 7. Integrated Gradients — and a faithfulness check

**Method.** Captum's `LayerIntegratedGradients` on the input embedding layer,
200 interpolation steps, PAD-token baseline. Target = positive class.

- For DNABERT-2 (BPE), per-token attribution is distributed back to
  nucleotides via the tokenizer's offset mapping.
- For HyenaDNA (single-nt), tokens and positions correspond 1:1.

**Sanity check — IG completeness (convergence δ).**
δ measures how well the per-token attributions sum to the model's actual
output change. Near zero means IG is faithful; large means the integration
path encountered strong non-linearities.

`[Visual: outputs/figures/convergence_delta.png — bar chart on a symlog axis]`

| Model | Mean \|IG δ\| |
|---|---|
| DNABERT-2 | **2.50** |
| HyenaDNA  | **0.0001** |

- HyenaDNA's IG is essentially exact; DNABERT-2's is approximate by ~4 orders
  of magnitude.
- **Why?** IG interpolates in *continuous embedding space* between baselines
  and inputs. HyenaDNA's 16-token vocab + long-conv (close to linear) gives
  IG a well-behaved path. DNABERT-2's 4k-token BPE embedding × full attention
  gives IG a wildly non-linear path.
- This is a structural property of the architecture pair — not a parameter
  to tune. Pay attention to it; it pays off at the end of the talk.

*Speaker note: This slide is the methods anchor. Spend ~90 seconds. The δ
chart is small but has a big payoff at slide 12.*

---

## 8. Attribution heatmap on a shared positive sequence

`[Visual: outputs/figures/attribution_heatmap.png — two stacked heatmaps,
RdBu_r, marked peak positions, sequence shown below]`

- Each row is one model's per-nucleotide IG score on the *same* 101 bp
  sequence that both classified as a high-confidence positive.
- Both models attend to roughly the same region (~mid-sequence), but with
  noticeably different distributions: DNABERT-2 spreads attribution across a
  wider neighborhood, HyenaDNA peaks more sharply on individual bases.
- This matches the δ result — diffuse IG path (DNABERT-2) ↔ spread
  attribution; exact IG path (HyenaDNA) ↔ peaked attribution.

*Speaker note: The first direct cross-model comparison. The figure should do
most of the work — point at the shared peak region, then at the difference
in spread. ~75 seconds.*

---

## 9. From attributions to motifs

**Initial attempt: TF-MoDISco**

- Industry-standard motif extractor used in DeepSEA, Basenji, ChromBPNet.
- Identified ~50 high-attribution seqlets per model but produced **zero
  final patterns** even after aggressive low-N parameter relaxation
  (target_seqlet_fdr → 0.3, min_metacluster_size → 10, final_min_cluster_size
  → 8, subcluster_perplexity → 12).
- 300 attribution sequences is below modisco's design point of thousands.

**Fallback: peak-window consensus PPMs**

1. For each sequence, locate the single nucleotide with maximum |IG|.
2. Extract a 21 bp window centered on it.
3. Pool all 300 windows; average → consensus PPM.
4. K-means (k=3) on flattened windows for distinct binding sub-modes.
5. Render sequence logos in information content (bits) via `logomaker`.

*Speaker note: Be straightforward about the modisco failure — honest methods
discussion makes the final results more credible. The peak-window approach
is less principled but doesn't pretend to be more than it is.*

---

## 10. Sequence logos — what each model "sees"

`[Visual: 2 × 3 grid, using existing PNGs from outputs/simple_*/:
  Row 1 — DNABERT-2: consensus_all_peaks, cluster_01, cluster_02
  Row 2 — HyenaDNA: consensus_all_peaks, cluster_01, cluster_02]`

**Key observations.**

- **DNABERT-2 cluster_02** (n = 88): consensus reads `GCGGCGC` at positions
  6–12, with ~0.5–1.0 bits per position. The sharpest motif extracted.
- **HyenaDNA cluster_01** (n = 102): consensus reads `GCGCG` at positions
  9–13, ~0.4–0.7 bits per position. Same family of pattern, lower IC.
- Both models recover the same nucleotide-pattern class: **GC-rich /
  CGCG-repeat** — the canonical zinc-finger TF binding signature.
- The other clusters (cluster_00 for both, plus cluster_02 for HyenaDNA)
  show weaker but related GC-rich patterns — likely the same TF binding
  at slightly different register, or related variants.

*Speaker note: This is the central result slide. Spend ~90 seconds. Walk
through the strongest motif (DNABERT-2 cluster_02) first, then point at
the HyenaDNA equivalent, then make the family claim.*

---

## 11. TOMTOM validation against JASPAR

`[Visual: outputs/figures/tomtom_top5_table.png — top-5 JASPAR matches per
model with shared targets highlighted]`

| Rank | DNABERT-2 match (q-value) | HyenaDNA match (q-value) |
|---|---|---|
| 1 | **PATZ1** (q = 8 × 10⁻⁴) | **ZNF341** (q = 9 × 10⁻⁴) |
| 2 | **ZNF341** (q = 2 × 10⁻³) | **PATZ1** (q = 1 × 10⁻³) |
| 3 | **KLF15** (q = 3 × 10⁻³) | **ZNF148** (q = 1 × 10⁻³) |
| 4 | **ZNF148** (q = 5 × 10⁻³) | **KLF15** (q = 2 × 10⁻³) |
| 5 | ZFX (q = 9 × 10⁻³) | ZFX (q = 3 × 10⁻²) |

- **The top-4 JASPAR matches are identical TFs in both models — just
  reordered.**
- All five are C2H2 zinc finger TFs binding GC-rich elements (the KLF/SP/ZNF
  family).
- The TF behind `human_tf_0` is almost certainly in this family — most
  likely PATZ1 or ZNF341 given that they appear in the top 2 for both models.

*Speaker note: This is the punchline. Different architectures, same biology.
~75 seconds. Spend extra time on the highlighted shared targets.*

---

## 12. Cross-model findings

**1. Architectures agree on the biology.**
- Both models recover GC-box zinc finger motifs.
- Identical top-4 JASPAR matches across DNABERT-2 and HyenaDNA.

**2. Architectures disagree on how attribution *looks*.**
- DNABERT-2 has spread attribution (consistent with δ = 2.5) and produces a
  *longer, sharper* motif.
- HyenaDNA has peaked attribution (consistent with δ ≈ 0) and produces a
  *shorter, weaker* motif.

**3. Attribution faithfulness is not motif quality.**
- DNABERT-2's IG fails the completeness check yet recovers the cleanest motif.
- HyenaDNA's IG is exact but its motif extraction is weaker.
- The two are decoupled: IG concentrates attribution along its integration
  path even when convergence δ is large.
- → Cautionary lesson: faithfulness metrics and downstream-usefulness
  metrics are not interchangeable, and choosing an attribution method
  requires thinking about both.

*Speaker note: The high-level finding. Frame point 3 as a useful caveat
rather than a contradiction.*

---

## 13. Limitations

- **Caduceus dropped.** Mamba CUDA kernels and Captum's gradient hooks
  are currently incompatible. A perturbation-based or post-hoc SHAP path
  would let us include it — out of scope here.
- **DNABERT-2 IG δ ≈ 2.5.** IG on transformer embedding layers is approximate
  by construction. The motifs are real but the per-position attribution
  scores should not be taken as exact.
- **N = 300 attribution sequences** is well below TF-MoDISco's design
  point; we had to fall back to a simpler extractor.
- **One TF, one subset.** `human_tf_0` is a single GC-binding zinc finger.
  The cross-architecture agreement might be family-specific.

*Speaker note: Don't apologize, just be specific. This is the slide that
makes the audience trust everything else.*

---

## 14. Future work

- **Re-include Caduceus** via a Mamba-compatible attribution method
  (gradient × input, occlusion-based SHAP).
- **All five human_tf_X subsets.** Does the agreement hold for non-GC-binding
  TFs (homeodomain, helix-turn-helix, basic-leucine-zipper)?
- **Cross-attribute DNABERT-2** with DeepSHAP and DeepLIFT to test whether
  IG is specifically the bottleneck for transformer attribution.
- **Higher-N attribution** to enable principled TF-MoDISco extraction with
  proper hypothetical contributions (per-base ablation).
- **Architectural probing.** Can attention features from DNABERT-2 transfer
  to a HyenaDNA-style head? Does that change the motif story?

*Speaker note: Three takeaways verbally; let the slide carry the rest.*

---

## 15. Acknowledgments & Q&A

- Code + outputs:  `[GitHub URL]`
- Tools: HuggingFace, JASPAR2022, MEME-suite (TOMTOM), Captum, TF-MoDISco-lite,
  logomaker
- Course staff
- **Questions welcome.**

*Speaker note: 10 seconds. Move on.*
