# Run log

Everything below was actually executed. The numbers are real.

## Environment

| Step | Result |
|---|---|
| `python3 -m venv` with system Python | **Changed** — system Python is 3.9.6; used python.org 3.12.1 instead |
| `pip install -r requirements.txt` | Worked. torch 2.15.0, sklearn 1.9.1, pandas 3.1.1 |
| MPS (Apple GPU) | Available and used |
| `pip install anticp2` | **Failed** — PyPI package requires Python ≤ 3.7 |
| Standalone AntiCP 2.0 on modern sklearn | **Failed** — `ModuleNotFoundError: sklearn.ensemble.forest` (renamed in sklearn 0.22) |
| `conda create -n anticp2 python=3.7 scikit-learn=0.21.3` under Rosetta | Worked. AntiCP 2.0 runs |

## Data

All four files downloaded (HTTP 200, plain ASCII, not HTML, not FASTA).

| | pos_train | neg_train | pos_test | neg_test |
|---|---|---|---|---|
| sequences | 689 | 689 | 172 | 172 |
| median length | 18 | 25 | 18 | 27 |

Cleaning dropped **nothing**: no non-standard residues, no duplicates, no
label conflicts, no exact train/val overlap.

Two findings the brief did not ask for but that matter:

- **66 training sequences have near-duplicates in validation** (≥90% shared
  6-mers) despite zero exact overlap. Example: `FAKKFAKKFKKFAKKFAKFAFAF` in
  train vs `VAKKFAKKFKKFAKKFAKFAFAF` in val. Held-out numbers read slightly
  optimistic, and the novelty check must use similarity, not exact matching.
- **AntiCP 2.0 has an off-by-one bug in its plain-text reader.** It measures
  each line including the newline, so a 50-residue peptide counts as 51 and is
  silently dropped. Empirical window for plain text is 5–49; FASTA is correct
  at 5–50. We always use FASTA.

## Model

Conditional VAE, PyTorch, 269k parameters (latent 32, hidden 128, max_len 25).

Peptides longer than 25 are dropped, not truncated: 851 of 1378 training
sequences survive (502 active, 349 inactive).

### Training experiments, in order

| Change | Train acc | Val acc | Note |
|---|---|---|---|
| First version (latent 64) | 78.4% | 54.2% | worked, but overfitting |
| **+ positional embedding in decoder** | 91.0% | 56.3% | big train gain, val flat |
| latent 32, dropout 0.2, β=0.05 | 85.2% | 55.8% | KL 67 |
| latent 32, dropout 0.2, β=0.2 | 73.9% | 54.7% | KL 30, better prior fit |
| **+ activity classifier (λ=5)** | 73.8% | 56.2% | activity acc 77.3% |

No posterior collapse in any run: active units stayed at the full latent
dimension and KL never approached zero.

The positional embedding was added because the decoder reads the same `[z, y]`
at every position and otherwise has to infer "where am I" from GRU dynamics
alone.

The activity classifier was added back after measuring that conditioning did
nothing without it. It is the `λ·L_act` term in the literature review's
equation 8.

## Phase 2 — rebuilding the decoder (this is what fixed it)

The results in Phase 1 below were poor. The diagnosis was the decoder: it was
non-autoregressive, reading the same `[z, y]` at every position, so it could
never build the local motifs that carry activity. Four changes:

| Change | Why |
|---|---|
| **Autoregressive decoder** | each residue is written knowing the one before it |
| **Word dropout (0.3)** | an autoregressive decoder can predict each residue from the previous one and ignore `z` entirely. Randomly replacing inputs with UNK forces it to consult `z` |
| **Label embedding (16-wide)** | a bare 0.0/1.0 scalar beside 32 latent dimensions was being ignored; now it can compete for attention |
| **Latent screening** | sample `z` from the prior, keep only those the activity head rates as active. Needs no dataset, so generation stays self-contained |
| **Analogue mode** | HydrAMP's other mode: encode a real active, drift the latent, decode |

600 epochs, latent 32, hidden 256, β=0.2, λ=5, word dropout 0.3, 344 seconds.

Final: train 96.6% (teacher-forced), val 58.7% (teacher-forced),
**val 55.0% (free-running)**, activity acc 77.6%, KL 26.0, 28/32 active units.

Note that free-running reconstruction (55.0%) barely changed from the old
model's 56.2%. **Reconstruction was never the problem.** What changed was the
decoder's ability to produce locally coherent sequences, which is what the
scoring picks up.

### Results after the rebuild

Scored by AntiCP 2.0, all sets under identical conditions:

| Set | mean | % ≥ 0.5 | % ≥ 0.8 |
|---|---|---|---|
| cVAE y=1 (prior) | 0.641 | 71.8% | 26.4% |
| cVAE y=1, screened | 0.675 | 78.2% | 34.8% |
| cVAE analogue (noise 0.5) | 0.834 | 90.8% | 66.2% |
| cVAE y=0 | 0.522 | 49.5% | 9.8% |
| random, same composition | 0.552 | 63.4% | 4.3% |
| scrambled real actives | 0.649 | 74.1% | 29.1% |
| real actives | 0.997 | 100% | 100% |

**Conditioning now works: +0.118** (was +0.020). Asking for active rather than
inactive peptides measurably changes what comes out.

### The novelty trap, and how we avoided it

The analogue score of 0.834 is **not** a real result. Checking novelty by noise
level showed why:

| noise | novel % | exact copies | mean similarity |
|---|---|---|---|
| 0.3 | 18.9% | **268** | 0.938 |
| 0.5 | 54.0% | **121** | 0.757 |
| 0.8 | 74.8% | 30 | 0.594 |
| 1.2 | 86.9% | 5 | 0.520 |
| 2.0 | 91.2% | 1 | 0.458 |

At noise 0.5, 121 of the 400 outputs were **exact copies of training peptides**.
The high score was memorisation.

So every number below scores **novel peptides only** — not an exact copy, and
under 90% 3-mer similarity to any training active:

| noise | novel | novel % | mean (novel only) | ≥ 0.8 | usable |
|---|---|---|---|---|---|
| 0.3 | 84 | 21.6% | 0.756 | 46.4% | 39 |
| 0.5 | 214 | 53.9% | 0.697 | 37.4% | 80 |
| 0.8 | 308 | 77.8% | 0.650 | 30.5% | 94 |
| 1.2 | 336 | 86.6% | 0.628 | 26.2% | 88 |
| **1.6** | **342** | **89.8%** | 0.633 | 28.1% | **96** |
| 2.0 | 340 | 90.2% | 0.616 | 22.4% | 76 |

And for prior sampling, which needs no template at all:

| mode | novel | novel % | mean (novel only) | ≥ 0.8 | usable |
|---|---|---|---|---|---|
| prior, y=1 | 529 | 88.3% | 0.614 | 17.8% | 94 |
| prior, screened 0.9 | 527 | 88.3% | 0.635 | 23.1% | 122 |
| **prior, screened 0.99** | **515** | **86.1%** | **0.647** | 25.4% | **131** |

**Where this lands.** Screened prior sampling produces peptides that are 86%
novel and score 0.647 on the novel ones — matching scrambled real peptides
(0.649) and clearly beating a composition-matched random baseline (0.552). The
final shortlist is **524 novel candidates, 142 scoring ≥ 0.8**, generated with
no template and no dataset at generation time.

Honest caveat: the top candidates sit at 0.78–0.89 similarity to training
actives. They are below our novelty cutoff but are recognisably variations on
known motifs — the `FAKKLAKK…` family is clearly related to the KLAKLAK
peptides in the training set. That is reasonable for peptide design and it is
what analogue-style generation is *for*, but it should not be described as
inventing something unprecedented.

### Latent space, after the rebuild

ARI 0.052 (was 0.009), silhouette 0.062, label agreement 61.6% (was 55.1%).
Better, still weak. Activity remains linearly decodable from `z` (the activity
head reaches 77.6%) without being the dominant axis of variation, so k-means
does not recover it.

---

## Phase 1 — the original non-autoregressive model (kept for the record)

## Generation and evaluation

500 sampled with y=1: **92.4% valid, 100% unique, 98.3% novel, 0 exact copies**.
454 candidates written and scored.

### The headline experiment

All four sets scored by AntiCP 2.0 (model 1, ACP vs AMP) under identical conditions:

| Set | n | mean score | % ≥ 0.5 | % ≥ 0.8 |
|---|---|---|---|---|
| cVAE, y=1 (asked for active) | 399 | 0.577 | 62.7% | 13.8% |
| cVAE, y=0 (asked for inactive) | 398 | 0.556 | 60.6% | 10.3% |
| random, ACP composition | 397 | 0.552 | 64.2% | 6.0% |
| **scrambled real actives** | 499 | **0.654** | 71.5% | 30.1% |
| real actives, unmodified | 499 | 0.997 | 100% | 100% |

**What this says, honestly:**

1. **Conditioning is weak.** y=1 beats y=0 by only +0.020. The decoder largely
   ignores the label. Adding the activity classifier made `z` carry activity
   (77.3% accuracy) but did not make the *decoder* use `y`.
2. **The model has not clearly learned sequence order.** It beats a
   composition-matched random baseline by only +0.024.
3. **Our output scores below scrambled real peptides** (0.577 vs 0.654). That is
   the most damning number here and the one to take seriously.
4. **The evaluator is partly a memorisation test.** Real actives score 0.997
   because they are AntiCP 2.0's own training data. Scrambling them — identical
   composition, random order — drops them to 0.654, yet 71.5% are still called
   active. So for novel sequences AntiCP is substantially composition-driven,
   which compresses the range we are trying to measure differences in.

### Latent space

PCA + k-means (k=2) on the 32-dimensional space, 851 training peptides:

- PC1 9.2%, PC2 7.7% of variance (16.9% together)
- adjusted Rand index **0.009** — clusters do not correspond to activity
- silhouette **0.066**, label agreement 55.1%

Not a contradiction with the 77.3% activity classifier: activity *is* linearly
decodable from `z`, but it is not the dominant axis of variation, so PCA's top
components and unsupervised k-means do not find it.

## What is not done

- The decoder is non-autoregressive. HydrAMP's is autoregressive, and that is
  the most likely cause of the 56% reconstruction ceiling.
- Reconstruction accuracy (56% val) is too low for convincing generation.
- No wet-lab validation; these are computational candidates only.
