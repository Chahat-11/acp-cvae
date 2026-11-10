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
