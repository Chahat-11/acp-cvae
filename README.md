# acp-cvae — designing anticancer peptides with a conditional VAE

DSC4804 Statistical Learning · Group 20 · Chahat Singh, Yakshita Yadav

---

## The short version

Anticancer peptides (ACPs) are short chains of amino acids that can kill cancer
cells while leaving healthy cells alone. Finding new ones in a laboratory is
slow and expensive, so the useful thing a computer can do is *propose
candidates* worth testing.

This project trains a small **conditional variational autoencoder (cVAE)** that
writes new peptide sequences on request, then scores them with an existing
published predictor.

---

## How the scope changed, and why

The project originally had **two halves**:

```
                 ┌──────────────────────────┐
                 │   AntiCP 2.0 dataset     │
                 └────────────┬─────────────┘
                              │
            ┌─────────────────┴──────────────────┐
            │                                    │
     ┌──────▼───────┐                   ┌────────▼────────┐
     │  THE JUDGE   │                   │   THE CREATOR   │
     │ predicts if  │                   │ invents new     │
     │ a peptide is │                   │ peptides        │
     │ anticancer   │                   │                 │
     └──────┬───────┘                   └────────┬────────┘
            │                                    │
            └─────────────────┬──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  score and rank    │
                    └────────────────────┘
```

**The judge** would have been our own classifier — logistic regression, SVM,
random forest, gradient boosting, plus a deep model on protein-language-model
embeddings — trained to predict whether a given peptide is anticancer.

**Our mentor told us to cut it.** Two reasons, and both are good ones:

1. **It is a solved problem that someone else already solved properly.**
   AntiCP 2.0 is published, peer reviewed, and trained on this exact dataset.
   Rebuilding a worse version of it would not teach us anything the paper does
   not already say, and it would not be a contribution.
2. **It doubles the work for no gain.** The interesting half is generation.
   Prediction scores peptides that already exist; it cannot invent one. The
   literature is almost entirely predictive, so the gap worth attacking is the
   generative side.

So the scope is now **creator only**:

```
   ┌──────────────────────────┐
   │   AntiCP 2.0 dataset     │
   │   cleaning and splits    │
   └────────────┬─────────────┘
                │
       ┌────────▼─────────┐
       │   THE CREATOR    │   ← we build this
       │   conditional    │
       │   VAE            │
       └────────┬─────────┘
                │ generated peptides
                │
       ┌────────▼─────────┐
       │  AntiCP 2.0      │   ← we use this, we do not train it
       │  (the judge,     │
       │   off the shelf) │
       └────────┬─────────┘
                │
       ┌────────▼─────────┐
       │ ranked shortlist │
       └──────────────────┘
```

We **do not train any classifier of our own.** The one exception is a tiny
activity head *inside* the VAE, explained below — it shapes the latent space
during training and never judges our output. AntiCP 2.0 does all the judging.

---

## What actually happens, step by step

### Step 1 — Data (`src/prepare_data.py`)

We use AntiCP 2.0's **main** dataset, four files downloaded from the Raghava
lab's site:

| File | Count | What it is |
|---|---|---|
| `pos_train_main` | 689 | anticancer peptides |
| `neg_train_main` | 689 | antimicrobial peptides that are **not** anticancer |
| `pos_test_main` | 172 | anticancer (validation) |
| `neg_test_main` | 172 | not anticancer (validation) |

**Why the negatives matter.** They are not random strings — they are real
antimicrobial peptides. So a model cannot succeed just by learning "this looks
like a peptide". It has to learn what makes one specifically *anticancer*. This
makes the task harder and the result meaningful.

The script uppercases, strips whitespace, drops anything containing a
non-standard amino acid, removes duplicates, removes sequences that appear with
both labels, and removes any training sequence that also appears in validation.
It prints every drop with a reason.

On this dataset it dropped **nothing** — the Raghava lab had already cleaned it.

Output: `data/processed/train.csv`, `val.csv`, `train_pos_only.csv`, and
`outputs/length_hist.png`.

### Step 2 — Encoding (`src/peptides.py`)

A peptide is a string like `FAKKFAKKFKKFAKK`. The model cannot read letters, so
each amino acid becomes a number:

```
0      = PAD (empty position, pads short peptides out to a fixed length)
1..20  = the twenty standard amino acids
```

That is a vocabulary of 21 symbols.

**Maximum length is 25**, following HydrAMP. Peptides longer than 25 are
**dropped, not cut short** — a truncated anticancer peptide is not an anticancer
peptide, and training on fragments would teach the model to write fragments.
This leaves 851 training peptides (502 active, 349 inactive).

### Step 3 — The model (`src/model.py`)

```
peptide x ──► encoder ──► mu, logvar ──► z = mu + sigma·eps ──┐
                                                  │            ├──► decoder ──► new peptide
                                                  │            │
label y (1 = anticancer) ──────────────────────────────────────┘
                                                  │
                                                  └──► activity head ──► is it active?
```

**Why the encoder outputs a distribution instead of a point.** A plain
autoencoder maps each peptide to one location in space. Sample anywhere between
two of those locations and you get noise, because nothing ever forced the space
in between to mean anything. A VAE instead outputs a *mean* and a *spread*, and
the KL term pulls every peptide's distribution toward a standard normal. That
makes the space continuous — which is exactly what lets us sample new points
from it later.

**Why `z = mu + sigma·eps`.** Sampling straight from `N(mu, sigma)` would break
backpropagation, because you cannot take a derivative through a random draw.
Written this way, the randomness lives in `eps`, which carries no gradient, so
`mu` and `sigma` stay trainable. This is the *reparameterisation trick*.

**Why the label is concatenated to `z`.** This is the "conditional" part. The
decoder sees both the latent vector and the label, so at generation time we can
set `y = 1` and ask for an anticancer peptide rather than taking whatever comes.
The label is repeated at every position so the decoder cannot forget it.

**The loss has three terms:**

```
L_total  =  L_rec  +  β · L_KL  +  λ · L_act
```

- `L_rec` — cross-entropy: did we rebuild the original peptide?
- `L_KL` — keeps the latent space continuous and samplable.
- `L_act` — binary cross-entropy from a small classifier reading `z`. This
  forces activity to be *represented in the latent space* during training
  rather than filtered for afterwards. It is the specific thing HydrAMP has
  that PepVAE does not.

**What we took from HydrAMP and what we dropped.** HydrAMP (Szymczak et al.
2023) is written in TensorFlow for *antimicrobial* peptides. We reimplemented
the ideas in PyTorch for *anticancer* peptides:

| | |
|---|---|
| **Kept** | encoder → mu/logvar, reparameterisation, label joined to `z`, rec + β·KL loss, the activity term |
| **Dropped** | the potency/MIC branch — we have one binary label, not a concentration |
| **Dropped** | Gumbel-softmax — only needed to push gradients through discrete samples; our classifier reads `z`, which is already continuous |
| **Simplified** | their autoregressive decoder became a plain GRU, for readability (this costs us accuracy — see Limitations) |

We also added a **positional embedding** in the decoder. Without it the decoder
receives an identical input at every step and has to work out "where am I in the
peptide" from the GRU's internal dynamics alone, which it does badly. Adding it
took training accuracy from 78% to 91%.

### Step 4 — Training (`src/train.py`)

Two things to watch, both printed every ten epochs:

**Reconstruction accuracy** — the fraction of amino acids put back correctly.
Chance is about 5% (1 in 21).

**Posterior collapse** — the classic VAE failure. If the KL term bites too hard
too early, the encoder gives up and returns the same distribution for every
peptide. `z` then carries no information and the decoder emits the same generic
sequence forever. The warning signs are KL falling toward zero and *active
units* dropping. The fix is **KL annealing**: hold β at 0 for the first 20
epochs so the model learns to reconstruct first, then raise it gradually.

We never saw collapse — active units stayed at the full latent dimension in
every run.

### Step 5 — Generation (`src/generate.py`)

Generation is three lines: draw `z` from `N(0, I)`, set `y = 1`, decode. No
encoder and no dataset are needed — that is the entire point of forcing the
latent space to match a known distribution.

Everything else in that file is *checking*:

- **validity** — right letters, sensible length
- **uniqueness** — distinct peptides, or does it repeat itself?
- **novelty** — genuinely new, or copied from training? We use sequence
  similarity rather than exact matching, because a peptide that changes one
  residue of a training peptide is not new
- **properties** — net charge and hydrophobicity against real ACPs
- **a baseline** — random peptides with the *same amino-acid composition and
  lengths* as real actives. That baseline knows which letters are common but
  nothing about their order. If the cVAE cannot beat it, it has not learned
  anything about sequence.

### Step 6 — Scoring (`src/score_with_anticp.py`)

AntiCP 2.0, unchanged, model 1 (ACP vs AMP — matching our dataset). Candidates
are ranked by its score.

### Step 7 — Latent analysis (`src/latent_analysis.py`)

PCA to see the shape of the space, k-means (k=2) to ask whether the
active/inactive split is something the model found *on its own*. Scored with
adjusted Rand index (do clusters match the labels?) and silhouette (are the
clusters tight?).

---

## Setup

You need **Python 3.12**. On a Mac, `python3` is often 3.9, which is too old —
check with `python3 -V`.

```bash
cd ~/acp-cvae
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Runs on CPU or Apple Silicon GPU (MPS). No CUDA, no TensorFlow.

### AntiCP 2.0 needs its own environment

`pip install anticp2` **does not work** — the PyPI package requires Python ≤ 3.7.

Use the standalone from <https://webs.iiitd.edu.in/raghava/anticp2/>. Its model
files were pickled with scikit-learn 0.21 and cannot be loaded by a modern
version (`ModuleNotFoundError: sklearn.ensemble.forest`), so it gets a separate
Python 3.7 environment running under Rosetta:

```bash
CONDA_SUBDIR=osx-64 conda create -y -n anticp2 python=3.7 "scikit-learn=0.21.3" numpy scipy
```

Our scripts call it automatically. If your standalone lives somewhere other
than `~/Downloads/anticp2`, pass `--anticp-dir`.

> **Always give AntiCP 2.0 FASTA input.** Its plain-text reader measures each
> line *including the newline character*, so a 50-residue peptide counts as 51
> and is silently dropped — no warning, it just vanishes. The FASTA reader is
> correct. Our code always writes FASTA.

---

## Running it

```bash
python src/prepare_data.py                     # clean the data

python src/train.py --epochs 600 --latent-dim 32 --hidden-dim 256 \
       --dropout 0.2 --beta-max 0.2 --word-dropout 0.3 --save-as cvae_ar.pt

python src/generate.py --n 600 --threshold 0.99 --checkpoint cvae_ar.pt
python src/score_with_anticp.py                # rank the shortlist
python src/evaluate_generation.py --n 400 --checkpoint cvae_ar.pt
python src/novelty_tradeoff.py                 # novelty vs activity curve
python src/latent_analysis.py --checkpoint cvae_ar.pt
```

Training takes about 6 minutes on an M1/M2.

`--threshold 0.99` turns on **latent screening**: sample `z` from the prior,
then keep only the ones the model's own activity head rates as confidently
active, before spending a decode on them. It needs no dataset, so generation
stays self-contained.

Or open `notebooks/pipeline.ipynb`, which runs all of it with explanations.

---

## Where it stands

**Final shortlist: 524 novel candidate peptides, 142 of them scoring ≥ 0.8**,
generated with no template and no dataset at generation time.

Generation quality — 99.8% valid, 99.7% unique, 87.8% novel, 9 exact copies out
of 600. Scored by AntiCP 2.0:

| Set | mean score | % ≥ 0.8 |
|---|---|---|
| cVAE asked for **active** | 0.641 | 26.4% |
| cVAE **active + latent screening** | **0.675** | 34.8% |
| cVAE asked for **inactive** | 0.522 | 9.8% |
| random, same composition | 0.552 | 4.3% |
| scrambled real actives | 0.649 | 29.1% |
| real actives, unmodified | 0.997 | 100% |

**Conditioning works: +0.118** between asking for active and asking for
inactive.

### Scoring only the novel peptides

A high score means nothing if the peptide is a copy. Analogue generation at low
noise scored 0.834 — but 121 of those 400 outputs were *exact copies* of
training peptides. So every headline number scores **novel peptides only** (not
a copy, under 90% 3-mer similarity to any training active):

| mode | novel % | mean (novel only) | candidates ≥ 0.8 |
|---|---|---|---|
| prior sampling | 88.3% | 0.614 | 94 |
| prior + screening 0.9 | 88.3% | 0.635 | 122 |
| **prior + screening 0.99** | **86.1%** | **0.647** | **131** |
| analogue, noise 1.6 | 89.8% | 0.633 | 96 |

Even counting only genuinely new peptides, we match scrambled real actives
(0.649) and clearly beat a composition-matched random baseline (0.552).

### Two caveats we will not hide

**The evaluator is partly a memory test.** Real actives score 0.997 because
they are AntiCP 2.0's own training data. Scramble them — same composition,
random order — and they fall to 0.649, yet 74% are *still* called active. On
novel sequences AntiCP leans heavily on composition.

**Our best candidates are variations, not inventions.** The top of the
shortlist sits at 0.78–0.89 similarity to training actives — below our novelty
cutoff, but recognisably related to known motifs (the `FAKKLAKK…` family echoes
the KLAKLAK peptides in the training set). That is what analogue-style peptide
design is *for*, but it is not discovery from nothing.

**Latent space:** ARI 0.052, silhouette 0.062, label agreement 61.6%. k-means
does not recover the activity split, even though activity *is* linearly
decodable from `z` (the internal activity head reaches 77.6%). Activity simply
is not the largest axis of variation. That is a reportable result, not a bug.

Every number, including the failures and the earlier non-autoregressive model,
is in [RUN_LOG.md](RUN_LOG.md).

## Limitations

1. **No wet-lab validation.** These are computational candidates, not drugs.
2. **The evaluator is not independent** — AntiCP 2.0 was trained on this dataset.
3. **The training set is small** — 502 active peptides after the length cap.
4. **Near-duplicates exist** between the published train and test splits (66
   training sequences share ≥90% of their 6-mers with a validation sequence,
   despite zero exact overlap), so held-out numbers read slightly optimistic.
5. **Free-running reconstruction is 55%** on held-out peptides. Interestingly
   this barely moved when we rebuilt the decoder (it was 56% before), yet the
   generated peptides improved a lot — so reconstruction accuracy is a poor
   proxy for generation quality here.
6. **The candidates are variations on known motifs**, not novel chemistry. See
   the caveat above.

## Layout

```
data/raw/          the four downloaded files, never modified
data/processed/    cleaned CSVs from src/prepare_data.py
src/               our code
notebooks/         pipeline.ipynb — runs everything with explanations
outputs/           model checkpoint, figures, generated peptides, scores
reference/         HydrAMP clone, read-only (git-ignored)
RUN_LOG.md         every experiment we ran, including the failures
```

## Credits

- **HydrAMP** — Szymczak, Mozejko, Grzegorzek et al., "Discovering highly potent
  antimicrobial peptides with deep generative model HydrAMP", *Nature
  Communications* 14:1453 (2023).
  <https://github.com/szczurek-lab/hydramp> · MIT licence, © 2022 University of
  Warsaw. Our cVAE follows their design; we reimplemented it in PyTorch.
- **AntiCP 2.0** — Agrawal P., Bhagat D., Mahalwal M., Sharma N., Raghava G.P.S.,
  "AntiCP 2.0: an updated model for predicting anticancer peptides",
  *Briefings in Bioinformatics* 22(3):bbaa153 (2021).
  doi:10.1093/bib/bbaa153. Used unmodified as our evaluator, and as the source
  of the dataset.

## AI assistance disclosure

This code was written with the help of an AI assistant (Claude). Both team
members have reviewed it and can explain how it works. The design decisions —
dropping the MIC branch, choosing 25 as the maximum length, dropping rather than
truncating long peptides, and adding the activity term back after measuring that
conditioning failed without it — were deliberate and are documented in the code
comments and in `RUN_LOG.md`.
