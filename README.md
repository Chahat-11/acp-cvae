# acp-cvae — generating anticancer peptides with a small conditional VAE

DSC4804 Statistical Learning, Group 20.

## What this project does

Anticancer peptides (ACPs) are short chains of amino acids that can kill cancer
cells. Finding new ones in a laboratory is slow, so we use a computer to propose
candidates instead.

The project has three parts:

1. **Generator.** A small conditional variational autoencoder (cVAE) that writes
   new peptide sequences. "Conditional" means we can ask it for a specific kind
   of peptide — we set the label `y = 1` (anticancer) and it generates
   accordingly, rather than producing random peptides and hoping.
2. **Evaluator.** We do **not** train our own activity classifier. We score the
   generated peptides with **AntiCP 2.0**, an existing published predictor.
3. **Analysis.** PCA and k-means on the cVAE's latent space, to see whether
   active and inactive peptides end up in different regions.

Scope is deliberately small: the point is to understand every part of it.

## Status

Data, model, generation, scoring and latent analysis all run end to end.

The honest summary: the pipeline works, but the generated peptides are not yet
convincing. The cVAE beats a random composition-matched baseline by only +0.024
mean AntiCP score, and scores *below* scrambled real peptides. Reconstruction
plateaus at 56% on held-out data. See [RUN_LOG.md](RUN_LOG.md) for every number
and [the limitations](#limitations).

## Setup

You need Python 3.12. On a Mac:

```bash
cd ~/acp-cvae
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This runs on CPU or Apple Silicon GPU (MPS). No CUDA, no TensorFlow.

### The AntiCP 2.0 evaluator needs its own environment

`pip install anticp2` **does not work** — the PyPI package requires Python ≤ 3.7,
which modern Macs do not ship. We use the standalone version instead, from
<https://webs.iiitd.edu.in/raghava/anticp2/>.

Its model files were pickled with scikit-learn 0.21, which needs Python 3.7, so
it gets a separate conda environment running under Rosetta:

```bash
CONDA_SUBDIR=osx-64 conda create -y -n anticp2 python=3.7 "scikit-learn=0.21.3" numpy scipy
```

Then run it from inside the standalone folder (it loads its models by relative
path, so the working directory matters):

```bash
cd ~/Downloads/anticp2
/opt/anaconda3/envs/anticp2/bin/python anticp2.py -i peptides.fa -o scores.csv -m 1 -d 2
```

**Always give it FASTA input.** Its plain-text reader has an off-by-one bug: it
measures each line including the newline character, so a 50-residue peptide
is counted as 51 and silently dropped. The FASTA reader does not have this
problem. Accepted length is 5–50 residues; anything longer is truncated to 50.

Model `-m 1` is the ACP-vs-AMP model, which matches the dataset we use.

## Running it

```bash
python src/prepare_data.py                                  # clean the data
python src/train.py --epochs 200 --latent-dim 32 --dropout 0.2 --beta-max 0.2
python src/generate.py --n 500                              # sample new peptides
python src/score_with_anticp.py                             # rank them
python src/evaluate_generation.py --n 400                   # the headline test
python src/latent_analysis.py                               # PCA + k-means
```

Or open `notebooks/pipeline.ipynb`, which runs all of it with explanations.

## Limitations

Worth stating plainly in the report:

1. **No wet-lab validation.** These are computational candidates, not drugs.
2. **The evaluator is not independent.** AntiCP 2.0 was trained on this dataset.
   Real actives score 0.997 from it; scrambling them (same composition, random
   order) drops them to 0.654, but 71.5% are *still* called active. So for novel
   sequences it is substantially composition-driven.
3. **The training set is small** — 502 active peptides after the length cap.
4. **Near-duplicates exist** between the published train and test splits, so
   held-out numbers read slightly optimistic.
5. **Conditioning is weak.** Asking for active peptides rather than inactive
   ones shifts the mean score by only +0.020.

## Data

AntiCP 2.0 "main" dataset, downloaded from
<https://webs.iiitd.edu.in/raghava/anticp2/>:

| File | Sequences | Meaning |
|---|---|---|
| `pos_train_main` | 689 | anticancer |
| `neg_train_main` | 689 | antimicrobial but not anticancer |
| `pos_test_main` | 172 | anticancer (validation) |
| `neg_test_main` | 172 | not anticancer (validation) |

The negatives are antimicrobial peptides, not random sequences. That makes the
task harder and more meaningful: the model has to learn what makes a peptide
*anticancer*, not merely what makes it a peptide.

We train on the train files only. The test files are used for checking.

## Layout

```
data/raw/          the four downloaded files, never modified
data/processed/    cleaned CSVs produced by src/prepare_data.py
src/               our code
notebooks/         one notebook that runs the whole pipeline
outputs/           figures and generated peptides
reference/         HydrAMP clone, for reading only (git-ignored)
```

## Credits

- **HydrAMP** — Szymczak et al., *Nature Communications* 14:1453 (2023),
  <https://github.com/szczurek-lab/hydramp>. Our cVAE design follows theirs
  (encoder → mu/sigma → reparameterisation → label concatenated with z →
  decoder). We dropped their potency/MIC branch and reimplemented in PyTorch.
- **AntiCP 2.0** — Agrawal P., Bhagat D., Mahalwal M., Sharma N., Raghava G.P.S.
  (2020), "AntiCP 2.0: an updated model for predicting anticancer peptides",
  *Briefings in Bioinformatics*, doi:10.1093/bib/bbaa153. Used as our evaluator
  and as the source of the dataset.

## AI assistance disclosure

This code was written with the help of an AI assistant (Claude). Both team
members have reviewed the code and can explain how it works. All design choices,
including the decision to drop the MIC branch and the choice of maximum sequence
length, were made deliberately and are documented in the code comments.

No wet-lab validation is involved. Generated peptides are computational
candidates only, not therapeutic compounds.
