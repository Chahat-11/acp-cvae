"""Generate new peptides and check whether they are any good.

Generation itself is three lines: sample z from N(0, I), set y = 1, decode.
Everything else in this file is checking.

Five checks, deliberately independent:
  validity     -- is it a sensible peptide at all (right letters, sane length)?
  uniqueness   -- how many distinct peptides did we get, or does it repeat?
  novelty      -- is it actually new, or copied from the training set?
  activity     -- what does AntiCP 2.0 say? (added by score_with_anticp.py)
  properties   -- net charge and hydrophobicity, compared with real ACPs

The baseline matters. We compare against random sequences built from the same
amino-acid composition as the real anticancer peptides. If the cVAE cannot beat
that, it has not learned anything about ordering -- only about which letters are
common, which is a much weaker claim.

Run:
    python src/generate.py --n 500
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import ConditionalVAE
from peptides import AMINO_ACIDS, MAX_LEN, decode, load_split, pick_device

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

# Kyte-Doolittle hydrophobicity, the standard scale.
KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# AntiCP 2.0 only scores peptides of 5 to 50 residues (see README), so anything
# shorter cannot be evaluated and we do not count it as a usable candidate.
MIN_SCORABLE = 5


def net_charge(sequence: str) -> float:
    """Rough charge at pH 7: K and R are +1, D and E are -1, H counts as +0.1."""
    return (sequence.count("K") + sequence.count("R")
            - sequence.count("D") - sequence.count("E")
            + 0.1 * sequence.count("H"))


def hydrophobicity(sequence: str) -> float:
    """Mean Kyte-Doolittle value over the peptide."""
    if not sequence:
        return 0.0
    return sum(KYTE_DOOLITTLE[aa] for aa in sequence) / len(sequence)


def similarity(a: str, b: str) -> float:
    """Shared 3-mers as a fraction of the shorter peptide's 3-mers.

    A cheap stand-in for sequence identity. We use it for the novelty check:
    a peptide that merely changes one residue of a training peptide is not new.
    """
    if len(a) < 3 or len(b) < 3:
        return 1.0 if a == b else 0.0
    ka = {a[i:i + 3] for i in range(len(a) - 2)}
    kb = {b[i:i + 3] for i in range(len(b) - 2)}
    return len(ka & kb) / min(len(ka), len(kb))


def max_similarity_to_training(sequence: str, training: list[str]) -> float:
    return max((similarity(sequence, t) for t in training), default=0.0)


def composition_baseline(training: list[str], n: int, rng) -> list[str]:
    """Random peptides with the same amino-acid composition and length profile.

    This is the control. It knows which letters are common and how long peptides
    tend to be, but nothing about the order they come in.
    """
    letters = Counter("".join(training))
    alphabet = list(letters)
    weights = np.array([letters[a] for a in alphabet], dtype=float)
    weights /= weights.sum()

    lengths = [len(t) for t in training]
    return ["".join(rng.choice(alphabet, size=rng.choice(lengths), p=weights))
            for _ in range(n)]


def describe(name: str, sequences: list[str], training: list[str]) -> dict:
    """Run the checks on one set of peptides and print a row."""
    valid = [s for s in sequences if len(s) >= MIN_SCORABLE]
    unique = sorted(set(valid))

    # novel = not an exact copy, and not a near-copy either
    exact_copies = sum(1 for s in unique if s in set(training))
    similarities = [max_similarity_to_training(s, training) for s in unique]
    novel = [s for s, sim in zip(unique, similarities) if sim < 0.9]

    stats = {
        "set": name,
        "generated": len(sequences),
        "valid": len(valid),
        "valid_pct": round(100 * len(valid) / max(1, len(sequences)), 1),
        "unique": len(unique),
        "unique_pct": round(100 * len(unique) / max(1, len(valid)), 1),
        "exact_copies": exact_copies,
        "novel": len(novel),
        "novel_pct": round(100 * len(novel) / max(1, len(unique)), 1),
        "mean_similarity_to_train": round(float(np.mean(similarities)) if similarities else 0.0, 3),
        "mean_length": round(float(np.mean([len(s) for s in valid])) if valid else 0.0, 1),
        "mean_charge": round(float(np.mean([net_charge(s) for s in valid])) if valid else 0.0, 2),
        "mean_hydrophobicity": round(float(np.mean([hydrophobicity(s) for s in valid])) if valid else 0.0, 2),
    }

    print(f"  {name:<22} valid {stats['valid_pct']:>5.1f}%   "
          f"unique {stats['unique_pct']:>5.1f}%   "
          f"novel {stats['novel_pct']:>5.1f}%   "
          f"copies {exact_copies:>3}   "
          f"len {stats['mean_length']:>4.1f}   "
          f"charge {stats['mean_charge']:>+5.2f}   "
          f"hyd {stats['mean_hydrophobicity']:>+5.2f}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="how many to sample")
    parser.add_argument("--checkpoint", default="cvae.pt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device()

    checkpoint = torch.load(OUTPUTS / args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = ConditionalVAE(
        max_len=config["max_len"], latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"], dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Loaded {args.checkpoint} (latent {config['latent_dim']})\n")

    print("Loading training peptides (for the novelty check)")
    _, _, train_df = load_split("train", MAX_LEN)
    actives = train_df[train_df["label"] == 1]["sequence"].tolist()
    print()

    # ---- generate ----
    active = [decode(row) for row in
              model.generate(args.n, label=1.0, device=device, temperature=args.temperature)]
    inactive = [decode(row) for row in
                model.generate(args.n, label=0.0, device=device, temperature=args.temperature)]
    baseline = composition_baseline(actives, args.n, rng)

    print("RESULTS")
    rows = [
        describe("cVAE (y=1, active)", active, actives),
        describe("cVAE (y=0, inactive)", inactive, actives),
        describe("composition baseline", baseline, actives),
        describe("real training actives", actives, actives),
    ]

    # ---- save ----
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUTS / "generation_summary.csv", index=False)

    candidates = sorted({s for s in active if len(s) >= MIN_SCORABLE})
    candidates = [s for s in candidates
                  if max_similarity_to_training(s, actives) < 0.9]
    frame = pd.DataFrame({
        "sequence": candidates,
        "length": [len(s) for s in candidates],
        "net_charge": [net_charge(s) for s in candidates],
        "hydrophobicity": [hydrophobicity(s) for s in candidates],
        "max_similarity_to_train": [
            round(max_similarity_to_training(s, actives), 3) for s in candidates
        ],
    })
    frame.to_csv(OUTPUTS / "generated_candidates.csv", index=False)

    # FASTA for AntiCP 2.0. It must be FASTA: the plain-text reader has an
    # off-by-one bug that silently drops 50-residue peptides.
    with open(OUTPUTS / "generated_candidates.fasta", "w") as fh:
        for i, sequence in enumerate(candidates):
            fh.write(f">cand_{i}\n{sequence}\n")

    print(f"\n{len(candidates)} novel candidates written to:")
    print("  outputs/generated_candidates.csv")
    print("  outputs/generated_candidates.fasta  (input for AntiCP 2.0)")
    print("  outputs/generation_summary.csv")

    if frame.empty:
        print("\nWARNING: no novel candidates survived the filters.")


if __name__ == "__main__":
    main()
