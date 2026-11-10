"""The novelty / activity trade-off, measured honestly.

Analogue generation starts from a real active peptide and drifts away from it
by `noise`. Small noise gives peptides that score well because they are nearly
copies of the training data -- which is worthless. Large noise gives genuinely
new peptides that may be worse.

HydrAMP exposes the same trade-off through a temperature parameter. This script
measures it, and the rule it enforces is simple:

    SCORE ONLY THE NOVEL ONES.

A peptide counts as novel if it is not an exact copy of a training sequence AND
shares less than 90% of its 3-mers with any of them. Anything else is
memorisation, and including it in the average would flatter us dishonestly.

Run:
    python src/novelty_tradeoff.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from generate import MIN_SCORABLE, max_similarity_to_training
from model import ConditionalVAE
from peptides import MAX_LEN, decode, load_split, pick_device
from score_with_anticp import DEFAULT_ANTICP_DIR, DEFAULT_ANTICP_PYTHON, run_anticp

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

NOVELTY_CUTOFF = 0.9


def score(sequences, tmp: Path, tag: str, anticp_dir, python) -> pd.DataFrame:
    fasta = tmp / f"{tag}.fasta"
    with open(fasta, "w") as fh:
        for i, s in enumerate(sequences):
            fh.write(f">s{i}\n{s}\n")
    return run_anticp(fasta, tmp / f"{tag}.csv", anticp_dir, python)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="cvae_ar.pt")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--noise", type=float, nargs="+",
                        default=[0.3, 0.5, 0.8, 1.2, 1.6, 2.0])
    parser.add_argument("--anticp-dir", type=Path, default=DEFAULT_ANTICP_DIR)
    parser.add_argument("--anticp-python", type=Path, default=DEFAULT_ANTICP_PYTHON)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()

    checkpoint = torch.load(OUTPUTS / args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = ConditionalVAE(
        max_len=config["max_len"], latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"], dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x, y, df = load_split("train", MAX_LEN)
    actives = df[df["label"] == 1]["sequence"].tolist()
    active_set = set(actives)
    templates = x[(y.squeeze(1) == 1)][:args.n]
    ones = torch.ones(len(templates), 1)

    tmp = OUTPUTS / "_tradeoff"
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"Scoring only peptides that are NOVEL "
          f"(not a copy, <{NOVELTY_CUTOFF:.0%} similar to any training active)\n")
    print(f"{'noise':>6} {'generated':>10} {'novel':>7} {'novel %':>8} "
          f"{'mean score':>11} {'>=0.5':>7} {'>=0.8':>7} {'usable':>7}")

    rows = []
    for noise in args.noise:
        sequences = [decode(r) for r in
                     model.generate_analogue(templates, ones, device, noise=noise)]
        unique = sorted({s for s in sequences if len(s) >= MIN_SCORABLE})

        novel = [s for s in unique
                 if s not in active_set
                 and max_similarity_to_training(s, actives) < NOVELTY_CUTOFF]
        if not novel:
            print(f"{noise:>6} {len(sequences):>10} {0:>7} {0.0:>7.1f}%  (none novel)")
            continue

        scores = score(novel, tmp, f"noise{noise}", args.anticp_dir, args.anticp_python)
        s = scores["Score"]

        # "usable" = novel AND confidently active. This is the number that
        # actually matters: how many candidates could we hand over?
        usable = int((s >= 0.8).sum())

        row = {
            "noise": noise,
            "generated": len(sequences),
            "unique": len(unique),
            "novel": len(novel),
            "novel_pct": round(100 * len(novel) / len(unique), 1),
            "mean_score_novel": round(float(s.mean()), 4),
            "pct_active_0.5": round(100 * float((s >= 0.5).mean()), 1),
            "pct_strong_0.8": round(100 * float((s >= 0.8).mean()), 1),
            "usable_candidates": usable,
        }
        rows.append(row)
        print(f"{noise:>6} {row['generated']:>10} {row['novel']:>7} "
              f"{row['novel_pct']:>7.1f}% {row['mean_score_novel']:>11.3f} "
              f"{row['pct_active_0.5']:>6.1f}% {row['pct_strong_0.8']:>6.1f}% "
              f"{usable:>7}")

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUTS / "novelty_tradeoff.csv", index=False)

    if not table.empty:
        best = table.loc[table["usable_candidates"].idxmax()]
        print(f"\nBest operating point: noise={best['noise']} -> "
              f"{int(best['usable_candidates'])} novel candidates scoring >= 0.8 "
              f"({best['novel_pct']:.0f}% of output was novel)")

        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(table["noise"], table["mean_score_novel"], "o-",
                 color="#8c3a52", label="mean AntiCP score (novel only)")
        ax1.set_xlabel("latent noise")
        ax1.set_ylabel("mean AntiCP score", color="#8c3a52")
        ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()
        ax2.plot(table["noise"], table["novel_pct"], "s--",
                 color="#4a6fa5", label="% novel")
        ax2.set_ylabel("% of output that is novel", color="#4a6fa5")
        ax2.set_ylim(0, 100)

        fig.suptitle("Novelty vs predicted activity")
        fig.tight_layout()
        fig.savefig(OUTPUTS / "novelty_tradeoff.png", dpi=150)
        plt.close(fig)
        print("Wrote outputs/novelty_tradeoff.csv and novelty_tradeoff.png")


if __name__ == "__main__":
    main()
