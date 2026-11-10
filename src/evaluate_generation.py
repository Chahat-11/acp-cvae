"""The headline experiment: does the conditioning actually do anything?

A conditional VAE is only worth building if the label changes the output. So we
generate three sets of peptides and score all of them with AntiCP 2.0 under
identical conditions:

  y = 1   ask the model for anticancer peptides
  y = 0   ask the same model for non-anticancer peptides
  random  composition baseline -- same amino-acid frequencies and lengths as
          the real actives, but the letters shuffled into random order

Two questions, and they are different:

  1. Does the label work?       y=1 should score higher than y=0.
     If the two are the same, the decoder is ignoring the condition and we have
     an ordinary VAE wearing a cVAE's clothes.

  2. Did it learn sequence?     y=1 should beat the composition baseline.
     The baseline already knows which amino acids anticancer peptides are made
     of. Beating it means the model learned something about ORDER, which is the
     part that actually carries activity.

Run:
    python src/evaluate_generation.py --n 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generate import MIN_SCORABLE, composition_baseline, hydrophobicity, net_charge
from model import ConditionalVAE
from peptides import MAX_LEN, decode, load_split, pick_device
from score_with_anticp import DEFAULT_ANTICP_DIR, DEFAULT_ANTICP_PYTHON, run_anticp

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"


def write_fasta(sequences: list[str], path: Path) -> None:
    with open(path, "w") as fh:
        for i, sequence in enumerate(sequences):
            fh.write(f">s{i}\n{sequence}\n")


def score_set(name: str, sequences: list[str], tmp: Path,
              anticp_dir: Path, python: Path) -> pd.DataFrame:
    usable = [s for s in sequences if len(s) >= MIN_SCORABLE]
    fasta = tmp / f"{name}.fasta"
    write_fasta(usable, fasta)
    scores = run_anticp(fasta, tmp / f"{name}_scores.csv", anticp_dir, python)
    scores["set"] = name
    return scores


def summarise(name: str, scores: pd.DataFrame) -> dict:
    s = scores["Score"]
    return {
        "set": name,
        "n_scored": len(s),
        "mean_score": round(float(s.mean()), 4),
        "median_score": round(float(s.median()), 4),
        "pct_active_0.5": round(100 * float((s >= 0.5).mean()), 1),
        "pct_strong_0.8": round(100 * float((s >= 0.8).mean()), 1),
        "mean_charge": round(float(np.mean([net_charge(x) for x in scores["Sequence"]])), 2),
        "mean_hydrophobicity": round(
            float(np.mean([hydrophobicity(x) for x in scores["Sequence"]])), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--checkpoint", default="cvae.pt")
    parser.add_argument("--anticp-dir", type=Path, default=DEFAULT_ANTICP_DIR)
    parser.add_argument("--anticp-python", type=Path, default=DEFAULT_ANTICP_PYTHON)
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
    print(f"Checkpoint: {args.checkpoint} (latent {config['latent_dim']})\n")

    _, _, train_df = load_split("train", MAX_LEN)
    actives = train_df[train_df["label"] == 1]["sequence"].tolist()

    sets = {
        "cvae_y1": [decode(r) for r in model.generate(args.n, 1.0, device)],
        "cvae_y0": [decode(r) for r in model.generate(args.n, 0.0, device)],
        "baseline_random": composition_baseline(actives, args.n, rng),
        "real_actives": actives,
    }

    tmp = OUTPUTS / "_scoring"
    tmp.mkdir(parents=True, exist_ok=True)

    print("Scoring each set with AntiCP 2.0 (model 1: ACP vs AMP)")
    all_scores, rows = [], []
    for name, sequences in sets.items():
        scores = score_set(name, sequences, tmp, args.anticp_dir, args.anticp_python)
        all_scores.append(scores)
        rows.append(summarise(name, scores))
        print(f"  {name:<16} scored {len(scores):>4}")

    table = pd.DataFrame(rows)
    print("\nRESULTS")
    print(table.to_string(index=False))

    pd.concat(all_scores).to_csv(OUTPUTS / "evaluation_raw_scores.csv", index=False)
    table.to_csv(OUTPUTS / "evaluation_summary.csv", index=False)

    # ---- the two questions, answered ----
    by_name = {r["set"]: r for r in rows}
    y1, y0 = by_name["cvae_y1"], by_name["cvae_y0"]
    base = by_name["baseline_random"]

    conditioning_gap = y1["mean_score"] - y0["mean_score"]
    baseline_gap = y1["mean_score"] - base["mean_score"]

    print("\n1. DOES THE LABEL WORK?")
    print(f"   y=1 mean {y1['mean_score']:.3f}   vs   y=0 mean {y0['mean_score']:.3f}"
          f"   gap {conditioning_gap:+.3f}")
    if conditioning_gap > 0.10:
        print("   Yes -- asking for active peptides gives measurably more active peptides.")
    elif conditioning_gap > 0.02:
        print("   Weakly. The label nudges the output but does not control it.")
    else:
        print("   No. The decoder is ignoring the label; this is effectively a plain VAE.")

    print("\n2. DID IT LEARN SEQUENCE, OR ONLY COMPOSITION?")
    print(f"   y=1 mean {y1['mean_score']:.3f}   vs   random-order baseline "
          f"{base['mean_score']:.3f}   gap {baseline_gap:+.3f}")
    if baseline_gap > 0.10:
        print("   Yes -- it beats a baseline with identical composition, so it learned order.")
    elif baseline_gap > -0.02:
        print("   Not clearly. Most of the signal may be amino-acid composition alone.")
    else:
        print("   No -- the random baseline scores HIGHER. The model is not adding value.")

    (OUTPUTS / "evaluation_verdict.json").write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "conditioning_gap": round(conditioning_gap, 4),
        "baseline_gap": round(baseline_gap, 4),
        "summary": rows,
    }, indent=2))
    print("\nWrote outputs/evaluation_summary.csv and outputs/evaluation_verdict.json")


if __name__ == "__main__":
    main()
