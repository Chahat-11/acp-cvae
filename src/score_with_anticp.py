"""Score generated peptides with AntiCP 2.0 and rank them.

We do not train our own activity classifier. AntiCP 2.0 is a published,
peer-reviewed predictor and we use it as-is.

Two awkward things about the standalone tool, both handled here:

1. It loads its model files by RELATIVE path, so it only works when the working
   directory is its own folder. We cd into it.
2. It cannot run on modern scikit-learn -- its pickles reference
   sklearn.ensemble.forest, a module renamed in scikit-learn 0.22. It needs its
   own Python 3.7 environment (see README).

We always hand it FASTA. Its plain-text reader measures each line including the
newline character, so a 50-residue peptide is counted as 51 and silently
dropped. FASTA avoids that bug entirely.

One honest caveat: AntiCP 2.0 was trained on this same dataset, so it is not an
independent judge of peptides drawn from this distribution. It is the standard
evaluator in this literature, and it is the right tool for ranking novel
candidates, but it is not external validation.

Run:
    python src/score_with_anticp.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

DEFAULT_ANTICP_DIR = Path.home() / "Downloads" / "anticp2"
DEFAULT_ANTICP_PYTHON = Path("/opt/anaconda3/envs/anticp2/bin/python")


def run_anticp(fasta: Path, output: Path, anticp_dir: Path, python: Path,
               model: int = 1) -> pd.DataFrame:
    """Call the standalone and read its CSV back.

    model 1 = ACP vs AMP, which matches the 'main' dataset we trained on.
    -d 2 means report every peptide, not only the ones it calls active.
    """
    if not anticp_dir.is_dir():
        sys.exit(f"AntiCP 2.0 folder not found: {anticp_dir}")
    if not python.exists():
        sys.exit(
            f"AntiCP 2.0 Python not found: {python}\n"
            "Create it with:\n"
            '  CONDA_SUBDIR=osx-64 conda create -y -n anticp2 python=3.7 '
            '"scikit-learn=0.21.3" numpy scipy'
        )

    result = subprocess.run(
        [str(python), "anticp2.py", "-i", str(fasta.resolve()),
         "-o", str(output.resolve()), "-m", str(model), "-d", "2"],
        cwd=anticp_dir,          # it loads its models by relative path
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"AntiCP 2.0 failed:\n{result.stdout}\n{result.stderr}")

    scores = pd.read_csv(output)
    scores.columns = [c.strip().lstrip("# ") for c in scores.columns]
    scores["Sequence"] = scores["Sequence"].str.strip()
    scores["Prediction"] = scores["Prediction"].str.strip()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path,
                        default=OUTPUTS / "generated_candidates.fasta")
    parser.add_argument("--candidates", type=Path,
                        default=OUTPUTS / "generated_candidates.csv")
    parser.add_argument("--anticp-dir", type=Path, default=DEFAULT_ANTICP_DIR)
    parser.add_argument("--anticp-python", type=Path, default=DEFAULT_ANTICP_PYTHON)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.fasta.exists():
        sys.exit(f"No candidates found at {args.fasta}. Run src/generate.py first.")

    print(f"Scoring {args.fasta.name} with AntiCP 2.0 (model 1: ACP vs AMP)")
    scores = run_anticp(args.fasta, OUTPUTS / "anticp_raw.csv",
                        args.anticp_dir, args.anticp_python)

    candidates = pd.read_csv(args.candidates)
    merged = candidates.merge(
        scores[["Sequence", "Score", "Prediction"]],
        left_on="sequence", right_on="Sequence", how="left",
    ).drop(columns=["Sequence"])

    skipped = merged["Score"].isna().sum()
    if skipped:
        print(f"  {skipped} peptide(s) were not scored "
              f"(AntiCP 2.0 skips anything under 5 residues)")
    merged = merged.dropna(subset=["Score"])

    merged = merged.sort_values("Score", ascending=False).reset_index(drop=True)
    merged.insert(0, "rank", merged.index + 1)
    merged.to_csv(OUTPUTS / "ranked_candidates.csv", index=False)

    active = merged[merged["Score"] >= args.threshold]
    strong = merged[merged["Score"] >= 0.8]

    print(f"\n  scored          {len(merged)}")
    print(f"  mean score      {merged['Score'].mean():.3f}")
    print(f"  >= {args.threshold} (active)  {len(active)} "
          f"({100 * len(active) / max(1, len(merged)):.1f}%)")
    print(f"  >= 0.8 (strong) {len(strong)} "
          f"({100 * len(strong) / max(1, len(merged)):.1f}%)")

    print("\nTOP 15 CANDIDATES")
    top = merged.head(15)
    print(top[["rank", "sequence", "length", "net_charge",
               "hydrophobicity", "max_similarity_to_train", "Score"]]
          .to_string(index=False))

    print(f"\nWrote outputs/ranked_candidates.csv")


if __name__ == "__main__":
    main()
