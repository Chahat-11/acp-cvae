"""Turn the four raw AntiCP 2.0 files into clean train/val CSVs.

The raw files are left untouched. Everything this script produces goes into
data/processed/ and outputs/.

What "clean" means here, in order:
  1. read the file (handles both FASTA and one-sequence-per-line)
  2. uppercase, strip whitespace
  3. drop sequences containing anything other than the 20 standard amino acids
  4. drop duplicate sequences
  5. drop sequences that appear with BOTH labels (we cannot trust either)
  6. drop any training sequence that also appears in validation

Step 6 matters most. If a peptide is in both train and val, then anything we
measure on val is partly measuring memorisation, not learning.

Run:
    python src/prepare_data.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to file, do not open a window
import matplotlib.pyplot as plt
import pandas as pd

# The 20 standard amino acids. Anything else (B, J, O, U, X, Z, digits,
# punctuation) means the sequence is ambiguous or not a plain peptide.
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed"
OUTPUTS = REPO / "outputs"

# Which raw file holds what. label 1 = anticancer, 0 = not anticancer.
SOURCES = {
    "train": [("pos_train_main", 1), ("neg_train_main", 0)],
    "val": [("pos_test_main", 1), ("neg_test_main", 0)],
}


def read_sequences(path: Path) -> list[str]:
    """Read one sequence per line, skipping FASTA headers if there are any."""
    sequences = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue  # blank line or FASTA header
        sequences.append(line.upper())
    return sequences


def report_raw(name: str, sequences: list[str]) -> None:
    lengths = pd.Series([len(s) for s in sequences])
    bad = [s for s in sequences if set(s) - STANDARD_AA]
    duplicates = len(sequences) - len(set(sequences))

    print(f"  {name:<16} {len(sequences):>4} sequences   "
          f"length {lengths.min()}-{lengths.max()} "
          f"(median {lengths.median():.0f})   "
          f"{duplicates} duplicate(s)   {len(bad)} with odd residues")

    if bad:
        odd = Counter(c for s in bad for c in set(s) - STANDARD_AA)
        print(f"{'':<18} odd residues seen: {dict(odd)}")


def build(part: str) -> tuple[pd.DataFrame, dict]:
    """Load one split, clean it, and return the frame plus a record of what was dropped."""
    rows, notes = [], {}

    for filename, label in SOURCES[part]:
        sequences = read_sequences(RAW / filename)
        report_raw(filename, sequences)
        rows += [{"sequence": s, "label": label} for s in sequences]

    df = pd.DataFrame(rows)
    notes["loaded"] = len(df)

    # 3. non-standard amino acids
    is_clean = df["sequence"].apply(lambda s: not (set(s) - STANDARD_AA))
    dropped = df[~is_clean]
    if len(dropped):
        for _, row in dropped.iterrows():
            odd = "".join(sorted(set(row["sequence"]) - STANDARD_AA))
            print(f"    dropping (non-standard '{odd}'): {row['sequence'][:40]}")
    df = df[is_clean]
    notes["dropped_non_standard"] = int((~is_clean).sum())

    # 5. same sequence with both labels -- ambiguous, so remove it entirely
    label_count = df.groupby("sequence")["label"].nunique()
    conflicted = set(label_count[label_count > 1].index)
    if conflicted:
        print(f"    dropping {len(conflicted)} sequence(s) labelled both 1 and 0")
    df = df[~df["sequence"].isin(conflicted)]
    notes["dropped_conflicting_label"] = len(conflicted)

    # 4. exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset="sequence")
    notes["dropped_duplicate"] = before - len(df)

    df = df.reset_index(drop=True)
    df["length"] = df["sequence"].str.len()
    return df[["sequence", "label", "length"]], notes


def histogram(train: pd.DataFrame, val: pd.DataFrame) -> Path:
    """Length distribution, split by label, to help choose the model's max length."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, (name, df) in zip(axes, [("train", train), ("val", val)]):
        for label, colour, text in [(1, "#8c3a52", "anticancer"), (0, "#b0a394", "not anticancer")]:
            ax.hist(df[df["label"] == label]["length"], bins=range(0, 55, 2),
                    alpha=0.7, color=colour, label=text)
        ax.set_title(f"{name}  (n={len(df)})")
        ax.set_xlabel("peptide length")
        ax.legend(frameon=False)

    axes[0].set_ylabel("count")
    fig.suptitle("Peptide length distribution")
    fig.tight_layout()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / "length_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def length_summary(df: pd.DataFrame, name: str) -> None:
    lengths = df["length"]
    print(f"  {name:<6} min {lengths.min():>3}   median {lengths.median():>5.1f}   "
          f"mean {lengths.mean():>5.1f}   max {lengths.max():>3}")
    for q in (0.90, 0.95, 0.99):
        print(f"{'':<9} {q:.0%} of sequences are <= {lengths.quantile(q):.0f} residues")


def main() -> None:
    print("RAW FILES")
    train, train_notes = build("train")
    val, val_notes = build("val")

    # 6. remove any training sequence that also shows up in validation
    overlap = set(train["sequence"]) & set(val["sequence"])
    if overlap:
        print(f"\n  {len(overlap)} sequence(s) appear in BOTH train and val -- "
              f"removing them from train")
        train = train[~train["sequence"].isin(overlap)].reset_index(drop=True)
    else:
        print("\n  no overlap between train and val")
    train_notes["dropped_in_val"] = len(overlap)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    val.to_csv(PROCESSED / "val.csv", index=False)

    train_pos = train[train["label"] == 1].reset_index(drop=True)
    train_pos.to_csv(PROCESSED / "train_pos_only.csv", index=False)

    print("\nWHAT WAS DROPPED")
    for part, notes in [("train", train_notes), ("val", val_notes)]:
        kept = len(train) if part == "train" else len(val)
        print(f"  {part}: loaded {notes['loaded']} -> kept {kept}")
        for reason, n in notes.items():
            if reason != "loaded" and n:
                print(f"      -{n} {reason.replace('_', ' ')}")

    print("\nFINAL COUNTS")
    for name, df in [("train", train), ("val", val), ("train_pos_only", train_pos)]:
        counts = df["label"].value_counts().to_dict()
        print(f"  {name:<15} {len(df):>4} rows   "
              f"label 1: {counts.get(1, 0):>3}   label 0: {counts.get(0, 0):>3}")

    print("\nLENGTH DISTRIBUTION")
    length_summary(train, "train")
    length_summary(val, "val")

    print("\nHEAD OF EACH FILE")
    for name, df in [("train.csv", train), ("val.csv", val), ("train_pos_only.csv", train_pos)]:
        print(f"\n  {name}")
        print(df.head().to_string(index=False).replace("\n", "\n  "))

    path = histogram(train, val)
    print(f"\nWrote {path.relative_to(REPO)}")
    print(f"Wrote {PROCESSED.relative_to(REPO)}/train.csv, val.csv, train_pos_only.csv")


if __name__ == "__main__":
    main()
