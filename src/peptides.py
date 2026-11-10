"""Turning peptides into numbers and back again.

A peptide is a string like "FAKKFAKKFKKFAKK". The model cannot read letters, so
we map each amino acid to an integer, and pad every sequence to the same length.

  index 0      = PAD (empty position, used to fill short peptides)
  index 1..20  = the twenty standard amino acids
  index 21     = BOS, the "start of peptide" marker
  index 22     = UNK, used only by word dropout during training

BOS and UNK exist because the decoder is autoregressive: it writes one residue
at a time, and each step is fed the residue before it. The first step has no
previous residue, so it gets BOS instead. UNK is explained in model.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

PAD = 0
BOS = len(AMINO_ACIDS) + 1  # 21
UNK = len(AMINO_ACIDS) + 2  # 22
VOCAB_SIZE = len(AMINO_ACIDS) + 3  # 23

# Maximum peptide length the model handles. HydrAMP uses 25 and we follow them.
# Peptides longer than this are DROPPED, not cut short: a truncated anticancer
# peptide is not an anticancer peptide, and training on fragments would teach
# the model to produce fragments.
MAX_LEN = 25

# letter -> index, and back
AA_TO_INDEX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
INDEX_TO_AA = {i + 1: aa for i, aa in enumerate(AMINO_ACIDS)}

REPO = Path(__file__).resolve().parent.parent


def encode(sequence: str, max_len: int = MAX_LEN) -> list[int]:
    """"FAKK" -> [5, 1, 9, 9, 0, 0, ...] padded out to max_len."""
    indices = [AA_TO_INDEX[aa] for aa in sequence[:max_len]]
    return indices + [PAD] * (max_len - len(indices))


def decode(indices) -> str:
    """[5, 1, 9, 9, 0, 0, ...] -> "FAKK". Stops at the first PAD."""
    letters = []
    for i in indices:
        i = int(i)
        if i == PAD:
            break  # padding means the peptide has ended
        letters.append(INDEX_TO_AA[i])
    return "".join(letters)


def load_split(name: str, max_len: int = MAX_LEN) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Load train.csv or val.csv, drop anything longer than max_len, and encode.

    Returns (x, y, dataframe):
      x -- LongTensor [n, max_len] of amino acid indices
      y -- FloatTensor [n, 1] of labels (1.0 = anticancer, 0.0 = not)
    """
    df = pd.read_csv(REPO / "data" / "processed" / f"{name}.csv")

    kept = df[df["length"] <= max_len].reset_index(drop=True)
    dropped = len(df) - len(kept)
    print(f"  {name}: {len(kept)} peptides "
          f"({dropped} dropped for being longer than {max_len})")

    x = torch.tensor([encode(s, max_len) for s in kept["sequence"]], dtype=torch.long)
    y = torch.tensor(kept["label"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    return x, y, kept


def pick_device() -> torch.device:
    """Apple GPU if we have one, otherwise CPU. No CUDA on a Mac."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
