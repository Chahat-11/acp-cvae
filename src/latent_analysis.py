"""Look inside the latent space: PCA and k-means.

The cVAE compresses every peptide into a vector z. This asks two questions
about that space:

  1. Do anticancer and non-anticancer peptides end up in different regions?
     We reduce z to two dimensions with PCA and colour by the true label.

  2. Would clustering find that split on its own, without being told the labels?
     We run k-means with k=2 and compare its clusters against the real labels
     using the adjusted Rand index (ARI). Silhouette score tells us separately
     how tight and well-separated the clusters are.

How to read the numbers:
  ARI         1.0 = clusters match the labels perfectly, 0.0 = no better than
              chance. Anything above ~0.3 on biological data is meaningful.
  Silhouette  +1 = tight, well-separated clusters; 0 = overlapping; negative =
              points are closer to the other cluster than their own.

A low ARI is not a failure of the project. It would mean activity is not the
dominant axis of variation -- length and composition may matter more -- which
is itself worth reporting.

Run:
    python src/latent_analysis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from model import ConditionalVAE
from peptides import MAX_LEN, load_split, pick_device

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

ACTIVE_COLOUR = "#8c3a52"
INACTIVE_COLOUR = "#b0a394"


@torch.no_grad()
def encode_all(model, x, device, batch_size: int = 128) -> np.ndarray:
    """Run every peptide through the encoder and keep mu (the centre of q(z|x))."""
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size].to(device)
        mu, _ = model.encode(batch)
        chunks.append(mu.cpu().numpy())
    return np.vstack(chunks)


def plot(coords, labels, clusters, variance, ari, silhouette) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left: coloured by the TRUE label
    for value, colour, name in [(1, ACTIVE_COLOUR, "anticancer"),
                                (0, INACTIVE_COLOUR, "not anticancer")]:
        mask = labels == value
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.7,
                        c=colour, label=name, edgecolors="none")
    axes[0].set_title("Latent space, coloured by true label")
    axes[0].legend(frameon=False)

    # right: coloured by what k-means found on its own
    for value, colour in [(0, "#4a6fa5"), (1, "#c88a3d")]:
        mask = clusters == value
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.7,
                        c=colour, label=f"cluster {value}", edgecolors="none")
    axes[1].set_title(f"k-means (k=2)   ARI {ari:.3f}   silhouette {silhouette:.3f}")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.set_xlabel(f"PC1 ({variance[0]:.1%} of variance)")
        ax.set_ylabel(f"PC2 ({variance[1]:.1%} of variance)")

    fig.suptitle("cVAE latent space")
    fig.tight_layout()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / "latent_space.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="cvae.pt")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = pick_device()
    checkpoint = torch.load(OUTPUTS / args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = ConditionalVAE(
        max_len=config["max_len"], latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"], dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    x, y, df = load_split(args.split, MAX_LEN)
    labels = y.squeeze(1).numpy().astype(int)

    z = encode_all(model, x, device)
    print(f"\nEncoded {len(z)} peptides into {z.shape[1]} dimensions")

    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(z)
    variance = pca.explained_variance_ratio_
    print(f"PC1 explains {variance[0]:.1%}, PC2 explains {variance[1]:.1%} "
          f"({variance.sum():.1%} together)")

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=args.seed)
    clusters = kmeans.fit_predict(z)

    ari = adjusted_rand_score(labels, clusters)
    sil = silhouette_score(z, clusters)
    # how cleanly does each k-means cluster line up with one label?
    purity = max(
        (clusters == labels).mean(),
        (clusters == 1 - labels).mean(),
    )

    print(f"\nCLUSTERING (k=2, on the full {z.shape[1]}-dimensional space)")
    print(f"  adjusted Rand index   {ari:.3f}   (1.0 = matches labels, 0.0 = chance)")
    print(f"  silhouette score      {sil:.3f}   (higher = tighter clusters)")
    print(f"  best label agreement  {purity:.1%}")

    if ari < 0.1:
        print("\n  The clusters do not correspond to activity. The latent space is")
        print("  organised by something else -- most likely length or composition.")

    path = plot(coords, labels, clusters, variance, ari, sil)

    (OUTPUTS / "latent_metrics.json").write_text(json.dumps({
        "split": args.split,
        "n_peptides": len(z),
        "latent_dim": int(z.shape[1]),
        "pc1_variance": round(float(variance[0]), 4),
        "pc2_variance": round(float(variance[1]), 4),
        "adjusted_rand_index": round(float(ari), 4),
        "silhouette": round(float(sil), 4),
        "label_agreement": round(float(purity), 4),
    }, indent=2))

    print(f"\nWrote {path.relative_to(REPO)} and outputs/latent_metrics.json")


if __name__ == "__main__":
    main()
