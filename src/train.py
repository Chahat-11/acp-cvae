"""Train the conditional VAE.

Two things to watch while this runs:

1. RECONSTRUCTION ACCURACY -- what fraction of amino acids the model puts back
   correctly. If this stays near 5% the model has learned nothing (1/21 is
   chance). We want it high.

2. POSTERIOR COLLAPSE -- the classic VAE failure. If the KL term is too strong
   too early, the encoder gives up: it outputs mu = 0 and sigma = 1 for every
   peptide, z carries no information, and the decoder produces the same generic
   sequence every time. The warning sign is KL falling to about zero and the
   number of ACTIVE UNITS (latent dimensions that actually carry signal)
   dropping toward zero.

   The remedy is KL annealing: start with beta near zero so the model first
   learns to reconstruct, then raise beta gradually.

Run:
    python src/train.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ConditionalVAE, vae_loss
from peptides import MAX_LEN, PAD, load_split, pick_device

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"


def reconstruction_accuracy(logits, x) -> float:
    """Fraction of NON-PAD positions predicted correctly.

    We ignore padding when scoring, because a model that only learned to emit
    padding would otherwise look good -- most positions in a short peptide are
    padding.
    """
    predicted = logits.argmax(dim=-1)
    real = x != PAD
    if real.sum() == 0:
        return 0.0
    return ((predicted == x) & real).sum().item() / real.sum().item()


def beta_at(epoch: int, warmup: int, beta_max: float) -> float:
    """KL annealing: 0 for the first few epochs, then a straight ramp up."""
    if epoch < warmup:
        return 0.0
    progress = (epoch - warmup) / max(1, warmup)
    return min(beta_max, beta_max * progress)


def evaluate(model, loader, device, beta, lam):
    """Teacher-forced metrics, plus the honest free-running one.

    Teacher forcing hands the decoder the true previous residue at every step,
    so it flatters the model. Free running makes it rely on its own output,
    which is what actually happens at generation time. We report both, and the
    free-running number is the one that matters.
    """
    model.eval()
    totals = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "acc": 0.0,
              "free_acc": 0.0, "act_acc": 0.0}
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, mu, logvar, activity = model(x, y)
            loss, rec, kl, _, _ = vae_loss(logits, x, mu, logvar, beta,
                                           activity, y, lam)

            # decode again from the same z, but without the crutch
            free_logits = model.decode(mu, y, x=None)

            totals["loss"] += loss.item()
            totals["rec"] += rec.item()
            totals["kl"] += kl.item()
            totals["acc"] += reconstruction_accuracy(logits, x)
            totals["free_acc"] += reconstruction_accuracy(free_logits, x)
            totals["act_acc"] += (((activity > 0).float() == y).float().mean().item())
            batches += 1
    return {k: v / batches for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=0.05,
                        help="final KL weight; small because reconstruction is summed")
    parser.add_argument("--warmup", type=int, default=20,
                        help="epochs with beta=0 before annealing starts")
    parser.add_argument("--lam", type=float, default=5.0,
                        help="weight on the activity classifier (lambda in L_act)")
    parser.add_argument("--word-dropout", type=float, default=0.3,
                        help="fraction of previous residues hidden from the decoder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-as", default="cvae.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}\n")

    print("Loading data")
    x_train, y_train, _ = load_split("train", MAX_LEN)
    x_val, y_val, _ = load_split("val", MAX_LEN)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size)

    model = ConditionalVAE(
        max_len=MAX_LEN, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters, latent dim {args.latent_dim}\n")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    started = time.time()
    print(f"{'epoch':>5} {'beta':>6} {'loss':>9} {'KL':>7} "
          f"{'train acc':>10} {'val acc':>8} {'free acc':>9} {'act acc':>8} {'active':>7}")

    for epoch in range(1, args.epochs + 1):
        beta = beta_at(epoch, args.warmup, args.beta_max)
        model.train()

        epoch_totals = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "acc": 0.0, "act_acc": 0.0}
        kl_per_dim_sum = torch.zeros(args.latent_dim, device=device)
        batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, mu, logvar, activity = model(x, y, word_dropout=args.word_dropout)
            loss, rec, kl, kl_per_dim, _ = vae_loss(logits, x, mu, logvar, beta,
                                                    activity, y, args.lam)

            optimiser.zero_grad()
            loss.backward()
            # keep a single bad batch from blowing up the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            epoch_totals["loss"] += loss.item()
            epoch_totals["rec"] += rec.item()
            epoch_totals["kl"] += kl.item()
            epoch_totals["acc"] += reconstruction_accuracy(logits, x)
            epoch_totals["act_acc"] += (
                ((activity > 0).float() == y).float().mean().item())
            kl_per_dim_sum += kl_per_dim.mean(dim=0).detach()
            batches += 1

        train_stats = {k: v / batches for k, v in epoch_totals.items()}

        # A latent dimension counts as "active" if it carries more than a
        # whisker of information. If this collapses to 0, the VAE has failed.
        active_units = int((kl_per_dim_sum / batches > 0.01).sum().item())

        val_stats = evaluate(model, val_loader, device, beta, args.lam)

        history.append({
            "epoch": epoch, "beta": round(beta, 4),
            "train_loss": round(train_stats["loss"], 3),
            "train_rec": round(train_stats["rec"], 3),
            "kl": round(train_stats["kl"], 3),
            "train_acc": round(train_stats["acc"], 4),
            "val_acc": round(val_stats["acc"], 4),
            "val_free_acc": round(val_stats["free_acc"], 4),
            "train_activity_acc": round(train_stats["act_acc"], 4),
            "val_activity_acc": round(val_stats["act_acc"], 4),
            "active_units": active_units,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>5} {beta:>6.3f} {train_stats['loss']:>9.2f} "
                  f"{train_stats['kl']:>7.2f} "
                  f"{train_stats['acc']:>9.1%} {val_stats['acc']:>8.1%} "
                  f"{val_stats['free_acc']:>9.1%} "
                  f"{val_stats['act_acc']:>8.1%} {active_units:>7}")

    elapsed = time.time() - started
    print(f"\nTrained {args.epochs} epochs in {elapsed:.0f}s")

    final = history[-1]
    print(f"Final: train acc {final['train_acc']:.1%}, "
          f"val acc {final['val_acc']:.1%} (teacher-forced), "
          f"val acc {final['val_free_acc']:.1%} (free-running), "
          f"activity acc {final['val_activity_acc']:.1%}, "
          f"KL {final['kl']:.2f}, active units {final['active_units']}/{args.latent_dim}")

    if final["active_units"] == 0:
        print("\nWARNING: posterior collapse -- no active latent units.")
    elif final["kl"] < 1.0:
        print("\nWARNING: KL is very low; the latent space may be near collapse.")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(),
         "config": {"max_len": MAX_LEN, "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim, "dropout": args.dropout}},
        OUTPUTS / args.save_as,
    )
    (OUTPUTS / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nSaved outputs/{args.save_as} and outputs/training_history.json")


if __name__ == "__main__":
    main()
