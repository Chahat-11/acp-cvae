"""A small conditional VAE for peptide sequences.

Adapted from HydrAMP (Szymczak et al. 2023, MIT licence), rewritten in PyTorch
and simplified for a course project. What we kept and what we dropped:

  KEPT    encoder produces mu and log-variance
  KEPT    reparameterisation  z = mu + sigma * eps
  KEPT    the label y is joined to z before decoding, and repeated at every
          position so the decoder cannot forget it
  KEPT    loss = reconstruction cross-entropy + beta * KL

  KEPT    a small activity classifier reading z, trained with binary
          cross-entropy. This is the lambda * L_act term. It is what forces
          activity to be REPRESENTED in the latent space during training
          instead of merely filtered for afterwards, and it is the specific
          thing HydrAMP has that PepVAE does not. Without it the decoder
          learns to ignore a one-dimensional label sitting next to 64 latent
          dimensions, and the conditioning does nothing -- we measured exactly
          that before adding it back.

  DROPPED the potency/MIC branch (we have one binary label, not a concentration)
  DROPPED Gumbel-softmax (only needed to pass gradients through discrete
          samples back into the classifier; our classifier reads z directly,
          which is continuous, so we do not need it)
  DROPPED the autoregressive decoder, replaced by a plain GRU that reads the
          same [z, y] at every position. Less powerful, but much easier to
          follow and it cannot drift the way free-running autoregression does.

Shapes, with B = batch size, L = MAX_LEN, V = 21:
  x  [B, L]      amino acid indices
  y  [B, 1]      label, 1.0 = anticancer
  z  [B, latent] latent vector
  logits [B, L, V]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from peptides import PAD, VOCAB_SIZE


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        max_len: int = 25,
        latent_dim: int = 64,    # HydrAMP LATENT_DIM
        hidden_dim: int = 128,   # HydrAMP HIDDEN_DIM
        embed_dim: int = 64,
        position_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(dropout)

        # ---------------- encoder:  x -> mu, logvar ----------------
        # Note: the encoder sees only x, not y. That is how HydrAMP does it.
        # A textbook cVAE would feed y here as well; we follow HydrAMP so the
        # comparison with their design stays honest.
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD)
        self.encoder_rnn = nn.GRU(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        # bidirectional, so the summary is 2 * hidden_dim wide
        self.to_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # ---------------- decoder:  [z, y] -> logits ----------------
        # +1 on the input width is the label, concatenated to z.
        self.latent_to_hidden = nn.Linear(latent_dim + 1, hidden_dim)
        # The decoder GRU reads [z, y] again at every position. Repeating the
        # label is the trick that keeps the condition from being forgotten --
        # HydrAMP does the same thing with its RepeatVector.
        #
        # We also give it a learned embedding of the position. Without this the
        # decoder gets an identical input at every step and has to work out
        # "where am I in the peptide" from the GRU's internal dynamics alone,
        # which it does badly. Telling it the position directly roughly doubles
        # reconstruction accuracy for 400 extra parameters.
        self.position = nn.Embedding(max_len, position_dim)
        self.decoder_rnn = nn.GRU(
            latent_dim + 1 + position_dim, hidden_dim, batch_first=True
        )
        self.to_logits = nn.Linear(hidden_dim, VOCAB_SIZE)

        # ---------------- activity classifier:  z -> is it anticancer? ----------
        # Used ONLY during training, to shape the latent space. We never use it
        # to judge our own output -- AntiCP 2.0 does that, independently.
        self.activity_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    # ---------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x)                  # [B, L, embed]
        _, hidden = self.encoder_rnn(embedded)        # hidden [2, B, hidden]
        # join the forward and backward final states
        summary = torch.cat([hidden[0], hidden[1]], dim=1)   # [B, 2*hidden]
        return self.to_mu(summary), self.to_logvar(summary)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, with eps drawn from N(0, 1).

        Sampling directly from N(mu, sigma) would block gradients. Writing it
        this way keeps the randomness in eps, which carries no gradient, so mu
        and sigma stay trainable. This is the reparameterisation trick.
        """
        if not self.training:
            return mu  # at eval time use the centre, so results are repeatable
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        conditioned = torch.cat([z, y], dim=1)             # [B, latent+1]
        # same vector at every position ...
        repeated = conditioned.unsqueeze(1).expand(-1, self.max_len, -1)

        # ... plus "this is position t"
        positions = torch.arange(self.max_len, device=z.device)
        position_embedded = self.position(positions)                    # [L, pos]
        position_embedded = position_embedded.unsqueeze(0).expand(z.size(0), -1, -1)

        decoder_input = torch.cat([repeated, position_embedded], dim=2)

        start = torch.tanh(self.latent_to_hidden(conditioned))  # [B, hidden]
        output, _ = self.decoder_rnn(decoder_input, start.unsqueeze(0))
        return self.to_logits(self.dropout(output))        # [B, L, V]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z, y), mu, logvar, self.activity_head(z)

    # ---------------------------------------------------------------
    @torch.no_grad()
    def generate(self, n: int, label: float, device, temperature: float = 1.0):
        """Draw n new peptides. No encoder and no dataset needed here.

        We sample z from the prior N(0, I) -- which is exactly why the KL term
        exists, to make that prior a sensible place to sample from -- set the
        label we want, and decode.

        temperature < 1 makes the output more conservative (closer to the most
        likely amino acid at each position), > 1 makes it more adventurous.
        """
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        y = torch.full((n, 1), float(label), device=device)

        logits = self.decode(z, y) / temperature
        probabilities = torch.softmax(logits, dim=-1)

        # sample one amino acid per position
        flat = probabilities.reshape(-1, probabilities.size(-1))
        sampled = torch.multinomial(flat, num_samples=1)
        return sampled.reshape(n, self.max_len).cpu()


def vae_loss(logits, x, mu, logvar, beta: float,
             activity_logit=None, y=None, lam: float = 0.0):
    """L_total = L_rec + beta * L_KL + lambda * L_act.

    Reconstruction is summed over positions and averaged over the batch, so it
    does not change meaning when the batch size changes.

    Padding IS included in the loss on purpose: the model has to learn where a
    peptide ends, and PAD is how it says so.

    The activity term is binary cross-entropy on a classifier reading z. It is
    what makes the conditioning work: it pushes active and inactive peptides
    into different regions of the latent space, so that setting y actually
    changes what comes out of the decoder.
    """
    batch_size = x.size(0)

    reconstruction = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        x.reshape(-1),
        reduction="sum",
    ) / batch_size

    # closed form for KL( N(mu, sigma) || N(0, I) )
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl_per_dim.sum() / batch_size

    if activity_logit is not None and lam > 0:
        activity = nn.functional.binary_cross_entropy_with_logits(activity_logit, y)
    else:
        activity = torch.zeros((), device=x.device)

    total = reconstruction + beta * kl + lam * activity
    return total, reconstruction, kl, kl_per_dim, activity
