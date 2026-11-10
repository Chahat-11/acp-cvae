"""A small conditional VAE for peptide sequences.

Adapted from HydrAMP (Szymczak et al. 2023, MIT licence), rewritten in PyTorch
and kept small enough to read in one sitting.

  KEPT    encoder produces mu and log-variance
  KEPT    reparameterisation  z = mu + sigma * eps
  KEPT    the label y is joined to z and fed to the decoder at every step
  KEPT    an AUTOREGRESSIVE decoder -- each residue is written knowing the one
          before it. This matters more than anything else here: activity lives
          in local motifs (the KLAKLAK-style repeats all over the real actives),
          and a decoder that cannot see what it just wrote cannot build them.
  KEPT    a small activity classifier reading z (the lambda * L_act term). It
          forces activity to be REPRESENTED in the latent space during training
          rather than filtered for afterwards.
  KEPT    word dropout, from Bowman et al. -- see decode() for why it is needed.

  DROPPED the potency/MIC branch (we have one binary label, not a concentration)
  DROPPED Gumbel-softmax (only needed to pass gradients through discrete
          samples back into the classifier; ours reads z, already continuous)

Shapes, with B = batch, L = MAX_LEN, V = 23:
  x  [B, L]      amino acid indices
  y  [B, 1]      label, 1.0 = anticancer
  z  [B, latent] latent vector
  logits [B, L, V]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from peptides import BOS, PAD, UNK, VOCAB_SIZE


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        max_len: int = 25,
        latent_dim: int = 64,    # HydrAMP LATENT_DIM
        hidden_dim: int = 128,   # HydrAMP HIDDEN_DIM
        embed_dim: int = 64,
        label_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(dropout)

        # ---------------- encoder:  x -> mu, logvar ----------------
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD)
        self.encoder_rnn = nn.GRU(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.to_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # ---------------- the condition ----------------
        # The label gets its own embedding rather than being a bare 0.0/1.0.
        # A single scalar sitting beside 64 latent dimensions is easy for the
        # decoder to ignore, and we measured it doing exactly that: asking for
        # active instead of inactive peptides moved the mean score by 0.02.
        # Giving the label its own 16-wide vector lets it compete for attention.
        self.label_embedding = nn.Embedding(2, label_dim)
        condition_dim = latent_dim + label_dim

        # ---------------- decoder:  [previous residue, z, y] -> next residue ----
        self.latent_to_hidden = nn.Linear(condition_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(
            embed_dim + condition_dim, hidden_dim, batch_first=True
        )
        self.to_logits = nn.Linear(hidden_dim, VOCAB_SIZE)

        # ---------------- activity classifier:  z -> is it anticancer? ----------
        # Used ONLY during training, to shape the latent space. We never use it
        # to judge our own output -- AntiCP 2.0 does that, separately.
        self.activity_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    # ---------------------------------------------------------------
    def condition(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Join the latent vector with the embedded label."""
        label = self.label_embedding(y.squeeze(-1).long())
        return torch.cat([z, label], dim=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x)                  # [B, L, embed]
        _, hidden = self.encoder_rnn(embedded)        # [2, B, hidden]
        summary = torch.cat([hidden[0], hidden[1]], dim=1)
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
        return mu + sigma * torch.randn_like(sigma)

    # ---------------------------------------------------------------
    def decode(self, z, y, x=None, word_dropout: float = 0.0):
        """Teacher forcing when x is given, free running when it is not.

        WORD DROPOUT. An autoregressive decoder has a seductive shortcut: it can
        predict each residue from the previous one alone and ignore z entirely.
        The reconstruction loss would look excellent and the latent space would
        be worthless -- sampling z would change nothing. The fix (Bowman et al.)
        is to randomly replace some of the previous-residue inputs with UNK, so
        the decoder cannot rely on them and has to consult z instead.
        """
        batch = z.size(0)
        conditioned = self.condition(z, y)                      # [B, cond]
        hidden = torch.tanh(self.latent_to_hidden(conditioned)).unsqueeze(0)

        positions = conditioned.unsqueeze(1).expand(-1, self.max_len, -1)

        if x is not None:
            # shift right: the input at step t is the true residue at t-1
            start = torch.full((batch, 1), BOS, device=z.device, dtype=torch.long)
            inputs = torch.cat([start, x[:, :-1]], dim=1)       # [B, L]

            if word_dropout > 0 and self.training:
                drop = torch.rand(inputs.shape, device=z.device) < word_dropout
                # never blank out the start marker
                drop[:, 0] = False
                inputs = inputs.masked_fill(drop, UNK)

            embedded = self.embedding(inputs)                   # [B, L, embed]
            rnn_input = torch.cat([embedded, positions], dim=2)
            output, _ = self.decoder_rnn(rnn_input, hidden)
            return self.to_logits(self.dropout(output))

        # free running: feed the model its own output, one step at a time
        token = torch.full((batch, 1), BOS, device=z.device, dtype=torch.long)
        collected = []
        for step in range(self.max_len):
            embedded = self.embedding(token)                    # [B, 1, embed]
            rnn_input = torch.cat([embedded, positions[:, step:step + 1]], dim=2)
            output, hidden = self.decoder_rnn(rnn_input, hidden)
            logits = self.to_logits(output)                     # [B, 1, V]
            collected.append(logits)
            token = logits.argmax(dim=-1)
        return torch.cat(collected, dim=1)

    def forward(self, x, y, word_dropout: float = 0.0):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        logits = self.decode(z, y, x=x, word_dropout=word_dropout)
        return logits, mu, logvar, self.activity_head(z)

    # ---------------------------------------------------------------
    @torch.no_grad()
    def sample_latents(self, n: int, device, threshold: float = 0.0,
                       max_tries: int = 200) -> torch.Tensor:
        """Draw z from the prior, optionally keeping only 'active-looking' ones.

        Sampling z blindly gives no guarantee of landing where active peptides
        live. The model already carries an opinion about that -- the activity
        head -- so we can screen our own latent samples before spending a
        decode on them. threshold=0 disables this and samples plainly.

        This still needs no dataset and no encoder, so it keeps the property
        that generation is self-contained.
        """
        if threshold <= 0:
            return torch.randn(n, self.latent_dim, device=device)

        kept = []
        total = 0
        for _ in range(max_tries):
            batch = torch.randn(n, self.latent_dim, device=device)
            probability = torch.sigmoid(self.activity_head(batch)).squeeze(-1)
            chosen = batch[probability >= threshold]
            if len(chosen):
                kept.append(chosen)
                total += len(chosen)
            if total >= n:
                break

        if not kept:  # threshold too strict; fall back rather than fail
            return torch.randn(n, self.latent_dim, device=device)
        return torch.cat(kept)[:n]

    @torch.no_grad()
    def generate_analogue(self, x, y, device, noise: float = 0.5,
                          temperature: float = 1.0):
        """HydrAMP's other mode: start from a real peptide and vary it.

        Encode a known active peptide, nudge its latent vector, and decode.
        Unlike sampling from the prior this starts somewhere we know is good,
        so the output stays closer to real peptides. The cost is that it needs
        a template, and the results are variations rather than inventions.

        `noise` is the drift: small values give near-copies, large values
        wander toward unconstrained sampling.
        """
        self.eval()
        mu, _ = self.encode(x.to(device))
        z = mu + noise * torch.randn_like(mu)
        return self._decode_tokens(z, y.to(device), temperature)

    @torch.no_grad()
    def generate(self, n: int, label: float, device, temperature: float = 1.0,
                 threshold: float = 0.0):
        """Draw n new peptides. No encoder and no dataset needed here.

        We sample z from the prior N(0, I) -- which is exactly why the KL term
        exists, to make that prior a sensible place to sample from -- set the
        label we want, and decode one residue at a time, feeding each choice
        back in as the input to the next step.

        temperature < 1 is more conservative, > 1 more adventurous.
        """
        self.eval()
        z = self.sample_latents(n, device, threshold)
        y = torch.full((len(z), 1), float(label), device=device)
        return self._decode_tokens(z, y, temperature)

    @torch.no_grad()
    def _decode_tokens(self, z, y, temperature: float = 1.0):
        """Free-running decode, sampling one residue at a time."""
        conditioned = self.condition(z, y)
        hidden = torch.tanh(self.latent_to_hidden(conditioned)).unsqueeze(0)
        positions = conditioned.unsqueeze(1)

        token = torch.full((len(z), 1), BOS, device=z.device, dtype=torch.long)
        collected = []
        for _ in range(self.max_len):
            embedded = self.embedding(token)
            rnn_input = torch.cat([embedded, positions], dim=2)
            output, hidden = self.decoder_rnn(rnn_input, hidden)
            logits = self.to_logits(output).squeeze(1) / temperature

            # BOS and UNK are inputs, never outputs
            logits[:, BOS] = float("-inf")
            logits[:, UNK] = float("-inf")

            token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            collected.append(token)

        return torch.cat(collected, dim=1).cpu()


def vae_loss(logits, x, mu, logvar, beta: float,
             activity_logit=None, y=None, lam: float = 0.0):
    """L_total = L_rec + beta * L_KL + lambda * L_act.

    Reconstruction is summed over positions and averaged over the batch, so it
    does not change meaning when the batch size changes.

    Padding IS included in the loss on purpose: the model has to learn where a
    peptide ends, and PAD is how it says so.
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
