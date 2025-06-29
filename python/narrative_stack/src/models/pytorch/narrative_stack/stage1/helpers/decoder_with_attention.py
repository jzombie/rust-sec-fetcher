from torch import nn


class DecoderWithAttention(nn.Module):
    """
    Latent-to-output decoder.

    Steps
    -----
    1. Expand latent vector and apply dropout.
    2. Run a lightweight self-attention pass.
    3. Split into two heads:
       * emb_head -> reconstructed embedding  [B, emb_dim]
       * val_head -> reconstructed scalar     [B, 1]
    """

    def __init__(self, latent_dim, emb_dim, dropout_rate=0.0):
        """
        Parameters
        ----------
        latent_dim : int
            Size of the bottleneck vector z.
        emb_dim : int
            Target dimension of the reconstructed embedding.
        dropout_rate : float, default 0.0
            Dropout applied after the expansion layer.
        """

        super().__init__()
        self.expand = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.LayerNorm(latent_dim * 4),
            nn.Dropout(p=dropout_rate),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim * 4, num_heads=4, batch_first=True
        )

        # Separate heads
        self.emb_head = nn.Sequential(
            nn.Linear(latent_dim * 4, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.val_head = nn.Sequential(
            nn.Linear(latent_dim * 4, emb_dim), nn.GELU(), nn.Linear(emb_dim, 1)
        )

    def forward(self, z):
        """
        Parameters
        ----------
        z : torch.Tensor
            Latent vector, shape [B, latent_dim].

        Returns
        -------
        recon_emb : torch.Tensor
            Reconstructed embedding, shape [B, emb_dim].
        recon_val : torch.Tensor
            Reconstructed scalar value, shape [B, 1].
        """

        h = self.expand(z)  # [B, 4D]
        h_attn, _ = self.attn(h.unsqueeze(1), h.unsqueeze(1), h.unsqueeze(1))
        h = h_attn.squeeze(1)

        recon_emb = self.emb_head(h)
        recon_val = self.val_head(h)
        return recon_emb, recon_val
