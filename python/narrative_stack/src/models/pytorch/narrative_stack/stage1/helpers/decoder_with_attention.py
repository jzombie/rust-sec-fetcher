from torch import nn


# from torchmetrics.regression import R2Score







class DecoderWithAttention(nn.Module):
    def __init__(self, latent_dim, emb_dim, dropout_rate=0.0):
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
        z: [B, latent_dim]
        Returns:
            recon_emb: [B, emb_dim]
            recon_val: [B, 1]
        """
        h = self.expand(z)  # [B, 4D]
        h_attn, _ = self.attn(h.unsqueeze(1), h.unsqueeze(1), h.unsqueeze(1))
        h = h_attn.squeeze(1)

        recon_emb = self.emb_head(h)
        recon_val = self.val_head(h)
        return recon_emb, recon_val
