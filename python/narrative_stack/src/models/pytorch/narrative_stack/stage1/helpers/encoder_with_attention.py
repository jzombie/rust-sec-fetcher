from torch import nn
from .value_attention_modulator import ValueAttentionModulator


class EncoderWithAttention(nn.Module):
    """
    Embedding-plus-value encoder.

    Combines an embedding with its scalar value using a learned attention gate
    and projects the result to a latent bottleneck.
    """

    def __init__(self, emb_dim, latent_dim, dropout_rate=0.0):
        """
        Parameters
        ----------
        emb_dim : int
            Dimension of the input embedding (not including the scalar).
        latent_dim : int
            Desired size of the bottleneck vector.
        dropout_rate : float, default 0.0
            Dropout applied after expansion.
        """

        super().__init__()
        self.expand = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.LayerNorm(emb_dim * 4),
            nn.Dropout(p=dropout_rate),
        )
        self.attn = ValueAttentionModulator(emb_dim * 4)
        self.bottleneck = nn.Linear(emb_dim * 4, latent_dim)

    def forward(self, x_emb, x_val):
        """
        Parameters
        ----------
        x_emb : torch.Tensor
            Input embedding, shape [B, emb_dim].
        x_val : torch.Tensor
            Scalar value, shape [B, 1].

        Returns
        -------
        z : torch.Tensor
            Latent representation, shape [B, latent_dim].
        """
        
        h = self.expand(x_emb)  # [B, 4D]
        h = self.attn(h, x_val)  # scalar-conditioned attention
        z = self.bottleneck(h)  # [B, latent_dim]
        return z
