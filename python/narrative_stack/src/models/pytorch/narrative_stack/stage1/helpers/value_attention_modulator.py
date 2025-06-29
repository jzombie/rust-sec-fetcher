import torch
from torch import nn


class ValueAttentionModulator(nn.Module):
    """
    Scalar-conditioned attention gate.

    Lifts the scalar to a query vector, computes a similarity with the
    embedding, and blends the original embedding with a value projection:
        y = alpha * V + (1 - alpha) * x_emb
    where alpha in (0, 1).
    """


    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Linear(1, emb_dim)  # scalar → query
        self.key = nn.Linear(emb_dim, emb_dim)  # x_emb → key
        self.value = nn.Linear(emb_dim, emb_dim)  # x_emb → value
        self.scale = emb_dim**0.5

    def forward(self, x_emb, x_val):
        """
        Parameters
        ----------
        x_emb : torch.Tensor
            Base embedding, shape [B, D].
        x_val : torch.Tensor
            Conditioning scalar, shape [B, 1].

        Returns
        -------
        torch.Tensor
            Modulated embedding, shape [B, D].
        """

        q = self.query(x_val)  # [B, D]
        k = self.key(x_emb)  # [B, D]
        v = self.value(x_emb)  # [B, D]

        attn = torch.sum(q * k, dim=-1, keepdim=True) / self.scale  # [B, 1]
        attn_weights = torch.sigmoid(attn)  # [B, 1] ∈ (0, 1)

        return attn_weights * v + (1 - attn_weights) * x_emb
