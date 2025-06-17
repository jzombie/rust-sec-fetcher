```python
# Perceiver IO
# https://arxiv.org/abs/2107.14795

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceiverStackEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128,
                 num_latents: int = 64, depth: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=4, batch_first=True
        )
        self.cross_proj = nn.Linear(input_dim, latent_dim)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=latent_dim * 4,
                batch_first=True
            ))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, N, D_in] (e.g., concept stack embeddings)
            mask: Optional [B, N] boolean mask (True for valid entries)

        Returns:
            Tensor of shape [B, latent_dim] summarizing the category stack
        """
        B, N, _ = x.shape
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)  # [B, M, D]

        x_proj = self.cross_proj(x)  # [B, N, latent_dim]
        latents, _ = self.cross_attn(latents, x_proj, x_proj,
                                     key_padding_mask=~mask if mask is not None else None)

        for block in self.blocks:
            latents = block(latents)

        return latents.mean(dim=1)  # Pool across latent tokens
```
