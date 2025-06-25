from torch import nn
import pytorch_lightning as pl

from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# from torchmetrics.regression import R2Score
from collections import defaultdict

from .value_attention_modulator import ValueAttentionModulator


class EncoderWithAttention(nn.Module):
    def __init__(self, emb_dim, latent_dim, dropout_rate=0.0):
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
        h = self.expand(x_emb)  # [B, 4D]
        h = self.attn(h, x_val)  # scalar-conditioned attention
        z = self.bottleneck(h)  # [B, latent_dim]
        return z
