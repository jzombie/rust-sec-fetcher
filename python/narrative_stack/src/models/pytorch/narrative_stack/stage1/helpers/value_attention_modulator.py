import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# from torchmetrics.regression import R2Score
from collections import defaultdict


class ValueAttentionModulator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Linear(1, emb_dim)  # scalar → query
        self.key = nn.Linear(emb_dim, emb_dim)  # x_emb → key
        self.value = nn.Linear(emb_dim, emb_dim)  # x_emb → value
        self.scale = emb_dim**0.5

    def forward(self, x_emb, x_val):
        q = self.query(x_val)  # [B, D]
        k = self.key(x_emb)  # [B, D]
        v = self.value(x_emb)  # [B, D]

        attn = torch.sum(q * k, dim=-1, keepdim=True) / self.scale  # [B, 1]
        attn_weights = torch.sigmoid(attn)  # [B, 1] ∈ (0, 1)

        return attn_weights * v + (1 - attn_weights) * x_emb

