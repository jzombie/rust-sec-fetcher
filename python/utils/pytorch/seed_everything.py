import random
import os
import json
import torch
import optuna
import numpy as np
import pytorch_lightning as pl

# === SEED ===
SEED = 42


def seed_everything(seed: int):
    """
    This function sets the seed for various libraries to ensure reproducibility.
    It seeds Python's built-in random module, NumPy, PyTorch (CPU and GPU), PyTorch Lightning, and MPS (Apple Silicon).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If using MPS (Apple Silicon), set the seed for MPS device
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)

    # If using CUDA (GPU), set the seed for CUDA device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # PyTorch Lightning seed (this handles distributed training if enabled)
    pl.seed_everything(seed, workers=True)


# Ensure it's called!
seed_everything(SEED)
