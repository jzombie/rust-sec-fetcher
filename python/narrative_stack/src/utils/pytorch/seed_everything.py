import os
import random
import torch
import numpy as np
import pytorch_lightning as pl
import logging

# === SEED ===
DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """
    This function sets the seed for various libraries to ensure reproducibility.
    It seeds Python's built-in random module, NumPy, PyTorch (CPU and GPU), PyTorch Lightning, and MPS (Apple Silicon).
    """

    logging.info("Everything seeded!")

    # Set PYTHONHASHSEED to make Python's hash-based operations deterministic
    # (e.g., dict key ordering, set ordering, hash() values).
    # Without this, Python may randomize hash seeds between runs,
    # which can affect any logic that relies on object ordering or hashing.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # PyTorch deterministic behavior
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # If using MPS (Apple Silicon), set the seed for MPS device
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)

    # If using CUDA (GPU), set the seed for CUDA device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # PyTorch Lightning seed (this handles distributed training if enabled)
    pl.seed_everything(seed, workers=True)
