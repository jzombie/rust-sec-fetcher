from typing import Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils.pytorch import seed_everything

class UsGaapAlignmentDataset(Dataset):
    def __init__(self, data_file: str, device: torch.device) -> None:
        """
        Initialize the dataset from a JSONL file containing variation and
        description embeddings, and place all tensors on the specified device.

        Args:
            data_file (str): Path to a line-delimited JSON file with fields:
                - variation_embedding: list[float]
                - description_embedding: list[float]
            device (torch.device): Target device to store tensors (e.g., "cpu", "cuda", "mps").
        """
        self.device = device

        self.data = pd.read_json(data_file, lines=True)

        # Extract embeddings from the data as NumPy arrays and cast them to float32
        self.input_embeddings = np.array(self.data["variation_embedding"].tolist(), dtype=np.float32)
        self.description_embeddings = np.array(self.data["description_embedding"].tolist(), dtype=np.float32)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """

        return len(self.input_embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a pair of (variation, description) embeddings as tensors.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - variation_embedding: Tensor of shape [input_dim]
                - description_embedding: Tensor of shape [input_dim]
        """

        # Convert the NumPy arrays to PyTorch tensors with requires_grad=True
        input_embedding = torch.tensor(self.input_embeddings[idx], dtype=torch.float32, requires_grad=False).to(self.device)
        description_embedding = torch.tensor(self.description_embeddings[idx], dtype=torch.float32, requires_grad=False).to(self.device)
        
        return input_embedding, description_embedding
