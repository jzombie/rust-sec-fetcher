import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils.pytorch import seed_everything, get_device

device = get_device()
print(f"Using device: {device}")

class UsGaapAlignmentDataset(Dataset):
    def __init__(self, data_file):
        """
        Loads the dataset and the precomputed embeddings directly from the JSONL file.
        """

        self.data = pd.read_json(data_file, lines=True)

        # Extract embeddings from the data as NumPy arrays and cast them to float32
        self.input_embeddings = np.array(self.data["variation_embedding"].tolist(), dtype=np.float32)
        self.description_embeddings = np.array(self.data["description_embedding"].tolist(), dtype=np.float32)

    def __len__(self):
        return len(self.input_embeddings)

    def __getitem__(self, idx):
        # Convert the NumPy arrays to PyTorch tensors with requires_grad=True
        input_embedding = torch.tensor(self.input_embeddings[idx], dtype=torch.float32, requires_grad=False).to(device)
        description_embedding = torch.tensor(self.description_embeddings[idx], dtype=torch.float32, requires_grad=False).to(device)
        
        return input_embedding, description_embedding
