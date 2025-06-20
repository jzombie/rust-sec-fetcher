from torch.utils.data import Dataset
import torch
import numpy as np
from utils.pytorch import seed_everything
from simd_r_drive_net_client import DataStoreNetClient
from models.pytorch.narrative_stack.common import UsGaapStore

seed_everything()


# Stage 1 dataset: concept+uom embedding + value
class ConceptValueDataset(Dataset):
    def __init__(self, websocket_address: str, return_scaler=False):
        # In __init__, we only store configuration and data that CAN be pickled.
        # We DO NOT create the network client here.
        self.address = websocket_address
        self.return_scaler = return_scaler

        # We set the client and store to None initially.
        # These will be initialized within each worker process.
        self.data_store_client = None
        self.us_gaap_store = None

        # You can get the count once in the main process to set the length.
        # This requires a temporary client.

        temp_client = DataStoreNetClient(self.address)

        temp_store = UsGaapStore(temp_client)
        self.triplet_count = temp_store.get_triplet_count()

    def __len__(self) -> int:
        return self.triplet_count

    def _init_worker(self):
        """Initializes the network client and store within the worker process."""
        # This function is called by __getitem__. If a client already exists
        # for this worker, it does nothing.
        if self.data_store_client is None:
            print(f"Initializing client for a new worker...")  # Helpful debug print

            self.data_store_client = DataStoreNetClient(self.address)
            self.us_gaap_store = UsGaapStore(self.data_store_client)

    def __getitem__(self, idx):
        # This is the key: initialize the client if it hasn't been already for this worker.
        self._init_worker()

        # Now, self.us_gaap_store is guaranteed to exist.
        item_data = self.us_gaap_store.lookup_by_index(idx)
        concept = item_data["concept"]
        unit = item_data["uom"]
        embedding = item_data["embedding"]
        value = item_data["scaled_value"]

        x = torch.tensor(
            np.concatenate([embedding, [value]]),
            dtype=torch.float32,
        )
        y = x.clone()

        if self.return_scaler:
            scaler_obj = item_data["scaler"]
            return x, y, scaler_obj, (concept, unit)
        return x, y, (concept, unit)

    # TODO: Remove
    # def __getitem__(self, idx):
    #     item_data = self.us_gaap_store.lookup_by_index(idx)
    #     concept = item_data["concept"]
    #     unit = item_data["uom"]

    #     # embedding = item_data["embedding"]
    #     embedding = np.array(item_data["embedding"], dtype=np.float32)  # shape (234,)
    #     value = np.array([item_data["scaled_value"]], dtype=np.float32)  # shape (1,)

    #     # concept, unit, value = self.rows[idx]

    #     # try:
    #     #     embedding = self.lookup[(concept, unit)]
    #     # except KeyError:
    #     #     raise ValueError(f"Missing embedding for ({concept}, {unit})")

    #     x = torch.tensor(np.concatenate([embedding, [value]]), dtype=torch.float32,
    #                      device=self.device)

    #     # For autoencoders, target y is typically the same as input x
    #     y = x.clone()

    #     if self.return_scaler:
    #         # scaler_obj = self.scalers.get((concept, unit))
    #         scaler_obj = item_data["scaler"]
    #         return x, y, scaler_obj, (concept, unit)
    #     return x, y, (concept, unit)


def collate_with_scaler(batch):
    xs, ys, scalers_list, concept_units = zip(*batch)
    return torch.stack(xs), torch.stack(ys), scalers_list, list(concept_units)
