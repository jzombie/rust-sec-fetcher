# from torch.utils.data import Dataset
# import torch
# import numpy as np
# from utils.pytorch import seed_everything
# from simd_r_drive_net_client import DataStoreNetClient
# from us_gaap_store import UsGaapStore

# seed_everything()


import torch
from torch.utils.data import IterableDataset

import numpy as np
import math
from config import SimdRDriveServerConfig
from simd_r_drive_ws_client import DataStoreWsClient
from us_gaap_store import UsGaapStore


def collate_with_scaler(batch):
    """
    Custom collate function that correctly handles a list of individual samples.
    Each sample is a tuple: (x, y, scaler, concept_unit).
    """

    # Unzip the list of tuples into separate lists
    xs, ys, scalers_list, concept_units = zip(*batch)

    # Stack the tensors to create a batch
    xs_batch = torch.stack(xs)
    ys_batch = torch.stack(ys)

    # The scalers and concept_units remain as lists
    return xs_batch, ys_batch, scalers_list, list(concept_units)


class IterableConceptValueDataset(IterableDataset):
    def __init__(
        self,
        simd_r_drive_server_config: SimdRDriveServerConfig,
        internal_batch_size: int = 1024,  # How many items to fetch from the DB at once
        return_scaler=True,
        shuffle=False,
    ):
        self.simd_r_drive_server_config = simd_r_drive_server_config
        self.internal_batch_size = internal_batch_size
        self.return_scaler = return_scaler
        self.shuffle = shuffle
        self.epoch = 0  # Initialize epoch count

        self.data_store_client = None
        self.us_gaap_store = None

        # Get the total count once in the main process
        temp_client = DataStoreWsClient(simd_r_drive_server_config.host, simd_r_drive_server_config.port)
        temp_store = UsGaapStore(temp_client)
        self.triplet_count = temp_store.get_triplet_count()

        # Disconnect temp client
        del temp_client

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        PyTorch Lightning will then use this total to display the
        epoch progress bar.
        """
        return self.triplet_count

    def _init_worker(self):
        """Initializes the client and store within the worker process."""
        if self.data_store_client is None:
            # Each worker gets its own client connection
            self.data_store_client = DataStoreWsClient(self.simd_r_drive_server_config.host, self.simd_r_drive_server_config.port)
            self.us_gaap_store = UsGaapStore(self.data_store_client)

    def __iter__(self):
        """
        The core logic for an iterable-style dataset. This iterator fetches data
        in large, efficient chunks and processes them with vectorized operations
        before yielding single items.
        """
        self._init_worker()

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        if worker_id == 0:
            self.epoch += 1

        all_indices = list(range(self.triplet_count))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(42 + self.epoch)
            perm = torch.randperm(self.triplet_count, generator=g).tolist()
            all_indices = [all_indices[i] for i in perm]

        items_per_worker = int(math.ceil(self.triplet_count / num_workers))
        start_idx = worker_id * items_per_worker
        end_idx = min(start_idx + items_per_worker, self.triplet_count)

        worker_indices = all_indices[start_idx:end_idx]

        for i in range(0, len(worker_indices), self.internal_batch_size):
            index_batch = worker_indices[i : i + self.internal_batch_size]
            if not index_batch:
                continue

            # Fetch all data for this internal batch of indices
            batch_data = self.us_gaap_store.batch_lookup_by_indices(index_batch)

            # --- Vectorized Processing with Pre-allocation ---
            valid_data = [
                item for item in batch_data if item.scaled_value is not None
            ]
            if not valid_data:
                continue

            num_valid = len(valid_data)
            # Assuming embedding dimension is 243
            embedding_dim = 243

            # Pre-allocate final NumPy arrays to avoid intermediate Python lists
            embeddings_np = np.empty((num_valid, embedding_dim), dtype=np.float32)
            values_np = np.empty((num_valid, 1), dtype=np.float32)

            # Fill the pre-allocated arrays in a loop
            for idx, item in enumerate(valid_data):
                embeddings_np[idx] = item.embedding
                values_np[idx] = item.scaled_value

            # Perform the concatenation as a single, fast matrix operation
            x_data_np = np.concatenate([embeddings_np, values_np], axis=1)

            # Convert the entire processed batch to a PyTorch tensor
            x_data_torch = torch.from_numpy(x_data_np)

            # --- Yield Individual Items ---
            for j in range(num_valid):
                x = x_data_torch[j]
                y = x.clone()

                item_meta = valid_data[j]
                scaler = item_meta.scaler if self.return_scaler else None
                concept_unit = (item_meta.concept, item_meta.uom)

                yield (x, y, scaler, concept_unit)


# --- Example of how to use it ---
#
# if __name__ == '__main__':
#     WEBSOCKET_ADDR = "ws://127.0.0.1:34129"
#     BATCH_SIZE = 64
#     NUM_WORKERS = 8
#
#     # 1. Instantiate the IterableDataset
#     iterable_dataset = IterableConceptValueDataset(
#         websocket_address=WEBSOCKET_ADDR,
#         batch_size=BATCH_SIZE
#     )
#
#     # You can now get the total number of batches for your trainer/progress bar
#     total_batches_per_epoch = iterable_dataset.num_batches
#     print(f"Total batches per epoch: {total_batches_per_epoch}")
#
#     # 2. Create the DataLoader.
#     #    CRITICAL: batch_size must be None because the dataset is already
#     #    creating and yielding batches. The collate_fn is also not needed.
#     train_loader = DataLoader(
#         iterable_dataset,
#         batch_size=None,  # Let the dataset handle batching
#         num_workers=NUM_WORKERS
#     )
#
#     # 3. The training loop now receives fully formed batches.
#     for batch in train_loader:
#         xs, ys, concept_units = batch
#         # xs and ys are already stacked tensors of shape [batch_size, 244]
#         # ... proceed with your training step ...


# TODO: Remove old implementation
# # Stage 1 dataset: concept+uom embedding + value
# class ConceptValueDataset(Dataset):
#     def __init__(self, websocket_address: str, return_scaler=False):
#         # In __init__, we only store configuration and data that CAN be pickled.
#         # We DO NOT create the network client here.
#         self.address = websocket_address
#         self.return_scaler = return_scaler

#         # We set the client and store to None initially.
#         # These will be initialized within each worker process.
#         self.data_store_client = None
#         self.us_gaap_store = None

#         # You can get the count once in the main process to set the length.
#         # This requires a temporary client.

#         temp_client = DataStoreNetClient(self.address)

#         temp_store = UsGaapStore(temp_client)
#         self.triplet_count = temp_store.get_triplet_count()

#     def __len__(self) -> int:
#         return self.triplet_count

#     def _init_worker(self):
#         """Initializes the network client and store within the worker process."""
#         # This function is called by __getitem__. If a client already exists
#         # for this worker, it does nothing.
#         if self.data_store_client is None:
#             print(f"Initializing client for a new worker...")  # Helpful debug print

#             self.data_store_client = DataStoreNetClient(self.address)
#             self.us_gaap_store = UsGaapStore(self.data_store_client)

#     def __getitem__(self, idx):
#         # This is the key: initialize the client if it hasn't been already for this worker.
#         self._init_worker()

#         # Now, self.us_gaap_store is guaranteed to exist.
#         item_data = self.us_gaap_store.lookup_by_index(idx)
#         concept = item_data["concept"]
#         unit = item_data["uom"]
#         embedding = item_data["embedding"]
#         value = item_data["scaled_value"]

#         x = torch.tensor(
#             np.concatenate([embedding, [value]]),
#             dtype=torch.float32,
#         )
#         y = x.clone()

#         if self.return_scaler:
#             scaler_obj = item_data["scaler"]
#             return x, y, scaler_obj, (concept, unit)
#         return x, y, (concept, unit)

#     # TODO: Remove
#     # def __getitem__(self, idx):
#     #     item_data = self.us_gaap_store.lookup_by_index(idx)
#     #     concept = item_data["concept"]
#     #     unit = item_data["uom"]

#     #     # embedding = item_data["embedding"]
#     #     embedding = np.array(item_data["embedding"], dtype=np.float32)  # shape (234,)
#     #     value = np.array([item_data["scaled_value"]], dtype=np.float32)  # shape (1,)

#     #     # concept, unit, value = self.rows[idx]

#     #     # try:
#     #     #     embedding = self.lookup[(concept, unit)]
#     #     # except KeyError:
#     #     #     raise ValueError(f"Missing embedding for ({concept}, {unit})")

#     #     x = torch.tensor(np.concatenate([embedding, [value]]), dtype=torch.float32,
#     #                      device=self.device)

#     #     # For autoencoders, target y is typically the same as input x
#     #     y = x.clone()

#     #     if self.return_scaler:
#     #         # scaler_obj = self.scalers.get((concept, unit))
#     #         scaler_obj = item_data["scaler"]
#     #         return x, y, scaler_obj, (concept, unit)
#     #     return x, y, (concept, unit)
