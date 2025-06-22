# from torch.utils.data import Dataset
# import torch
# import numpy as np
# from utils.pytorch import seed_everything
# from simd_r_drive_net_client import DataStoreNetClient
# from models.pytorch.narrative_stack.common import UsGaapStore

# seed_everything()

# TODO: Pre-concatenate torch tensor in store?

import torch
from torch.utils.data import IterableDataset

# import random
import numpy as np
import math
from simd_r_drive_ws_client import DataStoreWsClient
from models.pytorch.narrative_stack.common import (
    UsGaapStore,
)  # Assuming this path is correct


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
        websocket_address: str,
        internal_batch_size: int = 1024,  # How many items to fetch from the DB at once
        return_scaler=True,
        shuffle=False,
    ):
        self.address = websocket_address
        self.internal_batch_size = internal_batch_size
        self.return_scaler = return_scaler
        self.shuffle = shuffle
        self.epoch = 0  # Initialize epoch count

        self.data_store_client = None
        self.us_gaap_store = None

        # Get the total count once in the main process
        temp_client = DataStoreWsClient(self.address)
        temp_store = UsGaapStore(temp_client)
        self.triplet_count = temp_store.get_triplet_count()

    def _init_worker(self):
        """Initializes the client and store within the worker process."""
        if self.data_store_client is None:
            # Each worker gets its own client connection
            self.data_store_client = DataStoreWsClient(self.address)
            self.us_gaap_store = UsGaapStore(self.data_store_client)

    def _process_item(self, item_data: dict):
        """Helper function to convert a single data dict into a sample tuple."""
        concept = item_data["concept"]
        unit = item_data["uom"]
        embedding = item_data["embedding"]
        value = item_data["scaled_value"]

        # Ensure value is not None before processing
        if value is None:
            # Handle cases where a scaled value might be missing for a valid reason
            # For now, we'll skip this item. You could also log it or use a default.
            return None

        x = torch.tensor(
            np.concatenate([embedding, [value]]),
            dtype=torch.float32,
        )
        y = x.clone()  # For autoencoders
        scaler = item_data.get("scaler") if self.return_scaler else None
        concept_unit = (concept, unit)

        return (x, y, scaler, concept_unit)

    def __iter__(self):
        """
        The core logic for an iterable-style dataset. This iterator fetches data
        in large, efficient chunks but yields single, processed items one by one.
        """
        self._init_worker()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Increment epoch for the next iteration in the main process
            if worker_id == 0:
                self.epoch += 1

        # Create the full list of indices for the entire dataset
        all_indices = list(range(self.triplet_count))
        if self.shuffle:
            # FIX: All workers must use the same seed for an epoch to generate
            # the same shuffled list. Seeding with the epoch number ensures
            # the shuffle is different for each epoch.
            g = torch.Generator()
            g.manual_seed(42 + self.epoch)
            indices_for_shuffling = torch.randperm(
                self.triplet_count, generator=g
            ).tolist()
            all_indices = [all_indices[i] for i in indices_for_shuffling]

        # Determine which subset of the data this worker is responsible for
        # This slicing is now performed on the same shuffled list for all workers,
        # guaranteeing no data duplication.
        items_per_worker = int(math.ceil(self.triplet_count / num_workers))
        start_idx = worker_id * items_per_worker
        end_idx = min(start_idx + items_per_worker, self.triplet_count)

        worker_indices = all_indices[start_idx:end_idx]

        # Main loop: Fetch internal batches and yield single items
        for i in range(0, len(worker_indices), self.internal_batch_size):
            # Create a large, efficient batch of indices to fetch from the DB
            index_batch = worker_indices[i : i + self.internal_batch_size]

            if not index_batch:
                continue

            # Fetch all data for this internal batch of indices
            batch_data = self.us_gaap_store.batch_lookup_by_indices(index_batch)

            # Now, yield each item from the fetched batch one by one
            for item_data in batch_data:
                processed_item = self._process_item(item_data)
                if processed_item:
                    yield processed_item


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
