# from torch.utils.data import Dataset
# import torch
# import numpy as np
# from utils.pytorch import seed_everything
# from simd_r_drive_net_client import DataStoreNetClient
# from models.pytorch.narrative_stack.common import UsGaapStore

# seed_everything()

# # TODO: Convert to batch reader

from torch.utils.data import IterableDataset
import torch
import numpy as np
import math
from utils.pytorch import seed_everything
from simd_r_drive_net_client import DataStoreNetClient
from models.pytorch.narrative_stack.common import UsGaapStore

# def collate_with_scaler(batch):
#     xs, ys, scalers_list, concept_units = zip(*batch)
#     return torch.stack(xs), torch.stack(ys), scalers_list, list(concept_units)


seed_everything()


# Stage 1 dataset, refactored for batch reading.
class IterableConceptValueDataset(IterableDataset):
    def __init__(self, websocket_address: str, batch_size: int, return_scaler=False):
        # Store configuration.
        self.address = websocket_address
        self.batch_size = batch_size
        self.return_scaler = return_scaler

        # The client and store are initialized in each worker.
        self.data_store_client = None
        self.us_gaap_store = None

        # Get the total count once in the main process to calculate the number of batches.
        temp_client = DataStoreNetClient(self.address)
        temp_store = UsGaapStore(temp_client)
        self.triplet_count = temp_store.get_triplet_count()

        # Worker-specific info will be set in __iter__
        self.worker_id = None
        self.num_workers = None

    @property
    def num_batches(self) -> int:
        """
        Calculates and returns the total number of batches in the dataset for one epoch.
        This is crucial for progress bars and schedulers in training frameworks.
        """
        if self.batch_size is None or self.batch_size == 0:
            return 0
        return math.ceil(self.triplet_count / self.batch_size)

    def _init_worker(self):
        """Initializes the network client and store within the worker process."""
        if self.data_store_client is None:
            # This check ensures the client is created only once per worker.
            print(f"Initializing client for a new worker...")
            self.data_store_client = DataStoreNetClient(self.address)
            self.us_gaap_store = UsGaapStore(self.data_store_client)

    def _process_batch(self, batch_data: list[dict]):
        """Helper function to convert a list of data dicts into tensors."""
        xs, ys, scalers, concept_units = [], [], [], []

        for item_data in batch_data:
            concept = item_data["concept"]
            unit = item_data["uom"]
            embedding = item_data["embedding"]
            value = item_data["scaled_value"]

            x = torch.tensor(
                np.concatenate([embedding, [value]]),
                dtype=torch.float32,
            )

            xs.append(x)
            ys.append(x.clone())  # For autoencoders
            concept_units.append((concept, unit))
            if self.return_scaler:
                scalers.append(item_data["scaler"])

        if self.return_scaler:
            return torch.stack(xs), torch.stack(ys), scalers, concept_units
        return torch.stack(xs), torch.stack(ys), concept_units

    def __iter__(self):
        """The core logic for an iterable-style dataset."""
        # This method is called once per epoch for each worker.
        self._init_worker()

        # Determine which subset of the data this worker is responsible for.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process data loading
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

        # Calculate the start and end index for this specific worker.
        items_per_worker = int(math.ceil(self.triplet_count / self.num_workers))
        start_idx = self.worker_id * items_per_worker
        end_idx = min(start_idx + items_per_worker, self.triplet_count)

        # The main loop: create batches of indices and fetch them.
        for i in range(start_idx, end_idx, self.batch_size):
            # 1. Create a batch of indices for this worker's chunk of data.
            batch_indices = list(range(i, min(i + self.batch_size, end_idx)))

            if not batch_indices:
                continue

            # 2. Make ONE network call to fetch all data for this batch of indices.
            #    This is where you use your batch_read functionality.
            batch_data = self.us_gaap_store.batch_lookup_by_indices(batch_indices)

            # 3. Process the raw batch data into tensors.
            processed_batch = self._process_batch(batch_data)

            # 4. Yield the complete, processed batch.
            yield processed_batch


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
