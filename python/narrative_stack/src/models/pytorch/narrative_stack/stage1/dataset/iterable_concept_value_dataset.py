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
    i_cell_list, xs, ys, scalers_list, concept_units = zip(*batch)

    # Stack the tensors to create a batch
    xs_batch = torch.stack(xs)
    ys_batch = torch.stack(ys)

    # The scalers and concept_units remain as lists
    return i_cell_list, xs_batch, ys_batch, scalers_list, list(concept_units)


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
                i_cell = valid_data[j].i_cell
                scaler = item_meta.scaler if self.return_scaler else None
                concept_unit = (item_meta.concept, item_meta.uom)

                yield (i_cell, x, y, scaler, concept_unit)
