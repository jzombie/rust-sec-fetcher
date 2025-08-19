import torch
from models.pytorch.narrative_stack.common.dataset import BaseUsGaapIterableDataset

import numpy as np
from simd_r_drive_ws_client import DataStoreWsClient
from us_gaap_store import UsGaapStore


class Stage1CellDataset(BaseUsGaapIterableDataset):
    def _get_count(self) -> int:
        temp_client = DataStoreWsClient(
            self.simd_r_drive_server_config.host,
            self.simd_r_drive_server_config.port,
        )
        temp_store = UsGaapStore(temp_client)
        row_count = temp_store.get_triplet_count()
        del temp_client  # No longer needed

        return row_count


    def _yield_data(self, row_indices: list[int]):
        """
        The core logic for an iterable-style dataset. This iterator fetches data
        in large, efficient chunks and processes them with vectorized operations
        before yielding single items.
        """

        for i in range(0, len(row_indices), self.lookup_batch_size):
            index_batch = row_indices[i : i + self.lookup_batch_size]
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

            # FIXME: Preferably don't hardcode
            semantic_embedding_dim = 243

            # Pre-allocate final NumPy arrays to avoid intermediate Python lists
            embeddings_np = np.empty((num_valid, semantic_embedding_dim), dtype=np.float32)
            values_np = np.empty((num_valid, 1), dtype=np.float32)

            # Fill the pre-allocated arrays in a loop
            for idx, item in enumerate(valid_data):
                embeddings_np[idx] = item.embedding
                values_np[idx] = item.log_scaled_value

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
                scaler = item_meta.scaler
                concept_unit = (item_meta.concept, item_meta.uom)

                yield (i_cell, x, y, scaler, concept_unit)
