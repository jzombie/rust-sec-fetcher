import torch
from torch.utils.data import IterableDataset
from simd_r_drive_ws_client import DataStoreWsClient
from config import SimdRDriveServerConfig
from us_gaap_store import UsGaapStore


class Stage2StackDataset(IterableDataset):
    """
    Iterable dataset for Stage 2 autoencoder training.

    Each sample corresponds to a single document row (i_row), containing
    exactly six category-specific latent stacks. The categories are always
    returned in the following fixed order:

        0: credit::instant
        1: credit::duration
        2: debit::instant
        3: debit::duration
        4: none::instant
        5: none::duration

    Each output is a tensor of shape [N_i, latent_dim], where N_i may vary
    across stacks. Empty categories are returned as tensors with shape [0, D].
    """

    # TODO: Dedupe
    CATEGORY_ORDER = [
        "credit::instant",
        "credit::duration",
        "debit::instant",
        "debit::duration",
        "none::instant",
        "none::duration",
    ]

    def __init__(
        self,
        simd_r_drive_server_config: SimdRDriveServerConfig,
        shuffle: bool = False,
        lookup_batch_size: int = 64,
    ):
        """
        Args:
            simd_r_drive_server_config (SimdRDriveServerConfig): Connection info
                for the SIMD R Drive server.
            shuffle (bool): If True, rows are shuffled on each epoch.
        """
        super().__init__()
        self.simd_r_drive_server_config = simd_r_drive_server_config
        self.shuffle = shuffle
        self.lookup_batch_size = lookup_batch_size
        self.epoch = 0  # Used to seed per-epoch deterministic shuffling

        self.data_store_client = None
        self.us_gaap_store = None

        # Create a temporary client to fetch total row count
        temp_client = DataStoreWsClient(
            simd_r_drive_server_config.host,
            simd_r_drive_server_config.port,
        )
        temp_store = UsGaapStore(temp_client)
        self.row_count = temp_store.get_stage2_row_count()
        del temp_client  # No longer needed

    def __len__(self):
        """
        Returns:
            int: The total number of i_row samples.

        This allows PyTorch Lightning to display progress bars and
        manage epoch control.
        """
        return self.row_count

    def _init_worker(self):
        """
        Initializes the WebSocket client and store in each worker.

        Avoids cross-thread usage of the same socket connection.
        """
        if self.data_store_client is None:
            self.data_store_client = DataStoreWsClient(
                self.simd_r_drive_server_config.host,
                self.simd_r_drive_server_config.port,
            )
            self.us_gaap_store = UsGaapStore(self.data_store_client)

    def __iter__(self):
        """
        Iterates over the dataset, yielding a tuple of 6 latent stacks
        per document row.

        Yields:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] where
            each Tensor has shape [N_i, latent_dim] or [0, latent_dim].
        """
        self._init_worker()

        # Get worker identity for multi-process dataloaders
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        if worker_id == 0:
            self.epoch += 1

        row_indices = list(range(self.row_count))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(42 + self.epoch)
            row_indices = torch.randperm(self.row_count, generator=g).tolist()

        # Split rows evenly across workers
        per_worker = int(self.row_count / num_workers + 1)
        start = worker_id * per_worker
        end = min(start + per_worker, self.row_count)
        row_indices = row_indices[start:end]

        for batch_start in range(0, len(row_indices), self.lookup_batch_size):
            rows_batch = row_indices[batch_start : batch_start + self.lookup_batch_size]
            cell_map = self.us_gaap_store.get_stage2_category_stacks_cell_indices_by_rows(rows_batch)
            all_latents = self.us_gaap_store.get_cached_stage2_category_stacks_latents_by_cell_indices(cell_map)

            for sample in all_latents:
                stack_map = {k: None for k in self.CATEGORY_ORDER}

                latent_dim = None
                for cat in sample.categories:
                    if len(cat.latent_stack) > 0:
                        latent_dim = cat.latent_stack.shape[1]
                        break
                if latent_dim is None:
                    continue

                for k in stack_map:
                    stack_map[k] = torch.empty((0, latent_dim), dtype=torch.float32)

                for cat in sample.categories:
                    key = f"{cat.balance_type}::{cat.period_type}"
                    if key in stack_map and len(cat.latent_stack) > 0:
                        stack_map[key] = torch.tensor(cat.latent_stack, dtype=torch.float32)

                yield tuple(stack_map[k] for k in self.CATEGORY_ORDER)
