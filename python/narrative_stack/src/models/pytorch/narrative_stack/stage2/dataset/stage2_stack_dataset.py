import torch
from torch.utils.data import IterableDataset
from typing import List
from simd_r_drive_ws_client import DataStoreWsClient
from config import SimdRDriveServerConfig
from us_gaap_store import UsGaapStore


class Stage2StackDataset(IterableDataset):
    """
    Iterable dataset for Stage 2 autoencoder training.

    Yields per sample:
        - stacks: List[Tensor] of shape [N_i, latent_dim]
        - masks: List[BoolTensor] of shape [N_i]
        - balance_idxs: List[LongTensor] of shape [N_i]
        - period_idxs: List[LongTensor] of shape [N_i]
    """

    def __init__(
        self,
        simd_r_drive_server_config: SimdRDriveServerConfig,
        shuffle: bool = False,
    ):
        super().__init__()
        self.simd_r_drive_server_config = simd_r_drive_server_config
        self.shuffle = shuffle
        self.epoch = 0 # Initialize epoch count

        self.data_store_client = None
        self.us_gaap_store = None
        

        # Get the total count once in the main process
        temp_client = DataStoreWsClient(simd_r_drive_server_config.host, simd_r_drive_server_config.port)
        temp_store = UsGaapStore(temp_client)
        self.row_count = temp_store.get_stage2_row_count()

        # Disconnect temp client
        del temp_client

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        PyTorch Lightning will then use this total to display the
        epoch progress bar.
        """
        return self.row_count

    def _init_worker(self):
        """Initializes the client and store within the worker process."""
        if self.data_store_client is None:
            # Each worker gets its own client connection
            self.data_store_client = DataStoreWsClient(self.simd_r_drive_server_config.host, self.simd_r_drive_server_config.port)
            self.us_gaap_store = UsGaapStore(self.data_store_client)

    def __iter__(self):
        self._init_worker()

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        if worker_id == 0:
            self.epoch += 1

        # TODO: Fix; it doesn't work this way
        all_samples = self.us_gaap_store.get_cached_stage2_category_stacks_latents()

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(42 + self.epoch)
            indices = torch.randperm(len(all_samples), generator=g).tolist()
        else:
            indices = list(range(len(all_samples)))

        per_worker = int(len(indices) / num_workers + 1)
        start = worker_id * per_worker
        end = min(start + per_worker, len(indices))
        indices = indices[start:end]

        balance_map = {"credit": 0, "debit": 1, "none": 2}
        period_map = {"duration": 0, "instant": 1, "none": 2}

        for idx in indices:
            sample = all_samples[idx]

            stacks: List[torch.Tensor] = []
            masks: List[torch.Tensor] = []
            balance_idxs: List[torch.Tensor] = []
            period_idxs: List[torch.Tensor] = []

            for cat in sample.categories:
                latents = torch.tensor(cat.latent_stack, dtype=torch.float32)
                if latents.ndim != 2:
                    continue

                N = latents.size(0)
                stacks.append(latents)
                masks.append(torch.ones(N, dtype=torch.bool))

                bal_idx = balance_map[cat.balance_type]
                per_idx = period_map[cat.period_type]

                balance_idxs.append(torch.full((N,), bal_idx, dtype=torch.long))
                period_idxs.append(torch.full((N,), per_idx, dtype=torch.long))

            if len(stacks) > 0:
                yield stacks, masks, balance_idxs, period_idxs
