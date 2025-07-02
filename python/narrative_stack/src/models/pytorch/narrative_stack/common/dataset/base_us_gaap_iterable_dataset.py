import torch
from torch.utils.data import IterableDataset
from simd_r_drive_ws_client import DataStoreWsClient
from config import SimdRDriveServerConfig
from us_gaap_store import UsGaapStore

class BaseUsGaapIterableDataset(IterableDataset):
    def __init__(
        self,
        simd_r_drive_server_config: SimdRDriveServerConfig,
        shuffle: bool = False,
        lookup_batch_size: int = 64,
    ):
        self.simd_r_drive_server_config = simd_r_drive_server_config
        self.shuffle = shuffle
        self.lookup_batch_size = lookup_batch_size
        self.epoch = 0
        self.data_store_client = None
        self.us_gaap_store = None
        self._count = self._get_count()

    def _init_worker(self):
        if self.data_store_client is None:
            self.data_store_client = DataStoreWsClient(
                self.simd_r_drive_server_config.host,
                self.simd_r_drive_server_config.port,
            )
            self.us_gaap_store = UsGaapStore(self.data_store_client)

    def _get_count(self) -> int:
        """Must be overridden by subclasses"""
        raise NotImplementedError

    def _yield_data(self, row_indices: list[int]):
        """Must be overridden by subclasses"""
        raise NotImplementedError

    def __len__(self):
        return self._count

    def __iter__(self):
        self._init_worker()

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        if worker_id == 0:
            self.epoch += 1

        # Generate index list
        indices = list(range(self._count))

        # Apply deterministic shuffle per epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(42 + self.epoch)
            indices = torch.randperm(self._count, generator=g).tolist()

        # Split index space between workers (non-overlapping)
        # Each worker processes a distinct, contiguous subset
        per_worker = int(self._count / num_workers + 1)
        start = worker_id * per_worker
        end = min(start + per_worker, self._count)
        worker_indices = indices[start:end]

        # Yield items in lookup batches
        for i in range(0, len(worker_indices), self.lookup_batch_size):
            batch = worker_indices[i : i + self.lookup_batch_size]
            yield from self._yield_data(batch)
