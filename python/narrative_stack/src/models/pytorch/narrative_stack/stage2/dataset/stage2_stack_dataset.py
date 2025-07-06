import torch
from models.pytorch.narrative_stack.common.dataset import BaseUsGaapIterableDataset
from simd_r_drive_ws_client import DataStoreWsClient
from us_gaap_store import UsGaapStore, STAGE2_CATEGORY_STACKS


class Stage2StackDataset(BaseUsGaapIterableDataset):
    """
    Iterable dataset for Stage 2 autoencoder training.

    Each sample corresponds to a single document row (i_row), containing
    exactly six category-specific latent stacks. The categories are always
    returned in the same fixed order as `STAGE2_CATEGORY_STACKS`

    Each output is a tensor of shape [N_i, latent_dim], where N_i may vary
    across stacks. Empty categories are returned as tensors with shape [0, D].
    """

    def _get_count(self) -> int:
        temp_client = DataStoreWsClient(
            self.simd_r_drive_server_config.host,
            self.simd_r_drive_server_config.port,
        )
        temp_store = UsGaapStore(temp_client)
        row_count = temp_store.get_stage2_row_count()
        del temp_client  # No longer needed

        return row_count

    def _yield_data(self, row_indices: list[int]):
        """
        Iterates over the dataset, yielding a tuple of 6 latent stacks
        per document row.

        Yields:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] where
            each Tensor has shape [N_i, latent_dim] or [0, latent_dim].
        """

        for batch_start in range(0, len(row_indices), self.lookup_batch_size):
            rows_batch = row_indices[batch_start : batch_start + self.lookup_batch_size]
            cell_map = self.us_gaap_store.get_stage2_category_stacks_cell_indices_by_rows(rows_batch)
            all_latents = self.us_gaap_store.get_cached_stage2_category_stacks_latents_by_cell_indices(cell_map)

            for row in all_latents:
                stack_map = {k: None for k in STAGE2_CATEGORY_STACKS}

                latent_dim = None
                for key in STAGE2_CATEGORY_STACKS:
                    if key in row and len(row[key]) > 0:
                        latent_dim = row[key].shape[1]
                        break
                if latent_dim is None:
                    continue

                for k in STAGE2_CATEGORY_STACKS:
                    stack_map[k] = (
                        torch.tensor(row[k], dtype=torch.float32)
                        if k in row and len(row[k]) > 0
                        else torch.empty((0, latent_dim), dtype=torch.float32)
                    )

                    # TODO: Debug check; may need improving
                    assert not torch.isnan(stack_map[k]).any(), f"NaNs found in stack {k}"

                yield tuple(stack_map[k] for k in STAGE2_CATEGORY_STACKS)
