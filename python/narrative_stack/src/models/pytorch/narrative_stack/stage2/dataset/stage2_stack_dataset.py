import torch
from models.pytorch.narrative_stack.common.dataset import (
    BaseUsGaapIterableDataset,
)
from simd_r_drive_ws_client import DataStoreWsClient
from us_gaap_store import UsGaapStore, STAGE2_CATEGORY_STACKS


class Stage2StackDataset(BaseUsGaapIterableDataset):
    """
    Iterable dataset for Stage 2 autoencoder training.

    Each sample corresponds to a single document row (i_row) and yields a
    tuple of 6 tensors, one per category in STAGE2_CATEGORY_STACKS.
    Each tensor has shape [N_i, D]. Empty categories are [0, D].
    """

    def _get_count(self) -> int:
        """
        Return total number of Stage 2 rows as an int. If the backing
        store returns None or a non-int, coerce to 0 to avoid
        range(None) TypeError in the base iterator.
        """
        temp_client = DataStoreWsClient(
            self.simd_r_drive_server_config.host,
            self.simd_r_drive_server_config.port,
        )
        temp_store = UsGaapStore(temp_client)
        row_count = temp_store.get_stage2_row_count()
        del temp_client  # explicit close

        if row_count is None:
            return 0
        try:
            rc = int(row_count)
        except (TypeError, ValueError):
            rc = 0
        if rc < 0:
            rc = 0
        return rc

    def _yield_data(self, row_indices: list[int]):
        """
        Yield one tuple of 6 latent stacks per row index in row_indices.

        For each row:
          - Determine latent_dim from the first non-empty stack.
          - Yield tensors in fixed category order.
          - Skip rows with no data in all categories.
        """
        for start in range(0, len(row_indices), self.lookup_batch_size):
            rows_batch = row_indices[start : start + self.lookup_batch_size]
            if not rows_batch:
                continue

            cell_map = (
                self.us_gaap_store
                .get_stage2_category_stacks_cell_indices_by_rows(rows_batch)
            )
            all_latents = (
                self.us_gaap_store
                .get_cached_stage2_category_stacks_latents_by_cell_indices(
                    cell_map
                )
            )

            for row in all_latents:
                latent_dim = None
                for key in STAGE2_CATEGORY_STACKS:
                    if key in row and len(row[key]) > 0:
                        latent_dim = row[key].shape[1]
                        break

                if latent_dim is None:
                    # No data for this row in any category; skip.
                    continue

                out = []
                for k in STAGE2_CATEGORY_STACKS:
                    if k in row and len(row[k]) > 0:
                        t = torch.tensor(row[k], dtype=torch.float32)
                    else:
                        t = torch.empty((0, latent_dim), dtype=torch.float32)

                    # Debug guard against NaNs.
                    if torch.isnan(t).any():
                        raise ValueError("NaNs found in stack %s" % k)

                    out.append(t)

                yield tuple(out)
