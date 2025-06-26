import numpy as np
from collections import defaultdict


class AggregateStats:
    def __init__(self, device):
        self.device = device
        self._eps = 1e-8
        self._per_tag = defaultdict(
            lambda: {
                "mae_sum": 0.0,
                "abs_sum": 0.0,
                "n": 0,
                "sum_y_true": 0.0,
                "sum_y_pred": 0.0,
                "sum_y_true2": 0.0,
                "sum_y_pred2": 0.0,
                "sum_y_true_y_pred": 0.0,
            }
        )
        self.z_sum = 0.0
        self.z_sq_sum = 0.0
        self.z_count = 0

    def update(self, tags, y_pred_batch, y_true_batch, z_norm_batch):
        """
        tags: List[Tuple[str, str]]
        y_pred_batch, y_true_batch, z_norm_batch: Tensors of shape [B]
        """
        y_pred = y_pred_batch.detach().cpu().numpy().astype(np.float32)
        y_true = y_true_batch.detach().cpu().numpy().astype(np.float32)
        z_norm = z_norm_batch.detach().cpu().numpy().astype(np.float32)
        tags = np.array(tags)

        # Group indices by unique tag
        unique_tags, tag_indices = np.unique(tags, return_inverse=True)
        # num_samples = len(y_true)

        for i, tag in enumerate(unique_tags):
            idxs = np.where(tag_indices == i)[0]
            yp = y_pred[idxs]
            yt = y_true[idxs]

            mae_sum = np.abs(yp - yt).sum()
            abs_sum = np.abs(yt).sum()

            stats = self._per_tag[tuple(tag)]
            stats["mae_sum"] += mae_sum
            stats["abs_sum"] += abs_sum

            # Manual R² computation is used here instead of torchmetrics.R2Score
            # to address performance issues specific to this use case—
            # namely, avoiding per-sample `.update()` overhead and GPU sync stalls
            # during large-scale per-tag aggregation. This optimization is targeted
            # and does not imply that torchmetrics.R2Score is unsuitable
            # for other scenarios or tasks.

            stats["n"] += len(idxs)
            stats["sum_y_true"] += yt.sum()
            stats["sum_y_pred"] += yp.sum()
            stats["sum_y_true2"] += np.sum(np.square(yt.astype(np.float64)))
            stats["sum_y_pred2"] += np.sum(np.square(yp.astype(np.float64)))

            # TODO: Fix `RuntimeWarning: overflow encountered in multiply` error
            stats["sum_y_true_y_pred"] += np.sum(yt * yp)

        self.z_sum += z_norm.sum()
        self.z_sq_sum += np.sum(z_norm**2)
        self.z_count += z_norm.shape[0]

    def median_relative_mae(self):
        vals = []
        for v in self._per_tag.values():
            if v["abs_sum"] > 0:
                vals.append(v["mae_sum"] / (v["abs_sum"] + self._eps))
        return float(np.median(vals)) if vals else 0.0

    def worst_median_relative_mae(self, top_frac=0.05):
        vals = []
        for v in self._per_tag.values():
            if v["abs_sum"] > 0:
                rel_mae = v["mae_sum"] / (v["abs_sum"] + self._eps)
                vals.append(rel_mae)

        if not vals:
            return 0.0

        vals.sort(reverse=True)  # higher MAE is worse
        k = max(1, int(len(vals) * top_frac))
        return float(np.median(vals[:k]))

    def _compute_r2_values(self):
        """
        Compute R² values for all tags with at least 2 samples.
        Filters out NaN or infinite results caused by overflow or invalid math.
        Optimized for speed using NumPy vectorization.
        """
        # 1. Extract data into NumPy arrays from the defaultdict values
        # Filter out entries where n < 2 upfront
        valid_data_dicts = [v for v in self._per_tag.values() if v["n"] >= 2]

        if not valid_data_dicts:
            return []

        # Create a dictionary of arrays for vectorized operations
        # Ensure all data is float64 from the start for calculations
        n_fp = np.array([d["n"] for d in valid_data_dicts], dtype=np.float64)
        sum_y_true = np.array([d["sum_y_true"] for d in valid_data_dicts], dtype=np.float64)
        sum_y_true2 = np.array([d["sum_y_true2"] for d in valid_data_dicts], dtype=np.float64)
        sum_y_pred2 = np.array([d["sum_y_pred2"] for d in valid_data_dicts], dtype=np.float64)
        sum_y_true_y_pred = np.array([d["sum_y_true_y_pred"] for d in valid_data_dicts], dtype=np.float64)
        eps = np.float64(self._eps) # eps remains a scalar

        # 2. Perform vectorized calculations
        mean_y = sum_y_true / n_fp
        mean_y_sq = np.square(mean_y) # Using np.square for consistency and potential overflow handling if values are large

        # Initialize a mask for valid computations
        valid_mask = np.isfinite(mean_y_sq)

        product = n_fp * mean_y_sq
        valid_mask &= np.isfinite(product) # Update mask

        ss_tot = sum_y_true2 - product
        ss_res = sum_y_pred2 - 2 * sum_y_true_y_pred + sum_y_true2

        denom = ss_tot + eps

        # Apply conditions using boolean indexing
        valid_denom_mask = (denom > 0) & np.isfinite(denom)
        final_mask = valid_mask & valid_denom_mask

        # Apply the combined mask to the arrays before division
        ss_res_filtered = ss_res[final_mask]
        denom_filtered = denom[final_mask]

        # Avoid division by zero by ensuring denom_filtered is not empty if final_mask is True
        if denom_filtered.size == 0:
            return []

        r2_values = 1.0 - (ss_res_filtered / denom_filtered)

        # 3. Filter for finite r2 values and return as a list
        return r2_values[np.isfinite(r2_values)].tolist()

    def median_r2(self):
        """
        Compute the median R² across all valid tags.
        """
        vals = self._compute_r2_values()
        return float(np.median(vals)) if vals else 0.0

    def worst_median_r2(self, bottom_frac=0.05):
        """
        Compute the median R² among the bottom `bottom_frac` of tags.
        """
        vals = self._compute_r2_values()
        if not vals:
            return 0.0
        vals.sort()
        k = max(1, int(len(vals) * bottom_frac))
        return float(np.median(vals[:k]))

    def z_norm_mean_std(self):
        if self.z_count == 0:
            return 0.0, 0.0

        mean = self.z_sum / max(self.z_count, 1)
        mean_sq = self.z_sq_sum / max(self.z_count, 1)
        var = max(mean_sq - mean**2, 0.0)
        return mean, var**0.5

    def reset(self):
        self.__init__(self.device)
