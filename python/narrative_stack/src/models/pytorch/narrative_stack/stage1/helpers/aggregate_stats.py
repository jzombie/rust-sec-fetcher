import numpy as np
from collections import defaultdict


class AggregateStats:
    """
    Batch-wise accumulator for per-tag regression metrics.

    Keeps running totals for:
    - Mean Absolute Error (MAE) and relative MAE
    - Coefficient of determination (R2)
    - z-score mean and standard deviation

    Designed for large-scale evaluation where per-sample metric calls would
    create heavy Python overhead or GPU<->CPU synchronisation stalls.
    """

    def __init__(self, device):
        """
        Parameters
        ----------
        device : torch.device | str
            Torch device on which model tensors live.  Stored only for reference;
            all accumulation happens on CPU with NumPy.
        """

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
        Add a mini-batch to the running totals.

        Parameters
        ----------
        tags : list[tuple[str, str]]
            Identifier per sample, for example (concept, unit).  Used as dict key.
        y_pred_batch : torch.Tensor
            Model predictions, shape [B].
        y_true_batch : torch.Tensor
            Ground-truth targets, shape [B].
        z_norm_batch : torch.Tensor
            Pre-computed z-scores that track scale drift, shape [B].

        Notes
        -----
        * All tensors are detached, moved to CPU, then cast to float32 for MAE
        aggregation and float64 for R2 numerics.
        * A small epsilon (=1e-8) avoids divide-by-zero in relative MAE.
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
        """
        Return the median relative MAE across all tags.

        Relative MAE for a tag = sum|y_hat - y| / (sum|y| + eps).

        Returns
        -------
        float
            0 <= value < inf.  Zero means perfect reconstruction; higher is worse.
        """

        vals = []
        for v in self._per_tag.values():
            if v["abs_sum"] > 0:
                vals.append(v["mae_sum"] / (v["abs_sum"] + self._eps))
        return float(np.median(vals)) if vals else 0.0

    def worst_median_relative_mae(self, top_frac=0.05):
        """
        Median relative MAE for the worst-performing fraction of tags.

        Parameters
        ----------
        top_frac : float, default 0.05
            Fraction (0-1) of highest relative-MAE tags to include.

        Returns
        -------
        float
            Median of that subset; useful for spotting long-tail failure cases.
        """

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
        Vectorised R2 for every tag with >= 2 samples.

        Returns
        -------
        list[float]
            Finite R2 values; list may be empty if no valid tags exist.

        Notes
        -----
        * Uses float64 throughout to reduce numeric cancellation.
        * Non-finite or overflow results are filtered out with np.isfinite.
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
        Median R2 across all valid tags.

        Returns
        -------
        float
            Range is (-inf, 1].  One is perfect, zero matches the naïve mean,
            values below zero are worse than a constant predictor.
        """

        vals = self._compute_r2_values()
        return float(np.median(vals)) if vals else 0.0

    def worst_median_r2(self, bottom_frac=0.05):
        """
        Median R2 for the lowest-scoring fraction of tags.

        Parameters
        ----------
        bottom_frac : float, default 0.05
            Fraction (0-1) of tags with the poorest R2 to include.

        Returns
        -------
        float
            Median of that subset; highlights systematic under-performance.
        """

        vals = self._compute_r2_values()
        if not vals:
            return 0.0
        vals.sort()
        k = max(1, int(len(vals) * bottom_frac))
        return float(np.median(vals[:k]))

    def z_norm_mean_std(self):
        """
        Mean and standard deviation of observed z-scores.

        Returns
        -------
        tuple[float, float]
            (mean, std).  If no samples have been seen, both values are 0.0.
        """

        if self.z_count == 0:
            return 0.0, 0.0

        mean = self.z_sum / max(self.z_count, 1)
        mean_sq = self.z_sq_sum / max(self.z_count, 1)
        var = max(mean_sq - mean**2, 0.0)
        return mean, var**0.5

    def reset(self):
        """
        Clear all running totals and start from scratch.

        Equivalent to creating a new AggregateStats instance with the same device.
        """

        self.__init__(self.device)
