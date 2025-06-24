import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# from torchmetrics.regression import R2Score
from collections import defaultdict


class AggregateStats:
    def __init__(self, device):
        self.device = device
        self._eps = 1e-8
        self._per_tag = defaultdict(
            lambda: {
                "mae_sum": 0.0,
                "abs_sum": 0.0,
                # old: "r2": R2Score().to(device),
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

    # TODO: Remove old, slower update method
    # def update(self, tags, y_pred_batch, y_true_batch, z_norm_batch):
    #     """
    #     tags: List[Tuple[str, str]]
    #     y_pred_batch, y_true_batch, z_norm_batch: Tensors of shape [B]
    #     """
    #     y_pred_batch = y_pred_batch.detach().cpu()
    #     y_true_batch = y_true_batch.detach().cpu()
    #     z_norm_batch = z_norm_batch.detach().cpu()

    #     for i, tag in enumerate(tags):
    #         stats = self._per_tag[tag]
    #         abs_err = torch.abs(y_pred_batch[i] - y_true_batch[i]).item()
    #         abs_target = torch.abs(y_true_batch[i]).item()

    #         stats["mae_sum"] += abs_err
    #         stats["abs_sum"] += abs_target

    #         # Manual R² computation is used here instead of torchmetrics.R2Score
    #         # to address performance issues specific to this use case—
    #         # namely, avoiding per-sample `.update()` overhead and GPU sync stalls
    #         # during large-scale per-tag aggregation. This optimization is targeted
    #         # and does not imply that torchmetrics.R2Score is unsuitable
    #         # for other scenarios or tasks.

    #         # stats["r2"].update(y_pred_batch[i].unsqueeze(0), y_true_batch[i].unsqueeze(0))
    #         stats["n"] += 1
    #         yt = y_true_batch[i].item()
    #         yp = y_pred_batch[i].item()
    #         stats["sum_y_true"] += yt
    #         stats["sum_y_pred"] += yp
    #         stats["sum_y_true2"] += yt * yt
    #         stats["sum_y_pred2"] += yp * yp
    #         stats["sum_y_true_y_pred"] += yt * yp

    #     self.z_sum += z_norm_batch.sum().item()
    #     self.z_sq_sum += (z_norm_batch ** 2).sum().item()
    #     self.z_count += z_norm_batch.size(0)

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
            # stats["sum_y_true2"] += np.sum(yt ** 2) # TODO: Fix potential RuntimeWarning: overflow encountered in square
            # stats["sum_y_pred2"] += np.sum(yp ** 2) # TODO: Fix potential RuntimeWarning: overflow encountered in square
            stats["sum_y_true2"] += np.sum(np.square(yt.astype(np.float64)))
            stats["sum_y_pred2"] += np.sum(np.square(yp.astype(np.float64)))

            # TODO: Fix: RuntimeWarning: overflow encountered in multiply:  stats["sum_y_true_y_pred"] += np.sum(yt * yp)
             # Solution with clamping:
            # 1. Perform multiplication in float64 to avoid overflow during product calculation
            # 2. Clamp the result to FLOAT32_MAX (or any desired max value)
            # 3. Sum the clamped values
            # product_yt_yp_float64 = yt.astype(np.float64) * yp.astype(np.float64)
            # clamped_product = np.clip(product_yt_yp_float64, a_min=None, a_max=FLOAT32_MAX)
            # stats["sum_y_true_y_pred"] += np.sum(clamped_product)
            
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


class ValueAttentionModulator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Linear(1, emb_dim)  # scalar → query
        self.key = nn.Linear(emb_dim, emb_dim)  # x_emb → key
        self.value = nn.Linear(emb_dim, emb_dim)  # x_emb → value
        self.scale = emb_dim**0.5

    def forward(self, x_emb, x_val):
        q = self.query(x_val)  # [B, D]
        k = self.key(x_emb)  # [B, D]
        v = self.value(x_emb)  # [B, D]

        attn = torch.sum(q * k, dim=-1, keepdim=True) / self.scale  # [B, 1]
        attn_weights = torch.sigmoid(attn)  # [B, 1] ∈ (0, 1)

        return attn_weights * v + (1 - attn_weights) * x_emb


class EncoderWithAttention(nn.Module):
    def __init__(self, emb_dim, latent_dim, dropout_rate=0.0):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.LayerNorm(emb_dim * 4),
            nn.Dropout(p=dropout_rate),
        )
        self.attn = ValueAttentionModulator(emb_dim * 4)
        self.bottleneck = nn.Linear(emb_dim * 4, latent_dim)

    def forward(self, x_emb, x_val):
        h = self.expand(x_emb)  # [B, 4D]
        h = self.attn(h, x_val)  # scalar-conditioned attention
        z = self.bottleneck(h)  # [B, latent_dim]
        return z


class DecoderWithAttention(nn.Module):
    def __init__(self, latent_dim, emb_dim, dropout_rate=0.0):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.LayerNorm(latent_dim * 4),
            nn.Dropout(p=dropout_rate),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim * 4, num_heads=4, batch_first=True
        )

        # Separate heads
        self.emb_head = nn.Sequential(
            nn.Linear(latent_dim * 4, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.val_head = nn.Sequential(
            nn.Linear(latent_dim * 4, emb_dim), nn.GELU(), nn.Linear(emb_dim, 1)
        )

    def forward(self, z):
        """
        z: [B, latent_dim]
        Returns:
            recon_emb: [B, emb_dim]
            recon_val: [B, 1]
        """
        h = self.expand(z)  # [B, 4D]
        h_attn, _ = self.attn(h.unsqueeze(1), h.unsqueeze(1), h.unsqueeze(1))
        h = h_attn.squeeze(1)

        recon_emb = self.emb_head(h)
        recon_val = self.val_head(h)
        return recon_emb, recon_val


# LightningModule
class Stage1Autoencoder(pl.LightningModule):
    EPSILON = torch.finfo(torch.float32).eps

    def __init__(
        self,
        input_dim=244,
        latent_dim=256,
        encoder_dropout_rate=0.0,
        value_dropout_rate=0.0,
        # lr=0.00023072200683712404,
        lr=5e-5,
        min_lr=1e-6,
        # lr=0.00002307,
        batch_size=64,
        gradient_clip=0.5,
        alpha_embed=1.0,
        alpha_value=1.0,
        # embedding_noise_std=0.0, # 0.02 is roughly ~0.951 cosine sim difference for 243 dimeensions; 0.01 is roughly ~0.99
        weight_decay=5.220603379116996e-07,
        lr_annealing_epochs=15,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["median_scaled_val", "mean_emb"])

        # --> Register mean_s and mean_emb as buffers <--
        # self.register_buffer("median_scaled_val", median_scaled_val)
        # self.register_buffer("mean_emb", mean_emb)

        # self.value_proj = nn.Sequential(
        #     nn.Linear(1, 32),
        #     nn.GELU(),
        #     nn.Linear(32, self.hparams.latent_dim),
        #     nn.LayerNorm(self.hparams.latent_dim)
        # )

        # May 1, 2025 original
        # self.value_proj = nn.Sequential(
        #     nn.Linear(1, 32),
        #     nn.GELU(),
        #     nn.Linear(32, latent_dim),
        #     nn.LayerNorm(latent_dim)
        # )

        # self.value_proj = nn.Sequential(
        #     nn.Linear(1, 32),
        #     nn.GELU(),
        #     nn.Linear(32, 64),
        #     nn.GELU(),
        #     nn.Linear(64, latent_dim),
        #     nn.LayerNorm(latent_dim)
        # )

        # self.attended_interaction = nn.Sequential(
        #     nn.Linear(latent_dim * 2, latent_dim * 2),
        #     nn.GELU(),
        #     nn.Linear(latent_dim * 2, latent_dim),
        #     nn.LayerNorm(latent_dim)
        # )

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim - 1 + self.hparams.latent_dim, 256),
        #     nn.GELU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(256, latent_dim)
        # )

        # self.gate = nn.Sequential(
        #     nn.Linear(latent_dim * 2, latent_dim),
        #     nn.GELU(),
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.Sigmoid()
        # )

        # self.fusion_logits = nn.Parameter(torch.zeros(3))
        # self.fusion_dim = latent_dim * 3
        # self.post_fusion_norm = nn.LayerNorm(self.fusion_dim)

        # self.joint_input_dim = (input_dim - 1) * 2
        # self.joint_input_norm = nn.LayerNorm(self.joint_input_dim)

        # self.encoder = nn.Sequential(
        #     nn.Linear(self.joint_input_dim, latent_dim * 4),
        #     nn.GELU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(latent_dim * 4, latent_dim * 2),
        #     nn.GELU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(latent_dim * 2, latent_dim)
        # )

        # self.encoder = nn.Sequential(
        #     # nn.Linear(self.joint_input_dim, latent_dim * 4),
        #     # nn.GELU(),
        #     # nn.LayerNorm(latent_dim * 4),
        #     # nn.Dropout(p=encoder_dropout_rate),

        #     nn.Linear(input_dim - 1, input_dim * 4),
        #     nn.GELU(),
        #     nn.LayerNorm(input_dim * 4),
        #     nn.Dropout(p=encoder_dropout_rate),

        #     nn.Linear(input_dim * 4, latent_dim)  # Bottleneck
        # )

        self.encoder = EncoderWithAttention(
            emb_dim=input_dim - 1,
            latent_dim=latent_dim,
            dropout_rate=encoder_dropout_rate,
        )

        self.decoder = DecoderWithAttention(
            latent_dim=latent_dim,
            emb_dim=input_dim - 1,
            dropout_rate=value_dropout_rate,
        )

        # self.embedding_decoder = DecoderWithAttention(
        #     latent_dim=latent_dim,
        #     emb_dim=input_dim - 1,
        #     dropout_rate=encoder_dropout_rate
        # )

        # self.embedding_decoder = nn.Sequential(
        #     nn.Linear(latent_dim, (input_dim - 1) * 2),
        #     nn.GELU(),
        #     # nn.Dropout(p=dropout_rate),
        #     nn.Linear((input_dim - 1) * 2, input_dim - 1)
        # )

        # May 1, 2025 original
        # self.value_decoder = nn.Sequential(
        #     nn.Linear(latent_dim, 32),
        #     nn.GELU(),
        #     nn.Dropout(p=value_dropout_rate),
        #     nn.Linear(32, 1)
        # )

        # self.value_decoder = nn.Sequential(
        #     nn.Linear(latent_dim, (input_dim - 1) * 2),
        #     nn.GELU(),
        #     # nn.Dropout(p=dropout_rate),
        #     nn.Linear((input_dim - 1) * 2, 1)
        # )

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()  # MAELoss

        self._agg_train_stats = self.create_aggregate_stats()
        self._agg_val_stats = self.create_aggregate_stats()

    def create_aggregate_stats(self):
        return AggregateStats(self.device)

    def encode(self, x):
        x_emb = x[:, :-1]
        x_val = x[:, -1].unsqueeze(1)
        z = self.encoder(x_emb, x_val)
        return F.normalize(z, p=2, dim=1)

    # def encode(self, x):
    #     # x shape: [batch_size, input_dim]
    #     x_emb = x[:, :-1] # Non-scaled embeddings
    #     x_val = x[:, -1].unsqueeze(1) # Scaled values

    #     # Inject Gaussian noise into embedding (during training only)
    #     # if self.training:
    #     #     x_emb = x_emb + torch.randn_like(x_emb) * self.hparams.embedding_noise_std

    #     # val_proj = self.value_proj(x_val)
    #     value_modulated = self.value_attention(x_emb, x_val)
    #     joint_input = torch.cat([x_emb, value_modulated], dim=1)

    #     # joint_input = torch.cat([x_emb, value_weighted_x_emb], dim=1)
    #     # joint_input = self.joint_input_norm(joint_input)

    #     z = self.encoder(joint_input)

    #     # Apply L2 normalization along the feature dimension (dim=1)
    #     # p=2 is the default for L2 norm, but explicitly stated for clarity
    #     z = F.normalize(z, p=2, dim=1)

    #     return z

    # def decode(self, z):
    #     recon_emb = self.embedding_decoder(z)
    #     recon_val = self.value_decoder(z)

    #     return recon_emb, recon_val

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)

        recon_emb, recon_val = self.decode(z)
        return recon_emb, recon_val, z

    def compute_losses(self, x, target, scaler, concept_units, train=False):
        recon_emb, recon_val, z = self(x)

        target_emb = target[:, :-1]
        target_val = target[:, -1].unsqueeze(1)

        # TODO: Figure out how to work out a potentially missing scaler
        if scaler and isinstance(scaler, (list, tuple)):
            recon_val_np = recon_val.detach().cpu().numpy()
            target_val_np = target_val.detach().cpu().numpy()

            # Inverse transform per sample
            recon_val_orig = np.stack(
                [
                    s.inverse_transform(r.reshape(-1, 1)).flatten()
                    for s, r in zip(scaler, recon_val_np)
                ]
            )
            target_val_orig = np.stack(
                [
                    s.inverse_transform(t.reshape(-1, 1)).flatten()
                    for s, t in zip(scaler, target_val_np)
                ]
            )

            recon_val_orig = torch.tensor(
                recon_val_orig, dtype=torch.float32, device=recon_val.device
            )
            target_val_orig = torch.tensor(
                target_val_orig, dtype=torch.float32, device=target_val.device
            )
        else:
            raise Exception("Scaler not implemented")

        # non-scaled
        embedding_loss = self.loss_fn(recon_emb, target_emb)

        # scaled
        value_loss = self.loss_fn(recon_val, target_val)

        total_loss = (
            self.hparams.alpha_embed * embedding_loss
            + self.hparams.alpha_value * value_loss
        )

        # non-scaled
        cos_sim_emb = cosine_similarity(recon_emb, target_emb, dim=1).mean()
        euclidean_dist_emb = torch.norm(recon_emb - target_emb, dim=1).mean()

        # non-scaled
        z_norm = torch.norm(z, dim=1)

        agg_stats = self._agg_train_stats if train else self._agg_val_stats
        agg_stats.update(
            tags=concept_units,
            y_pred_batch=recon_val_orig.view(-1),
            y_true_batch=target_val_orig.view(-1),
            z_norm_batch=z_norm,
        )

        relative_mae_value = agg_stats.median_relative_mae()
        worst_relative_mae_value = agg_stats.worst_median_relative_mae()

        r2_value = agg_stats.median_r2()
        worst_r2_value = agg_stats.worst_median_r2()

        z_norm_mean, z_norm_std = agg_stats.z_norm_mean_std()

        return (
            total_loss,
            embedding_loss,
            value_loss,
            cos_sim_emb,
            euclidean_dist_emb,
            relative_mae_value,
            worst_relative_mae_value,
            r2_value,
            worst_r2_value,
            z_norm_mean,
            z_norm_std,
        )

    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            x, target, scaler, concept_units = batch
        elif len(batch) == 3:
            x, target, scaler = batch
            concept_units = None
        else:
            x, target = batch
            scaler = None
            concept_units = None

        (
            total_loss,
            embedding_loss,
            value_loss,
            cos_sim_emb,
            euclidean_dist_emb,
            relative_mae_value,
            worst_relative_mae_value,
            r2_value,
            worst_r2_value,
            z_norm_mean,
            z_norm_std,
        ) = self.compute_losses(x, target, scaler, concept_units, train=True)

        self.log(
            "train_loss", total_loss, prog_bar=True, batch_size=self.hparams.batch_size
        )
        # self.log("train_overlap_loss", overlap_loss, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log(
            "train_embedding_loss", embedding_loss, batch_size=self.hparams.batch_size
        )
        self.log("train_value_loss", value_loss, batch_size=self.hparams.batch_size)
        self.log(
            "train_embedding_cos_sim", cos_sim_emb, batch_size=self.hparams.batch_size
        )
        self.log(
            "train_embedding_euclidean",
            euclidean_dist_emb,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "train_value_relative_mae_running",
            relative_mae_value,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "train_worst_value_relative_mae_running",
            worst_relative_mae_value,
            batch_size=self.hparams.batch_size,
        )
        self.log("train_value_r2_running", r2_value, batch_size=self.hparams.batch_size)
        self.log(
            "train_worst_value_r2_running",
            worst_r2_value,
            batch_size=self.hparams.batch_size,
        )
        self.log("train_z_norm_mean", z_norm_mean, batch_size=self.hparams.batch_size)
        self.log("train_z_norm_std", z_norm_std, batch_size=self.hparams.batch_size)

        self.log(
            "train_loss_epoch",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            x, target, scaler, concept_units = batch
        elif len(batch) == 3:
            x, target, scaler = batch
            concept_units = None
        else:
            x, target = batch
            scaler = None
            concept_units = None

        (
            total_loss,
            embedding_loss,
            value_loss,
            cos_sim_emb,
            euclidean_dist_emb,
            relative_mae_value,
            worst_relative_mae_value,
            r2_value,
            worst_r2_value,
            z_norm_mean,
            z_norm_std,
        ) = self.compute_losses(x, target, scaler, concept_units, train=False)

        self.log(
            "val_loss", total_loss, prog_bar=True, batch_size=self.hparams.batch_size
        )
        # self.log("val_overlap_loss", overlap_loss, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log(
            "val_embedding_loss", embedding_loss, batch_size=self.hparams.batch_size
        )
        self.log("val_value_loss", value_loss, batch_size=self.hparams.batch_size)
        self.log(
            "val_embedding_cos_sim", cos_sim_emb, batch_size=self.hparams.batch_size
        )
        self.log(
            "val_embedding_euclidean",
            euclidean_dist_emb,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "val_value_relative_mae_running",
            relative_mae_value,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            "val_worst_value_relative_mae_running",
            worst_relative_mae_value,
            batch_size=self.hparams.batch_size,
        )
        self.log("val_value_r2_running", r2_value, batch_size=self.hparams.batch_size)
        self.log(
            "val_worst_value_r2_running",
            worst_r2_value,
            batch_size=self.hparams.batch_size,
        )
        self.log("val_z_norm_mean", z_norm_mean, batch_size=self.hparams.batch_size)
        self.log("val_z_norm_std", z_norm_std, batch_size=self.hparams.batch_size)

        self.log(
            "val_loss_epoch",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        return total_loss

    # Note: PyTorch Lightning doesn't support logging from `on_train_epoch_start`. Use `on_train_epoch_end` for logging, instead.
    def on_train_epoch_start(self):
        self._agg_train_stats.reset()

        print("Current LR: ", self.get_current_lr())

    def on_validation_start(self):
        self._agg_val_stats.reset()

    def on_train_epoch_end(self):
        # Log learning rate of first param group
        current_lr = self.get_current_lr()
        self.log("lr_adjusted", current_lr, prog_bar=True)

    def get_current_lr(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        return current_lr

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Use CosineAnnealingLR with T_max=15 and eta_min=1e-6 (matches your 15 epochs)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.lr_annealing_epochs,
            eta_min=self.hparams.min_lr,
        )

        # TODO: Replace scheduler with CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1)?

        return [optimizer], [scheduler]