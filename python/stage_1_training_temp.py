# %%
import os
from pathlib import Path

# This snippet ensures consistent import paths across environments.
# When running notebooks via JupyterLab's web UI, the current working
# directory is often different (e.g., /notebooks) compared to VS Code,
# which typically starts at the project root. This handles that by 
# retrying the import after changing to the parent directory.
# 
# Include this at the top of every notebook to standardize imports
# across development environments.

try:
    from utils.os import chdir_to_git_root
except ModuleNotFoundError:
    os.chdir(Path.cwd().parent)
    print(f"Retrying import from: {os.getcwd()}")
    from utils.os import chdir_to_git_root

chdir_to_git_root("python")

print(os.getcwd())

# %%
# # UMAP visualization
# umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
# umap_2d = umap_model.fit_transform(compressed)

# plt.figure(figsize=(10, 6))
# plt.scatter(umap_2d[:, 0], umap_2d[:, 1], c=labels, cmap="tab10", s=5)
# plt.title("Concept/UOM Embeddings Clustered")
# plt.show()

# %%
# df = pd.DataFrame({
#     "concept": [c for c, _ in concept_unit_pairs],
#     "unit": [u for _, u in concept_unit_pairs],
#     "cluster": labels
# })
# grouped = df.groupby("cluster")

# for cluster_id, group in grouped:
#     print(f"\nCluster {cluster_id} ({len(group)} items):")
#     print(group.head(10).to_string(index=False))

# noise = df[df["cluster"] == -1]

# print(f"Noise points: {len(noise)}")


# %%
# noise_points = df[df["cluster"] == -1][["concept", "unit"]].reset_index(drop=True)

# noise_points.to_csv("noise_points.csv")

# %%
# import numpy as np

# # Save both embeddings and tuples
# np.savez_compressed(
#     "data/stage1_latents.npz",
#     keys=np.array([f"{c}::{u}" for c, u in concept_unit_pairs]),
#     embeddings=compressed,
#     concept_unit_value_tuples=np.array(concept_unit_value_tuples, dtype=object)
# )

# print(f"✅ Saved {len(concept_unit_value_tuples):,} tuples and {len(compressed):,} embeddings to 'stage1_latents.npz'")


# %%
import numpy as np

# Load saved latent data
data = np.load("data/stage1_latents.npz", allow_pickle=True)

# Build embedding map
embedding_map = {
    tuple(key.split("::", 1)): vec
    for key, vec in zip(data["keys"], data["embeddings"])
}

# Load concept-unit-value tuples
concept_unit_value_tuples = data["concept_unit_value_tuples"].tolist()


# %%
# embedding_map

# %%
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.preprocessing import QuantileTransformer

# NOTE: Scaling is performed before train/val split to ensure that every
# (concept, unit) pair receives a fitted StandardScaler. If we split first,
# some (concept, unit) groups might not appear in the training set at all,
# making it impossible to fit their scalers later — leading to missing
# or unscalable entries downstream.
#
# IMPORTANT: The goal of this Stage 1 encoder is not to "predict" values,
# but to learn meaningful latent representations of (concept, unit, value)
# tuples. These embeddings are intended for use in downstream models and
# alignment stages, not for direct forecasting or regression tasks.

# Step 1: Group values per (concept, unit)
grouped = defaultdict(list)
for concept, unit, value in concept_unit_value_tuples:
    grouped[(concept, unit)].append(value)

# Step 2: Fit individual scalers and transform
scalers = {}
scaled_tuples = []

for key, vals in tqdm(grouped.items(), desc="Scaling per concept/unit"):
    vals_np = np.array(vals).reshape(-1, 1)

    n_quantiles_val = min(len(vals), 1000)
    if n_quantiles_val < 2 and len(vals) >= 2:
        n_quantiles_val = 2

    scaler = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=n_quantiles_val,
        subsample=len(vals),
        random_state=42
    )

    scaled_vals = scaler.fit_transform(vals_np).flatten()
    scalers[key] = scaler

    scaled_tuples.extend((key[0], key[1], v) for v in scaled_vals)

# %%
scalers

# %%
# Debug scaled stats

from collections import defaultdict
import numpy as np

# Reconstruct per (concept, unit) from scaled_tuples
reconstructed = defaultdict(list)
all_vals = []

for concept, unit, val in scaled_tuples:
    key = (concept, unit)
    reconstructed[key].append(val)
    all_vals.append(val)

total_items = 0
total_no_variance_items = 0

# Analyze per group
for key, vals in reconstructed.items():
    vals_np = np.array(vals)

    if not np.all(np.isfinite(vals_np)):
        print(f"[BAD SCALE] {key} has non-finite scaled values!")
        print("  Sample:", vals_np[:5])
        continue

    max_val = np.max(vals_np)
    min_val = np.min(vals_np)
    mean_val = np.mean(vals_np)

    if max_val > 1e6 or min_val < -1e6:
        print(f"[OUTLIER SCALE] {key} range is extreme:")
        print("  min:", min_val, "max:", max_val, "mean:", mean_val)

    if np.allclose(max_val, min_val):
        # print(f"[FLAT SCALE] {key} has no variance ({len(vals)} item(s)):")
        # print("  value:", max_val)
        total_no_variance_items += len(vals)

    total_items += len(vals)

# Global stats
all_vals_np = np.array(all_vals)

print("\n=== GLOBAL STATS ===")
print(f"Global max: {np.max(all_vals_np)}")
print(f"Global min: {np.min(all_vals_np)}")
print(f"Global mean: {np.mean(all_vals_np)}")
print(f"Global std: {np.std(all_vals_np)}")

print(f"Total items: {total_items}")
print(f"Total no variance items: {total_no_variance_items}")


# %%
import math
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset
from utils.pytorch import seed_everything
import numpy as np
from torch.nn.functional import cosine_similarity, l1_loss
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchmetrics.regression import R2Score
from collections import defaultdict

class AggregateStats:
    def __init__(self, device):
        self.device = device
        self._eps = 1e-8
        self._per_tag = defaultdict(lambda: {
            "mae_sum": 0.0,
            "abs_sum": 0.0,
            # old: "r2": R2Score().to(device),
            "n": 0,
            "sum_y_true": 0.0,
            "sum_y_pred": 0.0,
            "sum_y_true2": 0.0,
            "sum_y_pred2": 0.0,
            "sum_y_true_y_pred": 0.0,
        })
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
            stats["sum_y_true_y_pred"] += np.sum(yt * yp)

        self.z_sum += z_norm.sum()
        self.z_sq_sum += np.sum(z_norm ** 2)
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
        Filters out NaN or infinite results caused by invalid divisions.
        """
        vals = []
        for v in self._per_tag.values():
            n = v["n"]
            if n < 2:
                continue  # R² is undefined for n < 2

            mean_y = v["sum_y_true"] / n
            ss_tot = v["sum_y_true2"] - n * mean_y**2
            ss_res = (
                v["sum_y_pred2"]
                - 2 * v["sum_y_true_y_pred"]
                + v["sum_y_true2"]
            )

            denom = ss_tot + self._eps
            r2 = 1.0 - (ss_res / denom) if denom > 0 else -np.inf

            # Append only valid finite values (skip NaN, +inf, -inf)
            if math.isfinite(r2):
                vals.append(r2)
        return vals

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



# Stage 1 dataset: concept+uom embedding + value
class ConceptValueDataset(Dataset):
    def __init__(self, scaled_tuples, embedding_lookup, device: torch.tensor, scalers=None, return_scaler=False):
        """
        Dataset for (concept, unit, value) triplets with optional per-sample scaler.

        :param scaled_tuples: List of (concept, unit, scaled_value) tuples
        :param embedding_lookup: Dict[(concept, unit)] -> embedding np.array
        :param device: torch device tensor to place tensors on
        :param scalers: Optional dict of (concept, unit) -> QuantileTransformer
        :param return_scaler: If True, return the scaler used per sample
        """
        self.rows = scaled_tuples
        self.lookup = embedding_lookup
        self.device = device
        self.scalers = scalers
        self.return_scaler = return_scaler

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        concept, unit, value = self.rows[idx]

        try:
            embedding = self.lookup[(concept, unit)]
        except KeyError:
            raise ValueError(f"Missing embedding for ({concept}, {unit})")

        x = torch.tensor(np.concatenate([embedding, [value]]), dtype=torch.float32,
                         device=self.device)

        # For autoencoders, target y is typically the same as input x
        y = x.clone() 

        if self.return_scaler:
            scaler_obj = self.scalers.get((concept, unit))
            return x, y, scaler_obj, (concept, unit)
        return x, y, (concept, unit)


def collate_with_scaler(batch):
    xs, ys, scalers_list, concept_units = zip(*batch)
    return torch.stack(xs), torch.stack(ys), scalers_list, list(concept_units)


class ValueAttentionModulator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Linear(1, emb_dim)        # scalar → query
        self.key = nn.Linear(emb_dim, emb_dim)    # x_emb → key
        self.value = nn.Linear(emb_dim, emb_dim)  # x_emb → value
        self.scale = emb_dim ** 0.5

    def forward(self, x_emb, x_val):
        q = self.query(x_val)                    # [B, D]
        k = self.key(x_emb)                      # [B, D]
        v = self.value(x_emb)                    # [B, D]

        attn = torch.sum(q * k, dim=-1, keepdim=True) / self.scale  # [B, 1]
        attn_weights = torch.sigmoid(attn)       # [B, 1] ∈ (0, 1)

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
        h = self.expand(x_emb)                  # [B, 4D]
        h = self.attn(h, x_val)                 # scalar-conditioned attention
        z = self.bottleneck(h)                  # [B, latent_dim]
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
            embed_dim=latent_dim * 4,
            num_heads=4,
            batch_first=True
        )

        # Separate heads
        self.emb_head = nn.Sequential(
            nn.Linear(latent_dim * 4, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )
        self.val_head = nn.Sequential(
            nn.Linear(latent_dim * 4, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, 1)
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
            lr_annealing_epochs=15
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['median_scaled_val', 'mean_emb'])

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
            dropout_rate=encoder_dropout_rate
        )

        self.decoder = DecoderWithAttention(
            latent_dim=latent_dim,
            emb_dim=input_dim - 1,
            dropout_rate=value_dropout_rate
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
        self.loss_fn = nn.L1Loss() # MAELoss

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

        if scaler and isinstance(scaler, (list, tuple)):
            recon_val_np = recon_val.detach().cpu().numpy()
            target_val_np = target_val.detach().cpu().numpy()

            # Inverse transform per sample
            recon_val_orig = np.stack([
                s.inverse_transform(r.reshape(-1, 1)).flatten()
                for s, r in zip(scaler, recon_val_np)
            ])
            target_val_orig = np.stack([
                s.inverse_transform(t.reshape(-1, 1)).flatten()
                for s, t in zip(scaler, target_val_np)
            ])

            recon_val_orig = torch.tensor(recon_val_orig, dtype=torch.float32,
                                        device=recon_val.device)
            target_val_orig = torch.tensor(target_val_orig, dtype=torch.float32,
                                        device=target_val.device)
        else:
            raise Exception("Scaler not implemented")

        # non-scaled
        embedding_loss = self.loss_fn(recon_emb, target_emb)

        # scaled
        value_loss = self.loss_fn(recon_val, target_val)
    
        total_loss = (
            self.hparams.alpha_embed * embedding_loss +
            self.hparams.alpha_value * value_loss
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
            z_norm_batch=z_norm
        )

        relative_mae_value = agg_stats.median_relative_mae()
        worst_relative_mae_value = agg_stats.worst_median_relative_mae()

        r2_value = agg_stats.median_r2()
        worst_r2_value = agg_stats.worst_median_r2()

        z_norm_mean, z_norm_std = agg_stats.z_norm_mean_std()

        return total_loss, embedding_loss, value_loss, cos_sim_emb, euclidean_dist_emb, relative_mae_value, worst_relative_mae_value, r2_value, worst_r2_value, z_norm_mean, z_norm_std


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

        total_loss, embedding_loss, value_loss, cos_sim_emb, euclidean_dist_emb, relative_mae_value, worst_relative_mae_value, r2_value, worst_r2_value, z_norm_mean, z_norm_std = (
            self.compute_losses(x, target, scaler, concept_units, train=True)
        )

        self.log("train_loss", total_loss, prog_bar=True, batch_size=self.hparams.batch_size)
        # self.log("train_overlap_loss", overlap_loss, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("train_embedding_loss", embedding_loss, batch_size=self.hparams.batch_size)
        self.log("train_value_loss", value_loss, batch_size=self.hparams.batch_size)
        self.log("train_embedding_cos_sim", cos_sim_emb, batch_size=self.hparams.batch_size)
        self.log("train_embedding_euclidean", euclidean_dist_emb, batch_size=self.hparams.batch_size)
        self.log("train_value_relative_mae_running", relative_mae_value, batch_size=self.hparams.batch_size)
        self.log("train_worst_value_relative_mae_running", worst_relative_mae_value, batch_size=self.hparams.batch_size)
        self.log("train_value_r2_running", r2_value, batch_size=self.hparams.batch_size)
        self.log("train_worst_value_r2_running", worst_r2_value, batch_size=self.hparams.batch_size)
        self.log("train_z_norm_mean", z_norm_mean, batch_size=self.hparams.batch_size)
        self.log("train_z_norm_std", z_norm_std, batch_size=self.hparams.batch_size)

        self.log("train_loss_epoch", total_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)
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

        total_loss, embedding_loss, value_loss, cos_sim_emb, euclidean_dist_emb, relative_mae_value, worst_relative_mae_value, r2_value, worst_r2_value, z_norm_mean, z_norm_std = (
            self.compute_losses(x, target, scaler, concept_units, train=False)
        )

        self.log("val_loss", total_loss, prog_bar=True, batch_size=self.hparams.batch_size)
        # self.log("val_overlap_loss", overlap_loss, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("val_embedding_loss", embedding_loss, batch_size=self.hparams.batch_size)
        self.log("val_value_loss", value_loss, batch_size=self.hparams.batch_size)
        self.log("val_embedding_cos_sim", cos_sim_emb, batch_size=self.hparams.batch_size)
        self.log("val_embedding_euclidean", euclidean_dist_emb, batch_size=self.hparams.batch_size)
        self.log("val_value_relative_mae_running", relative_mae_value, batch_size=self.hparams.batch_size)
        self.log("val_worst_value_relative_mae_running", worst_relative_mae_value, batch_size=self.hparams.batch_size)
        self.log("val_value_r2_running", r2_value, batch_size=self.hparams.batch_size)
        self.log("val_worst_value_r2_running", worst_r2_value, batch_size=self.hparams.batch_size)
        self.log("val_z_norm_mean", z_norm_mean, batch_size=self.hparams.batch_size)
        self.log("val_z_norm_std", z_norm_std, batch_size=self.hparams.batch_size)

        self.log("val_loss_epoch", total_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        # Use CosineAnnealingLR with T_max=15 and eta_min=1e-6 (matches your 15 epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.lr_annealing_epochs, eta_min=self.hparams.min_lr)

        # TODO: Replace scheduler with CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1)?
        
        return [optimizer], [scheduler]


# %%
# Tuning

# import os
# import optuna
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# from torch.utils.data import DataLoader
# from utils.pytorch import get_device

# device = get_device()

# # === CONFIG ===
# OUTPUT_PATH = "data/stage1"
# os.makedirs(OUTPUT_PATH, exist_ok=True)
# OPTUNA_DB_PATH = os.path.join(OUTPUT_PATH, "optuna_study.db")
# EPOCHS = 3
# PATIENCE = 5
# VAL_SPLIT = 0.2

# def objective(trial):
#     batch_size = trial.suggest_int("batch_size", 8, 64, step=8)
#     lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
#     latent_dim = trial.suggest_int("latent_dim", 32, 128, step=32)
#     dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2, step=0.1)
#     weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
#     gradient_clip = trial.suggest_float("gradient_clip", 0.0, 1.0, step=0.1)

#     # # 80/20 Train/Val Split
#     # split = int(len(scaled_tuples) * (1 - VAL_SPLIT))
#     # train_data = scaled_tuples[:split]
#     # val_data = scaled_tuples[split:]

#      # === Sample Subset for Faster Debugging ===
#     SAMPLE_SIZE = 500_000
#     subset = scaled_tuples[:SAMPLE_SIZE]
    
#     # 80/20 Train/Val Split
#     split = int(len(subset) * (1 - VAL_SPLIT))
#     train_data = subset[:split]
#     val_data = subset[split:]

#     train_loader = DataLoader(
#         ConceptValueDataset(train_data, embedding_map, device=device, value_noise_std=0.005, train=True),
#         batch_size=batch_size,
#         shuffle=True
#     )
    
#     val_loader = DataLoader(
#         ConceptValueDataset(val_data, embedding_map, device=device, value_noise_std=0.00, train=False),
#         batch_size=batch_size,
#         shuffle=False
#     )

#     input_dim = len(next(iter(embedding_map.values()))) + 1

#     model = Stage1Autoencoder(
#         input_dim=input_dim,
#         latent_dim=latent_dim,
#         dropout_rate=dropout_rate,
#         lr=lr,
#         batch_size=batch_size,
#         weight_decay=weight_decay,
#         gradient_clip=gradient_clip
#     )

#     early_stop_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE, verbose=True, mode="min")

#     model_checkpoint = ModelCheckpoint(
#         dirpath=OUTPUT_PATH,
#         filename="best_model_trial_{trial.number}",
#         monitor="val_loss",
#         mode="min",
#         save_top_k=1,
#         verbose=True
#     )

#     trainer = pl.Trainer(
#         max_epochs=EPOCHS,
#         logger=TensorBoardLogger(OUTPUT_PATH, name="stage1_autoencoder"),
#         callbacks=[early_stop_callback, model_checkpoint],
#         accelerator="auto",
#         devices=1,
#         gradient_clip_val=gradient_clip
#     )

#     trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
#     return trainer.callback_metrics["val_loss"].item()

# # === Optuna Study ===
# study = optuna.create_study(direction="minimize",
#                             storage=f"sqlite:///{OPTUNA_DB_PATH}",
#                             load_if_exists=True)
# study.optimize(objective, n_trials=25)

# print("Best params:", study.best_params)
# print("Best trial value:", study.best_trial.value)


# %%
# from torch.utils.data import DataLoader

# # Instantiate dataset
# dataset = ConceptValueDataset(scaled_tuples, embedding_map)

# # Sample inspection
# sample_x, sample_y = dataset[0]
# print("Sample input:", sample_x)
# print("Min:", sample_x.min().item(), "Max:", sample_x.max().item())
# print("Mean:", sample_x.mean().item(), "Std:", sample_x.std().item())
# print("Input dim:", sample_x.shape[0], "Target dim:", sample_y.shape[0])

# # Optional: test batch loading
# loader = DataLoader(dataset, batch_size=4)
# for xb, yb in loader:
#     print("Batch shape:", xb.shape)
#     break


# %%
train_data = list(scaled_tuples)
val_data = train_data

# # TODO: Get rid of most of this and just evaluate on the full train_data set

# from collections import defaultdict
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split # Make sure this is imported
# import numpy as np # For potential use, though train_test_split handles counts

# # Target fraction for the validation set from each group
# VALIDATION_FRACTION = 0.25
# RANDOM_STATE = 42

# # Assume `scaled_tuples` is the list of (concept, unit, scaled_value)
# # generated by your previous preprocessing script. It contains ALL your scaled data.

# # 1. Training data will be 100% of scaled_tuples.
# # We create a list copy for clarity, though direct use is also possible.
# # The DataLoader for training will typically shuffle this.
# train_data = list(scaled_tuples)

# # 2. Validation data will be a stratified sample, aiming for ~25% of each group's items.
# val_data = []

# # Group all tuples by (concept, unit) to perform stratified sampling for the validation set.
# # This grouping is based on the full scaled_tuples list.
# grouped_for_val_sampling = defaultdict(list)
# for concept, unit, value in scaled_tuples: # Iterate over the full dataset
#     grouped_for_val_sampling[(concept, unit)].append(value)

# # For each group, select a VALIDATION_FRACTION sample of its values to contribute to val_data.
# # These samples will also be part of the 100% train_data.
# for (concept, unit), values_in_group in grouped_for_val_sampling.items():
#     if not values_in_group: # Skip if group is empty
#         continue

#     num_samples_in_group = len(values_in_group)
#     group_val_values = [] # To store validation values from this specific group

#     if num_samples_in_group == 1:
#         # If a group has only one sample, and we want validation data (VALIDATION_FRACTION > 0),
#         # we include this single sample in the validation set.
#         # This means all single-sample groups will be represented in this stratified validation set.
#         if VALIDATION_FRACTION > 0:
#             group_val_values = list(values_in_group) # Take the single value
#     elif num_samples_in_group > 1:
#         # For groups with more than one sample, use train_test_split to get
#         # the desired fraction for the validation set from this group.
#         # The '_group_train_dummy' part is not used for constructing val_data here.
#         # We ensure shuffling before split if not done by train_test_split, but it shuffles by default.
#         _group_train_dummy, sampled_val_values_for_group = train_test_split(
#             values_in_group,       # Values from the current (concept, unit) group
#             test_size=VALIDATION_FRACTION,
#             random_state=RANDOM_STATE,
#             shuffle=True          # Ensure shuffling for random sampling within the group
#         )
#         group_val_values = sampled_val_values_for_group
    
#     # Extend the global val_data list with the (concept, unit, value) tuples from this group's sample
#     if group_val_values: # If any values were selected for validation from this group
#         val_data.extend([(concept, unit, v) for v in group_val_values])

# print(f"Total items for training (100%): {len(train_data)}")
# print(f"Total items for validation (stratified ~25% sample): {len(val_data)}")

# # Now you have:
# # - `train_data`: A list of all your (concept, unit, scaled_value) tuples (100%).
# # - `val_data`: A list containing a stratified sample, where each (concept, unit) group
# #               contributes approximately 25% of its items to this validation set.
# #               The total size will be roughly 25% of the original dataset, but may vary
# #               slightly due to per-group integer rounding and handling of small groups.

# # You would then use `train_data` and `val_data` in your PyTorch DataLoaders.
# # The `train_loader` will use `train_data` (100%) with shuffle=True.
# # The `val_loader` will use `val_data` (the ~25% stratified sample) with shuffle=False.

# %%
# For debugging, only

# Example: keep only 10,000 training and 2,000 validation samples
# train_data = train_data[:100_000]
# val_data = val_data[:2_000]
# val_data = train_data

# print(f"Truncated train_data: {len(train_data)}")
# print(f"Truncated val_data: {len(val_data)}")

# %%
# import torch
# from utils.pytorch import get_device # Assuming you have this

# device = get_device() # Make sure device is defined

# # Define Median Scaled Value (which is 0.0 after RobustScaler)
# median_scaled_val_tensor = torch.tensor(0.0, dtype=torch.float32, device=device)
# print(f"Using median_scaled_val: {median_scaled_val_tensor.item()}")

# # === Calculate Mean Embedding (mean_emb) ===
# # (Calculation for mean_emb_tensor remains the same as before)
# # Get all unique (concept, unit) keys from train_data
# train_keys = set((item[0], item[1]) for item in train_data)
# # Get the corresponding embeddings
# train_embeddings = [embedding_map[key] for key in train_keys if key in embedding_map]

# if not train_embeddings:
#     raise ValueError("No valid embeddings found for training data keys!")

# # Stack embeddings into a numpy array and calculate the mean vector
# train_embeddings_np = np.stack(train_embeddings)
# mean_emb_numpy = np.mean(train_embeddings_np, axis=0)
# # Convert to a tensor on the correct device
# mean_emb_tensor = torch.tensor(mean_emb_numpy, dtype=torch.float32, device=device)
# print(f"Calculated mean_emb shape: {mean_emb_tensor.shape}")
# print(f"Calculated mean_emb value (first 10 elements): {mean_emb_tensor[:10]}")

# %%
# Training

import os
import optuna
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from utils.pytorch import get_device

device = get_device()

# === CONFIG ===
OUTPUT_PATH = "data/stage1_23_(no_pre_dedupe)"
os.makedirs(OUTPUT_PATH, exist_ok=True)
OPTUNA_DB_PATH = os.path.join(OUTPUT_PATH, "optuna_study.db")
EPOCHS = 1000
PATIENCE = 20 # Compensate for long annealing period + some

ckpt_path = f"{OUTPUT_PATH}/stage1_resume-v4.ckpt"
# ckpt_path = f"{OUTPUT_PATH}/manual_resumed_checkpoint.ckpt"
# ckpt_path = None

model = Stage1Autoencoder.load_from_checkpoint(ckpt_path,
    # lr=5e-5,
    lr=2.5e-6,
    # min_lr=1e-6,
    min_lr=1e-7
)
# model = Stage1Autoencoder()

batch_size = model.hparams.batch_size
gradient_clip = model.hparams.gradient_clip

# train_loader = DataLoader(
#     ConceptValueDataset(train_data, embedding_map, device=device, value_noise_std=0.005, train=True),
#     batch_size=batch_size,
#     shuffle=True
# )

# val_loader = DataLoader(
#     ConceptValueDataset(val_data, embedding_map, device=device, value_noise_std=0.00, train=False),
#     batch_size=batch_size,
#     shuffle=False
# )
train_loader = DataLoader(
    ConceptValueDataset(
        train_data,
        embedding_map,
        device=device,
        scalers=scalers,
        return_scaler=True
    ),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_with_scaler
)

val_loader = DataLoader(
    ConceptValueDataset(
        val_data,
        embedding_map,
        device=device,
        scalers=scalers,
        return_scaler=True
    ),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_with_scaler
)


input_dim = len(next(iter(embedding_map.values()))) + 1

early_stop_callback = EarlyStopping(monitor="val_loss_epoch", patience=PATIENCE, verbose=True, mode="min")

model_checkpoint = ModelCheckpoint(
    dirpath=OUTPUT_PATH,
    filename="stage1_resume",
    monitor="val_loss_epoch",
    mode="min",
    save_top_k=1,
    verbose=True
)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    logger=TensorBoardLogger(OUTPUT_PATH, name="stage1_autoencoder"),
    callbacks=[early_stop_callback, model_checkpoint],
    accelerator="auto",
    devices=1,
    gradient_clip_val=gradient_clip,
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    #
    # ckpt_path=ckpt_path # TODO: Uncomment if resuming training AND wanting to restore existing model configuration
)



# %%
# trainer.save_checkpoint(f"{OUTPUT_PATH}/manual_resumed_checkpoint.ckpt")

# %% [markdown]
# # Conceptual Draft
# 
# Stage 1 learns semantic+quantitative embeddings for individual concept/unit/value triplets.
# 
# Stage 2 learns how to aggregate and contextualize those embeddings into higher-order units (i.e., financial statements).
# 
# Stage 3 learns how to model temporal dynamics and structural evolution across filings — a full hierarchy of understanding.
# 
# This pipeline could encode an entire company's financial narrative into vector space.
# 
# It’s structured like language modeling, but for accounting — and that’s what makes it potentially groundbreaking.

# %%
# import numpy as np
# import torch

# import numpy as np
# import torch

# # Where correlation matrix is on the full z, and the `corr_value` is derived specifically from the input value dimension
# # IMPORTANT: This should only be used with this "stage 1" model
# def analyze_latent_correlation_matrix_streaming(model, val_loader, device):
#     model.eval()
#     model.to(device)

#     latent_dim = model.hparams.latent_dim

#     count = 0
#     mean_z = np.zeros(latent_dim)
#     m2_z = np.zeros(latent_dim)
#     cov_z = np.zeros((latent_dim, latent_dim))

#     # For value correlation
#     mean_val = 0.0
#     m2_val = 0.0
#     cov_val = np.zeros(latent_dim)

#     for batch in val_loader:
#         x, y, _ = batch
#         x = x.to(device)
#         y = y.to(device)

#         z = model.encode(x).detach().cpu().numpy()  # [B, D]
#         v = y[:, -1].detach().cpu().numpy()         # [B]

#         for zi, vi in zip(z, v):
#             count += 1

#             # === Update latent stats (Welford) ===
#             delta_z = zi - mean_z
#             mean_z += delta_z / count
#             m2_z += delta_z * (zi - mean_z)

#             # === Update cov_z (outer product) ===
#             cov_z += np.outer(delta_z, zi - mean_z)

#             # === Update value stats ===
#             delta_v = vi - mean_val
#             mean_val += delta_v / count
#             m2_val += delta_v * (vi - mean_val)

#             # === Update cov_val ===
#             cov_val += delta_z * (vi - mean_val)

#         # break

#     var_z = m2_z / (count - 1)
#     var_val = m2_val / (count - 1)
#     cov_z /= (count - 1)
#     cov_val /= (count - 1)

#     std_z = np.sqrt(var_z + 1e-8)
#     std_val = np.sqrt(var_val + 1e-8)

#     corr_matrix = cov_z / (std_z[:, None] * std_z[None, :])
#     corr_value = cov_val / (std_z * std_val)

#     return corr_matrix, corr_value


# print("\n=== Computing latent correlation matrix... ===")
# corr_matrix, corr_value = analyze_latent_correlation_matrix_streaming(model, val_loader, device=device)

# import matplotlib.pyplot as plt

# def plot_correlation_matrix(corr_matrix, title="Latent Dimension Correlation Matrix"):
#     plt.figure(figsize=(10, 8))
#     plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
#     plt.colorbar(shrink=0.5)
#     plt.title(title)
#     plt.xlabel("Latent Dim")
#     plt.ylabel("Latent Dim")
#     plt.tight_layout()
#     plt.show()

# plot_correlation_matrix(corr_matrix)

# top_k = 128

# def print_top_latent_correlations(corr_matrix, top_k=top_k):
#     dim = corr_matrix.shape[0]
#     pairs = []

#     for i in range(dim):
#         for j in range(i + 1, dim):
#             corr = corr_matrix[i, j]
#             pairs.append(((i, j), corr))

#     top_corrs = sorted(pairs, key=lambda x: -abs(x[1]))[:top_k]

#     print(f"\nTop {top_k} most correlated latent dimension pairs:")
#     for (i, j), corr in top_corrs:
#         print(f"z[{i:03d}] ↔ z[{j:03d}]: corr = {corr:.4f}")


# print_top_latent_correlations(corr_matrix, top_k=top_k)


# top_dims = sorted(enumerate(corr_value), key=lambda x: -abs(x[1]))[:top_k]
# print("\nTop latent dimensions most correlated with scaled value:")
# for i, c in top_dims:
#     print(f"z[{i:03d}]: corr = {c:.4f}")


# plt.figure(figsize=(12, 4))
# plt.plot(np.sort(np.abs(corr_value))[::-1], marker='o')
# plt.title("Absolute Correlation of Latent Dims with Value")
# plt.xlabel("Sorted Latent Dimension")
# plt.ylabel("Absolute Correlation")
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# %%



