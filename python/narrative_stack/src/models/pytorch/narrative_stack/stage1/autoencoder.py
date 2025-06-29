import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .helpers import AggregateStats, DecoderWithAttention, EncoderWithAttention



class Stage1Autoencoder(pl.LightningModule):
    EPSILON = torch.finfo(torch.float32).eps

    def __init__(
        self,
        input_dim=244,
        latent_dim=256,
        encoder_dropout_rate=0.0,
        value_dropout_rate=0.0,
        lr=5e-5,
        min_lr=1e-6,
        batch_size=64,
        gradient_clip=0.5,
        alpha_embed=1.0,
        alpha_value=1.0,
        weight_decay=5.220603379116996e-07,
        lr_annealing_epochs=15,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Access through `self.hparams`; `save_hyperparameters` fills it.
        # These locals exist only to prevent “unused” linter warnings.
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.encoder_dropout_rate=encoder_dropout_rate
        self.value_dropout_rate=value_dropout_rate
        self.lr=lr
        self.min_lr=min_lr
        self.batch_size=batch_size
        self.gradient_clip=gradient_clip
        self.alpha_embed=alpha_embed
        self.alpha_value=alpha_value
        self.weight_decay=weight_decay
        self.lr_annealing_epochs=lr_annealing_epochs

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

    def decode(self, z):
        return self.decoder(z)

    # pylint: disable=arguments-differ
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
            # FIXME: A more precise error would be beneficial
            raise NotImplementedError("A scaler was not implemented for a concept/unit pair")

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

    def training_step(self, batch, _batch_idx):
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

    def validation_step(self, batch, _batch_idx):
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.lr_annealing_epochs,
            eta_min=self.hparams.min_lr,
        )


        return [optimizer], [scheduler]
