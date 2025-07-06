import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Tuple, Dict
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

# --- BUILDING BLOCK MODULES ---

class PerceiverStackEncoder(nn.Module):
    """
    The encoder for a single stack. Its job is to take a variable-length sequence of items 
    (a "stack") and distill it into a single, fixed-size latent vector. This is one of the 
    six independent encoders that creates an "anchor vector".
    """
    def __init__(self, input_dim: int, latent_dim: int, num_latents: int, depth: int, dropout_rate: float):
        super().__init__()
        # A small, learnable array of vectors that will query the input data.
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        
        # The cross-attention layer where the learnable latents attend to the input stack.
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True, dropout=dropout_rate)
        
        # A linear layer to project the input data to the same dimension as the latents.
        self.cross_proj = nn.Linear(input_dim, latent_dim)
        
        # LayerNorms are applied BEFORE the attention operation for training stability (Pre-Norm).
        self.query_norm = nn.LayerNorm(latent_dim)
        self.kv_norm = nn.LayerNorm(latent_dim) 

        # A stack of standard Transformer blocks to process the information in the latent space.
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=latent_dim, nhead=4, dim_feedforward=latent_dim * 4,
                    batch_first=True, activation=F.gelu, norm_first=True,
                    dropout=dropout_rate
                ) for _ in range(depth)
            ]
        )
        # Final normalization layer before outputting the result.
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Processes the input stack to produce a single latent vector. This version
        solves numerical instability by segregating valid and empty samples before
        computation.

        Args:
            x (torch.Tensor): The input stack tensor of shape [B, N, D].
            mask (torch.Tensor): The boolean mask of shape [B, N] indicating valid items.

        Returns:
            torch.Tensor: A single latent vector of shape [B, latent_dim].
        """
        B, N, D = x.shape
        device = x.device

        # 1. Identify valid and fully masked (empty) items in the batch
        valid_item_indices = torch.where(mask.any(dim=1))[0]
        # masked_item_indices = torch.where(~mask.any(dim=1))[0]

        # If the whole batch is empty, return zeros.
        if len(valid_item_indices) == 0:
            return torch.zeros(B, self.latents.size(-1), device=device)

        # 2. Create a "sub-batch" containing only the data from valid items
        valid_x = x[valid_item_indices]
        valid_mask = mask[valid_item_indices]
        
        # --- Run the entire encoder pipeline ONLY on the valid sub-batch ---
        sub_batch_size = valid_x.shape[0]

        latents = self.latents.unsqueeze(0).repeat(sub_batch_size, 1, 1)
        x_proj = self.cross_proj(valid_x)

        latents_norm = self.query_norm(latents)
        x_proj_norm = self.kv_norm(x_proj)
        
        key_padding_mask = ~valid_mask
        
        attn_output, _ = self.cross_attn(
            query=latents_norm, key=x_proj_norm, value=x_proj_norm, key_padding_mask=key_padding_mask
        )
        
        # The sub-batch is guaranteed to be free of fully masked items, so this is stable.
        latents = latents + attn_output

        for block in self.blocks:
            latents = block(latents)
            
        processed_latents = self.output_norm(latents)
        valid_outputs = processed_latents.mean(dim=1)
        
        # 3. Create a full-size output tensor and scatter the results back
        full_output = torch.zeros(B, self.latents.size(-1), device=device)
        full_output[valid_item_indices] = valid_outputs

        # Any NaNs from other sources are still sanitized as a final guardrail.
        return torch.nan_to_num(full_output)


class PerceiverDecoder(nn.Module):
    """
    The decoder module. Its job is to take a context vector (combined from the shared latent
    and a task-specific anchor) and a query, and reconstruct the original stack.
    """
    def __init__(self, shared_dim: int, anchor_dim: int, query_dim: int, output_dim: int, depth: int, dropout_rate: float):
        super().__init__()
        
        # Projection layer to combine the shared and anchor vectors into a single context.
        self.context_proj = nn.Linear(shared_dim + anchor_dim, query_dim)
        
        # The cross-attention layer where the output query attends to the combined context.
        self.cross_attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=4, batch_first=True, dropout=dropout_rate)
        
        # LayerNorms for pre-normalization.
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(query_dim) 

        # A stack of standard Transformer blocks for deep processing.
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=query_dim, nhead=4, dim_feedforward=query_dim * 4,
                    batch_first=True, activation=F.gelu, norm_first=True,
                    dropout=dropout_rate
                ) for _ in range(depth)
            ]
        )
        # The final linear layer to project the output to the original input dimension.
        self.output_head = nn.Linear(query_dim, output_dim)

    def forward(self, shared_vector: torch.Tensor, anchor_vector: torch.Tensor, output_query: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs a stack from the shared and anchor vectors.
        
        Args:
            shared_vector: The single latent vector representing all inputs.
            anchor_vector: The specific latent vector from the corresponding encoder.
            output_query: The query specifying the stack to reconstruct (e.g., from balance/period).

        Returns:
            The reconstructed stack. Shape: [Batch, Sequence Length, Output Dim].
        """
        # Combine the global (shared) and local (anchor) information.
        combined_context = torch.cat([shared_vector, anchor_vector], dim=-1)
        projected_context = self.context_proj(combined_context)
        
        # Prepare the context to be attended to by the query.
        latent = projected_context.unsqueeze(1).expand(-1, output_query.size(1), -1)

        # Apply pre-normalization.
        query_norm = self.query_norm(output_query)
        latent_norm = self.context_norm(latent)

        # Perform cross-attention.
        attn_output, _ = self.cross_attn(
            query=query_norm, 
            key=latent_norm, 
            value=latent_norm
        )

        # Add the residual connection.
        x = output_query + attn_output

        # Process through the deep latent transformer blocks.
        for block in self.blocks:
            x = block(x)
        return self.output_head(x)

# --- MAIN LIGHTNING MODULE (ANCHOR VECTOR ARCHITECTURE) ---

class Stage2Autoencoder(pl.LightningModule):
    """
    This model implements the full autoencoder using the "Anchor Vector" architecture.
    It processes multiple independent data stacks, combines them into a single shared latent
    vector, and then uses that shared vector along with task-specific "anchor" vectors
    to reconstruct the original stacks.
    """
    def __init__(
        self,
        loss_alpha: float = 0.5,
        dropout_rate: float = 0.1,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        num_categories: int = 6,
        input_dim: int = 256,
        encoder_latent_dim: int = 128,
        shared_latent_dim: int = 512, # The dimension of the single vector
        num_latents: int = 64,
        encoder_depth: int = 2,
        query_dim: int = 256,
        decoder_depth: int = 2,
        batch_size: int = 4,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.stats = {
            'train': self._get_initial_stats_dict(),
            'val': self._get_initial_stats_dict()
        }
        
        
        # A list of independent encoders, one for each of the 6 categories.
        self.encoders = nn.ModuleList(
            [
                PerceiverStackEncoder(input_dim, encoder_latent_dim, num_latents, encoder_depth, dropout_rate)
                for _ in range(num_categories)
            ]
        )
        # A single linear layer to create the shared bottleneck vector from all encoder outputs.
        self.encoder_to_shared = nn.Linear(num_categories * encoder_latent_dim, shared_latent_dim)

        # Embeddings used to create the decoder query.
        self.balance_embedding = nn.Embedding(num_embeddings=3, embedding_dim=query_dim)
        self.period_embedding = nn.Embedding(num_embeddings=3, embedding_dim=query_dim)
        
        # The single, shared decoder. It takes both the shared latent and an anchor vector.
        self.decoder = PerceiverDecoder(shared_latent_dim, encoder_latent_dim, query_dim, input_dim, decoder_depth, dropout_rate)
        self.loss_fn = nn.MSELoss()

    def _get_initial_stats_dict(self):
        """Helper method to create a clean stats dictionary."""
        return {
            'running_loss': 0.0,
            'running_cosine_sim': 0.0,
            'batches_seen': 0,
            'non_finite_grad_count': 0
        }

    def on_train_epoch_start(self):
        """Resets running loss stats at the beginning of each training epoch."""
        self.stats['train'] = self._get_initial_stats_dict()

    def on_validation_epoch_start(self):
        """Resets running loss stats at the beginning of each validation run."""
        self.stats['val'] = self._get_initial_stats_dict()

    def forward(
        self,
        stacks: List[torch.Tensor],
        masks: List[torch.Tensor],
        balance_batch: List[torch.Tensor],
        period_batch: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        The main forward pass of the model.
        """
        
        # 1. Create the task-specific "anchor" vectors by encoding each stack independently.
        anchor_vectors = []
        batch_size = stacks[0].size(0) if stacks else 1
        for i in range(self.hparams.num_categories):
            # Handle cases where a category might be missing from a batch.
            if i < len(stacks):
                output = self.encoders[i](stacks[i], masks[i])
                anchor_vectors.append(output)
            else:
                # Use a zero tensor as a placeholder if a category is missing.
                zero_output = torch.zeros(batch_size, self.hparams.encoder_latent_dim, device=self.device)
                anchor_vectors.append(zero_output)
        
        # 2. Create the SINGLE shared latent vector from the anchors, meeting the design constraint.
        concatenated_anchors = torch.cat(anchor_vectors, dim=1)
        shared_latent = self.encoder_to_shared(concatenated_anchors)

        # === Normalize shared latent before decoding ===
        assert not torch.isnan(shared_latent).any(), "NaN before normalization"
        shared_latent = F.normalize(shared_latent, p=2, dim=-1, eps=1e-8)
        assert not torch.isnan(shared_latent).any(), "NaN after normalization"

        # 3. Decode each category using the shared vector AND its specific anchor.
        reconstructions = []
        # Note: `balance_batch` and `period_batch` lengths should always be the same.
        # This loop correctly iterates over the number of categories present in the current batch.
        for i in range(len(balance_batch)):
            # Select the corresponding anchor vector for this category.
            anchor_vector = anchor_vectors[i]
            
            # --- FIX: Skip decoder for categories with no items to reconstruct ---
            # If the tensor of indices is empty, the query would be empty, causing an MPS crash.
            if balance_batch[i].numel() == 0:
                # Append a placeholder. The shape must match what the loss function expects.
                # The sequence length is 0, which will be handled correctly in _shared_step.
                batch_size_for_cat = anchor_vector.size(0)
                empty_recon = torch.empty(batch_size_for_cat, 0, self.hparams.input_dim, device=self.device)
                reconstructions.append(empty_recon)
                continue

            # Create the reconstruction query from balance and period info.
            balance_embed = self.balance_embedding(balance_batch[i])
            period_embed = self.period_embedding(period_batch[i])
            query = balance_embed + period_embed
            
            # Decode using both the global shared vector and the local anchor vector.
            recon = self.decoder(shared_latent, anchor_vector, query)
            reconstructions.append(recon)
            
        # --- Sanity Checks (for debugging, no performance impact in production) ---
        # 1. Ensure the shared latent is a single vector per batch item (i.e., a 2D tensor).
        assert shared_latent.dim() == 2, \
            f"Shared latent must be a 2D tensor [batch, dim], but got shape {shared_latent.shape}"

        # 2. Ensure the number of reconstructions matches the number of categories present IN THIS BATCH.
        #    This number can be less than self.hparams.num_categories if the batch is sparse.
        assert len(reconstructions) == len(balance_batch), \
            f"Expected {len(balance_batch)} reconstructions for this batch, but got {len(reconstructions)}"
            
        # 4. Return both reconstructions and the single shared vector for downstream use.
        return reconstructions, shared_latent

    def _shared_step(self, batch):
        stacks, masks, balance_batch, period_batch = batch
        reconstructions, shared_latent = self.forward(stacks, masks, balance_batch, period_batch)

        # Normalize shared latent
        shared_latent = F.normalize(shared_latent, p=2, dim=1)

        total_loss = 0.0
        total_cos_sim = 0.0
        per_stack_cosine = []

        for i in range(self.hparams.num_categories):
            original = stacks[i][masks[i]]
            recon = reconstructions[i][masks[i]]
            if original.numel() == 0:
                per_stack_cosine.append(0.0)
                continue

            loss = self.loss_fn(recon, original)
            cos_sim = F.cosine_similarity(recon, original, dim=-1).mean()

            total_loss += loss
            total_cos_sim += cos_sim
            per_stack_cosine.append(cos_sim.item())

        avg_cosine = total_cos_sim / self.hparams.num_categories

        return {
            "loss": total_loss,
            "mse_loss": total_loss,
            "cosine_sim": avg_cosine,
            "stack_cosine_sims": per_stack_cosine,
        }

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        losses = self._shared_step(batch)
        batch_total_loss = losses['loss']

        if not torch.isfinite(batch_total_loss):
            self.log("train_step_skipped", 1, on_step=True)
            # Return a dummy tensor
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        train_stats = self.stats['train']
        train_stats['running_loss'] += batch_total_loss.item()
        train_stats['running_cosine_sim'] += float(losses['cosine_sim'])
        train_stats['batches_seen'] += 1

        running_avg_loss = train_stats['running_loss'] / train_stats['batches_seen']
        running_avg_sim = train_stats['running_cosine_sim'] / train_stats['batches_seen']

        self.log("train_loss_running", float(running_avg_loss), on_step=True, prog_bar=True)
        self.log("train_cosine_sim_running", float(running_avg_sim), on_step=True)

        for i, sim in enumerate(losses["stack_cosine_sims"]):
            self.log(f"train_cosine_sim_stack_{i}", float(sim), on_step=True, on_epoch=True)

        self.log("train_loss_epoch", batch_total_loss, on_step=False, on_epoch=True)
        self.log("train_mse_epoch", float(losses["mse_loss"]), on_step=False, on_epoch=True)
        self.log("train_cosine_sim_epoch", float(losses["cosine_sim"]), on_step=False, on_epoch=True)

        return batch_total_loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        losses = self._shared_step(batch)
        if losses is None or not torch.isfinite(losses['loss']):
            return None

        val_stats = self.stats['val']
        val_stats['running_loss'] += losses['loss'].item()
        val_stats['running_cosine_sim'] += float(losses['cosine_sim'])
        val_stats['batches_seen'] += 1

        for i, sim in enumerate(losses["stack_cosine_sims"]):
            self.log(f"val_cosine_sim_stack_{i}", float(sim), on_step=False, on_epoch=True)

        return None

    def on_validation_epoch_end(self):
        """Calculates and logs the average validation loss and similarity at the end of the epoch."""
        val_stats = self.stats['val']
        if val_stats['batches_seen'] > 0:
            avg_val_loss = val_stats['running_loss'] / val_stats['batches_seen']
            # Calculate and log the average validation cosine similarity
            avg_val_sim = val_stats['running_cosine_sim'] / val_stats['batches_seen']
            
            self.log("val_loss_epoch", avg_val_loss, prog_bar=True)
            self.log("val_cosine_sim_epoch", avg_val_sim, prog_bar=True)

    # TODO: Remove
    # def on_before_optimizer_step(self, optimizer):
    #     """
    #     Called by PyTorch Lightning after loss.backward() but before optimizer.step().
    #     This hook inspects gradients, fixes non-finite values, and updates a
    #     cumulative count in self.stats.
    #     """
    #     non_finite_found_this_step = False
        
    #     for group in optimizer.param_groups:
    #         for param in group['params']:
    #             if param.grad is not None:
    #                 if not torch.isfinite(param.grad).all():
    #                     non_finite_found_this_step = True
    #                     param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    #     if non_finite_found_this_step:
    #         # Increment the counter in our stats dictionary
    #         self.stats['train']['non_finite_grad_count'] += 1
            
    #         # Log the new cumulative count for the current epoch to TensorBoard
    #         self.log(
    #             "instability/epoch_non_finite_grad_count",
    #             float(self.stats['train']['non_finite_grad_count']),
    #             on_step=True,
    #             on_epoch=False, # We log on the step to see it update live
    #             prog_bar=False,
    #             logger=True
    #         )

    def configure_optimizers(self):
        """Sets up the AdamW optimizer and a learning rate scheduler with a warm-up phase."""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # The scheduler linearly increases the LR for `warmup_steps`, then decays it.
        try:
            total_steps = self.trainer.estimated_stepping_batches
        except AttributeError:
            total_steps = 100_000 # Fallback for when trainer is not fully initialized
            
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.hparams.warmup_steps)
        main_scheduler_t_max = max(1, total_steps - self.hparams.warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=main_scheduler_t_max, eta_min=1e-7)
        
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[self.hparams.warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step", # The scheduler is updated at every training step.
            },
        }
