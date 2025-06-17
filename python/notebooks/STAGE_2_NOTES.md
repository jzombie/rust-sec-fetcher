```python
# Perceiver IO
# https://arxiv.org/abs/2107.14795

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceiverStackEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128,
                 num_latents: int = 64, depth: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=4, batch_first=True
        )
        self.cross_proj = nn.Linear(input_dim, latent_dim)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=latent_dim * 4,
                batch_first=True
            ))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, N, D_in] (e.g., concept stack embeddings)
            mask: Optional [B, N] boolean mask (True for valid entries)

        Returns:
            Tensor of shape [B, latent_dim] summarizing the category stack
        """
        B, N, _ = x.shape
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)  # [B, M, D]

        x_proj = self.cross_proj(x)  # [B, N, latent_dim]
        latents, _ = self.cross_attn(latents, x_proj, x_proj,
                                     key_padding_mask=~mask if mask is not None else None)

        for block in self.blocks:
            latents = block(latents)

        return latents.mean(dim=1)  # Pool across latent tokens
```


# Draft model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from typing import List

# --- SECTION 1: BUILDING BLOCK MODULES ---
# These are the reusable components of our architecture.

class PerceiverStackEncoder(nn.Module):
    """
    The encoder for a single stack of input items (e.g., all 'credit/duration' transactions).
    It uses cross-attention to distill a variable-length input stack into a fixed-size
    latent representation, which is then processed by a standard Transformer.
    """
    def __init__(self, input_dim: int, latent_dim: int, num_latents: int, depth: int):
        super().__init__()
        # The learnable latent array that acts as the bottleneck. It's the "interviewer".
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # First layer: Cross-attention where the latents attend to the input data.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=4, batch_first=True
        )
        # A linear layer to project the input data to the same dimension as the latents.
        self.cross_proj = nn.Linear(input_dim, latent_dim)

        # The main processing block: a standard Transformer encoder that operates
        # only in the latent space. Its computational cost is independent of input size.
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=4,
                    dim_feedforward=latent_dim * 4,
                    batch_first=True,
                    activation=F.gelu # GELU is common in modern transformers
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, D_in] (Batch, Num Items, Input Dim)
            mask: Boolean tensor of shape [B, N] where True indicates a valid (non-padded) item.

        Returns:
            A tensor of shape [B, latent_dim] summarizing the entire stack.
        """
        B, N, _ = x.shape
        # Create a batch of the learnable latents for processing.
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)  # Shape: [B, num_latents, latent_dim]

        # Project the input data to the latent dimension.
        x_proj = self.cross_proj(x)  # Shape: [B, N, latent_dim]

        # The key Perceiver step: cross-attention.
        # Queries come from our learnable latents.
        # Keys/Values come from the projected input data.
        # The key_padding_mask ensures we don't attend to padded parts of the input.
        latents, _ = self.cross_attn(
            latents, x_proj, x_proj, key_padding_mask=~mask
        )

        # Process the distilled information through the deep latent transformer.
        for block in self.blocks:
            latents = block(latents)

        # Pool the final latent vectors into a single summary vector for this stack.
        return latents.mean(dim=1)


class PerceiverDecoder(nn.Module):
    """
    The shared decoder module.
    It takes a shared latent vector (containing info from all 6 stacks) and a specific
    output query to reconstruct a single target stack.
    """
    def __init__(self, shared_dim: int, query_dim: int, output_dim: int, depth: int):
        super().__init__()
        self.output_dim = output_dim

        # Projects the shared latent vector to the decoder's internal dimension.
        self.latent_proj = nn.Linear(shared_dim, query_dim)

        # Cross-attention where the output query "interrogates" the processed latent vector.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=4, batch_first=True
        )
        
        # A standard Transformer encoder for deep processing.
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=query_dim,
                    nhead=4,
                    dim_feedforward=query_dim * 4,
                    batch_first=True,
                    activation=F.gelu
                )
                for _ in range(depth)
            ]
        )
        # Final linear layer to project the output to the desired reconstruction dimension.
        self.output_head = nn.Linear(query_dim, output_dim)

    def forward(self, shared_latent: torch.Tensor, output_query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_latent: The compressed vector from the encoders. Shape [B, shared_dim].
            output_query: The query specifying which stack to reconstruct. Shape [B, M, query_dim].
                          M is the max number of items to reconstruct.
        Returns:
            The reconstructed stack. Shape [B, M, output_dim].
        """
        # Project the single latent vector to the right dimension and repeat it
        # to match the query sequence length.
        latent = self.latent_proj(shared_latent).unsqueeze(1).repeat(1, output_query.size(1), 1)

        # Cross-attention: The output_query asks questions of the latent vector.
        x, _ = self.cross_attn(output_query, latent, latent)

        # Deep processing of the resulting information.
        for block in self.blocks:
            x = block(x)

        # Project to the final output dimension (e.g., 256).
        return self.output_head(x)


# --- SECTION 2: THE MAIN PYTORCH LIGHTNING MODULE ---
# This class orchestrates the entire autoencoder architecture.

class FinancialAutoencoder(pl.LightningModule):
    def __init__(
        self,
        num_categories: int = 6,
        input_dim: int = 256,
        encoder_latent_dim: int = 128,
        num_latents: int = 64,
        encoder_depth: int = 4,
        shared_latent_dim: int = 512,
        query_dim: int = 192,
        decoder_depth: int = 4,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        # Save all hyperparameters (like input_dim, lr, etc.) to self.hparams
        # This makes them accessible for logging and checkpointing.
        self.save_hyperparameters()

        # --- Encoders ---
        # Create 6 separate instances of the PerceiverStackEncoder.
        self.encoders = nn.ModuleList(
            [
                PerceiverStackEncoder(input_dim, encoder_latent_dim, num_latents, encoder_depth)
                for _ in range(num_categories)
            ]
        )
        # A layer to project the concatenated encoder outputs into the shared latent space.
        self.encoder_to_shared = nn.Linear(num_categories * encoder_latent_dim, shared_latent_dim)

        # --- Compositional Query Embeddings ---
        # This is the "recipe card" generator for the decoder.
        # We learn an embedding for each attribute value.
        # num_embeddings=3 for ["credit", "debit", "none"]
        self.balance_embedding = nn.Embedding(num_embeddings=3, embedding_dim=query_dim)
        # num_embeddings=3 for ["duration", "instant", "none"]
        self.period_embedding = nn.Embedding(num_embeddings=3, embedding_dim=query_dim)

        # --- Decoder ---
        # A single, shared decoder instance.
        self.decoder = PerceiverDecoder(shared_latent_dim, query_dim, input_dim, decoder_depth)

        # --- Loss Function ---
        self.loss_fn = nn.MSELoss()

    def _create_queries(self, batch_size: int, max_output_len: int) -> List[torch.Tensor]:
        """Helper function to construct the 6 compositional queries."""
        queries = []
        # Define the 6 categories by their attribute indices.
        # (balance_idx, period_idx)
        categories = [
            (0, 0), # credit, duration
            (0, 1), # credit, instant
            (1, 0), # debit, duration
            (1, 1), # debit, instant
            (0, 2), # credit, none
            (1, 2), # debit, none
        ]
        
        for bal_idx, per_idx in categories:
            # Get the embedding vector for each attribute.
            balance_vec = self.balance_embedding(torch.tensor(bal_idx, device=self.device))
            period_vec = self.period_embedding(torch.tensor(per_idx, device=self.device))
            
            # Combine them to form the query for this category.
            # We then expand this single query vector to create a sequence,
            # which will be used to generate a sequence of reconstructed items.
            query_vec = (balance_vec + period_vec).unsqueeze(0).repeat(max_output_len, 1)
            queries.append(query_vec.unsqueeze(0).repeat(batch_size, 1, 1))
            
        return queries

    def forward(self, stacks: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        """The full forward pass of the autoencoder."""
        # --- Encoding ---
        # 1. Run each input stack through its dedicated encoder.
        encoder_outputs = [
            self.encoders[i](stacks[i], masks[i]) for i in range(self.hparams.num_categories)
        ]
        
        # 2. Concatenate the 6 summary vectors.
        concatenated_latents = torch.cat(encoder_outputs, dim=1)
        
        # 3. Project into the final shared latent space.
        shared_latent = self.encoder_to_shared(concatenated_latents) # Shape: [B, shared_latent_dim]

        # --- Decoding ---
        # 4. Generate the 6 queries needed for reconstruction.
        # We need to know the max length of the output to generate the right query shape.
        max_len = max(stack.size(1) for stack in stacks)
        queries = self._create_queries(batch_size=shared_latent.size(0), max_output_len=max_len)
        
        # 5. Call the shared decoder 6 times with the same latent but different queries.
        reconstructions = [
            self.decoder(shared_latent, queries[i]) for i in range(self.hparams.num_categories)
        ]
        
        return reconstructions

    def training_step(self, batch, batch_idx):
        """
        Defines the logic for a single training step.
        """
        # The batch consists of the original input stacks and their corresponding masks.
        stacks, masks = batch
        
        # Get the 6 reconstructed stacks from the model.
        reconstructions = self.forward(stacks, masks)
        
        # --- Loss Calculation ---
        # Calculate the reconstruction loss for each stack and sum them up.
        total_loss = 0
        for i in range(self.hparams.num_categories):
            # We only calculate loss on the non-padded parts of the input.
            original = stacks[i][masks[i]]
            recon = reconstructions[i][masks[i]]
            
            # This handles cases where a mask might be all False for a sample in the batch
            if original.numel() > 0:
                loss = self.loss_fn(recon, original)
                total_loss += loss
                # Log individual losses for better monitoring.
                self.log(f"train_loss_stack_{i}", loss, on_step=True, on_epoch=True, prog_bar=False)

        # Log the total combined loss.
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def configure_optimizers(self):
        """Sets up the optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


# --- SECTION 3: EXAMPLE USAGE ---

if __name__ == '__main__':
    # --- 1. Create Dummy Data ---
    # In a real scenario, this would come from your database and pre-processing pipeline.
    BATCH_SIZE = 4
    INPUT_DIM = 256
    
    # Create 6 stacks of tensors with variable lengths.
    dummy_stacks = [
        torch.randn(BATCH_SIZE, 10, INPUT_DIM), # stack 0 has 10 items
        torch.randn(BATCH_SIZE, 5, INPUT_DIM),  # stack 1 has 5 items
        torch.randn(BATCH_SIZE, 15, INPUT_DIM), # ...
        torch.randn(BATCH_SIZE, 8, INPUT_DIM),
        torch.randn(BATCH_SIZE, 12, INPUT_DIM),
        torch.randn(BATCH_SIZE, 7, INPUT_DIM),
    ]
    
    # To handle variable lengths in a batch, we pad them to the max length in the batch.
    max_len = max(s.size(1) for s in dummy_stacks)
    
    padded_stacks = []
    masks = []
    for stack in dummy_stacks:
        B, N, D = stack.shape
        # Create a padded tensor of shape [B, max_len, D]
        padded_tensor = torch.zeros(B, max_len, D)
        padded_tensor[:, :N, :] = stack
        padded_stacks.append(padded_tensor)
        
        # Create the corresponding mask. True for real data, False for padding.
        mask = torch.zeros(B, max_len, dtype=torch.bool)
        mask[:, :N] = True
        masks.append(mask)

    # Use a simple TensorDataset and DataLoader for training.
    # We wrap the lists of tensors directly. The dataset will return a tuple of lists.
    dataset = TensorDataset(
        torch.stack(padded_stacks, dim=1), # Shape: [B, 6, max_len, D]
        torch.stack(masks, dim=1)          # Shape: [B, 6, max_len]
    )
    
    # Custom collate_fn to keep the data as a list of 6 tensors.
    def collate_fn(batch):
        # batch is a list of tuples, each tuple is (stacks_tensor, masks_tensor)
        # We want to transform it from a list of [B, 6, ...] to a tuple of (list of 6 [B, ...], list of 6 [B, ...])
        all_stacks = [torch.stack([item[0][i] for item in batch]) for i in range(6)]
        all_masks = [torch.stack([item[1][i] for item in batch]) for i in range(6)]
        return all_stacks, all_masks

    # Since we only have one batch of dummy data, we'll just use it directly.
    # In a real setup, you'd have a large dataset.
    train_loader = DataLoader([dataset[i] for i in range(BATCH_SIZE)], batch_size=BATCH_SIZE, collate_fn=collate_fn)


    # --- 2. Initialize and Train the Model ---
    print("Initializing model...")
    model = FinancialAutoencoder(
        input_dim=INPUT_DIM
    )

    print("Setting up trainer...")
    # Use the PyTorch Lightning Trainer for boilerplate-free training.
    # We'll run it on CPU for a few steps for this demonstration.
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="cpu",
        # fast_dev_run=True, # Uncomment for a quick test run of one batch
        logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="financial_autoencoder"),
        enable_checkpointing=False
    )

    print("Starting dummy training run...")
    trainer.fit(model, train_dataloaders=train_loader)
    print("Dummy training run complete.")
```
