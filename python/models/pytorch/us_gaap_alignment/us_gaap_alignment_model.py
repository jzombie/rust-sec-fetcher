from ..pretrained_io import PretrainedIO
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from utils.pytorch import seed_everything

class UsGaapAlignmentModel(pl.LightningModule, PretrainedIO):
    def __init__(self, dropout_rate=0.2, hidden_size=256, num_heads=8, lr=1e-5, batch_size=36, gradient_clip=1.0, input_size = 1024):
        super(UsGaapAlignmentModel, self).__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip

        # Ensure that hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            # Adjust hidden_size to make it divisible by num_heads
            hidden_size = (hidden_size // num_heads) * num_heads

        # Now we ensure that `hidden_size` is compatible with the number of attention heads
        # Calculate the dimension per attention head (hidden_size per head)
        self.head_dim = hidden_size // num_heads

        # Ensure that `head_dim` is valid (greater than 0) 
        if self.head_dim == 0:
            raise ValueError(f"Hidden size {hidden_size} is too small for {num_heads} attention heads!")

        # Save hyperparameters
        self.save_hyperparameters()

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Ensure this matches your input size
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Attention layer
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(hidden_size, input_size)  # Adjust to match input size

    def forward(self, variation_embeddings):      
        # Pass the embeddings through the fully connected layers
        variation_embeddings = self.fc(variation_embeddings)

        # Apply attention mechanism
        attn_output, _ = self.attn(variation_embeddings.unsqueeze(0), variation_embeddings.unsqueeze(0), variation_embeddings.unsqueeze(0))
        attn_output = attn_output.squeeze(0) + variation_embeddings  # Add the input to the attention output

        # Apply normalization and dropout
        attn_output = self.layer_norm(attn_output)
        attn_output = self.dropout(attn_output)

        variation_embeddings = attn_output

        # Final output layer
        transformed_variation_embeddings = self.fc_out(variation_embeddings)

        return transformed_variation_embeddings

    def training_step(self, batch, batch_idx):
        # Get the variation and description embeddings from the batch
        variation_embeddings, description_embeddings = batch
    
        # Pass the variation embedding through the model to transform it
        transformed_variation_embeddings = self(variation_embeddings)

        # Original cosine similarity loss (non-trained)
        non_trained_cosine_loss = self.cosine_similarity_loss(variation_embeddings, description_embeddings)
    
        # Compute the cosine similarity loss
        cosine_loss = self.cosine_similarity_loss(transformed_variation_embeddings, description_embeddings)
    
        # Compute the Euclidean distance loss
        euclidean_loss = self.euclidean_distance(transformed_variation_embeddings, description_embeddings)

        # Calculate the mean Euclidean loss across the batch
        mean_euclidean_loss = euclidean_loss.mean()
    
        # Combine both losses
        total_loss = cosine_loss + mean_euclidean_loss
    
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_cosine_loss", cosine_loss, prog_bar=True)
        self.log("train_non_trained_cosine_loss", non_trained_cosine_loss, prog_bar=True)
        self.log("train_euclidean_loss", mean_euclidean_loss, prog_bar=True)
    
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Get the variation and description embeddings from the batch
        variation_embeddings, description_embeddings = batch
        
        # Pass the variation embedding through the model to transform it
        transformed_variation_embedding = self(variation_embeddings)

        # Original cosine similarity loss (non-trained)
        non_trained_cosine_loss = self.cosine_similarity_loss(variation_embeddings, description_embeddings)
    
        # Compute the cosine similarity loss
        cosine_loss = self.cosine_similarity_loss(transformed_variation_embedding, description_embeddings)
    
        # Compute the Euclidean distance loss
        euclidean_loss = self.euclidean_distance(transformed_variation_embedding, description_embeddings)

        # Calculate the mean Euclidean loss across the batch
        mean_euclidean_loss = euclidean_loss.mean()
    
        # Combine both losses
        total_loss = cosine_loss + mean_euclidean_loss
    
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_cosine_loss", cosine_loss, prog_bar=True)
        self.log("val_non_trained_cosine_loss", non_trained_cosine_loss, prog_bar=True)
        self.log("val_euclidean_loss", mean_euclidean_loss, prog_bar=True)
        
    
        return total_loss

    def cosine_similarity_loss(self, embedding1, embedding2):
        """
        Calculate the Cosine similarity between two embeddings using F.pairwise_distance, as a loss (lower is better).
        """
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(embedding1, embedding2, dim=-1)
        
        # Convert cosine similarity to a loss value (1 - similarity)
        loss = 1 - cosine_sim.mean()  # Lower is better, as we want the transformed variation to be closer to the description
        return loss

    def euclidean_distance(self, embedding1, embedding2):
        """
        Calculate the Euclidean distance between two embeddings using F.pairwise_distance.
        """
        return F.pairwise_distance(embedding1, embedding2, p=2)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


    # TODO: Implement; For inference (Leave here or place elsewhere)
    # def transform(self, variation_embeddings: torch.Tensor) -> torch.Tensor:
    #     """
    #     Transforms input variation embeddings using the trained model.
    #     Equivalent to forward() but semantically clearer for inference.

    #     Args:
    #         variation_embeddings (torch.Tensor): Input tensor of shape (N, D)

    #     Returns:
    #         torch.Tensor: Transformed embeddings of shape (N, D)
    #     """
    #     self.eval()
    #     with torch.no_grad():
    #         return self.forward(variation_embeddings)