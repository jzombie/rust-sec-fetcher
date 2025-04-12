from functools import lru_cache
from typing import Tuple
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel
from ..pretrained_io import PretrainedIO
from utils.pytorch import seed_everything

# Note:
# The BGE encoder model (e.g., BAAI/bge-large-en-v1.5) is *not* included directly
# in this architecture. This is an intentional design decision for efficiency:
#
# 1. The BGE model is large (~560M parameters) and memory-intensive.
# 2. We assume embeddings from BGE are precomputed and passed in as input.
# 3. This reduces runtime memory usage and avoids redundant computation.
# 4. Keeping the encoder separate enables embedding caching and flexible updates.
#
# If needed, the encoder can be integrated as a frozen submodule,
# but doing so would significantly increase memory and load time,
# and would require retraining to maintain compatibility.
class UsGaapAlignmentModel(pl.LightningModule, PretrainedIO):
    @staticmethod
    @lru_cache(maxsize=4)
    def get_base_encoder(device: torch.device) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """
        Load the tokenizer and transformer encoder used to generate text embeddings
        prior to alignment. This method ensures consistent preprocessing across
        training, inference, and evaluation workflows.

        Args:
            device (torch.device): Target device for encoder placement 
                (e.g., torch.device("cuda"), "cpu", or "mps").

        Returns:
            Tuple[PreTrainedTokenizer, PreTrainedModel]: Tokenizer and transformer
            model (e.g., BAAI/bge-large-en-v1.5) moved to the target device.
        """

        model_name = "BAAI/bge-large-en-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)

        # Move the encoder to the specified device
        encoder = encoder.to(device)

        encoder.eval()

        return tokenizer, encoder


    def __init__(
            self,
            dropout_rate=0.0,
            hidden_size=1024,
            num_heads=4,
            lr=0.00045519680860224305,
            batch_size=24,
            gradient_clip=0.7000000000000001,
            input_size=1024
        ):
        """
        Initialize the US GAAP alignment model with an attention-based architecture
        for embedding transformation.

        Args:
            dropout_rate (float): Dropout rate applied after activation layers.
            hidden_size (int): Dimension of the hidden layer and attention embedding.
            num_heads (int): Number of attention heads.
            lr (float): Learning rate for the optimizer.
            batch_size (int): Training batch size.
            gradient_clip (float): Gradient clipping value.
            input_size (int): Dimensionality of input embeddings.
        """

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
        """
        Perform a forward pass to transform variation embeddings into aligned
        semantic space using fully connected layers and self-attention.

        Args:
            variation_embeddings (torch.Tensor): Input tensor of shape 
                [batch_size, input_size].

        Returns:
            torch.Tensor: Transformed embedding tensor of shape 
                [batch_size, input_size].
        """

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
        """
        Perform a single training step using a batch of variation and description 
        embeddings. Computes the cosine and Euclidean losses for alignment.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of variation and
                corresponding description embeddings.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The total training loss for the batch.
        """

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
        """
        Perform a validation step on a batch, computing the total loss and tracking
        loss metrics including cosine and Euclidean distances.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of variation and
                corresponding description embeddings.
            batch_idx (int): Index of the current validation batch.

        Returns:
            torch.Tensor: The total validation loss for the batch.
        """

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
        Compute the mean cosine similarity loss between two embedding tensors. 
        Converts similarity into loss via (1 - cosine_sim).

        Args:
            embedding1 (torch.Tensor): First batch of embeddings.
            embedding2 (torch.Tensor): Second batch of embeddings.

        Returns:
            torch.Tensor: Scalar loss value representing average 1 - cosine similarity.
        """
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(embedding1, embedding2, dim=-1)
        
        # Convert cosine similarity to a loss value (1 - similarity)
        loss = 1 - cosine_sim.mean()  # Lower is better, as we want the transformed variation to be closer to the description
        return loss

    def euclidean_distance(self, embedding1, embedding2):
        """
        Compute the pairwise Euclidean distance between two batches of embeddings.

        Args:
            embedding1 (torch.Tensor): First batch of embeddings.
            embedding2 (torch.Tensor): Second batch of embeddings.

        Returns:
            torch.Tensor: Element-wise Euclidean distance vector for each pair.
        """

        return F.pairwise_distance(embedding1, embedding2, p=2)

    def configure_optimizers(self):
        """
        Configure and return the optimizer for the model training process.

        Returns:
            torch.optim.Optimizer: An instance of AdamW optimizer.
        """

        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
