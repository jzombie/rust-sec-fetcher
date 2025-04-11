from typing import List, Dict
import torch
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import generate_us_gaap_description
from ..us_gaap_alignment_model import UsGaapAlignmentModel

# TODO: Memoize
def load_jsonl_embeddings(path):
    """
    Load precomputed embeddings and metadata from a JSONL file.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            row["description_embedding"] = np.array(row["description_embedding"], dtype=np.float32)
            rows.append(row)
    return rows

def embed_text(text: str, model: UsGaapAlignmentModel, device: str) -> torch.Tensor:
    """
    Convert raw text to aligned embedding using BGE encoder and fine-tuned model.

    Args:
        text (str): Input variation or description text.
        model (UsGaapAlignmentModel): Alignment model to transform BGE embeddings.
        device (str): Device to perform inference on.

    Returns:
        torch.Tensor: Transformed embedding as a 1D tensor.
    """

    tokenizer, encoder = UsGaapAlignmentModel.get_base_encoder(device)

    model = model.to(device).eval()
    encoder.to(device)
    encoder.eval()

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = encoder(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, hidden]
        transformed = model(embedding).squeeze(0)       # shape: [hidden]

    return transformed  # still a torch.Tensor



def find_closest_match(
    us_gaap_concept: str,
    model: UsGaapAlignmentModel,
    concept_type: str,
    balance_type: str,
    period_type: str,
    dataset_path: str,
    top_k: int = 1,
    device: str = "cpu"
) -> List[Dict]:
    """
    Find the top-k closest aligned US GAAP concepts by cosine similarity,
    using a fine-tuned vector alignment model.

    Args:
        us_gaap_concept (str): Raw US GAAP concept name (e.g. tag like "AccountsReceivable").
        model (UsGaapAlignmentModel): Loaded alignment model for transforming embeddings.
        concept_type (str): Required concept type filter (e.g. "monetaryItemType").
        balance_type (str): Required balance type filter (e.g. "debit", "credit").
        period_type (str): Required period type filter (e.g. "instant", "duration").
        dataset_path (str): Path to the JSONL dataset with precomputed description embeddings.
        top_k (int): Number of top matching results to return.
        device (str): Device for inference (e.g. "cpu", "cuda", "mps").

    Returns:
        List[dict]: Top matching entries from the dataset, ranked by cosine similarity.
    """
    _, encoder = UsGaapAlignmentModel.get_base_encoder(device)

    model = model.to(device).eval()
    encoder.to(device)

    description = generate_us_gaap_description(us_gaap_concept)

    # Step 1: Encode the text with the BGE encoder
    variation_embedding = embed_text(description, model, device)

    # Step 2: Transform using the fine-tuned model
    with torch.no_grad():
        transformed = model(variation_embedding.unsqueeze(0)).squeeze(0).cpu().numpy()

    # Step 3: Load and filter the reference dataset
    rows = load_jsonl_embeddings(dataset_path)
    filtered_rows = [
        r for r in rows
        if r.get("concept_type") == concept_type and
            r.get("balance_type") == balance_type and
            r.get("period_type") == period_type
    ]

    if not filtered_rows:
        return []

    reference_matrix = np.stack([r["description_embedding"] for r in filtered_rows])

    # Step 4: Cosine similarity
    sims = cosine_similarity([transformed], reference_matrix)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    return [filtered_rows[i] for i in top_indices]
