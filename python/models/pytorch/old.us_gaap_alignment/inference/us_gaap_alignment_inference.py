from functools import lru_cache
from typing import List, Dict
import torch
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import generate_us_gaap_description
from ..us_gaap_alignment_model import UsGaapAlignmentModel

@lru_cache(maxsize=None)
def load_jsonl_embeddings(path: str) -> List[Dict]:
    """
    Load precomputed JSONL-encoded embedding data from disk and convert each
    row's 'description_embedding' into a NumPy array.

    NOTE: The result is memoized and should be treated as read-only to prevent
    unintended side effects across calls.

    Args:
        path (str): Path to the JSONL file containing embedding records.

    Returns:
        List[Dict]: List of deserialized rows with NumPy embeddings attached.
    """

    rows = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            row["description_embedding"] = np.array(
                row["description_embedding"], dtype=np.float32
            )
            rows.append(row)
    return rows

def embed_text(text: str, model: UsGaapAlignmentModel, device: torch.device) -> torch.Tensor:
    """
    Encode raw input text using the BGE encoder and alignment model into a
    single semantic embedding.

    Args:
        text (str): Raw description or variation text to encode.
        model (UsGaapAlignmentModel): The trained transformation model.
        device (torch.device): The device on which to run the encoder and model.

    Returns:
        torch.Tensor: Transformed 1D tensor representing the aligned embedding.
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
    device: torch.device = torch.device("cpu")
) -> List[Dict]:
    """
    Find the top-k most similar concepts from the precomputed dataset by 
    computing cosine similarity between the input concept and reference entries.

    Args:
        us_gaap_concept (str): The input GAAP concept name to match.
        model (UsGaapAlignmentModel): Trained alignment model.
        concept_type (str): Required concept type filter (e.g., monetaryItemType).
        balance_type (str): Required balance filter (e.g., debit, credit).
        period_type (str): Required period type filter (e.g., instant, duration).
        dataset_path (str): Path to the JSONL dataset with reference embeddings.
        top_k (int): Number of top matches to return.
        device (torch.device): Torch device for inference computations.

    Returns:
        List[Dict]: Top-k most similar entries from the dataset.
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
