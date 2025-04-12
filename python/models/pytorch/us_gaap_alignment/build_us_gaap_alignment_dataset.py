import logging
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import pandas as pd
import numpy as np
from db import DB
from utils import generate_us_gaap_description
from tqdm import tqdm
import torch
from utils.pytorch import seed_everything
from .us_gaap_alignment_model import UsGaapAlignmentModel

def build_us_gaap_alignment_dataset(output_file: str, device: torch.device) -> None:
    """
    Construct a dataset of US GAAP variation embeddings and descriptions,
    enriched with concept metadata and OFSS category associations. The dataset
    is saved in JSONL format.

    Steps:
    - Queries the database for concept variations and metadata.
    - Generates BGE embeddings for variation text and auto-descriptions.
    - Adds OFSS category IDs as multi-label targets.
    - Saves to disk with variation and description embeddings.

    Args:
        output_file (str): Path to save the output JSONL dataset file.
        device (torch.device): Device to run inference (e.g., "cuda", "cpu", "mps").
    """

    tokenizer, encoder = UsGaapAlignmentModel.get_base_encoder(device)

    # Database setup
    db = DB()

    query = """
        SELECT
            t.id AS us_gaap_concept_id,
            t.name AS us_gaap_concept_name,
            ct.concept_type AS concept_type,
            v.text AS variation_text,
            bt.balance AS balance_type,
            pt.period_type AS period_type,
            GROUP_CONCAT(DISTINCT m.ofss_category_id ORDER BY m.ofss_category_id) AS ofss_category_ids
            -- GROUP_CONCAT(DISTINCT s.us_gaap_statement_type_id ORDER BY s.us_gaap_statement_type_id) AS statement_type_ids
        FROM us_gaap_concept t
        JOIN us_gaap_concept_description_variation v ON v.us_gaap_concept_id = t.id
        LEFT JOIN us_gaap_concept_ofss_category m ON m.us_gaap_concept_id = t.id AND m.is_manually_mapped = 1
        -- LEFT JOIN us_gaap_concept_statement_type s ON s.us_gaap_concept_id = t.id AND s.is_manually_mapped = 1
        LEFT JOIN us_gaap_balance_type bt ON t.balance_type_id = bt.id
        LEFT JOIN us_gaap_period_type pt ON t.period_type_id = pt.id
        LEFT JOIN us_gaap_concept_type ct ON t.concept_type_id = ct.id
        WHERE m.ofss_category_id IS NOT NULL
        GROUP BY t.id, v.text
    """

    # Fetch data from the database
    df = db.get(query, [
        "us_gaap_concept_id",
        "us_gaap_concept_name",
        "concept_type",
        "variation_text",
        "balance_type",
        "period_type",
        "ofss_category_ids",
        # "statement_type_ids"
    ])

    # Apply generate_us_gaap_description to the concept names
    df["us_gaap_concept_description"] = df["us_gaap_concept_name"].apply(generate_us_gaap_description)

    # Generate embeddings for variation text and description
    logging.info("Generating embeddings for variation text...")
    variation_embeddings = generate_embeddings(df["variation_text"].tolist(), tokenizer, encoder, device)

    # Add embeddings as columns directly to the DataFrame
    logging.info("Generating embeddings for concept descriptions...")
    description_embeddings = generate_embeddings(df["us_gaap_concept_description"].tolist(), tokenizer, encoder, device)

    # Optionally process the `ofss_category_ids` if needed
    df["variation_embedding"] = list(variation_embeddings)
    df["description_embedding"] = list(description_embeddings)

    # Optionally process the `ofss_category_ids` if needed
    df["ofss_category_ids"] = df["ofss_category_ids"].apply(lambda s: [int(x) for x in s.split(",")] if s else [])
    
    # Limit to a maximum of 2 labels per row
    df["ofss_category_ids"] = df["ofss_category_ids"].apply(lambda x: x[:2])

    # df["statement_type_ids"] = df["statement_type_ids"].apply(
    #     lambda s: [int(x) for x in s.split(",")] if s else []
    # )

    # Save the dataset as JSONL (with embeddings and category IDs included)
    df.to_json(output_file, orient="records", lines=True)

    logging.info(f"Dataset saved to {output_file} with {len(df)} rows, including embeddings and metadata.")

def generate_embeddings(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    encoder: PreTrainedModel,
    device: torch.device,
    batch_size: int = 16
) -> np.ndarray:
    """
    Generate BGE embeddings for a list of input texts using the transformer model.

    Args:
        texts (list[str]): List of raw text strings to embed.
        tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer compatible with encoder model.
        encoder (PreTrainedModel): HuggingFace transformer model to produce embeddings.
        device (torch.device): Device to run inference (e.g., "cuda", "cpu", "mps").
        batch_size (int): Number of samples per inference batch.

    Returns:
        np.ndarray: 2D NumPy array of shape [len(texts), embedding_dim].
    """

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch_texts = texts[i:i+batch_size]

        # Tokenize the batch of texts
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(device)

        # No gradients required for inference
        with torch.no_grad():
            outputs = encoder(**inputs)
        
        # Extract embeddings (use [CLS] token, first token in the sequence)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)
