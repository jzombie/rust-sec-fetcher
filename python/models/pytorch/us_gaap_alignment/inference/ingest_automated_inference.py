import logging
import torch
import pandas as pd
from db import DB
from tqdm import tqdm
from .us_gaap_alignment_inference import find_closest_match
from .. import UsGaapAlignmentModel
from utils.pytorch import seed_everything

def ingest_automated_inference(db: DB, dataset_path: str, model: UsGaapAlignmentModel, device: torch.device):
    """
    Ingests automated inference results into the database.
    
    Parameters:
    - db: The DB instance from your ORM.
    - dataset_path (str): Path to the dataset file.
    - model: The trained model for inference.
    - device: TODO: Document

    """
    # Query unmapped concepts
    query = """
    SELECT
        c.id AS concept_id,
        c.name AS concept_name,
        ct.concept_type,
        bt.balance,
        pt.period_type
    FROM us_gaap_concept c
    JOIN us_gaap_concept_type ct ON ct.id = c.concept_type_id
    LEFT JOIN us_gaap_balance_type bt ON bt.id = c.balance_type_id
    LEFT JOIN us_gaap_period_type pt ON pt.id = c.period_type_id
    LEFT JOIN us_gaap_concept_ofss_category m ON m.us_gaap_concept_id = c.id
    WHERE m.ofss_category_id IS NULL
    """

    df = db.get(query, ["concept_id", "concept_name", "concept_type", "balance", "period_type"])
    logging.info(f"Found {len(df)} unmapped concepts.")

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Mapping Concepts"):
        result = find_closest_match(
            us_gaap_concept=row.concept_name,
            model=model,
            concept_type=row.concept_type,
            balance_type=row.balance,
            period_type=row.period_type,
            dataset_path=dataset_path,
            top_k=1,
            device=device
        )

        if not result:
            continue

        closest = result[0]
        us_gaap_concept_id = row.concept_id

        for ofss_category_id in closest["ofss_category_ids"]:
            db.upsert_entity(
                table_name="us_gaap_concept_ofss_category",
                field_dict={
                    "us_gaap_concept_id": us_gaap_concept_id,
                    "ofss_category_id": ofss_category_id,
                    "is_manually_mapped": 0,
                },
                unique_fields=["us_gaap_concept_id", "ofss_category_id"]
            )

    logging.info("Inference and upsert complete.")
