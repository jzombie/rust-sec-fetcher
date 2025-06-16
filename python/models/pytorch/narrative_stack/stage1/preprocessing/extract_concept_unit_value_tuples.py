import os
import logging
from typing import List, Dict, Set, Tuple, Union
from utils.os import to_path
from pydantic import BaseModel
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from db import DB
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from utils import generate_us_gaap_description
from utils.pytorch import seed_everything, model_hash

ConceptUnitValueTuples = List[Tuple[str, str, float]]
# UnitValues = Dict[str, List[float]] # TODO: Remove
# UnitConcepts = Dict[str, Set[str]] # TODO: Remove
NonNumericUnits = Set[str]
CsvFiles = List[str]
Concepts = List[str]
ConceptUnitPair = Tuple[str, str]


class ExtractedConceptUnitValueData(BaseModel):
    concept_unit_value_tuples: ConceptUnitValueTuples
    # unit_values: UnitValues # TODO: Remove
    # unit_concepts: UnitConcepts # TODO: Remove
    non_numeric_units: NonNumericUnits
    csv_files: CsvFiles


def extract_concept_unit_value_tuples(
    data_dir: str | Path, valid_concepts: Concepts
) -> ExtractedConceptUnitValueData:
    concept_unit_value_tuples = []
    # unit_values = defaultdict(list) # TODO: Remove
    # unit_concepts = defaultdict(set) # TODO: Remove
    non_numeric_units = set()

    csv_files = []
    # Note: If this were to span multiple sub-directories, dirs should be presorted as well
    for root, _, files in os.walk(to_path(data_dir, as_str=True)):
        for file in sorted(files):  # Ensures files in each directory are read in order
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    for path in tqdm(csv_files, desc="Scanning CSV files"):
        try:
            df = pd.read_csv(path, low_memory=False)
            tag_columns = [col for col in df.columns if col in valid_concepts]
            if not tag_columns:
                continue

            for col in tag_columns:
                for val in df[col].dropna().astype(str):
                    # TODO: Use common token constant for "::"
                    if "::" not in val:
                        continue
                    val_part, unit_part = val.split("::", 1)
                    unit_part = unit_part.strip().upper()
                    try:
                        num_val = float(val_part.strip())
                        concept_unit_value_tuples.append((col, unit_part, num_val))
                        # unit_values[unit_part].append(num_val) # TODO: Remove
                        # unit_concepts[unit_part].add(col) # TODO: Remove
                    except ValueError:
                        non_numeric_units.add(unit_part)
        except Exception as e:
            logging.warning(f"Skipped {path}: {e}")

    return ExtractedConceptUnitValueData(
        concept_unit_value_tuples=concept_unit_value_tuples,
        # unit_values=unit_values, # TODO: Remove
        # unit_concepts=unit_concepts, # TODO: Remove
        non_numeric_units=non_numeric_units,
        csv_files=csv_files,
    )


def get_valid_concepts(db: DB) -> Concepts:
    concept_df = db.get("SELECT name FROM us_gaap_concept", ["name"])
    valid_concepts = set(concept_df["name"].values)

    return valid_concepts


def collect_concept_unit_pairs(
    extracted_concept_unit_value_data: ExtractedConceptUnitValueData,
) -> List[ConceptUnitPair]:
    seen = set()
    concept_unit_pairs = []
    for concept, unit, _ in extracted_concept_unit_value_data.concept_unit_value_tuples:
        pair = (concept, unit)
        if pair not in seen:
            seen.add(pair)
            concept_unit_pairs.append(pair)

    return concept_unit_pairs


# TODO: Document return type
def generate_concept_unit_embeddings(
    concept_unit_pairs: List[ConceptUnitPair], device: torch.device
):
    input_texts = [
        f"{generate_us_gaap_description(concept)} measured in {unit}"
        for concept, unit in concept_unit_pairs
    ]

    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    model.eval()  # IMPORTANT
    model.to(device)

    logging.info(f"Embedding model hash: {model_hash(model)}")

    def encode_on_device(texts, model, batch_size=64):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i : i + batch_size]
            tokens = model.tokenize(batch)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                output = model.forward(tokens)
                embeddings = output["sentence_embedding"]
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings).numpy()

    concept_unit_embeddings = encode_on_device(input_texts, model)

    return concept_unit_embeddings


# TODO: Remove old version
# def generate_concepts_report(
#     extracted_concept_unit_value_data: ExtractedConceptUnitValueData,
# ):
#     print(f"\n✅ Scanned {len(extracted_concept_unit_value_data.csv_files)} files.")
#     print(
#         f"📦 Found {len(extracted_concept_unit_value_data.unit_values)} numeric units and {len(extracted_concept_unit_value_data.non_numeric_units)} non-numeric units."
#     )

#     for unit, values in sorted(extracted_concept_unit_value_data.unit_values.items()):
#         arr = np.array(values)
#         print(f"🔹 {unit}")
#         print(f"   Count: {len(arr)}")
#         print(f"   Min:   {arr.min():,.4f}")
#         print(f"   Max:   {arr.max():,.4f}")
#         print(f"   Mean:  {arr.mean():,.4f}")
#         print(f"   Std:   {arr.std():,.4f}")
#         print(
#             f"   Concepts: {', '.join(sorted(extracted_concept_unit_value_data.unit_concepts[unit]))}"
#         )

#     if extracted_concept_unit_value_data.non_numeric_units:
#         print("\n⚠️ Non-numeric units encountered:")
#         for unit in sorted(extracted_concept_unit_value_data.non_numeric_units):
#             print(f"  - {unit}")

#     print(
#         f"\n🧮 Total values extracted: {len(extracted_concept_unit_value_data.concept_unit_value_tuples):,}"
#     )


# TODO: Validate this matches old version
def generate_concepts_report(
    extracted_concept_unit_value_data: ExtractedConceptUnitValueData,
):
    print(f"\n✅ Scanned {len(extracted_concept_unit_value_data.csv_files)} files.")

    unit_stats = defaultdict(list)
    concept_by_unit = defaultdict(set)

    for (
        concept,
        unit,
        value,
    ) in extracted_concept_unit_value_data.concept_unit_value_tuples:
        unit_stats[unit].append(value)
        concept_by_unit[unit].add(concept)

    print(
        f"📦 Found {len(unit_stats)} numeric units and "
        f"{len(extracted_concept_unit_value_data.non_numeric_units)} non-numeric units."
    )

    for unit, values in sorted(unit_stats.items()):
        arr = np.array(values)
        print(f"🔹 {unit}")
        print(f"   Count: {len(arr)}")
        print(f"   Min:   {arr.min():,.4f}")
        print(f"   Max:   {arr.max():,.4f}")
        print(f"   Mean:  {arr.mean():,.4f}")
        print(f"   Std:   {arr.std():,.4f}")
        print(f"   Concepts: {', '.join(sorted(concept_by_unit[unit]))}")

    if extracted_concept_unit_value_data.non_numeric_units:
        print("\n⚠️ Non-numeric units encountered:")
        for unit in sorted(extracted_concept_unit_value_data.non_numeric_units):
            print(f"  - {unit}")

    print(
        f"\n🧮 Total values extracted: "
        f"{len(extracted_concept_unit_value_data.concept_unit_value_tuples):,}"
    )
