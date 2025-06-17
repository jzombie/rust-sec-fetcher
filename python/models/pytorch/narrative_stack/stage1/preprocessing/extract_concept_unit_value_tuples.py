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
from simd_r_drive import DataStore, NamespaceHasher
import msgpack

ConceptUnitValueTuples = List[Tuple[str, str, float]]
# UnitValues = Dict[str, List[float]] # TODO: Remove
# UnitConcepts = Dict[str, Set[str]] # TODO: Remove
NonNumericUnits = Set[str]
CsvFiles = List[str]
Concepts = List[str]
ConceptUnitPair = Tuple[str, str]

NON_EMBEDDED_NAMESPACE = NamespaceHasher(b"non-embedded-concept-unit-value-tuples")


# TODO: Remove
class ExtractedConceptUnitValueData(BaseModel):
    concept_unit_value_tuples: ConceptUnitValueTuples
    # unit_values: UnitValues # TODO: Remove
    # unit_concepts: UnitConcepts # TODO: Remove
    non_numeric_units: NonNumericUnits
    csv_files: CsvFiles


# TODO: Restructure as an iterator, with option to yield single values, or an entire row, and don't integrate the store here
def extract_concept_unit_value_tuples(
    data_dir: str | Path, valid_concepts: Concepts, data_store: DataStore
) -> None:
    # concept_unit_value_tuples = []
    # unit_values = defaultdict(list) # TODO: Remove
    # unit_concepts = defaultdict(set) # TODO: Remove
    non_numeric_units = set()

    csv_files = []
    # Note: If this were to span multiple sub-directories, dirs should be presorted as well
    for root, _, files in os.walk(to_path(data_dir, as_str=True)):
        for file in sorted(files):  # Ensures files in each directory are read in order
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    # TODO: Namespace and iterate
    namespace = NON_EMBEDDED_NAMESPACE

    i = -1

    # TODO: Extract so this can be used to obtain symbol & year & statement type
    for path in tqdm(csv_files, desc="Scanning CSV files"):
        try:
            # Note: To ensure no mixed types either set False, or specify the type with the dtype parameter.
            # These files are not large enough on their own to need `low_memory` set to True.
            df = pd.read_csv(path, low_memory=False)

            # TODO: Iterate over rows or values

            # TODO: Handle
            for row in df.itertuples(index=False):
                print(row)  # Access via row.ColumnName
                break

            # TODO: Remove
            return df

            # TODO: Rename `col` to `concept`
            tag_columns = [col for col in df.columns if col in valid_concepts]
            if not tag_columns:
                continue

            for col in tag_columns:
                batch_entries = []

                for val in df[col].dropna().astype(str):
                    # TODO: Use common token constant for "::"
                    if "::" not in val:
                        continue
                    val_part, unit_part = val.split("::", 1)

                    # TODO: Rename `unit_part` to `uom`
                    unit_part = unit_part.strip().upper()
                    try:
                        # Will raise a `ValueError`` cannot be parsed as a float
                        num_val = float(val_part.strip())

                        # Increment only before write
                        i = i + 1

                        # concept_unit_value_tuples.append((col, unit_part, num_val))
                        payload = msgpack.packb(
                            {"concept": col, "uom": unit_part, "value": num_val}
                        )

                        # Numeric index → data mapping
                        batch_entries.append(
                            (
                                namespace.namespace(i.to_bytes(4, byteorder="little")),
                                payload,
                            )
                        )

                        # Data hash → numeric index reverse mapping
                        batch_entries.append(
                            (
                                namespace.namespace(payload),
                                i.to_bytes(4, byteorder="little"),
                            )
                        )

                    except ValueError:
                        non_numeric_units.add(unit_part)

                # TODO: Uncomment if leaving in
                # data_store.batch_write(batch_entries)
        except Exception as e:
            logging.warning(f"Skipped {path}: {e}")

        total_entries = i + 1

        # TODO: Uncomment if leaving in
        # data_store.write(
        #     namespace.namespace(
        #         b"__triplet_count__",
        #     ),
        #     total_entries.to_bytes(4, byteorder="little"),
        # )

    # TODO: Remove
    # return ExtractedConceptUnitValueData(
    #     concept_unit_value_tuples=concept_unit_value_tuples,
    #     # unit_values=unit_values, # TODO: Remove
    #     # unit_concepts=unit_concepts, # TODO: Remove
    #     non_numeric_units=non_numeric_units,
    #     csv_files=csv_files,
    # )


# TODO: Move/refactor
# Assuming that DB-stored concepts are "valid"
def get_valid_concepts(db: DB) -> Concepts:
    concept_df = db.get("SELECT name FROM us_gaap_concept", ["name"])
    valid_concepts = set(concept_df["name"].values)

    return valid_concepts


# TODO: Remove
# def collect_concept_unit_pairs(
#     extracted_concept_unit_value_data: ExtractedConceptUnitValueData,
# ) -> List[ConceptUnitPair]:
#     seen = set()
#     concept_unit_pairs = []
#     for concept, unit, _ in extracted_concept_unit_value_data.concept_unit_value_tuples:
#         pair = (concept, unit)
#         if pair not in seen:
#             seen.add(pair)
#             concept_unit_pairs.append(pair)

#     return concept_unit_pairs


# TODO: Refactor to work with drive
# TODO: Document return type
# TODO: Accept an iterator instead of a list
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


# TODO: Refactor to work with drive
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
