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


class ExtractedConceptData(BaseModel):
    concept_unit_value_tuples: List[Tuple[str, str, float]]
    unit_values: Dict[str, List[float]]
    unit_concepts: Dict[str, Set[str]]
    non_numeric_units: Set[str]
    csv_files: List[str]


def extract_concept_unit_value_tuples(
    data_dir: str | Path, valid_concepts: list[str]
) -> ExtractedConceptData:
    concept_unit_value_tuples = []
    unit_values = defaultdict(list)
    unit_concepts = defaultdict(set)
    non_numeric_units = set()

    csv_files = []
    for root, _, files in os.walk(to_path(data_dir, as_str=True)):
        for file in files:
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
                        unit_values[unit_part].append(num_val)
                        unit_concepts[unit_part].add(col)
                    except ValueError:
                        non_numeric_units.add(unit_part)
        except Exception as e:
            logging.warning(f"Skipped {path}: {e}")

    return ExtractedConceptData(
        concept_unit_value_tuples=concept_unit_value_tuples,
        unit_values=unit_values,
        unit_concepts=unit_concepts,
        non_numeric_units=non_numeric_units,
        csv_files=csv_files,
    )


def get_valid_concepts(db: DB) -> list[str]:
    concept_df = db.get("SELECT name FROM us_gaap_concept", ["name"])
    valid_concepts = set(concept_df["name"].values)

    return valid_concepts


def generate_concepts_report(data_dir: str | Path, db: DB):
    valid_concepts = get_valid_concepts(db)

    extracted_concept_data = extract_concept_unit_value_tuples(data_dir, valid_concepts)

    print(f"\n✅ Scanned {len(extracted_concept_data.csv_files)} files.")
    print(
        f"📦 Found {len(extracted_concept_data.unit_values)} numeric units and {len(extracted_concept_data.non_numeric_units)} non-numeric units."
    )

    for unit, values in sorted(extracted_concept_data.unit_values.items()):
        arr = np.array(values)
        print(f"🔹 {unit}")
        print(f"   Count: {len(arr)}")
        print(f"   Min:   {arr.min():,.4f}")
        print(f"   Max:   {arr.max():,.4f}")
        print(f"   Mean:  {arr.mean():,.4f}")
        print(f"   Std:   {arr.std():,.4f}")
        print(
            f"   Concepts: {', '.join(sorted(extracted_concept_data.unit_concepts[unit]))}"
        )

    if extracted_concept_data.non_numeric_units:
        print("\n⚠️ Non-numeric units encountered:")
        for unit in sorted(extracted_concept_data.non_numeric_units):
            print(f"  - {unit}")

    print(
        f"\n🧮 Total values extracted: {len(extracted_concept_data.concept_unit_value_tuples):,}"
    )
