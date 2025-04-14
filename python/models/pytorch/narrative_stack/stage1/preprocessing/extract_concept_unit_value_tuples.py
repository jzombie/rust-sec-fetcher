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
    rows: List[Tuple[str, str, float]]
    unit_values: Dict[str, List[float]]
    unit_concepts: Dict[str, Set[str]]
    non_numeric_units: Set[str]
    csv_files: List[str]

def extract_concept_unit_value_tuples(data_dir: str | Path, valid_concepts: list[str]) -> ExtractedConceptData:
    rows = []
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
                    if "::" not in val:
                        continue
                    val_part, unit_part = val.split("::", 1)
                    unit_part = unit_part.strip().upper()
                    try:
                        num_val = float(val_part.strip())
                        rows.append((col, unit_part, num_val))
                        unit_values[unit_part].append(num_val)
                        unit_concepts[unit_part].add(col)
                    except ValueError:
                        non_numeric_units.add(unit_part)
        except Exception as e:
            logging.warning(f"Skipped {path}: {e}")

    return ExtractedConceptData(
        rows=List[Tuple[str, str, float]],
        unit_values=Dict[str, List[float]],
        unit_concepts=Dict[str, Set[str]],
        non_numeric_units=Set[str],
        csv_files=List[str]
    )

def get_valid_concepts(db: DB) -> list[str]:
    concept_df = db.get("SELECT name FROM us_gaap_concept", ["name"])
    valid_concepts = set(concept_df["name"].values)

    return valid_concepts
