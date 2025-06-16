import os
import logging
from pathlib import Path
from typing import Generator, Union, Dict, List, Literal
from tqdm import tqdm
from utils.os import to_path
import pandas as pd

UsGaapConcept = str


# TODO: Add return type
def walk_us_gaap_csvs(
    data_dir: str | Path,
    valid_concepts: List[UsGaapConcept],
    walk_type: Literal["row", "cell"] = "cell",
) -> Generator[Union[pd.Series, Dict[str, Union[str, float]]], None, set]:
    non_numeric_units = set()

    csv_files = []
    # Note: If this were to span multiple sub-directories, dirs should be presorted as well
    for root, _, files in os.walk(to_path(data_dir, as_str=True)):
        for file in sorted(files):  # Ensures files in each directory are read in order
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    # TODO: Extract so this can be used to obtain symbol & year & statement type
    for path in tqdm(csv_files, desc="Scanning CSV files"):
        try:
            # Note: To ensure no mixed types either set False, or specify the type with the dtype parameter.
            # These files are not large enough on their own to need `low_memory` set to True.
            df = pd.read_csv(path, low_memory=False)

            if walk_type == "row":
                for row in df.itertuples(index=False):
                    yield row

            elif walk_type == "cell":
                # TODO: Rename `col` to `concept`
                tag_columns = [col for col in df.columns if col in valid_concepts]

                if not tag_columns:
                    continue

                for col in tag_columns:

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

                            # TODO: Use Pydantic here?
                            yield {"concept": col, "uom": unit_part, "value": num_val}

                        except ValueError:
                            non_numeric_units.add(unit_part)

        except Exception as e:
            logging.warning(f"Skipped {path}: {e}")

        return non_numeric_units
