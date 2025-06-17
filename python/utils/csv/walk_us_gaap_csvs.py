import os
import logging
from pathlib import Path
from typing import Generator, Union, Dict, List, Literal, Tuple
from tqdm import tqdm
from utils.os import to_path
import pandas as pd

UsGaapConcept = str

# TODO: Update types
RowYield = pd.Series
CellYield = Dict[str, Union[str, float]]
PairYield = Tuple[str, str]
WalkYield = Union[RowYield, CellYield, PairYield]
WalkGenerator = Generator[WalkYield, None, set]


# TODO: Add return type
def walk_us_gaap_csvs(
    data_dir: str | Path,
    valid_concepts: List[UsGaapConcept],
    walk_type: Literal["row", "cell", "pair"] = "cell",
) -> WalkGenerator:
    non_numeric_units = set()
    seen_pairs = set()  # Only populated if `walk_type` == `pair`

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

            # if walk_type == "row":
            #     for row in df.itertuples(index=False):
            #         yield row

            if walk_type == "row":
                tag_columns = [col for col in df.columns if col in valid_concepts]
                if not tag_columns:
                    continue

                for _, row in df.iterrows():
                    entries = []
                    for col in tag_columns:
                        val = str(row[col])
                        # TODO: Use common token constant for "::"
                        if "::" not in val:
                            continue

                        # TODO: Use common token constant for "::"
                        val_part, unit_part = val.split("::", 1)
                        unit_part = unit_part.strip().upper()
                        try:
                            num_val = float(val_part.strip())
                            # TODO: Use pydantic here
                            entries.append(
                                {"concept": col, "uom": unit_part, "value": num_val}
                            )
                        except ValueError:
                            non_numeric_units.add(unit_part)
                    if entries:
                        yield {
                            "ticker_symbol": os.path.splitext(os.path.basename(path))[
                                0
                            ],
                            "form": row["form"],
                            "filed": row["filed"],
                            # "columns": tag_columns,
                            "entries": entries,
                        }

            elif walk_type in {"cell", "pair"}:
                # TODO: Rename `col` to `concept`
                tag_columns = [col for col in df.columns if col in valid_concepts]

                if not tag_columns:
                    continue

                for col in tag_columns:

                    for val in df[col].dropna().astype(str):
                        # TODO: Use common token constant for "::"
                        if "::" not in val:
                            continue
                        # TODO: Use common token constant for "::"
                        val_part, unit_part = val.split("::", 1)

                        # TODO: Rename `unit_part` to `uom`
                        unit_part = unit_part.strip().upper()
                        try:
                            # Will raise a `ValueError`` cannot be parsed as a float
                            num_val = float(val_part.strip())

                        except ValueError:
                            non_numeric_units.add(unit_part)
                            continue

                        if walk_type == "cell":
                            # TODO: Use Pydantic here
                            yield {"concept": col, "uom": unit_part, "value": num_val}
                        elif walk_type == "pair":
                            pair = (col, unit_part)
                            if pair not in seen_pairs:
                                seen_pairs.add(pair)
                                yield pair

        except Exception as e:
            logging.warning(f"Skipped {path}: {e}")

        return non_numeric_units
