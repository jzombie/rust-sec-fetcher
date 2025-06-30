import os
import logging
from pathlib import Path
from typing import Generator, Union, List, Literal, Tuple, Set
from tqdm import tqdm
from utils.os import to_path
import pandas as pd
from pydantic import BaseModel
from db import DbUsGaap
import math


# Explicit domain-focused types
UsGaapCsvConceptUomPair = Tuple[str, str]


class UsGaapCsvTriplet(BaseModel):
    concept: str
    uom: str
    value: float | int
    balance_type: Literal["credit", "debit"] | None
    period_type: Literal["duration", "instant"] | None

    def as_key(self) -> str:
        return f"{self.concept}::{self.uom}::{self.value}"


class UsGaapCsvRowRecord(BaseModel):
    ticker_symbol: str
    form: str
    filed: str
    entries: List[UsGaapCsvTriplet]


class UsGaapCsvWalkSummary(BaseModel):
    non_numeric_units: Set[str]
    csv_files: List[str]


UsGaapCsvYield = Union[UsGaapCsvTriplet, UsGaapCsvConceptUomPair, UsGaapCsvRowRecord, str]
UsGaapCsvIterator = Generator[UsGaapCsvYield, None, UsGaapCsvWalkSummary]

def walk_us_gaap_csvs(
    data_dir: str | Path,
    db_us_gaap: DbUsGaap,
    # TODO: Consider removing `cell` walk type (and use row entries instead)
    walk_type: Literal["row", "cell", "pair", "ticker_symbol"] = "cell",
    filtered_symbols: set[str] | None = None,
    tqdm_desc: str = "Scanning CSV files",
) -> UsGaapCsvIterator:
    """
    Iterate over SEC US-GAAP CSV exports in a *flat* directory.

    The function is deterministic **IF** all CSVs live in the same
    directory (no sub-folders).  File order is fixed by an explicit
    `sorted(files)` call; row and column order are preserved by pandas.

    Args:
        data_dir: Path to the directory containing the CSV files.  Only
            one level deep is supported; nested folders break ordering
            guarantees.
        db_us_gaap: Database handle used for concept validation and for
            querying balance/period metadata.
        walk_type: Selects the yield granularity:
            * "row"  - yield a ``UsGaapCsvRowRecord`` per table row.
            * "cell" - yield a ``UsGaapCsvTriplet`` per numeric cell.
            * "pair" - yield each unique ``(concept, uom)`` pair once.
            * "ticker_symbol" - yield the file's ticker symbol only.
        filtered_symbols: If given, restrict traversal to these tickers.
        tqdm_desc: Progress-bar label.

    Yields:
        Objects defined by ``UsGaapCsvYield`` that match *walk_type*.

    Returns:
        UsGaapCsvWalkSummary: Lists all CSV paths scanned and the set of
        units that appeared with non-numeric values.

    Notes:
        * The generator consumes every CSV fully before moving to the
          next, so memory footprint stays small.
        * In "pair" mode, uniqueness is enforced with an in-memory set.
          In CPython ≥3.7 dicts preserve insertion order, so first-seen
          order is stable.
    """

    valid_concepts = db_us_gaap.get_valid_concepts()

    non_numeric_units = set()
    seen_pairs = set()  # Only populated if `walk_type` == `pair`

    csv_files = []
    # Note: If this were to span multiple sub-directories, dirs should be presorted as well
    for root, _, files in os.walk(to_path(data_dir, as_str=True)):
        for file in sorted(files):  # Ensures files in each directory are read in order
            if not file.endswith(".csv"):
                continue

            ticker_symbol = os.path.splitext(file)[0]
            if filtered_symbols is not None and ticker_symbol not in filtered_symbols:
                continue

            csv_files.append(os.path.join(root, file))

    for path in tqdm(csv_files, desc=tqdm_desc):
        ticker_symbol = os.path.splitext(os.path.basename(path))[0]

        if walk_type == "ticker_symbol":
            yield ticker_symbol
            continue

        try:
            # Note: To ensure no mixed types either set False, or specify the type with the dtype parameter.
            # These files are not large enough on their own to need `low_memory` set to True.
            df = pd.read_csv(path, low_memory=False)

            if walk_type == "row":
                tag_columns = [col for col in df.columns if col in valid_concepts]
                if not tag_columns:
                    continue

                # Fetch balance_type and period_type for all tag columns in one batch
                concept_meta_list = (
                    db_us_gaap.get_balance_and_period_types_for_concepts(tag_columns)
                )
                concept_meta_map = dict(zip(tag_columns, concept_meta_list))

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

                            # Ensure `NaN`'s are skipped
                            if math.isnan(num_val):
                                continue

                            # Make this optional
                            balance_type, period_type = concept_meta_map.get(
                                col, (None, None)
                            )

                            entries.append(
                                UsGaapCsvTriplet(
                                    concept=col,
                                    uom=unit_part,
                                    value=num_val,
                                    balance_type=balance_type,
                                    period_type=period_type,
                                )
                            )
                        except ValueError:
                            non_numeric_units.add(unit_part)
                    if entries:
                        yield UsGaapCsvRowRecord(
                            ticker_symbol=ticker_symbol,
                            form=row["form"],
                            filed=row["filed"],
                            entries=entries,
                        )

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

                            # Ensure `NaN`'s are skipped
                            if math.isnan(num_val):
                                continue

                        except ValueError:
                            non_numeric_units.add(unit_part)
                            continue

                        if walk_type == "cell":
                            yield UsGaapCsvTriplet(
                                concept=col,
                                uom=unit_part,
                                value=num_val,
                                balance_type=None,  # Currently not supported in `cell` operation; use `row` operation instead
                                period_type=None,  # Currently not supported in `cell` operation; use `row` operation instead
                            )
                        elif walk_type == "pair":
                            pair = (col, unit_part)
                            if pair not in seen_pairs:
                                seen_pairs.add(pair)
                                yield pair

        except Exception as e:
            logging.warning(f"Skipped {path}: {e}")

    return UsGaapCsvWalkSummary(csv_files=csv_files, non_numeric_units=non_numeric_units)


def get_filtered_us_gaap_form_rows_for_symbol(
    data_dir: str | Path,
    db_us_gaap: DbUsGaap,
    symbol: str,
    form_types: set[str] | None = None,
) -> Generator[UsGaapCsvRowRecord, None, None]:
    """
    Yields parsed row records for a given ticker symbol, optionally filtered by form.

    This function loads all CSV rows associated with the specified ticker symbol
    and yields only those rows that match the allowed form types, if provided.

    Args:
        data_dir (str | Path): Directory containing flat SEC US GAAP CSV files.
        db_us_gaap (DbUsGaap): Database interface used to fetch concept metadata.
        symbol (str): The ticker symbol to filter on.
        form_types (set[str] | None, optional): A set of form types to include
            (e.g., {"10-K", "10-Q"}). If None, all forms are included.

    Yields:
        UsGaapCsvRowRecord: One record per qualifying row in the CSV.

    Notes:
        * This function uses the "row" walk mode and performs type checks on each
          yielded item to ensure it is a `UsGaapCsvRowRecord`.
        * Form filtering is applied after row parsing.
    """

    rows = walk_us_gaap_csvs(
        data_dir=data_dir,
        db_us_gaap=db_us_gaap,
        walk_type="row",
        filtered_symbols={symbol},
    )

    for row in rows:
        if isinstance(row, UsGaapCsvRowRecord):
            if form_types and row.form not in form_types:
                continue
            yield row
