import pandas as pd
from io import BytesIO
from pathlib import Path


def excel_to_csv(
    excel_path: BytesIO | str | Path, sheet_name: str, output_csv_path: str | Path
) -> None:
    """
    Load a specific sheet from an Excel file and save it as CSV.

    Args:
        excel_path: Path to the .xlsx or .xls file, or a BytesIO object.
        sheet_name: Name of the worksheet to extract.
        output_csv_path: Path where the CSV will be saved.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    df.to_csv(output_csv_path, index=False)
