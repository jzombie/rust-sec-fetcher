import json
import csv

# Optionally handle multi-line files here?
def load_json(file_path):
    """
    Loads the contents from a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: Parsed JSON data.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def load_csv(file_path: str):
    """
    Loads the data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - list: Parsed data from the CSV file as a list of dictionaries.
    """
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data
