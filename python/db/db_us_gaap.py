from .db import DB
from typing import List


class DbUsGaap(DB):
    # TODO: Use common type
    def get_valid_concepts(self) -> List[str]:
        concept_df = self.get("SELECT name FROM us_gaap_concept", ["name"])
        valid_concepts = set(concept_df["name"].values)

        return valid_concepts
