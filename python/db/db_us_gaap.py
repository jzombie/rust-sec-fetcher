from .db import DB
from typing import List, Tuple


class DbUsGaap(DB):
    # TODO: Use common type
    def get_valid_concepts(self) -> List[str]:
        concept_df = self.get("SELECT name FROM us_gaap_concept", ["name"])
        valid_concepts = set(concept_df["name"].values)

        return valid_concepts

    def get_balance_and_period_types_for_concepts(
        self, concepts: list[str]
    ) -> list[tuple[str | None, str | None]]:
        """
        Returns a list of (balance_type, period_type) for each concept in the input list.
        For concepts not found in the database, (None, None) is returned in their place.
        The output list preserves the exact order of the input list.
        """

        # Query all matches at once using IN clause
        result_df = self.get(
            """
            SELECT c.name, bt.balance_type, pt.period_type
            FROM us_gaap_concept AS c
            LEFT JOIN us_gaap_balance_type AS bt ON c.balance_type_id = bt.id
            LEFT JOIN us_gaap_period_type AS pt ON c.period_type_id = pt.id
            WHERE c.name IN (%s)
            """
            % ", ".join(
                ["%s"] * len(concepts)
            ),  # Generates placeholders like %s, %s, ...
            ["name", "balance_type", "period_type"],
            params=tuple(concepts),
        )

        # Build a map from concept name to (balance, period_type)
        result_map = {
            row["name"]: (row["balance_type"], row["period_type"])
            for _, row in result_df.iterrows()
        }

        # Iterate over input list to preserve order
        # If a concept is missing, insert (None, None)
        return [result_map.get(concept, (None, None)) for concept in concepts]
