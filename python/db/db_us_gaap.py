from .db import DB
from typing import List, Tuple


class DbUsGaap(DB):
    # TODO: Use common type
    def get_valid_concepts(self) -> List[str]:
        concept_df = self.get("SELECT name FROM us_gaap_concept", ["name"])
        valid_concepts = set(concept_df["name"].values)

        return valid_concepts

    def get_concept_balance_and_period_type(
        self, concept: str
    ) -> Tuple[str | None, str | None]:
        result_df = self.get(
            """
            SELECT bt.balance_type, pt.period_type
            FROM us_gaap_concept AS c
            LEFT JOIN us_gaap_balance_type AS bt ON c.balance_type_id = bt.id
            LEFT JOIN us_gaap_period_type AS pt ON c.period_type_id = pt.id
            WHERE c.name = %s
            """,
            ["balance_type", "period_type"],
            params=(concept,),
        )

        if result_df.empty:
            return None, None

        row = result_df.iloc[0]
        return row["balance_type"], row["period_type"]
