# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run with vane-ai[typesafe] installed and TYPESAFE_API_KEY set.

Each row sends its state and all three questions together. Results are JSON
text (VARCHAR), including the complete probability distributions and token usage.
"""

from typesafe_sdk import Choice, Noul, Score

import vane
from vane.ai import jev


def main() -> None:
    questions = {
        "billing": Noul(instructions="Is this ticket about billing?"),
        "team": Choice(
            instructions="Which team should handle this ticket?",
            criteria={"billing": "Charges, invoices, refunds", "technical": "Bugs or outages", "other": "Neither"},
        ),
        "urgency": Score(
            instructions="How urgently does this ticket need attention?",
            criteria=["Can wait until next week", "Needs attention this week", "Needs attention today"],
        ),
    }
    with vane.connect() as connection:
        source = connection.sql(
            "SELECT * FROM (VALUES (1, 'I was charged twice. Please refund the extra payment.'), "
            "(2, 'Our production service has been unavailable all morning.'), (3, NULL)) AS t(id, text)"
        )
        result = jev(source, vane.col("text"), questions=questions, max_concurrency_per_actor=8)
        # Fetch once; accessing a lazy relation again may repeat inference.
        for row in (
            result.order("id")
            .select(
                vane.col("id"),
                vane.sql_expr("response ->> '$.answers.team.choice'").alias("team"),
                vane.sql_expr("(response ->> '$.answers.billing.noul')::DOUBLE").alias("billing_probability"),
                vane.sql_expr("(response ->> '$.answers.urgency.score')::DOUBLE").alias("urgency"),
            )
            .fetchall()
        ):
            print(row)


if __name__ == "__main__":
    main()
