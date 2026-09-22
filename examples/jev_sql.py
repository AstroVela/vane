# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run Jev directly in SQL with vane-ai[typesafe] and TYPESAFE_API_KEY configured.

ai_jev(state, questions, model := 'jev-latest', on_error := 'raise', options := NULL)
returns JSON text (VARCHAR) with answers, model, and usage. State is evaluated
per row: text stays text, and JSON objects, STRUCTs, and lists stay structured.
NULL state returns NULL. Questions accepts a constant JSON string or SQL STRUCT;
model, on_error, and the options STRUCT must also be constant.

Options: batch_size, actor_number, max_concurrency_per_actor, max_retries,
base_url, timeout. SQL uses the connection's actor backend (local subprocess
or Ray). Configure credentials in the application environment before planning;
inline credentials are rejected. on_error='ignore' turns failed rows into NULL
after SDK retries; planning and client initialization failures still propagate.
"""

import vane


def main() -> None:
    with vane.connect() as connection:
        result = connection.sql("""
            WITH judged AS MATERIALIZED (
                SELECT id, ai_jev(
                    text,
                    questions := {
                        'billing': {'type': 'noul', 'instructions': 'Is this ticket about billing?'},
                        'team': {
                            'type': 'choice',
                            'instructions': 'Which team should handle this ticket?',
                            'criteria': {'billing': 'Charges or refunds', 'technical': 'Bugs or outages'}
                        }
                    },
                    options := {max_concurrency_per_actor: 8}
                ) AS judgment
                FROM (VALUES (1, 'I was charged twice.'), (2, 'Our service is down.'), (3, NULL)) t(id, text)
            )
            SELECT id,
                   judgment ->> '$.answers.team.choice' AS team,
                   (judgment ->> '$.answers.billing.noul')::DOUBLE AS billing_probability
            FROM judged
            ORDER BY id
        """)
        for row in result.fetchall():
            print(row)


if __name__ == "__main__":
    main()
