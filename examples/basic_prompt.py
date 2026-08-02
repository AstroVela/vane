# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Basic text/image Prompt examples for Python and SQL."""

import vane


connection = vane.connect()
relation = connection.sql(
    """
    SELECT *
    FROM (VALUES
        ('Summarize this image.', read_blob('example.png'))
    ) input(prompt, image)
    """
)

# Functional Relation API. Message expressions are evaluated left-to-right;
# NULL items are omitted for each row.
responses = vane.ai.prompt(
    relation,
    [vane.col("prompt"), vane.col("image")],
    system_message="Answer briefly.",
    model="gpt-4o-mini",
    output_column="response",
    max_output_tokens=128,
)
responses.show()

# The equivalent typed SQL overload accepts one BLOB or BLOB[] image input.
connection.sql(
    """
    SELECT AI_PROMPT(
        prompt,
        image,
        system_message := 'Answer briefly.',
        model := 'gpt-4o-mini',
        options := struct_pack(max_output_tokens := 128)
    ) AS response
    FROM relation
    """
).show()
