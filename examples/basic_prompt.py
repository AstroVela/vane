# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Basic, structured, and raw text/image Prompt examples for Python and SQL."""

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

# The portable schema subset produces a native STRUCT response. For OpenAI GPT
# models every object is closed and every property is required.
answer_schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "confidence": {"type": ["number", "null"]},
    },
    "required": ["summary", "confidence"],
    "additionalProperties": False,
}
structured = vane.ai.prompt(
    relation,
    [vane.col("prompt"), vane.col("image")],
    return_format=answer_schema,
    model="gpt-4o-mini",
    output_column="answer",
)
structured.show()

# Combining both parameters still constrains generation with the schema, but
# returns the provider response body JSON as VARCHAR.
raw = vane.ai.prompt(
    relation,
    vane.col("prompt"),
    return_format=answer_schema,
    return_raw_response=True,
    model="gpt-4o-mini",
    output_column="raw_response",
)
raw.show()

# The equivalent typed SQL overload accepts one BLOB or BLOB[] image input.
connection.sql(
    """
    SELECT AI_PROMPT(
        prompt,
        image,
        return_format := json '{
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "confidence": {"type": ["number", "null"]}
            },
            "required": ["summary", "confidence"],
            "additionalProperties": false
        }',
        system_message := 'Answer briefly.',
        model := 'gpt-4o-mini',
        options := struct_pack(max_output_tokens := 128)
    ) AS response
    FROM relation
    """
).show()
