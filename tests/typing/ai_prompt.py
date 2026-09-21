# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

from typing_extensions import assert_type

import vane
from vane.ai import prompt
from vane.ai._sql import build_ai_prompt_sql_spec

schema: vane.ai.JSONSchema = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

text = vane.col("text")
image = vane.col("image")
relation = cast(vane.Relation, None)


class FakePydanticModel:
    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        return schema


assert_type(prompt(text), vane.Expression)
assert_type(prompt(messages=[text, image], max_output_tokens=64), vane.Expression)
assert_type(prompt(text, return_format=schema), vane.Expression)
assert_type(prompt(text, return_format=schema, return_raw_response=True), vane.Expression)
assert_type(prompt(relation, text), vane.Relation)
assert_type(prompt(relation, text, return_format=schema), vane.Relation)
assert_type(prompt(rel=relation, messages=[text, image]), vane.Relation)
assert_type(relation.prompt([text, image], output_column="answer"), vane.Relation)
relation.prompt(text, return_format=FakePydanticModel)  # type: ignore[arg-type]

assert_type(build_ai_prompt_sql_spec(input_kind="text"), dict[str, Any])
assert_type(build_ai_prompt_sql_spec(input_kind="blob"), dict[str, Any])
assert_type(build_ai_prompt_sql_spec(input_kind="blob_list"), dict[str, Any])
assert_type(build_ai_prompt_sql_spec(input_kind="file"), dict[str, Any])
assert_type(build_ai_prompt_sql_spec(input_kind="file_list"), dict[str, Any])
build_ai_prompt_sql_spec(input_kind="image")  # type: ignore[arg-type]
build_ai_prompt_sql_spec(input_kind="video")  # type: ignore[arg-type]
