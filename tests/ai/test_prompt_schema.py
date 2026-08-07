# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pyarrow as pa
import pytest

import duckdb
from vane.ai._schema import OutputValidationError, SchemaValidationError, compile_return_format

PORTABLE_SCHEMA = {
    "title": "Answer",
    "type": "object",
    "properties": {
        "active": {"type": "boolean"},
        "items": {"type": "array", "items": {"type": "integer"}},
        "meta": {
            "type": "object",
            "properties": {"score": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
            "required": ["score"],
            "additionalProperties": False,
        },
        "status": {"type": "string", "enum": ["ok", "failed"]},
    },
    "required": ["active", "items", "meta", "status"],
    "additionalProperties": False,
}


def test_finite_schema_maps_to_duckdb_arrow_and_validated_json():
    spec = compile_return_format(PORTABLE_SCHEMA)

    assert spec is not None
    assert spec.name == "Answer"
    assert spec.duckdb_type == (
        'STRUCT("active" BOOLEAN, "items" BIGINT[], "meta" STRUCT("score" DOUBLE), "status" VARCHAR)'
    )
    assert spec.arrow_type == pa.struct(
        [
            pa.field("active", pa.bool_(), nullable=False),
            pa.field("items", pa.list_(pa.field("item", pa.int64(), nullable=False)), nullable=False),
            pa.field(
                "meta",
                pa.struct([pa.field("score", pa.float64(), nullable=True)]),
                nullable=False,
            ),
            pa.field("status", pa.string(), nullable=False),
        ]
    )
    assert json.loads(spec.validate_json('{"active":true,"items":[1,2],"meta":{"score":null},"status":"ok"}')) == {
        "active": True,
        "items": [1, 2],
        "meta": {"score": None},
        "status": "ok",
    }


def test_local_non_recursive_refs_are_supported():
    spec = compile_return_format(
        {
            "$defs": {
                "Detail": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                    "required": ["label"],
                }
            },
            "type": "object",
            "properties": {"detail": {"$ref": "#/$defs/Detail"}},
            "required": ["detail"],
        }
    )

    assert spec is not None
    assert spec.duckdb_type == 'STRUCT("detail" STRUCT("label" VARCHAR))'
    assert json.loads(spec.validate_json({"detail": {"label": "x"}})) == {"detail": {"label": "x"}}


def test_legacy_definitions_are_canonicalized_for_provider_requests():
    spec = compile_return_format(
        {
            "definitions": {
                "Detail": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                    "required": ["label"],
                }
            },
            "type": "object",
            "properties": {"detail": {"$ref": "#/definitions/Detail"}},
            "required": ["detail"],
        }
    )

    assert spec is not None
    assert "definitions" not in spec.schema
    assert spec.schema["properties"]["detail"]["$ref"] == "#/$defs/Detail"
    assert "Detail" in spec.schema["$defs"]
    assert json.loads(spec.validate_json({"detail": {"label": "x"}})) == {"detail": {"label": "x"}}


def test_allowed_additional_properties_are_projected_to_the_struct_shape():
    spec = compile_return_format(
        {
            "type": "object",
            "properties": {
                "detail": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                    },
                },
            },
            "required": ["detail", "items"],
        }
    )
    assert spec is not None

    validated = spec.validate_json(
        {
            "detail": {"label": "x", "nested_extra": True},
            "items": [{"value": 1, "item_extra": "ignored"}],
            "root_extra": 2,
        }
    )

    assert json.loads(validated) == {
        "detail": {"label": "x"},
        "items": [{"value": 1}],
    }


def test_missing_optional_properties_are_materialized_as_null():
    spec = compile_return_format(
        {
            "type": "object",
            "properties": {
                "required": {"type": "string"},
                "optional": {"type": "integer"},
                "nested": {
                    "type": "object",
                    "properties": {"optional": {"type": "boolean"}},
                },
            },
            "required": ["required", "nested"],
            "additionalProperties": False,
        }
    )
    assert spec is not None

    assert json.loads(spec.validate_json({"required": "x", "nested": {}})) == {
        "required": "x",
        "optional": None,
        "nested": {"optional": None},
    }


def test_nullable_enum_preserves_json_schema_intersection_semantics():
    type_array = compile_return_format(
        {
            "type": "object",
            "properties": {"status": {"type": ["string", "null"], "enum": ["ok"]}},
        }
    )
    nullable_branch = compile_return_format(
        {
            "type": "object",
            "properties": {
                "status": {
                    "anyOf": [
                        {"type": "string", "enum": ["ok"]},
                        {"type": "null"},
                    ]
                }
            },
        }
    )
    assert type_array is not None and nullable_branch is not None

    with pytest.raises(OutputValidationError, match="one of"):
        type_array.validate_json({"status": None})
    assert json.loads(nullable_branch.validate_json({"status": None})) == {"status": None}


def test_pydantic_model_and_equivalent_json_schema_compile_identically():
    pydantic = pytest.importorskip("pydantic")

    class Answer(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")

        active: bool
        count: int

    from_model = compile_return_format(Answer)
    from_schema = compile_return_format(Answer.model_json_schema())

    assert from_model is not None and from_schema is not None
    assert from_model.schema == from_schema.schema
    assert from_model.duckdb_type == from_schema.duckdb_type
    assert from_model.arrow_type == from_schema.arrow_type


def test_non_pydantic_class_with_schema_method_is_rejected():
    pytest.importorskip("pydantic")

    class FakeModel:
        @classmethod
        def model_json_schema(cls):
            return PORTABLE_SCHEMA

    with pytest.raises(TypeError, match="Pydantic BaseModel subclass"):
        compile_return_format(FakeModel)


@pytest.mark.parametrize(
    ("schema", "match"),
    [
        ({"type": "array", "items": {"type": "string"}}, "root type"),
        ({"type": "object", "properties": {}}, "at least one"),
        (
            {"type": "object", "properties": {"bad.name": {"type": "string"}}},
            "property names",
        ),
        (
            {"type": "object", "properties": {"x": {"type": "string", "format": "date"}}},
            "unsupported constraint",
        ),
        (
            {
                "type": "object",
                "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
            },
            "T \\| null",
        ),
        (
            {
                "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
                "type": "object",
                "properties": {"node": {"$ref": "#/$defs/Node"}},
            },
            "recursive",
        ),
        (
            {"type": "object", "properties": {"x": {"$ref": "https://example.test/schema"}}},
            "only local",
        ),
        (
            {
                "type": "object",
                "properties": {"meta": {"type": "object", "properties": {}}},
            },
            "at least one",
        ),
        (
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}, "ANSWER": {"type": "integer"}},
            },
            "differ only by case",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}, "VALUE": {"type": "integer"}},
                    }
                },
            },
            "differ only by case",
        ),
    ],
)
def test_schema_subset_rejects_unsupported_shapes(schema, match):
    with pytest.raises(SchemaValidationError, match=match):
        compile_return_format(schema)


def test_case_insensitive_collision_in_definitions_is_rejected():
    with pytest.raises(SchemaValidationError, match="differ only by case"):
        compile_return_format(
            {
                "$defs": {
                    "Detail": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}, "LABEL": {"type": "string"}},
                    }
                },
                "type": "object",
                "properties": {"detail": {"$ref": "#/$defs/Detail"}},
            }
        )


def test_empty_nested_object_in_array_items_is_rejected():
    with pytest.raises(SchemaValidationError, match="at least one property"):
        compile_return_format(
            {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {}}}},
            }
        )


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
            },
        },
        {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}}},
        {
            "type": "object",
            "properties": {
                "a": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "string"}}}}
            },
        },
        {
            "type": "object",
            "properties": {"a": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}},
        },
        {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}}},
        {"type": "object", "properties": {"x": {"type": ["integer", "null"]}}},
        {"type": "object", "properties": {"e": {"type": "string", "enum": ["a", "b"]}}},
        {"type": "object", "properties": {"e": {"type": ["string", "null"], "enum": ["a"]}}},
        {
            "$defs": {"Detail": {"type": "object", "properties": {"y": {"type": "integer"}}}},
            "type": "object",
            "properties": {"d": {"$ref": "#/$defs/Detail"}},
        },
        {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "object",
                            "properties": {"c": {"type": "object", "properties": {"d": {"type": "string"}}}},
                        }
                    },
                }
            },
        },
    ],
)
def test_every_supported_schema_shape_derives_a_duckdb_bindable_type(schema):
    spec = compile_return_format(schema)
    assert spec is not None

    duckdb.sqltype(spec.duckdb_type)
    duckdb.sql(f"SELECT CAST(NULL AS {spec.duckdb_type})")


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ('{"active":true,"items":[],"meta":{"score":1},"status":"other"}', "one of"),
        ('{"active":true,"items":[],"meta":{},"status":"ok"}', "missing required"),
        ('{"active":true,"items":[true],"meta":{"score":1},"status":"ok"}', "integer"),
        ('{"active":true,"items":[],"meta":{"score":1},"status":"ok","extra":1}', "unexpected"),
        (
            '{"active":true,"items":[],"meta":{"score":1e400},"status":"ok"}',
            "finite",
        ),
        ("not-json", "valid JSON"),
    ],
)
def test_runtime_validation_rejects_invalid_provider_output(value, match):
    spec = compile_return_format(PORTABLE_SCHEMA)
    assert spec is not None

    with pytest.raises(OutputValidationError, match=match):
        spec.validate_json(value)


def test_openai_gpt_requires_closed_objects_and_all_properties_required():
    spec = compile_return_format(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }
    )
    assert spec is not None

    with pytest.raises(SchemaValidationError, match="additionalProperties"):
        spec.validate_openai_gpt_contract("gpt-5-mini")
    spec.validate_openai_gpt_contract("compatible-non-gpt-model")

    spec = compile_return_format(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": False,
        }
    )
    assert spec is not None
    with pytest.raises(SchemaValidationError, match="every property"):
        spec.validate_openai_gpt_contract("gpt-5-mini")
