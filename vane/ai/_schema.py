# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Finite JSON Schema support for Prompt structured outputs.

The public Prompt contract deliberately accepts a small, provider-portable
subset of JSON Schema.  This module is the single implementation of that
subset: request schemas, DuckDB/Arrow types, and runtime validation are all
derived from the same compiled tree.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import pyarrow as pa


class SchemaValidationError(ValueError):
    """A Prompt ``return_format`` is outside Vane's finite schema subset."""


class OutputValidationError(TypeError):
    """A provider's structured output does not satisfy ``return_format``."""


class RawResponseSerializationError(TypeError):
    """A provider response body cannot be represented as strict JSON."""


_PROPERTY_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SCHEMA_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SCALAR_TYPES = frozenset({"string", "integer", "number", "boolean"})
_SUPPORTED_TYPES = _SCALAR_TYPES | {"object", "array"}
_ANNOTATION_KEYS = frozenset(
    {
        "$comment",
        "$id",
        "$schema",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }
)
_SCHEMA_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "allOf",
        "anyOf",
        "definitions",
        "enum",
        "items",
        "oneOf",
        "properties",
        "required",
        "type",
    }
)
_UNSUPPORTED_CONSTRAINTS = frozenset(
    {
        "contains",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxContains",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minContains",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "pattern",
        "patternProperties",
        "propertyNames",
        "uniqueItems",
    }
)
_BIGINT_MIN = -(2**63)
_BIGINT_MAX = 2**63 - 1
_OPENAI_STRUCTURED_OUTPUT_MODELS = frozenset(
    {
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-search-preview",
        "gpt-4o-search-preview-2025-03-11",
        "gpt-4o-mini-search-preview",
        "gpt-4o-mini-search-preview-2025-03-11",
        "gpt-4.1",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano",
        "gpt-4.1-nano-2025-04-14",
        "o1",
        "o1-2024-12-17",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-pro",
        "o1-pro-2025-03-19",
        "o3",
        "o3-2025-04-16",
        "o3-mini",
        "o3-mini-2025-01-31",
        "o3-pro",
        "o3-pro-2025-06-10",
        "o4-mini",
        "o4-mini-2025-04-16",
        "gpt-5",
        "gpt-5-2025-08-07",
        "gpt-5-mini",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano",
        "gpt-5-nano-2025-08-07",
        "gpt-5-pro",
        "gpt-5-pro-2025-10-06",
        "gpt-5-chat-latest",
        "gpt-5.1",
        "gpt-5.1-2025-11-13",
    }
)
_OPENAI_UNSUPPORTED_STRUCTURED_OUTPUT_MODELS = frozenset(
    {
        "o1-mini",
        "o1-mini-2024-09-12",
    }
)


def _openai_structured_outputs_capability(model: str) -> Literal["supported", "unsupported", "unknown"]:
    normalized = model.strip().casefold()
    if normalized in _OPENAI_STRUCTURED_OUTPUT_MODELS:
        return "supported"
    if normalized in _OPENAI_UNSUPPORTED_STRUCTURED_OUTPUT_MODELS:
        return "unsupported"
    return "unknown"


def _is_known_openai_prompt_model(model: str) -> bool:
    return _openai_structured_outputs_capability(model) != "unknown"


def _uses_openai_strict_structured_outputs(model: str) -> bool:
    return _openai_structured_outputs_capability(model) == "supported"


def _schema_error(path: str, message: str) -> SchemaValidationError:
    return SchemaValidationError(f"Invalid Prompt return_format at {path}: {message}")


def _output_error(path: str, message: str) -> OutputValidationError:
    return OutputValidationError(f"Prompt structured output at {path} {message}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not supported")


def _strict_json_loads(value: str) -> Any:
    return json.loads(value, parse_constant=_reject_json_constant)


@dataclass(frozen=True)
class _Property:
    name: str
    node: _Node
    required: bool


@dataclass(frozen=True)
class _Node:
    kind: str
    nullable: bool = False
    properties: tuple[_Property, ...] = ()
    additional_properties: bool = True
    items: _Node | None = None
    enum_values: tuple[Any, ...] | None = None

    @property
    def duckdb_type(self) -> str:
        if self.kind == "string":
            return "VARCHAR"
        if self.kind == "integer":
            return "BIGINT"
        if self.kind == "number":
            return "DOUBLE"
        if self.kind == "boolean":
            return "BOOLEAN"
        if self.kind == "array":
            assert self.items is not None
            return f"{self.items.duckdb_type}[]"
        assert self.kind == "object"
        fields = ", ".join(f'"{prop.name}" {prop.node.duckdb_type}' for prop in self.properties)
        return f"STRUCT({fields})"

    @property
    def arrow_type(self) -> pa.DataType:
        if self.kind == "string":
            return pa.string()
        if self.kind == "integer":
            return pa.int64()
        if self.kind == "number":
            return pa.float64()
        if self.kind == "boolean":
            return pa.bool_()
        if self.kind == "array":
            assert self.items is not None
            return pa.list_(pa.field("item", self.items.arrow_type, nullable=self.items.nullable))
        assert self.kind == "object"
        return pa.struct(
            [
                pa.field(prop.name, prop.node.arrow_type, nullable=not prop.required or prop.node.nullable)
                for prop in self.properties
            ]
        )

    def iter_objects(self) -> list[_Node]:
        objects: list[_Node] = []
        if self.kind == "object":
            objects.append(self)
            for prop in self.properties:
                objects.extend(prop.node.iter_objects())
        elif self.kind == "array":
            assert self.items is not None
            objects.extend(self.items.iter_objects())
        return objects

    def validate(self, value: Any, path: str = "$") -> Any:
        if value is None:
            if not self.nullable:
                raise _output_error(path, "must not be NULL")
            if self.enum_values is not None and None not in self.enum_values:
                raise _output_error(path, f"must be one of {list(self.enum_values)!r}")
            return None

        if self.kind == "string":
            if not isinstance(value, str):
                raise _output_error(path, f"must be a string, got {type(value).__name__}")
        elif self.kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise _output_error(path, f"must be an integer, got {type(value).__name__}")
            if not _BIGINT_MIN <= value <= _BIGINT_MAX:
                raise _output_error(path, "is outside the signed BIGINT range")
        elif self.kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _output_error(path, f"must be a number, got {type(value).__name__}")
            try:
                finite = math.isfinite(float(value))
            except OverflowError:
                finite = False
            if not finite:
                raise _output_error(path, "must be finite")
        elif self.kind == "boolean":
            if not isinstance(value, bool):
                raise _output_error(path, f"must be a boolean, got {type(value).__name__}")
        elif self.kind == "array":
            if not isinstance(value, list):
                raise _output_error(path, f"must be an array, got {type(value).__name__}")
            assert self.items is not None
            value = [self.items.validate(item, f"{path}[{index}]") for index, item in enumerate(value)]
        else:
            assert self.kind == "object"
            if not isinstance(value, dict):
                raise _output_error(path, f"must be an object, got {type(value).__name__}")
            known = {prop.name: prop for prop in self.properties}
            missing = [prop.name for prop in self.properties if prop.required and prop.name not in value]
            if missing:
                raise _output_error(path, f"is missing required properties: {', '.join(missing)}")
            extras = [key for key in value if key not in known]
            if extras and not self.additional_properties:
                raise _output_error(path, f"contains unexpected properties: {', '.join(map(str, extras))}")
            for key in value:
                if not isinstance(key, str):
                    raise _output_error(path, "must use string property names")
            validated: dict[str, Any] = {}
            for prop in self.properties:
                if prop.name in value:
                    validated[prop.name] = prop.node.validate(value[prop.name], f"{path}.{prop.name}")
                elif not prop.required:
                    validated[prop.name] = None
            value = validated

        if self.enum_values is not None and value not in self.enum_values:
            raise _output_error(path, f"must be one of {list(self.enum_values)!r}")
        return value


@dataclass(frozen=True)
class StructuredOutputSpec:
    """One validated schema and all types derived from it."""

    schema: dict[str, Any]
    root: _Node
    name: str

    @property
    def duckdb_type(self) -> str:
        return self.root.duckdb_type

    @property
    def arrow_type(self) -> pa.DataType:
        return self.root.arrow_type

    def validate_json(self, value: Any) -> str:
        if isinstance(value, str):
            try:
                decoded = _strict_json_loads(value)
            except (TypeError, ValueError):
                raise OutputValidationError("Prompt structured output is not valid JSON") from None
        elif isinstance(value, Mapping):
            decoded = dict(value)
        else:
            raise OutputValidationError(
                f"Prompt structured output must be a JSON string or object, got {type(value).__name__}"
            )
        validated = self.root.validate(decoded)
        try:
            return json.dumps(validated, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError):
            raise OutputValidationError("Prompt structured output is not strict JSON") from None

    def validate_openai_gpt_contract(self, model: str) -> None:
        """Enforce object rules for known OpenAI Structured Outputs models."""
        if not _uses_openai_strict_structured_outputs(model):
            return
        for node in self.root.iter_objects():
            names = {prop.name for prop in node.properties}
            required = {prop.name for prop in node.properties if prop.required}
            if node.additional_properties:
                raise SchemaValidationError(
                    "OpenAI structured output requires additionalProperties=false on every object"
                )
            if required != names:
                missing = ", ".join(sorted(names - required))
                raise SchemaValidationError(
                    "OpenAI structured output requires every property to be required"
                    + (f"; missing: {missing}" if missing else "")
                )


class _Compiler:
    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema
        self.definitions: dict[str, Mapping[str, Any]] = {}
        for keyword in ("$defs", "definitions"):
            definitions = schema.get(keyword, {})
            if not isinstance(definitions, Mapping):
                raise _schema_error(f"$.{keyword}", "must be an object")
            for name, definition in definitions.items():
                if not isinstance(name, str) or not isinstance(definition, Mapping):
                    raise _schema_error(f"$.{keyword}", "must map string names to schema objects")
                self.definitions[f"#/{keyword}/{_pointer_escape(name)}"] = definition

    def compile(self) -> _Node:
        root = self._compile_node(self.schema, "$", ())
        if root.kind != "object" or root.nullable:
            raise _schema_error("$", "root type must be a non-nullable object")
        for reference in self.definitions:
            self._compile_reference(reference, f"definition {reference}", ())
        return root

    def _compile_reference(self, reference: str, path: str, stack: tuple[str, ...]) -> _Node:
        if reference not in self.definitions:
            raise _schema_error(path, f"unresolved local $ref {reference!r}")
        if reference in stack:
            raise _schema_error(path, f"recursive $ref {reference!r} is not supported")
        return self._compile_node(self.definitions[reference], path, (*stack, reference))

    def _compile_node(self, raw: Any, path: str, stack: tuple[str, ...]) -> _Node:
        if not isinstance(raw, Mapping):
            raise _schema_error(path, "must be an object")
        schema = dict(raw)
        unsupported = sorted(set(schema) & _UNSUPPORTED_CONSTRAINTS)
        if unsupported:
            raise _schema_error(path, f"unsupported constraint(s): {', '.join(unsupported)}")
        unknown = sorted(set(schema) - _SCHEMA_KEYS - _ANNOTATION_KEYS)
        if unknown:
            raise _schema_error(path, f"unsupported keyword(s): {', '.join(unknown)}")
        if "allOf" in schema:
            raise _schema_error(path, "allOf is not supported")
        if "$defs" in schema and path != "$":
            raise _schema_error(path, "nested $defs is not supported")
        if "definitions" in schema and path != "$":
            raise _schema_error(path, "nested definitions is not supported")

        if "$ref" in schema:
            allowed = {"$ref"} | _ANNOTATION_KEYS
            siblings = sorted(set(schema) - allowed)
            if siblings:
                raise _schema_error(path, f"$ref cannot have schema siblings: {', '.join(siblings)}")
            reference = schema["$ref"]
            if not isinstance(reference, str) or not reference.startswith("#/"):
                raise _schema_error(path, "only local $defs/definitions $ref values are supported")
            return self._compile_reference(reference, path, stack)

        combinators = [key for key in ("anyOf", "oneOf") if key in schema]
        if combinators:
            if len(combinators) != 1:
                raise _schema_error(path, "only one nullable anyOf/oneOf may be used")
            keyword = combinators[0]
            allowed = {keyword} | _ANNOTATION_KEYS
            siblings = sorted(set(schema) - allowed)
            if siblings:
                raise _schema_error(path, f"nullable {keyword} cannot have schema siblings: {', '.join(siblings)}")
            branches = schema[keyword]
            if not isinstance(branches, list) or len(branches) != 2:
                raise _schema_error(path, f"{keyword} is supported only as exactly T | null")
            null_indices = [index for index, branch in enumerate(branches) if _is_null_schema(branch)]
            if len(null_indices) != 1:
                raise _schema_error(path, f"{keyword} is supported only as exactly T | null")
            node = self._compile_node(branches[1 - null_indices[0]], path, stack)
            enum_values = node.enum_values
            if enum_values is not None and None not in enum_values:
                enum_values = (*enum_values, None)
            return replace(node, nullable=True, enum_values=enum_values)

        schema_type = schema.get("type")
        nullable = False
        if isinstance(schema_type, list):
            if len(schema_type) != 2 or schema_type.count("null") != 1:
                raise _schema_error(path, "type arrays are supported only as exactly T | null")
            non_null = [item for item in schema_type if item != "null"]
            if len(non_null) != 1 or non_null[0] not in _SUPPORTED_TYPES:
                raise _schema_error(path, "type arrays are supported only as exactly T | null")
            schema_type = non_null[0]
            nullable = True
        if schema_type not in _SUPPORTED_TYPES:
            raise _schema_error(path, f"type must be one of {', '.join(sorted(_SUPPORTED_TYPES))}")

        structural_keys = {"properties", "required", "additionalProperties", "items"}
        allowed_structural = (
            {"properties", "required", "additionalProperties"}
            if schema_type == "object"
            else {"items"}
            if schema_type == "array"
            else set()
        )
        misplaced = sorted((set(schema) & structural_keys) - allowed_structural)
        if misplaced:
            raise _schema_error(path, f"keyword(s) invalid for type {schema_type}: {', '.join(misplaced)}")

        enum_values = self._compile_enum(schema, schema_type, path, nullable)
        if schema_type == "object":
            node = self._compile_object(schema, path, stack)
        elif schema_type == "array":
            if "items" not in schema:
                raise _schema_error(f"{path}.items", "array schemas must define items")
            node = _Node(kind="array", items=self._compile_node(schema["items"], f"{path}.items", stack))
        else:
            node = _Node(kind=schema_type)
        return replace(node, nullable=nullable, enum_values=enum_values)

    def _compile_object(self, schema: dict[str, Any], path: str, stack: tuple[str, ...]) -> _Node:
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            raise _schema_error(f"{path}.properties", "object schemas must define a properties object")
        for name in properties:
            if not isinstance(name, str) or _PROPERTY_NAME.fullmatch(name) is None:
                raise _schema_error(
                    f"{path}.properties", "property names must match [A-Za-z0-9_-] and contain 1-64 characters"
                )
        if not properties:
            raise _schema_error(f"{path}.properties", "object schemas must define at least one property")
        seen: dict[str, str] = {}
        for name in properties:
            folded = name.casefold()
            if folded in seen:
                raise _schema_error(
                    f"{path}.properties",
                    f"property names differ only by case, which DuckDB treats as the same STRUCT member: "
                    f"{seen[folded]!r} and {name!r}",
                )
            seen[folded] = name
        required_raw = schema.get("required", [])
        if not isinstance(required_raw, list) or any(not isinstance(name, str) for name in required_raw):
            raise _schema_error(f"{path}.required", "must be a string array")
        if len(required_raw) != len(set(required_raw)):
            raise _schema_error(f"{path}.required", "must not contain duplicate property names")
        unknown_required = sorted(set(required_raw) - set(properties))
        if unknown_required:
            raise _schema_error(
                f"{path}.required", f"contains names absent from properties: {', '.join(unknown_required)}"
            )
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, bool):
            raise _schema_error(f"{path}.additionalProperties", "must be a boolean")
        required = set(required_raw)
        compiled = tuple(
            _Property(
                name=name,
                node=self._compile_node(value, f"{path}.properties.{name}", stack),
                required=name in required,
            )
            for name, value in properties.items()
        )
        return _Node(kind="object", properties=compiled, additional_properties=additional)

    def _compile_enum(
        self,
        schema: dict[str, Any],
        schema_type: str,
        path: str,
        nullable: bool,
    ) -> tuple[Any, ...] | None:
        if "enum" not in schema:
            return None
        if schema_type not in _SCALAR_TYPES:
            raise _schema_error(f"{path}.enum", "enum is supported only for primitive scalar types")
        values = schema["enum"]
        if not isinstance(values, list) or not values:
            raise _schema_error(f"{path}.enum", "must be a non-empty array")
        if len({json.dumps(value, sort_keys=True) for value in values}) != len(values):
            raise _schema_error(f"{path}.enum", "must not contain duplicate values")
        non_null_values = [value for value in values if value is not None]
        if len(non_null_values) != len(values) and not nullable:
            raise _schema_error(f"{path}.enum", "NULL requires an explicit nullable T | null schema")
        probe = _Node(kind=schema_type)
        for index, value in enumerate(non_null_values):
            try:
                probe.validate(value, f"{path}.enum[{index}]")
            except OutputValidationError as exc:
                raise _schema_error(f"{path}.enum[{index}]", str(exc)) from exc
        return tuple(values)


def _is_null_schema(value: Any) -> bool:
    if not isinstance(value, Mapping) or value.get("type") != "null":
        return False
    return not (set(value) - ({"type"} | _ANNOTATION_KEYS))


def _pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _copy_json_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, type):
        try:
            from pydantic import BaseModel
        except ImportError:
            raise TypeError("return_format requires pydantic when passing a Pydantic BaseModel class") from None
        if not issubclass(value, BaseModel):
            raise TypeError("return_format class must be a Pydantic BaseModel subclass")
        value = value.model_json_schema()
    elif not isinstance(value, Mapping):
        raise TypeError("return_format must be a Pydantic BaseModel class, a JSON schema mapping, or None")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("Prompt return_format must be strict JSON-compatible data") from exc
    if not isinstance(copied, dict):
        raise SchemaValidationError("Prompt return_format must be a JSON schema object")
    return copied


def _canonicalize_legacy_definitions(schema: dict[str, Any]) -> None:
    legacy = schema.pop("definitions", None)
    if legacy is None:
        return
    if not isinstance(legacy, Mapping):
        raise _schema_error("$.definitions", "must be an object")
    current = schema.get("$defs", {})
    if not isinstance(current, Mapping):
        raise _schema_error("$.$defs", "must be an object")

    definitions = dict(current)
    for name, definition in legacy.items():
        if name in definitions and definitions[name] != definition:
            raise _schema_error("$", f"conflicting $defs/definitions entry {name!r}")
        definitions[name] = definition
    schema["$defs"] = definitions

    def rewrite_refs(value: Any) -> None:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/definitions/"):
                value["$ref"] = "#/$defs/" + reference.removeprefix("#/definitions/")
            for item in value.values():
                rewrite_refs(item)
        elif isinstance(value, list):
            for item in value:
                rewrite_refs(item)

    rewrite_refs(schema)


def compile_return_format(value: Any) -> StructuredOutputSpec | None:
    """Validate and compile a public Prompt ``return_format`` value."""
    if value is None:
        return None
    schema = _copy_json_schema(value)
    _canonicalize_legacy_definitions(schema)
    root = _Compiler(schema).compile()
    title = schema.get("title")
    name = title if isinstance(title, str) and _SCHEMA_NAME.fullmatch(title) else "vane_response"
    return StructuredOutputSpec(schema=schema, root=root, name=name)


def serialize_raw_response(response: Any, *, exclude: set[str] | None = None) -> str:
    """Serialize an official SDK response body without transport metadata."""
    excluded = exclude or set()
    if isinstance(response, Mapping):
        body: Any = {key: value for key, value in response.items() if key not in excluded}
    elif isinstance(response, list):
        body = response
    else:
        model_dump = getattr(response, "model_dump", None)
        if not callable(model_dump):
            raise RawResponseSerializationError(
                f"Provider raw response body must support model_dump(), got {type(response).__name__}"
            )
        try:
            body = model_dump(mode="json", by_alias=True, exclude_unset=True, exclude=excluded)
        except Exception:
            raise RawResponseSerializationError("Provider raw response body model_dump() failed") from None
    try:
        return json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        raise RawResponseSerializationError("Provider raw response body is not strict JSON") from None


def validate_raw_response_json(value: Any) -> str:
    """Validate the public raw-response VARCHAR contract without rewriting it."""
    if not isinstance(value, str):
        raise RawResponseSerializationError(f"Prompt raw response must be a JSON string, got {type(value).__name__}")
    try:
        _strict_json_loads(value)
    except (TypeError, ValueError):
        raise RawResponseSerializationError("Prompt raw response is not valid JSON") from None
    return value
