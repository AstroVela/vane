# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed, whole-document Turbopuffer upserts.

The adapter intentionally exposes a narrow contract. Document IDs are explicit
Arrow ``uint64`` or UTF-8 strings, the vector is a fixed-size list of
``float32``, and every stored non-vector attribute is selected through an
explicit mapping. Successful worker batches are durable Turbopuffer writes,
but batches are independent and Vane does not provide rollback or exactly-once
delivery.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

from vane.datasink import (
    BoundKeyedUpsertSink,
    DataSink,
    DataSinkExecutionOptions,
    DataSinkWorker,
    EnvironmentSecret,
    WriteContext,
    WriteResult,
)

if TYPE_CHECKING:
    from vane import DuckDBPyRelation


_MAX_INT64 = (1 << 63) - 1
_MAX_UINT64 = (1 << 64) - 1
_MAX_ID_BYTES = 64
_MAX_ATTRIBUTE_NAME_BYTES = 128
_MAX_ATTRIBUTE_NAMES = 1_024
_MAX_VECTOR_DIMENSIONS = 10_752
_MAX_WRITE_REQUEST_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_BATCH_ROWS = 1_000
_DEFAULT_MAX_BATCH_BYTES = 16 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 30.0
_NAMESPACE_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_REGION_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_DISTANCE_METRICS = frozenset({"cosine_distance", "euclidean_squared"})
_PARTIAL_VISIBILITY_WARNING = (
    "Turbopuffer applies worker batches independently; a later operation failure can leave this "
    "whole-document overwrite visible"
)


def _name(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if value != value.strip() or "\x00" in value:
        raise ValueError(f"{name} must not contain surrounding whitespace or NUL characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid UTF-8") from error
    return value


def _namespace(value: object) -> str:
    normalized = _name("namespace", value)
    if _NAMESPACE_PATTERN.fullmatch(normalized) is None:
        raise ValueError("namespace must match [A-Za-z0-9-_.]{1,128}")
    return normalized


def _region(value: object) -> str:
    normalized = _name("region", value)
    if _REGION_PATTERN.fullmatch(normalized) is None:
        raise ValueError("region must be a lowercase DNS label of at most 63 characters")
    return normalized


def _attribute_name(value: object) -> str:
    normalized = _name("attribute_mapping target", value)
    if len(normalized.encode("utf-8")) > _MAX_ATTRIBUTE_NAME_BYTES:
        raise ValueError("attribute_mapping targets must be at most 128 UTF-8 bytes")
    if normalized.startswith("$"):
        raise ValueError("attribute_mapping targets must not start with the reserved '$' prefix")
    if normalized in {"id", "vector"}:
        raise ValueError(f"attribute_mapping target {normalized!r} is reserved")
    return normalized


def _positive_int(name: str, value: object, *, maximum: int = _MAX_INT64) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0 or value > _MAX_INT64:
        raise ValueError(f"{name} must be a non-negative signed 64-bit integer")
    return value


def _positive_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return normalized


def _attribute_mapping(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise TypeError("attribute_mapping must be a mapping")
    normalized: list[tuple[str, str]] = []
    for source, target in value.items():
        normalized.append((_name("attribute_mapping source", source), _attribute_name(target)))
    sources = [source.casefold() for source, _ in normalized]
    targets = [target for _, target in normalized]
    if len(set(sources)) != len(sources):
        raise ValueError("attribute_mapping sources must be unique ignoring case")
    if len(set(targets)) != len(targets):
        raise ValueError("attribute_mapping targets must be unique")
    if len(normalized) + 2 > _MAX_ATTRIBUTE_NAMES:
        raise ValueError("attribute_mapping exceeds Turbopuffer's 1024 attribute-name limit")
    return tuple(normalized)


class _IDKind(str, Enum):
    UINT = "uint"
    STRING = "string"


class _ScalarKind(str, Enum):
    BOOL = "bool"
    INT = "int"
    UINT = "uint"
    FLOAT = "float"
    STRING = "string"


@dataclass(frozen=True)
class _SourceField:
    name: str
    data_type: pa.DataType


@dataclass(frozen=True)
class _VectorField(_SourceField):
    dimension: int


@dataclass(frozen=True)
class _AttributeField(_SourceField):
    target_name: str
    scalar_kind: _ScalarKind
    is_array: bool
    schema_type: str


def _id_kind(field: pa.Field) -> _IDKind:
    if pa.types.is_uint64(field.type):
        return _IDKind.UINT
    if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
        return _IDKind.STRING
    raise ValueError("Turbopuffer ID source must be Arrow uint64 or string")


def _vector_dimension(field: pa.Field) -> int:
    data_type = field.type
    if not pa.types.is_fixed_size_list(data_type) or not pa.types.is_float32(data_type.value_type):
        raise ValueError("Turbopuffer vector source must be a fixed-size list of Arrow float32")
    dimension = data_type.list_size
    if dimension <= 0 or dimension > _MAX_VECTOR_DIMENSIONS:
        raise ValueError(f"Turbopuffer vector dimension must be between 1 and {_MAX_VECTOR_DIMENSIONS}")
    return dimension


def _scalar_kind(data_type: pa.DataType) -> _ScalarKind:
    if pa.types.is_boolean(data_type):
        return _ScalarKind.BOOL
    if pa.types.is_signed_integer(data_type):
        return _ScalarKind.INT
    if pa.types.is_unsigned_integer(data_type):
        return _ScalarKind.UINT
    if pa.types.is_float32(data_type) or pa.types.is_float64(data_type):
        return _ScalarKind.FLOAT
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return _ScalarKind.STRING
    raise ValueError(f"unsupported Arrow attribute type {data_type}")


def _attribute_field(source: pa.Field, target: str) -> _AttributeField:
    data_type = source.type
    is_array = (
        pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type)
    )
    value_type = data_type.value_type if is_array else data_type
    try:
        kind = _scalar_kind(value_type)
    except ValueError as error:
        raise ValueError(
            f"TurbopufferSink does not support Arrow attribute {source.name!r} with type {data_type}"
        ) from error
    schema_type = f"[]{kind.value}" if is_array else kind.value
    return _AttributeField(source.name, data_type, target, kind, is_array, schema_type)


def _resolve_field(schema: pa.Schema, requested: str, role: str) -> pa.Field:
    matches = [field for field in schema if field.name.casefold() == requested.casefold()]
    if len(matches) != 1:
        raise ValueError(f"{role} {requested!r} must match exactly one input column")
    return matches[0]


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _load_turbopuffer_sdk() -> tuple[type[Any], type[Any]]:
    try:
        from turbopuffer import Turbopuffer  # type: ignore[import-not-found, import-untyped, unused-ignore]
        from turbopuffer.types import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
            NamespaceWriteResponse,
        )
    except ModuleNotFoundError as error:
        if error.name != "turbopuffer":
            raise
        raise ImportError("TurbopufferSink requires the Turbopuffer SDK; install vane-ai[turbopuffer]") from error
    return Turbopuffer, NamespaceWriteResponse


def _compact_json_size(value: object, maximum: int) -> int:
    """Return current-SDK compact JSON bytes without materializing one large body."""

    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    total = 0
    try:
        for chunk in encoder.iterencode(value):
            total += len(chunk.encode("utf-8"))
            if total > maximum:
                raise ValueError("Turbopuffer write request exceeds max_request_bytes")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error) == "Turbopuffer write request exceeds max_request_bytes":
            raise
        raise ValueError("Turbopuffer write request contains a value that cannot be encoded as JSON") from error
    return total


class TurbopufferSink(DataSink):
    """Write relation batches to one Turbopuffer namespace.

    ``id_column`` and ``vector_column`` identify relation columns that become
    Turbopuffer's reserved ``id`` and ``vector`` fields. ``attribute_mapping``
    explicitly selects all other stored columns and maps source names to target
    attribute names. Unmapped relation columns are not written.

    The API key reference is resolved only in a worker. Both SDK retries and
    Vane framework retries default to disabled; setting ``max_retries`` enables
    full-input Vane replay with the same operation ID.
    """

    def __init__(
        self,
        namespace: str,
        *,
        region: str,
        api_key: EnvironmentSecret,
        id_column: str,
        vector_column: str,
        distance_metric: str,
        attribute_mapping: Mapping[str, str],
        worker_count: int = 1,
        max_batch_rows: int = _DEFAULT_MAX_BATCH_ROWS,
        max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
        max_request_bytes: int = _MAX_WRITE_REQUEST_BYTES,
        max_retries: int = 0,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.namespace = _namespace(namespace)
        self.region = _region(region)
        self.id_column = _name("id_column", id_column)
        self.vector_column = _name("vector_column", vector_column)
        if self.id_column.casefold() == self.vector_column.casefold():
            raise ValueError("id_column and vector_column must name different input columns")
        if not isinstance(api_key, EnvironmentSecret):
            raise TypeError("api_key must be an EnvironmentSecret")
        normalized_metric = _name("distance_metric", distance_metric)
        if normalized_metric not in _DISTANCE_METRICS:
            raise ValueError("distance_metric must be 'cosine_distance' or 'euclidean_squared'")
        mapping = _attribute_mapping(attribute_mapping)
        reserved_sources = {self.id_column.casefold(), self.vector_column.casefold()}
        if any(source.casefold() in reserved_sources for source, _ in mapping):
            raise ValueError("attribute_mapping sources must not reuse id_column or vector_column")

        self.distance_metric = normalized_metric
        self.worker_count = _positive_int("worker_count", worker_count)
        self.max_batch_rows = _positive_int("max_batch_rows", max_batch_rows)
        self.max_request_bytes = _positive_int("max_request_bytes", max_request_bytes, maximum=_MAX_WRITE_REQUEST_BYTES)
        self.max_batch_bytes = _positive_int("max_batch_bytes", max_batch_bytes)
        if self.max_batch_bytes > self.max_request_bytes:
            raise ValueError("max_batch_bytes must not exceed max_request_bytes")
        self.max_retries = _non_negative_int("max_retries", max_retries)
        self.timeout = _positive_number("timeout", timeout)
        self._api_key = api_key
        self._attribute_mapping = mapping

    def bind(self, schema: pa.Schema) -> BoundKeyedUpsertSink:
        if not isinstance(schema, pa.Schema):
            raise TypeError("schema must be pyarrow.Schema")
        folded_names = [name.casefold() for name in schema.names]
        if len(set(folded_names)) != len(folded_names):
            raise ValueError("TurbopufferSink requires input column names to be unique ignoring case")

        id_field = _resolve_field(schema, self.id_column, "id_column")
        vector_field = _resolve_field(schema, self.vector_column, "vector_column")
        id_kind = _id_kind(id_field)
        dimension = _vector_dimension(vector_field)

        attributes: list[_AttributeField] = []
        for source_name, target_name in self._attribute_mapping:
            source = _resolve_field(schema, source_name, "attribute_mapping source")
            attributes.append(_attribute_field(source, target_name))

        selected = [id_field.name, vector_field.name, *(field.name for field in attributes)]
        if len({name.casefold() for name in selected}) != len(selected):
            raise ValueError("id, vector, and attribute source columns must be distinct")
        return _BoundTurbopufferSink(
            self,
            _SourceField(id_field.name, id_field.type),
            id_kind,
            _VectorField(vector_field.name, vector_field.type, dimension),
            tuple(attributes),
        )


class _BoundTurbopufferSink(BoundKeyedUpsertSink):
    def __init__(
        self,
        sink: TurbopufferSink,
        id_field: _SourceField,
        id_kind: _IDKind,
        vector_field: _VectorField,
        attributes: tuple[_AttributeField, ...],
    ) -> None:
        self._sink = sink
        self._id_field = id_field
        self._id_kind = id_kind
        self._vector_field = vector_field
        self._attributes = attributes

    @property
    def execution_options(self) -> DataSinkExecutionOptions:
        return DataSinkExecutionOptions(
            worker_count=self._sink.worker_count,
            batch_size=self._sink.max_batch_rows,
            target_max_batch_bytes=self._sink.max_batch_bytes,
            max_retries=self._sink.max_retries,
        )

    @property
    def key_columns(self) -> Sequence[str]:
        return (self._id_field.name,)

    def prepare_input(self, relation: DuckDBPyRelation) -> DuckDBPyRelation:
        selected = (self._id_field, self._vector_field, *self._attributes)
        return relation.project(", ".join(_quote_identifier(field.name) for field in selected))

    def open_worker(self, _context: WriteContext) -> DataSinkWorker:
        return _TurbopufferWorker(
            self._sink,
            self._id_field,
            self._id_kind,
            self._vector_field,
            self._attributes,
        )


class _TurbopufferWorker(DataSinkWorker):
    def __init__(
        self,
        sink: TurbopufferSink,
        id_field: _SourceField,
        id_kind: _IDKind,
        vector_field: _VectorField,
        attributes: tuple[_AttributeField, ...],
    ) -> None:
        client_type, response_type = _load_turbopuffer_sdk()
        api_key = sink._api_key.resolve()
        if not api_key or api_key != api_key.strip() or "\x00" in api_key:
            raise RuntimeError("Turbopuffer api_key environment secret must be a non-empty credential")
        self._sink = sink
        self._id_field = id_field
        self._id_kind = id_kind
        self._vector_field = vector_field
        self._attributes = attributes
        self._response_type = response_type
        self._client: Any | None = None
        self._namespace_resource: Any | None = None
        self._warning_pending = True
        self._schema = {
            "id": id_kind.value,
            "vector": {"type": f"[{vector_field.dimension}]f32", "ann": True},
            **{field.target_name: field.schema_type for field in attributes},
        }
        self._base_metadata = {
            "provider": "turbopuffer",
            "namespace": sink.namespace,
            "write_mode": "overwrite",
        }
        try:
            self._client = client_type(
                api_key=api_key,
                region=sink.region,
                base_url="https://{region}.turbopuffer.com",
                timeout=sink.timeout,
                max_retries=0,
                compression=False,
            )
            self._namespace_resource = self._client.namespace(sink.namespace)
            WriteResult(
                rows_received=0,
                rows_affected=0,
                metadata={**self._base_metadata, "request_bytes": 0},
                warnings=(_PARTIAL_VISIBILITY_WARNING,),
            )
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    try:
                        add_note(f"Turbopuffer client cleanup also failed: {type(close_error).__name__}")
                    except BaseException:
                        pass
            raise

    def _namespace_or_raise(self) -> Any:
        if self._client is None or self._namespace_resource is None:
            raise RuntimeError("TurbopufferSink worker is closed")
        return self._namespace_resource

    def _validate_schema(self, table: pa.Table) -> None:
        if not isinstance(table, pa.Table):
            raise TypeError(f"TurbopufferSink expected pyarrow.Table, got {type(table).__name__}")
        expected = (self._id_field, self._vector_field, *self._attributes)
        if len(table.schema) != len(expected):
            raise ValueError("Turbopuffer batch schema does not match the bound input schema")
        for index, binding in enumerate(expected):
            field = table.schema.field(index)
            if field.name != binding.name or field.type != binding.data_type:
                raise ValueError("Turbopuffer batch schema does not match the bound input schema")

    def _ids(self, table: pa.Table) -> list[int | str]:
        raw_values = table.column(self._id_field.name).to_pylist()
        values: list[int | str] = []
        for value in raw_values:
            if value is None:
                raise ValueError("Turbopuffer document IDs must not be null")
            if self._id_kind is _IDKind.UINT:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > _MAX_UINT64:
                    raise ValueError("Turbopuffer uint document ID is outside the unsigned 64-bit range")
            else:
                if not isinstance(value, str):
                    raise ValueError("Turbopuffer string document ID contains an invalid value")
                try:
                    encoded = value.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise ValueError("Turbopuffer string document ID contains invalid UTF-8") from error
                if len(encoded) > _MAX_ID_BYTES:
                    raise ValueError("Turbopuffer string document ID exceeds 64 UTF-8 bytes")
            values.append(value)
        if len(set(values)) != len(values):
            raise ValueError("Turbopuffer batch contains duplicate document IDs")
        return values

    def _vectors(self, table: pa.Table) -> list[list[float]]:
        values = table.column(self._vector_field.name).to_pylist()
        for vector in values:
            if not isinstance(vector, list) or len(vector) != self._vector_field.dimension:
                raise ValueError("Turbopuffer vector contains a null or invalid dimension")
            if any(
                isinstance(element, bool) or not isinstance(element, (int, float)) or not math.isfinite(float(element))
                for element in vector
            ):
                raise ValueError("Turbopuffer vector contains a null or non-finite value")
        return values

    @staticmethod
    def _validate_scalar(field: _AttributeField, value: object) -> None:
        if field.scalar_kind is _ScalarKind.BOOL:
            valid = isinstance(value, bool)
        elif field.scalar_kind is _ScalarKind.INT:
            valid = isinstance(value, int) and not isinstance(value, bool) and -(1 << 63) <= value <= _MAX_INT64
        elif field.scalar_kind is _ScalarKind.UINT:
            valid = isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_UINT64
        elif field.scalar_kind is _ScalarKind.FLOAT:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        else:
            if isinstance(value, str):
                valid = True
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError:
                    valid = False
            else:
                valid = False
        if not valid:
            raise ValueError(f"Turbopuffer attribute {field.target_name!r} contains an invalid value")

    def _attribute_values(self, table: pa.Table, field: _AttributeField) -> list[object]:
        values = table.column(field.name).to_pylist()
        for value in values:
            if value is None:
                continue
            if field.is_array:
                if not isinstance(value, list) or any(element is None for element in value):
                    raise ValueError(f"Turbopuffer array attribute {field.target_name!r} contains an invalid value")
                for element in value:
                    self._validate_scalar(field, element)
            else:
                self._validate_scalar(field, value)
        return values

    def _request(self, table: pa.Table) -> tuple[dict[str, Sequence[object]], int, int, int]:
        self._validate_schema(table)
        row_count = table.num_rows
        batch_bytes = table.nbytes
        if row_count > self._sink.max_batch_rows:
            raise ValueError("Turbopuffer batch exceeds max_batch_rows")
        if batch_bytes > self._sink.max_batch_bytes:
            raise ValueError("Turbopuffer batch exceeds max_batch_bytes")

        columns: dict[str, Sequence[object]] = {
            "id": self._ids(table),
            "vector": self._vectors(table),
        }
        for field in self._attributes:
            columns[field.target_name] = self._attribute_values(table, field)
        request = {
            "distance_metric": self._sink.distance_metric,
            "schema": self._schema,
            "upsert_columns": columns,
        }
        request_bytes = _compact_json_size(request, self._sink.max_request_bytes)
        return columns, row_count, batch_bytes, request_bytes

    def write(self, table: pa.Table) -> WriteResult:
        columns, row_count, batch_bytes, request_bytes = self._request(table)
        metadata = {**self._base_metadata, "request_bytes": request_bytes}
        if row_count == 0:
            return WriteResult(
                rows_received=0,
                rows_affected=0,
                bytes_received=batch_bytes,
                metadata=metadata,
            )
        warnings = (_PARTIAL_VISIBILITY_WARNING,) if self._warning_pending else ()
        result = WriteResult(
            rows_received=row_count,
            rows_affected=row_count,
            bytes_received=batch_bytes,
            metadata=metadata,
            warnings=warnings,
        )
        response = self._namespace_or_raise().write(
            upsert_columns=columns,
            distance_metric=self._sink.distance_metric,
            schema=self._schema,
            timeout=self._sink.timeout,
        )
        if not isinstance(response, self._response_type) or response.status != "OK":
            raise RuntimeError("Turbopuffer write returned an invalid response")
        affected = response.rows_affected
        upserted = response.rows_upserted
        if (
            isinstance(affected, bool)
            or not isinstance(affected, int)
            or affected != row_count
            or isinstance(upserted, bool)
            or not isinstance(upserted, int)
            or upserted != row_count
        ):
            raise RuntimeError("Turbopuffer write returned an invalid affected-row count")
        self._warning_pending = False
        return result

    def abort(self, _error: BaseException) -> None:
        self.close()

    def close(self) -> None:
        client = self._client
        if client is None:
            return
        client.close()
        self._client = None
        self._namespace_resource = None


__all__ = ["TurbopufferSink"]
