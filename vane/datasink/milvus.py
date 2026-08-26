# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed, full-row Milvus upserts.

The adapter intentionally supports a narrow delivery contract: collections
must use a caller-assigned INT64 or VARCHAR primary key, AutoID and dynamic
fields must be disabled, and collection functions are not supported. Each
successful worker batch has been acknowledged by Milvus, but batches are not a
transaction and read visibility follows the consistency selected by Milvus
readers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

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

_MAX_INT64 = (1 << 63) - 1
_DEFAULT_MAX_BATCH_ROWS = 1_000
_DEFAULT_MAX_BATCH_BYTES = 16 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 30.0
_PARTIAL_VISIBILITY_WARNING = (
    "Milvus applies worker batches independently; a later operation failure can leave this batch applied, "
    "and query visibility follows the reader's Milvus consistency level"
)


def _name(name: str, value: object, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    suffix = " or None" if optional else ""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string{suffix}")
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if value != value.strip() or "\x00" in value:
        raise ValueError(f"{name} must not contain surrounding whitespace or NUL characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid UTF-8") from error
    return value


def _uri(value: object) -> str:
    uri = _name("uri", value)
    assert uri is not None
    try:
        parsed = urlsplit(uri)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise ValueError("uri must be a valid HTTP or HTTPS Milvus endpoint") from error
    if parsed.scheme not in {"http", "https"} or hostname is None:
        raise ValueError("uri must be an absolute HTTP or HTTPS Milvus endpoint")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError(
            "uri must not contain credentials, query parameters, or fragments; use token=EnvironmentSecret(...)"
        )
    if parsed.path not in {"", "/"}:
        raise ValueError("uri must not contain a database path; use database=... instead")
    return uri


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0 or value > _MAX_INT64:
        raise ValueError(f"{name} must be a positive signed 64-bit integer")
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


def _field_mapping(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("field_mapping must be a mapping or None")
    normalized: list[tuple[str, str]] = []
    for source, target in value.items():
        source_name = _name("field_mapping source", source)
        target_name = _name("field_mapping target", target)
        assert source_name is not None and target_name is not None
        normalized.append((source_name, target_name))
    sources = [source for source, _ in normalized]
    targets = [target for _, target in normalized]
    if len(set(sources)) != len(sources):
        raise ValueError("field_mapping sources must be unique")
    if len(set(targets)) != len(targets):
        raise ValueError("field_mapping targets must be unique")
    return tuple(normalized)


class _ArrowKind(str, Enum):
    BOOL = "BOOL"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    STRING = "STRING"
    FLOAT_VECTOR = "FLOAT_VECTOR"


@dataclass(frozen=True)
class _ArrowField:
    source_name: str
    target_name: str
    data_type: pa.DataType
    kind: _ArrowKind
    fixed_dimension: int | None = None


def _arrow_kind(field: pa.Field) -> tuple[_ArrowKind, int | None]:
    data_type = field.type
    if pa.types.is_boolean(data_type):
        return _ArrowKind.BOOL, None
    if pa.types.is_int8(data_type):
        return _ArrowKind.INT8, None
    if pa.types.is_int16(data_type):
        return _ArrowKind.INT16, None
    if pa.types.is_int32(data_type):
        return _ArrowKind.INT32, None
    if pa.types.is_int64(data_type):
        return _ArrowKind.INT64, None
    if pa.types.is_float32(data_type):
        return _ArrowKind.FLOAT, None
    if pa.types.is_float64(data_type):
        return _ArrowKind.DOUBLE, None
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return _ArrowKind.STRING, None
    if (
        pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type)
    ) and pa.types.is_float32(data_type.value_type):
        dimension = data_type.list_size if pa.types.is_fixed_size_list(data_type) else None
        if dimension is not None and dimension <= 0:
            raise ValueError(f"Milvus FLOAT_VECTOR source field {field.name!r} must have a positive dimension")
        return _ArrowKind.FLOAT_VECTOR, dimension
    raise ValueError(f"MilvusSink does not support Arrow field {field.name!r} with type {data_type}")


@dataclass(frozen=True)
class _MilvusTypes:
    bool: int
    int8: int
    int16: int
    int32: int
    int64: int
    float: int
    double: int
    varchar: int
    text: int
    float_vector: int

    @classmethod
    def from_sdk(cls, data_type: Any) -> _MilvusTypes:
        return cls(
            bool=int(data_type.BOOL),
            int8=int(data_type.INT8),
            int16=int(data_type.INT16),
            int32=int(data_type.INT32),
            int64=int(data_type.INT64),
            float=int(data_type.FLOAT),
            double=int(data_type.DOUBLE),
            varchar=int(data_type.VARCHAR),
            text=int(data_type.TEXT),
            float_vector=int(data_type.FLOAT_VECTOR),
        )

    def expected(self, kind: _ArrowKind) -> tuple[int, ...]:
        if kind is _ArrowKind.STRING:
            return (self.varchar, self.text)
        return {
            _ArrowKind.BOOL: (self.bool,),
            _ArrowKind.INT8: (self.int8,),
            _ArrowKind.INT16: (self.int16,),
            _ArrowKind.INT32: (self.int32,),
            _ArrowKind.INT64: (self.int64,),
            _ArrowKind.FLOAT: (self.float,),
            _ArrowKind.DOUBLE: (self.double,),
            _ArrowKind.FLOAT_VECTOR: (self.float_vector,),
        }[kind]

    def label(self, value: int) -> str:
        labels = {
            self.bool: "BOOL",
            self.int8: "INT8",
            self.int16: "INT16",
            self.int32: "INT32",
            self.int64: "INT64",
            self.float: "FLOAT",
            self.double: "DOUBLE",
            self.varchar: "VARCHAR",
            self.text: "TEXT",
            self.float_vector: "FLOAT_VECTOR",
        }
        return labels.get(value, f"type code {value}")


def _load_milvus_sdk() -> tuple[type[Any], Any]:
    try:
        from pymilvus import DataType, MilvusClient  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ModuleNotFoundError as error:
        if error.name != "pymilvus":
            raise
        raise ImportError("MilvusSink requires PyMilvus; install vane-ai[milvus]") from error
    return MilvusClient, DataType


@dataclass(frozen=True)
class _RemoteField:
    data_type: int
    nullable: bool
    has_default: bool
    max_length: int | None
    dimension: int | None

    @property
    def accepts_null(self) -> bool:
        return self.nullable or self.has_default


def _remote_type(field: Mapping[str, Any]) -> int:
    value = field.get("type")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Milvus field {field.get('name')!r} has an invalid type")
    return int(value)


def _field_parameter(field: Mapping[str, Any], parameter: str) -> int:
    params = field.get("params")
    if not isinstance(params, Mapping):
        raise ValueError(f"Milvus field {field.get('name')!r} has invalid parameters")
    value = params.get(parameter)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Milvus field {field.get('name')!r} has an invalid {parameter}")
    return value


class MilvusSink(DataSink):
    """Write relation batches to one Milvus collection with override upserts.

    ``field_mapping`` maps source relation columns to collection fields. The
    primary key names a collection field and must map from exactly one source
    column. ``token`` is resolved from the worker environment and is never
    stored as plaintext in the serialized sink plan.

    Framework retries are disabled unless ``max_retries`` is positive. A retry
    replays the full input using the same Vane operation ID. Replaying a row
    replaces the same Milvus primary key, but Vane does not coordinate with
    concurrent external writers and does not provide exactly-once delivery.
    """

    def __init__(
        self,
        collection_name: str,
        *,
        uri: str,
        primary_key: str,
        token: EnvironmentSecret | None = None,
        database: str | None = None,
        field_mapping: Mapping[str, str] | None = None,
        worker_count: int = 1,
        max_batch_rows: int = _DEFAULT_MAX_BATCH_ROWS,
        max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
        max_retries: int = 0,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        normalized_collection = _name("collection_name", collection_name)
        normalized_primary_key = _name("primary_key", primary_key)
        normalized_database = _name("database", database, optional=True)
        assert normalized_collection is not None and normalized_primary_key is not None
        if token is not None and not isinstance(token, EnvironmentSecret):
            raise TypeError("token must be an EnvironmentSecret or None")
        self.collection_name = normalized_collection
        self.uri = _uri(uri)
        self.primary_key = normalized_primary_key
        self.database = normalized_database
        self.worker_count = _positive_int("worker_count", worker_count)
        self.max_batch_rows = _positive_int("max_batch_rows", max_batch_rows)
        self.max_batch_bytes = _positive_int("max_batch_bytes", max_batch_bytes)
        self.max_retries = _non_negative_int("max_retries", max_retries)
        self.timeout = _positive_number("timeout", timeout)
        self._token = token
        self._field_mapping = _field_mapping(field_mapping)

    def bind(self, schema: pa.Schema) -> BoundKeyedUpsertSink:
        if not isinstance(schema, pa.Schema):
            raise TypeError("schema must be pyarrow.Schema")
        if len(set(schema.names)) != len(schema.names):
            raise ValueError("MilvusSink requires unique input column names")
        mapping = dict(self._field_mapping)
        unknown_sources = set(mapping).difference(schema.names)
        if unknown_sources:
            raise ValueError(f"field_mapping contains unknown input columns: {sorted(unknown_sources)!r}")

        fields: list[_ArrowField] = []
        for field in schema:
            kind, dimension = _arrow_kind(field)
            fields.append(
                _ArrowField(
                    source_name=field.name,
                    target_name=mapping.get(field.name, field.name),
                    data_type=field.type,
                    kind=kind,
                    fixed_dimension=dimension,
                )
            )
        if len({field.target_name for field in fields}) != len(fields):
            raise ValueError("field_mapping must produce unique Milvus field names")
        primary_sources = [field for field in fields if field.target_name == self.primary_key]
        if len(primary_sources) != 1:
            raise ValueError("primary_key must map from exactly one input column")
        if primary_sources[0].kind not in {_ArrowKind.INT64, _ArrowKind.STRING}:
            raise ValueError("Milvus primary key source must be Arrow int64 or string")
        return _BoundMilvusSink(self, tuple(fields), primary_sources[0].source_name)


class _BoundMilvusSink(BoundKeyedUpsertSink):
    def __init__(self, sink: MilvusSink, fields: tuple[_ArrowField, ...], primary_source: str) -> None:
        self._sink = sink
        self._fields = fields
        self._primary_source = primary_source

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
        return (self._primary_source,)

    def open_worker(self, _context: WriteContext) -> DataSinkWorker:
        return _MilvusWorker(self._sink, self._fields)


class _MilvusWorker(DataSinkWorker):
    def __init__(self, sink: MilvusSink, fields: tuple[_ArrowField, ...]) -> None:
        client_type, data_type = _load_milvus_sdk()
        self._types = _MilvusTypes.from_sdk(data_type)
        client_options: dict[str, Any] = {
            "uri": sink.uri,
            "timeout": sink.timeout,
            "dedicated": True,
        }
        if sink._token is not None:
            client_options["token"] = sink._token.resolve()
        if sink.database is not None:
            client_options["db_name"] = sink.database
        self._sink = sink
        self._bindings = fields
        self._client: Any | None = client_type(**client_options)
        self._warning_pending = True
        try:
            self._remote_fields = self._validate_collection()
            self._result_metadata = {
                "provider": "milvus",
                "collection": self._sink.collection_name,
                "write_mode": "override",
            }
            # Validate all static result payloads before the first remote side
            # effect, so an applied upsert cannot be followed by a local result
            # serialization failure.
            WriteResult(
                rows_received=0,
                rows_affected=0,
                metadata=self._result_metadata,
                warnings=(_PARTIAL_VISIBILITY_WARNING,),
            )
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    try:
                        add_note(f"Milvus client cleanup also failed: {type(close_error).__name__}")
                    except BaseException:
                        pass
            raise

    def _client_or_raise(self) -> Any:
        if self._client is None:
            raise RuntimeError("MilvusSink worker is closed")
        return self._client

    def _validate_collection(self) -> Mapping[str, _RemoteField]:
        description = self._client_or_raise().describe_collection(
            collection_name=self._sink.collection_name,
            timeout=self._sink.timeout,
        )
        if not isinstance(description, Mapping):
            raise ValueError("Milvus describe_collection returned an invalid description")
        if description.get("collection_name") != self._sink.collection_name:
            raise ValueError("Milvus describe_collection returned a different collection")
        if description.get("auto_id") is not False:
            raise ValueError("MilvusSink requires AutoID to be disabled")
        if description.get("enable_dynamic_field") is not False:
            raise ValueError("MilvusSink requires dynamic fields to be disabled")
        functions = description.get("functions")
        if not isinstance(functions, list):
            raise ValueError("Milvus describe_collection returned invalid function metadata")
        if functions:
            raise ValueError("MilvusSink does not support collections with functions")
        raw_fields = description.get("fields")
        if not isinstance(raw_fields, list) or not raw_fields:
            raise ValueError("Milvus describe_collection returned no fields")

        fields: dict[str, Mapping[str, Any]] = {}
        for raw_field in raw_fields:
            if not isinstance(raw_field, Mapping):
                raise ValueError("Milvus describe_collection returned an invalid field")
            field_name = raw_field.get("name")
            if not isinstance(field_name, str) or not field_name:
                raise ValueError("Milvus describe_collection returned an invalid field name")
            if field_name in fields:
                raise ValueError(f"Milvus describe_collection returned duplicate field {field_name!r}")
            if raw_field.get("is_function_output") is True:
                raise ValueError("MilvusSink does not support function output fields")
            fields[field_name] = raw_field

        primary_fields = [field for field in fields.values() if field.get("is_primary") is True]
        if len(primary_fields) != 1 or primary_fields[0].get("name") != self._sink.primary_key:
            raise ValueError("Milvus collection primary key does not match MilvusSink.primary_key")
        primary_type = _remote_type(primary_fields[0])
        if primary_type not in {self._types.int64, self._types.varchar}:
            raise ValueError("Milvus collection primary key must be INT64 or VARCHAR")
        if primary_fields[0].get("auto_id") is True:
            raise ValueError("MilvusSink requires AutoID to be disabled")

        target_names = {binding.target_name for binding in self._bindings}
        unknown_targets = target_names.difference(fields)
        if unknown_targets:
            raise ValueError(f"input fields are absent from the Milvus collection: {sorted(unknown_targets)!r}")
        missing_required = [
            name
            for name, field in fields.items()
            if name not in target_names and field.get("nullable") is not True and field.get("default_value") is None
        ]
        if missing_required:
            raise ValueError(f"input is missing required Milvus fields: {sorted(missing_required)!r}")

        validated: dict[str, _RemoteField] = {}
        for binding in self._bindings:
            raw_field = fields[binding.target_name]
            remote_type = _remote_type(raw_field)
            if remote_type not in self._types.expected(binding.kind):
                raise ValueError(
                    f"Milvus field {binding.target_name!r} type {self._types.label(remote_type)} "
                    f"does not match Arrow field {binding.source_name!r} type {binding.data_type}"
                )
            max_length = _field_parameter(raw_field, "max_length") if remote_type == self._types.varchar else None
            dimension = _field_parameter(raw_field, "dim") if remote_type == self._types.float_vector else None
            if binding.fixed_dimension is not None and dimension != binding.fixed_dimension:
                raise ValueError(
                    f"Milvus vector field {binding.target_name!r} dimension {dimension} "
                    f"does not match Arrow fixed dimension {binding.fixed_dimension}"
                )
            validated[binding.target_name] = _RemoteField(
                data_type=remote_type,
                nullable=raw_field.get("nullable") is True,
                has_default=raw_field.get("default_value") is not None,
                max_length=max_length,
                dimension=dimension,
            )
        return validated

    def _records(self, table: pa.Table) -> tuple[list[dict[str, Any]], int, int]:
        if not isinstance(table, pa.Table):
            raise TypeError(f"MilvusSink expected pyarrow.Table, got {type(table).__name__}")
        if len(table.schema) != len(self._bindings):
            raise ValueError("Milvus batch schema does not match the bound input schema")
        for index, binding in enumerate(self._bindings):
            batch_field = table.schema.field(index)
            if batch_field.name != binding.source_name or batch_field.type != binding.data_type:
                raise ValueError("Milvus batch schema does not match the bound input schema")
        row_count = table.num_rows
        batch_bytes = table.nbytes
        if row_count > self._sink.max_batch_rows:
            raise ValueError("Milvus batch exceeds max_batch_rows")
        if batch_bytes > self._sink.max_batch_bytes:
            raise ValueError("Milvus batch exceeds max_batch_bytes")

        records: list[dict[str, Any]] = []
        for row in table.to_pylist():
            record: dict[str, Any] = {}
            for binding in self._bindings:
                value = row[binding.source_name]
                remote = self._remote_fields[binding.target_name]
                if value is None:
                    if binding.target_name == self._sink.primary_key or not remote.accepts_null:
                        raise ValueError(f"Milvus field {binding.target_name!r} does not accept null values")
                elif remote.data_type == self._types.varchar:
                    try:
                        encoded = value.encode("utf-8")
                    except (AttributeError, UnicodeEncodeError) as error:
                        raise ValueError(
                            f"Milvus VARCHAR field {binding.target_name!r} contains invalid UTF-8"
                        ) from error
                    assert remote.max_length is not None
                    if len(encoded) > remote.max_length:
                        raise ValueError(f"Milvus VARCHAR field {binding.target_name!r} exceeds max_length")
                elif remote.data_type == self._types.float_vector:
                    assert remote.dimension is not None
                    if not isinstance(value, list) or len(value) != remote.dimension:
                        raise ValueError(f"Milvus vector field {binding.target_name!r} has an invalid dimension")
                    if any(
                        isinstance(element, bool)
                        or not isinstance(element, (int, float))
                        or not math.isfinite(float(element))
                        for element in value
                    ):
                        raise ValueError(f"Milvus vector field {binding.target_name!r} contains an invalid value")
                record[binding.target_name] = value
            records.append(record)
        return records, row_count, batch_bytes

    def write(self, table: pa.Table) -> WriteResult:
        records, row_count, batch_bytes = self._records(table)
        response = self._client_or_raise().upsert(
            collection_name=self._sink.collection_name,
            data=records,
            timeout=self._sink.timeout,
            partial_update=False,
        )
        count = response.get("upsert_count") if isinstance(response, Mapping) else None
        if isinstance(count, bool) or not isinstance(count, int) or count != row_count:
            raise RuntimeError("Milvus upsert returned an invalid affected-row count")
        warnings = (_PARTIAL_VISIBILITY_WARNING,) if self._warning_pending else ()
        self._warning_pending = False
        return WriteResult(
            rows_received=row_count,
            rows_affected=count,
            bytes_received=batch_bytes,
            metadata=self._result_metadata,
            warnings=warnings,
        )

    def abort(self, _error: BaseException) -> None:
        self.close()

    def close(self) -> None:
        client = self._client
        if client is None:
            return
        client.close()
        self._client = None


__all__ = ["MilvusSink"]
