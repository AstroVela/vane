# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Replay-safe, full-row Milvus upserts for :mod:`vane.datasink`.

Install with ``pip install vane-ai[milvus]``.  This adapter intentionally
supports only collections with an explicit INT64 or VARCHAR primary key and
ordinary override-mode upserts.  A failed distributed operation can therefore
leave some batches visible; it is not an atomic transaction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import pyarrow as pa

from vane.datasink import (
    BoundKeyedUpsertSink,
    DataSink,
    DataSinkCapabilities,
    DataSinkExecutionOptions,
    DataSinkWorker,
    EnvironmentSecret,
    WriteContext,
    WriteResult,
)

_INT64 = 5
_VARCHAR = 21
_FLOAT_VECTOR = 101
_TYPE_NAMES = {
    1: "BOOL",
    2: "INT8",
    3: "INT16",
    4: "INT32",
    5: "INT64",
    10: "FLOAT",
    11: "DOUBLE",
    21: "VARCHAR",
    101: "FLOAT_VECTOR",
}


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_text(name: str, value: object | None) -> str | None:
    if value is None:
        return None
    return _require_text(name, value)


def _uri(value: object) -> str:
    uri = _require_text("uri", value)
    parsed = urlsplit(uri)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("uri must not contain credentials; use token=EnvironmentSecret(...) instead")
    return uri


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _milvus_type(field: Mapping[str, Any]) -> int:
    value = field.get("type")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Milvus field {field.get('name')!r} has an invalid type")
    return value


def _arrow_milvus_type(field: pa.Field) -> int:
    data_type = field.type
    if pa.types.is_boolean(data_type):
        return 1
    if pa.types.is_int8(data_type):
        return 2
    if pa.types.is_int16(data_type):
        return 3
    if pa.types.is_int32(data_type):
        return 4
    if pa.types.is_int64(data_type):
        return _INT64
    if pa.types.is_float32(data_type):
        return 10
    if pa.types.is_float64(data_type):
        return 11
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return _VARCHAR
    if (
        pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type)
    ) and pa.types.is_float32(data_type.value_type):
        return _FLOAT_VECTOR
    raise ValueError(f"MilvusSink does not support Arrow field {field.name!r} with type {data_type}")


def _field_dimension(field: Mapping[str, Any]) -> int:
    params = field.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError(f"Milvus vector field {field.get('name')!r} has invalid params")
    return _positive_int(f"Milvus vector field {field.get('name')!r} dim", params.get("dim"))


def _field_max_length(field: Mapping[str, Any]) -> int:
    params = field.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError(f"Milvus VARCHAR field {field.get('name')!r} has invalid params")
    return _positive_int(f"Milvus VARCHAR field {field.get('name')!r} max_length", params.get("max_length"))


def _load_milvus_client() -> type[Any]:
    try:
        from pymilvus import MilvusClient  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("MilvusSink requires pymilvus; install vane-ai[milvus]") from error
    return MilvusClient


@dataclass(frozen=True)
class _FieldBinding:
    source_name: str
    target_name: str
    arrow_type: pa.DataType


class MilvusSink(DataSink):
    """Write a relation into one Milvus collection using full-row upserts.

    ``token`` accepts only :class:`EnvironmentSecret`, keeping credential values
    out of serialized plans.  ``field_mapping`` maps input relation columns to
    collection fields; without it, names must match exactly.
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
        max_batch_rows: int = 1_000,
        max_batch_bytes: int = 16 * 1024 * 1024,
        timeout: float | None = None,
    ) -> None:
        self.collection_name = _require_text("collection_name", collection_name)
        self.uri = _uri(uri)
        self.primary_key = _require_text("primary_key", primary_key)
        if token is not None and not isinstance(token, EnvironmentSecret):
            raise TypeError("token must be an EnvironmentSecret or None")
        self.token = token
        self.database = _optional_text("database", database)
        self.worker_count = _positive_int("worker_count", worker_count)
        self.max_batch_rows = _positive_int("max_batch_rows", max_batch_rows)
        self.max_batch_bytes = _positive_int("max_batch_bytes", max_batch_bytes)
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number or None")
        self.timeout = None if timeout is None else float(timeout)
        mapping = dict(field_mapping or {})
        for source, target in mapping.items():
            _require_text("field_mapping source", source)
            _require_text("field_mapping target", target)
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("field_mapping values must be unique")
        self.field_mapping = mapping

    def bind(self, schema: pa.Schema) -> BoundKeyedUpsertSink:
        if not isinstance(schema, pa.Schema):
            raise TypeError("schema must be pyarrow.Schema")
        source_names = set(schema.names)
        unknown_sources = set(self.field_mapping).difference(source_names)
        if unknown_sources:
            raise ValueError(f"field_mapping contains unknown input columns: {sorted(unknown_sources)!r}")
        bindings = tuple(
            _FieldBinding(field.name, self.field_mapping.get(field.name, field.name), field.type) for field in schema
        )
        if len({binding.target_name for binding in bindings}) != len(bindings):
            raise ValueError("field_mapping must produce unique collection field names")
        primary_sources = [binding.source_name for binding in bindings if binding.target_name == self.primary_key]
        if len(primary_sources) != 1:
            raise ValueError("primary_key must map from exactly one input column")
        primary_field = schema.field(primary_sources[0])
        if _arrow_milvus_type(primary_field) not in {_INT64, _VARCHAR}:
            raise ValueError("Milvus primary_key input column must be Arrow int64 or string")
        for field in schema:
            _arrow_milvus_type(field)
        return _BoundMilvusSink(self, bindings, primary_sources[0])


class _BoundMilvusSink(BoundKeyedUpsertSink):
    def __init__(self, sink: MilvusSink, bindings: tuple[_FieldBinding, ...], key_column: str) -> None:
        self._sink = sink
        self._bindings = bindings
        self._key_column = key_column

    @property
    def capabilities(self) -> DataSinkCapabilities:
        from vane.datasink import CommitProtocol, RetryMode

        return DataSinkCapabilities(CommitProtocol.IMMEDIATE, RetryMode.IDEMPOTENT)

    @property
    def execution_options(self) -> DataSinkExecutionOptions:
        return DataSinkExecutionOptions(
            worker_count=self._sink.worker_count,
            batch_size=self._sink.max_batch_rows,
            target_max_batch_bytes=self._sink.max_batch_bytes,
        )

    @property
    def key_columns(self) -> Sequence[str]:
        return (self._key_column,)

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _MilvusWorker(self._sink, self._bindings)


class _MilvusWorker(DataSinkWorker):
    def __init__(self, sink: MilvusSink, bindings: tuple[_FieldBinding, ...]) -> None:
        client_kwargs: dict[str, Any] = {"uri": sink.uri}
        if sink.token is not None:
            client_kwargs["token"] = sink.token.resolve()
        if sink.database is not None:
            client_kwargs["db_name"] = sink.database
        self._sink = sink
        self._bindings = bindings
        self._client = _load_milvus_client()(**client_kwargs)
        self._closed = False
        try:
            self._fields = self._validate_collection()
        except BaseException:
            self.close()
            raise

    def _validate_collection(self) -> Mapping[str, Mapping[str, Any]]:
        description = self._client.describe_collection(
            collection_name=self._sink.collection_name, timeout=self._sink.timeout
        )
        if not isinstance(description, Mapping):
            raise ValueError("Milvus describe_collection returned an invalid schema")
        if description.get("auto_id") is not False:
            raise ValueError("MilvusSink requires a collection with auto_id disabled")
        raw_fields = description.get("fields")
        if not isinstance(raw_fields, list):
            raise ValueError("Milvus describe_collection returned no fields")
        fields: dict[str, Mapping[str, Any]] = {}
        for raw_field in raw_fields:
            if not isinstance(raw_field, Mapping) or not isinstance(raw_field.get("name"), str):
                raise ValueError("Milvus describe_collection returned an invalid field")
            if raw_field["name"] in fields:
                raise ValueError(f"Milvus describe_collection returned duplicate field {raw_field['name']!r}")
            fields[raw_field["name"]] = raw_field
        primary = [field for field in fields.values() if field.get("is_primary") is True]
        if len(primary) != 1 or primary[0].get("name") != self._sink.primary_key:
            raise ValueError("Milvus collection primary key does not match MilvusSink.primary_key")
        if _milvus_type(primary[0]) not in {_INT64, _VARCHAR}:
            raise ValueError("Milvus collection primary key must be INT64 or VARCHAR")
        bound_targets = {binding.target_name for binding in self._bindings}
        unknown = bound_targets.difference(fields)
        if unknown:
            raise ValueError(f"input fields are absent from Milvus collection: {sorted(unknown)!r}")
        if description.get("enable_dynamic_field") is True:
            raise ValueError("MilvusSink does not support collections with dynamic fields enabled")
        missing_required = [
            name
            for name, field in fields.items()
            if name not in bound_targets and field.get("nullable") is not True and "default_value" not in field
        ]
        if missing_required:
            raise ValueError(f"input is missing required Milvus fields: {sorted(missing_required)!r}")
        for binding in self._bindings:
            remote = fields[binding.target_name]
            expected_type = _arrow_milvus_type(pa.field(binding.source_name, binding.arrow_type))
            actual_type = _milvus_type(remote)
            if expected_type != actual_type:
                raise ValueError(
                    f"Milvus field {binding.target_name!r} type {_TYPE_NAMES.get(actual_type, actual_type)} "
                    f"does not match Arrow field {binding.source_name!r}"
                )
            if actual_type == _FLOAT_VECTOR:
                _field_dimension(remote)
            if actual_type == _VARCHAR:
                _field_max_length(remote)
        return fields

    def _records(self, table: pa.Table) -> list[dict[str, Any]]:
        if not isinstance(table, pa.Table):
            raise TypeError(f"MilvusSink expected pyarrow.Table, got {type(table).__name__}")
        expected_names = tuple(binding.source_name for binding in self._bindings)
        if tuple(table.column_names) != expected_names:
            raise ValueError("Milvus batch schema does not match the bound input schema")
        for binding in self._bindings:
            if table.schema.field(binding.source_name).type != binding.arrow_type:
                raise ValueError("Milvus batch schema does not match the bound input schema")
        if table.num_rows > self._sink.max_batch_rows:
            raise ValueError("Milvus batch exceeds max_batch_rows")
        if table.nbytes > self._sink.max_batch_bytes:
            raise ValueError("Milvus batch exceeds max_batch_bytes")
        records: list[dict[str, Any]] = []
        for row in table.to_pylist():
            record: dict[str, Any] = {}
            for binding in self._bindings:
                value = row[binding.source_name]
                remote = self._fields[binding.target_name]
                if value is None and remote.get("nullable") is not True:
                    raise ValueError(f"Milvus field {binding.target_name!r} is not nullable")
                if value is not None and _milvus_type(remote) == _VARCHAR:
                    if len(value.encode("utf-8")) > _field_max_length(remote):
                        raise ValueError(f"Milvus field {binding.target_name!r} exceeds max_length")
                if value is not None and _milvus_type(remote) == _FLOAT_VECTOR:
                    if not isinstance(value, list) or len(value) != _field_dimension(remote):
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
        return records

    def write(self, table: pa.Table) -> WriteResult:
        records = self._records(table)
        response = self._client.upsert(
            collection_name=self._sink.collection_name,
            data=records,
            timeout=self._sink.timeout,
        )
        count = response.get("upsert_count") if isinstance(response, Mapping) else None
        if isinstance(count, bool) or not isinstance(count, int) or count != table.num_rows:
            raise RuntimeError("Milvus upsert returned an invalid affected-row count")
        return WriteResult(
            rows_received=table.num_rows,
            rows_affected=count,
            bytes_received=table.nbytes,
            metadata={"provider": "milvus", "collection": self._sink.collection_name},
        )

    def abort(self, error: BaseException) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True


__all__ = ["MilvusSink"]
