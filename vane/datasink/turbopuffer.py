# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Turbopuffer upsert sink."""

from __future__ import annotations

import binascii
import hashlib
import math
import os
import re
import struct
import threading
import uuid
from base64 import b64decode, b64encode
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from vane.datasink import (
    BoundDataSink,
    CommitProtocol,
    DataSink,
    DataSinkCapabilities,
    DataSinkExecutionOptions,
    DataSinkWriter,
    RetryMode,
    WriteContext,
    WriteResult,
    WriteState,
    _append_write_result_warning,
    _bounded_warning,
    _freeze_json_mapping,
    _json_mapping,
    _safe_exception_message,
    _strict_non_negative_int,
)

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

    from vane import DuckDBPyRelation


_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_MAX_TURBOPUFFER_REQUEST_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_BATCH_BYTES = 256 * 1024 * 1024
_MAX_ATTRIBUTE_NAMES = 1_024
_MAX_VECTOR_DIMENSIONS = 10_752
_MAX_EMBEDDED_ATTRIBUTES = 4
_MAX_EMBEDDED_ATTRIBUTE_BATCH_ROWS = 30
_CLIENT_CACHE_SIZE = 8
_CLIENTS = threading.local()
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DENSE_VECTOR_TYPE_RE = re.compile(r"\[(\d+)\](f16|f32|i8)")
_ID_COUNT_COLUMN = "$vane_turbopuffer_id_count"


def _positive_int(name: str, value: object) -> int:
    checked = _strict_non_negative_int(name, value)
    assert checked is not None
    if checked == 0:
        raise ValueError(f"{name} must be positive")
    return checked


def _finite_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _client_cache() -> OrderedDict[tuple[object, ...], Any]:
    cache = getattr(_CLIENTS, "cache", None)
    if cache is None:
        cache = OrderedDict()
        _CLIENTS.cache = cache
    return cache


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _turbopuffer_client(
    *,
    api_key_env: str,
    region: str,
    max_retries: int,
    timeout: float,
    compression: bool,
) -> Any:
    api_key = os.environ.get(api_key_env)
    if api_key is None or not api_key:
        raise RuntimeError(f"Turbopuffer API key environment variable {api_key_env!r} is not set")
    try:
        import turbopuffer
    except ImportError as exc:
        raise ImportError(
            "TurbopufferSink requires the 'turbopuffer' extra: pip install 'vane-ai[turbopuffer]'"
        ) from exc

    secret_digest = hashlib.sha256(api_key.encode("utf-8")).digest()
    key = (secret_digest, region, max_retries, timeout, compression)
    cache = _client_cache()
    client = cache.pop(key, None)
    if client is None:
        client = turbopuffer.Turbopuffer(
            api_key=api_key,
            region=region,
            max_retries=max_retries,
            timeout=timeout,
            compression=compression,
        )
    cache[key] = client
    while len(cache) > _CLIENT_CACHE_SIZE:
        _, evicted = cache.popitem(last=False)
        _close_client(evicted)
    return client


def _validate_attribute_type(name: str, dtype: pa.DataType, *, nested: bool = False) -> None:
    import pyarrow as pa

    if isinstance(dtype, pa.BaseExtensionType):
        extension_name = getattr(dtype, "extension_name", "")
        if extension_name == "arrow.fixed_shape_tensor":
            _validate_attribute_type(name, dtype.storage_type, nested=nested)
            return
        if extension_name in {"arrow.uuid", "uuid"}:
            return
        raise TypeError(f"TurbopufferSink column {name!r} has unsupported Arrow extension type {extension_name!r}")
    if pa.types.is_dictionary(dtype):
        _validate_attribute_type(name, dtype.value_type, nested=nested)
        return
    if (
        pa.types.is_boolean(dtype)
        or pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_date(dtype)
        or pa.types.is_timestamp(dtype)
    ):
        return
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        if nested:
            raise TypeError(f"TurbopufferSink column {name!r} contains nested arrays, which are not supported")
        _validate_attribute_type(name, dtype.value_type, nested=True)
        return
    raise TypeError(f"TurbopufferSink column {name!r} has unsupported Arrow type {dtype}")


def _inferred_attribute_type(dtype: pa.DataType) -> str | None:
    import pyarrow as pa

    if isinstance(dtype, pa.BaseExtensionType):
        extension_name = getattr(dtype, "extension_name", "")
        if extension_name in {"arrow.uuid", "uuid"}:
            return "uuid"
        return None
    if pa.types.is_dictionary(dtype):
        return _inferred_attribute_type(dtype.value_type)
    if pa.types.is_boolean(dtype):
        return "bool"
    if pa.types.is_unsigned_integer(dtype):
        return "uint"
    if pa.types.is_signed_integer(dtype):
        return "int"
    if pa.types.is_floating(dtype):
        return "float"
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "string"
    if pa.types.is_date(dtype) or pa.types.is_timestamp(dtype):
        return "datetime"
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        value_type = _inferred_attribute_type(dtype.value_type)
        return None if value_type is None else f"[]{value_type}"
    return None


def _normalize_id_values(values: list[Any], *, id_column: str) -> list[str | int]:
    normalized: list[str | int] = []
    seen: set[str | int] = set()
    for index, value in enumerate(values):
        if value is None:
            raise ValueError(f"TurbopufferSink ID column {id_column!r} contains NULL at batch row {index}")
        if isinstance(value, bool):
            raise TypeError(f"TurbopufferSink ID column {id_column!r} contains a boolean at batch row {index}")
        if isinstance(value, int):
            if value < 0 or value > (1 << 64) - 1:
                raise ValueError(
                    f"TurbopufferSink ID column {id_column!r} must contain unsigned 64-bit integers, UUIDs, or strings"
                )
            normalized_value: str | int = value
        elif isinstance(value, str):
            if not value:
                raise ValueError(
                    f"TurbopufferSink ID column {id_column!r} contains an empty string at batch row {index}"
                )
            if len(value.encode("utf-8")) > 64:
                raise ValueError(
                    f"TurbopufferSink ID column {id_column!r} contains a string longer than 64 UTF-8 bytes"
                )
            normalized_value = value
        elif isinstance(value, uuid.UUID):
            normalized_value = str(value)
        else:
            raise TypeError(
                f"TurbopufferSink ID column {id_column!r} contains unsupported "
                f"{type(value).__name__} at batch row {index}"
            )
        if normalized_value in seen:
            raise ValueError(
                f"TurbopufferSink ID column {id_column!r} contains duplicate ID "
                f"{normalized_value!r} at batch row {index}"
            )
        seen.add(normalized_value)
        normalized.append(normalized_value)
    return normalized


def _normalize_attribute_value(name: str, value: Any, *, row: int) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"TurbopufferSink column {name!r} contains a non-finite value at batch row {row}")
    if isinstance(value, (list, tuple)):
        return [_normalize_attribute_value(name, item, row=row) for item in value]
    return value


def _validate_vector_storage(values: list[float], *, row: int, element_type: str) -> None:
    if element_type == "i8":
        if any(not item.is_integer() or item < -128 or item > 127 for item in values):
            raise ValueError(
                f"TurbopufferSink i8 vector at batch row {row} must contain integers from -128 through 127"
            )
        return
    if element_type == "f16":
        try:
            struct.pack(f"<{len(values)}e", *values)
        except (OverflowError, struct.error) as exc:
            raise ValueError(f"TurbopufferSink f16 vector at batch row {row} contains an out-of-range value") from exc
        return
    if element_type != "f32":
        raise RuntimeError(f"unsupported Turbopuffer dense vector element type {element_type!r}")


def _encode_vector(value: Any, *, row: int, dimensions: int, element_type: str) -> Any:
    if value is None:
        return value
    if isinstance(value, str):
        expected_bytes = dimensions * 4
        expected_encoded_length = 4 * ((expected_bytes + 2) // 3)
        if len(value) != expected_encoded_length:
            raise ValueError(f"TurbopufferSink vector at batch row {row} does not contain {dimensions} float32 values")
        try:
            packed = b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"TurbopufferSink vector at batch row {row} is not valid base64") from exc
        if len(packed) != expected_bytes:
            raise ValueError(f"TurbopufferSink vector at batch row {row} does not contain {dimensions} float32 values")
        numeric_values = [item[0] for item in struct.iter_unpack("<f", packed)]
        if any(not math.isfinite(item) for item in numeric_values):
            raise ValueError(f"TurbopufferSink vector contains a non-finite value at batch row {row}")
        _validate_vector_storage(numeric_values, row=row, element_type=element_type)
        return value
    if not isinstance(value, list):
        raise TypeError(f"TurbopufferSink vector at batch row {row} must be a string or numeric array")
    if len(value) != dimensions:
        raise ValueError(f"TurbopufferSink vector at batch row {row} must contain exactly {dimensions} values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"TurbopufferSink vector at batch row {row} must contain only numeric values")
    try:
        numeric_values = [float(item) for item in value]
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"TurbopufferSink vector at batch row {row} cannot be represented as float32") from exc
    if any(not math.isfinite(item) for item in numeric_values):
        raise ValueError(f"TurbopufferSink vector contains a non-finite value at batch row {row}")
    _validate_vector_storage(numeric_values, row=row, element_type=element_type)
    try:
        packed = struct.pack(f"<{len(numeric_values)}f", *numeric_values)
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"TurbopufferSink vector at batch row {row} cannot be represented as float32") from exc
    return b64encode(packed).decode("ascii")


def _serialized_request_bytes(request: Mapping[str, Any], *, max_bytes: int) -> int:
    from turbopuffer.lib import json as turbopuffer_json

    encoded = turbopuffer_json.dumps(request)
    if not isinstance(encoded, bytes):
        raise TypeError("Turbopuffer SDK JSON serializer must return bytes")
    size = len(encoded)
    if size > max_bytes:
        raise ValueError(
            f"TurbopufferSink serialized request exceeds max_batch_bytes={max_bytes}; "
            "reduce batch_size or increase max_batch_bytes"
        )
    return size


def _validate_vector_type(name: str, dtype: pa.DataType) -> None:
    import pyarrow as pa

    if isinstance(dtype, pa.BaseExtensionType) and getattr(dtype, "extension_name", "") == "arrow.fixed_shape_tensor":
        dtype = dtype.storage_type
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        if pa.types.is_floating(dtype.value_type) or pa.types.is_integer(dtype.value_type):
            return
    raise TypeError(f"TurbopufferSink vector column {name!r} must contain strings or one-dimensional numeric arrays")


def _fixed_vector_dimensions(dtype: pa.DataType) -> int | None:
    import pyarrow as pa

    if isinstance(dtype, pa.BaseExtensionType) and getattr(dtype, "extension_name", "") == "arrow.fixed_shape_tensor":
        dtype = dtype.storage_type
    if pa.types.is_fixed_size_list(dtype):
        return int(dtype.list_size)
    return None


def _configured_attribute_type(value: Any) -> str | None:
    raw_type = value.get("type") if isinstance(value, Mapping) else value
    return raw_type if isinstance(raw_type, str) else None


def _dense_vector_spec(name: str, value: Any) -> tuple[int, str] | None:
    raw_type = _configured_attribute_type(value)
    if raw_type is None:
        return None
    if re.fullmatch(r"\[\]\[\d+\]f32", raw_type):
        raise ValueError(f"TurbopufferSink multi-vector attribute {name!r} is not supported")
    match = _DENSE_VECTOR_TYPE_RE.fullmatch(raw_type)
    if match is None:
        return None
    if not isinstance(value, Mapping) or value.get("ann") is not True:
        raise ValueError(f"TurbopufferSink dense vector schema for {name!r} must be a mapping with ann=True")
    return int(match.group(1)), match.group(2)


def _dense_vector_dimensions(name: str, value: Any) -> int | None:
    spec = _dense_vector_spec(name, value)
    return None if spec is None else spec[0]


def _response_attribute(response: Any, name: str, warnings: list[str]) -> Any:
    try:
        return getattr(response, name, None)
    except Exception as error:
        warnings.append(
            _bounded_warning(
                f"Turbopuffer response field {name!r} could not be read: "
                f"{type(error).__name__}: {_safe_exception_message(error)}"
            )
        )
        return None


def _response_non_negative_int(
    response: Any,
    name: str,
    warnings: list[str],
    *,
    required: bool = False,
) -> int | None:
    value = _response_attribute(response, name, warnings)
    if value is None:
        if required:
            warnings.append(_bounded_warning(f"Turbopuffer response is missing required field {name!r}"))
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        warnings.append(
            _bounded_warning(
                f"Turbopuffer response field {name!r} was omitted because it is not a non-negative integer"
            )
        )
        return None
    return value


def _response_metadata(response: Any, namespace: str, warnings: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"namespace": namespace, "provider": "turbopuffer"}
    for attribute in ("rows_upserted", "rows_patched", "rows_deleted"):
        value = _response_non_negative_int(response, attribute, warnings)
        if value is not None:
            metadata[attribute] = value
    billing = _response_attribute(response, "billing", warnings)
    billable_bytes = (
        None if billing is None else _response_non_negative_int(billing, "billable_logical_bytes_written", warnings)
    )
    if billable_bytes is not None:
        metadata["billable_logical_bytes_written"] = billable_bytes
    performance = _response_attribute(response, "performance", warnings)
    server_total_ms = (
        None if performance is None else _response_non_negative_int(performance, "server_total_ms", warnings)
    )
    if server_total_ms is not None:
        metadata["server_total_ms"] = server_total_ms
    return metadata


@dataclass
class _TurbopufferWriter(DataSinkWriter):
    namespace: str
    region: str
    id_column: str
    distance_metric: str | None
    namespace_schema: dict[str, Any]
    disable_backpressure: bool
    api_key_env: str
    max_retries: int
    timeout: float
    compression: bool
    max_batch_bytes: int
    vector_specs: dict[str, tuple[int, str]]

    def write(self, table: pa.Table) -> WriteResult:
        if _ID_COUNT_COLUMN not in table.column_names:
            raise RuntimeError("TurbopufferSink worker input is missing its global ID uniqueness marker")
        id_counts = table.column(_ID_COUNT_COLUMN).to_pylist()
        invalid_count_index = next(
            (
                row
                for row, count in enumerate(id_counts)
                if isinstance(count, bool) or not isinstance(count, int) or count != 1
            ),
            None,
        )
        if invalid_count_index is not None:
            duplicate_id = table.column(self.id_column)[invalid_count_index].as_py()
            raise ValueError(
                f"TurbopufferSink requires globally unique IDs; {duplicate_id!r} occurs "
                f"{id_counts[invalid_count_index]!r} times"
            )
        source_table = table.select([name for name in table.column_names if name != _ID_COUNT_COLUMN])
        if source_table.nbytes > self.max_batch_bytes:
            raise ValueError(
                f"TurbopufferSink Arrow batch is {source_table.nbytes} bytes, "
                f"exceeding max_batch_bytes={self.max_batch_bytes}"
            )
        if source_table.num_rows == 0:
            return WriteResult(
                state=WriteState.APPLIED,
                rows_received=0,
                rows_affected=0,
                bytes_received=0,
                metadata={"namespace": self.namespace, "provider": "turbopuffer"},
            )

        columns: dict[str, list[Any]] = {}
        for name in source_table.column_names:
            output_name = "id" if name == self.id_column else name
            values = source_table.column(name).to_pylist()
            columns[output_name] = [
                _normalize_attribute_value(output_name, value, row=row) for row, value in enumerate(values)
            ]
        columns["id"] = _normalize_id_values(columns["id"], id_column=self.id_column)
        for name, (dimensions, element_type) in self.vector_specs.items():
            if name in columns:
                columns[name] = [
                    _encode_vector(value, row=row, dimensions=dimensions, element_type=element_type)
                    for row, value in enumerate(columns[name])
                ]

        client = _turbopuffer_client(
            api_key_env=self.api_key_env,
            region=self.region,
            max_retries=self.max_retries,
            timeout=self.timeout,
            compression=self.compression,
        )
        request: dict[str, Any] = {
            "upsert_columns": columns,
            "disable_backpressure": self.disable_backpressure,
        }
        if self.distance_metric is not None:
            request["distance_metric"] = self.distance_metric
        if self.namespace_schema:
            request["schema"] = self.namespace_schema
        request_bytes = _serialized_request_bytes(request, max_bytes=self.max_batch_bytes)
        response = client.namespace(self.namespace).write(**request)
        response_warnings: list[str] = []
        rows_affected = _response_non_negative_int(
            response,
            "rows_affected",
            response_warnings,
            required=True,
        )
        metadata = _response_metadata(response, self.namespace, response_warnings)
        metadata["request_bytes"] = request_bytes
        bounded_response_warnings: tuple[str, ...] = ()
        for warning in response_warnings:
            bounded_response_warnings = _append_write_result_warning(bounded_response_warnings, warning)
        return WriteResult(
            state=WriteState.APPLIED,
            rows_received=source_table.num_rows,
            rows_affected=rows_affected,
            bytes_received=source_table.nbytes,
            metadata=metadata,
            warnings=bounded_response_warnings,
        )


@dataclass(frozen=True)
class _BoundTurbopufferSink(BoundDataSink):
    namespace: str
    region: str
    id_column: str
    distance_metric: str | None
    namespace_schema: dict[str, Any]
    disable_backpressure: bool
    api_key_env: str
    max_retries: int
    timeout: float
    compression: bool
    batch_size: int
    max_batch_bytes: int
    vector_specs: dict[str, tuple[int, str]]

    @property
    def capabilities(self) -> DataSinkCapabilities:
        return DataSinkCapabilities(
            commit_protocol=CommitProtocol.IMMEDIATE,
            retry_mode=RetryMode.IDEMPOTENT,
        )

    @property
    def execution_options(self) -> DataSinkExecutionOptions:
        return DataSinkExecutionOptions(
            batch_size=self.batch_size,
            cpus=1.0,
            target_max_batch_bytes=self.max_batch_bytes,
            task_input_max_bytes=self.max_batch_bytes,
        )

    def prepare_input(self, relation: DuckDBPyRelation) -> DuckDBPyRelation:
        id_column = _quoted_identifier(self.id_column)
        count_column = _quoted_identifier(_ID_COUNT_COLUMN)
        return relation.project(f"*, count(*) OVER (PARTITION BY {id_column})::UBIGINT AS {count_column}")

    def open_writer(self, context: WriteContext) -> DataSinkWriter:
        del context
        return _TurbopufferWriter(
            namespace=self.namespace,
            region=self.region,
            id_column=self.id_column,
            distance_metric=self.distance_metric,
            namespace_schema=self.namespace_schema,
            disable_backpressure=self.disable_backpressure,
            api_key_env=self.api_key_env,
            max_retries=self.max_retries,
            timeout=self.timeout,
            compression=self.compression,
            max_batch_bytes=self.max_batch_bytes,
            vector_specs=self.vector_specs,
        )


@dataclass(frozen=True)
class TurbopufferSink(DataSink):
    """Idempotently upsert relation batches into one Turbopuffer namespace.

    The API key itself is never serialized into the plan. Every worker reads it
    from ``api_key_env`` (``TURBOPUFFER_API_KEY`` by default).

    Vane validates global ID uniqueness in the same lazy plan before affected
    batches reach Turbopuffer, retaining replay-safe semantics even when worker
    completion order changes.
    """

    namespace: str
    region: str = "gcp-us-central1"
    id_column: str = "id"
    distance_metric: str | None = None
    schema: Mapping[str, Any] = field(default_factory=dict)
    disable_backpressure: bool = False
    api_key_env: str = "TURBOPUFFER_API_KEY"
    max_retries: int = 2
    timeout: float = 60.0
    compression: bool = False
    batch_size: int = 10_000
    max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str):
            raise TypeError("TurbopufferSink namespace must be a string")
        namespace = self.namespace.strip()
        if not _NAMESPACE_RE.fullmatch(namespace):
            raise ValueError("Turbopuffer namespace must match [A-Za-z0-9-_.]{1,128}")
        object.__setattr__(self, "namespace", namespace)

        for name in ("region", "id_column", "api_key_env"):
            raw_value = getattr(self, name)
            if not isinstance(raw_value, str):
                raise TypeError(f"TurbopufferSink {name} must be a string")
            value = raw_value.strip()
            if not value:
                raise ValueError(f"TurbopufferSink {name} must not be empty")
            object.__setattr__(self, name, value)
        if not _ENV_NAME_RE.fullmatch(self.api_key_env):
            raise ValueError("TurbopufferSink api_key_env must be a valid environment variable name")
        if self.id_column == "id":
            pass
        elif self.id_column.startswith("$") or len(self.id_column.encode("utf-8")) > 128:
            raise ValueError("TurbopufferSink id_column must be at most 128 UTF-8 bytes and must not start with '$'")

        if self.distance_metric not in {None, "cosine_distance", "euclidean_squared"}:
            raise ValueError("distance_metric must be 'cosine_distance', 'euclidean_squared', or None")
        if not isinstance(self.disable_backpressure, bool):
            raise TypeError("disable_backpressure must be a boolean")
        if not isinstance(self.compression, bool):
            raise TypeError("compression must be a boolean")
        object.__setattr__(self, "max_retries", _strict_non_negative_int("max_retries", self.max_retries))
        object.__setattr__(self, "timeout", _finite_positive_float("timeout", self.timeout))
        object.__setattr__(self, "batch_size", _positive_int("batch_size", self.batch_size))
        max_batch_bytes = _positive_int("max_batch_bytes", self.max_batch_bytes)
        if max_batch_bytes > _MAX_TURBOPUFFER_REQUEST_BYTES:
            raise ValueError("max_batch_bytes must not exceed Turbopuffer's 512 MiB request limit")
        object.__setattr__(self, "max_batch_bytes", max_batch_bytes)
        namespace_schema = _json_mapping("TurbopufferSink.schema", self.schema)
        if len(namespace_schema) > _MAX_ATTRIBUTE_NAMES:
            raise ValueError(f"TurbopufferSink schema must contain at most {_MAX_ATTRIBUTE_NAMES} attributes")
        for attribute_name in namespace_schema:
            if attribute_name.startswith("$") or len(attribute_name.encode("utf-8")) > 128:
                raise ValueError(
                    "TurbopufferSink schema attribute names must be at most 128 UTF-8 bytes and must not start with '$'"
                )
        object.__setattr__(self, "schema", _freeze_json_mapping(namespace_schema))

    def bind(self, schema: pa.Schema) -> BoundDataSink:
        import pyarrow as pa

        if not isinstance(schema, pa.Schema):
            raise TypeError(f"TurbopufferSink.bind() requires pyarrow.Schema, got {type(schema).__name__}")
        if self.id_column not in schema.names:
            raise ValueError(f"TurbopufferSink requires ID column {self.id_column!r}")
        namespace_schema = _json_mapping("TurbopufferSink.schema", self.schema)
        for name, value in namespace_schema.items():
            _dense_vector_dimensions(name, value)
        output_names: set[str] = set()
        vector_specs: dict[str, tuple[int, str]] = {}
        for field_value in schema:
            output_name = "id" if field_value.name == self.id_column else field_value.name
            if output_name in output_names:
                raise ValueError(f"TurbopufferSink columns collide after mapping {self.id_column!r} to 'id'")
            output_names.add(output_name)
            if output_name.startswith("$"):
                raise ValueError(f"Turbopuffer attribute {output_name!r} must not start with '$'")
            if len(output_name.encode("utf-8")) > 128:
                raise ValueError(f"Turbopuffer attribute {output_name!r} exceeds 128 UTF-8 bytes")
            _validate_attribute_type(output_name, field_value.type)
            if output_name == "id" and (
                pa.types.is_integer(field_value.type)
                or pa.types.is_string(field_value.type)
                or pa.types.is_large_string(field_value.type)
                or (
                    isinstance(field_value.type, pa.BaseExtensionType)
                    and getattr(field_value.type, "extension_name", "") in {"arrow.uuid", "uuid"}
                )
            ):
                id_schema_type = (
                    "uint"
                    if pa.types.is_integer(field_value.type)
                    else "string"
                    if pa.types.is_string(field_value.type) or pa.types.is_large_string(field_value.type)
                    else "uuid"
                )
                if output_name in namespace_schema:
                    configured_id_type = _configured_attribute_type(namespace_schema[output_name])
                    if configured_id_type != id_schema_type:
                        raise ValueError(
                            f"TurbopufferSink schema type for 'id' must be {id_schema_type!r} "
                            f"for Arrow type {field_value.type}"
                        )
                else:
                    namespace_schema[output_name] = id_schema_type
            elif output_name == "vector" and output_name not in namespace_schema:
                _validate_vector_type(output_name, field_value.type)
                fixed_dimensions = _fixed_vector_dimensions(field_value.type)
                if fixed_dimensions is None:
                    raise ValueError(
                        "TurbopufferSink variable-length or base64 'vector' columns require an explicit "
                        "schema mapping such as {'type': '[1536]f32', 'ann': True}"
                    )
                if fixed_dimensions < 1 or fixed_dimensions > _MAX_VECTOR_DIMENSIONS:
                    raise ValueError(
                        f"TurbopufferSink vector dimensions must be between 1 and {_MAX_VECTOR_DIMENSIONS}"
                    )
                namespace_schema["vector"] = {"type": f"[{fixed_dimensions}]f32", "ann": True}
            elif output_name not in namespace_schema:
                inferred_type = _inferred_attribute_type(field_value.type)
                if inferred_type is not None:
                    namespace_schema[output_name] = inferred_type
            configured_spec = _dense_vector_spec(output_name, namespace_schema.get(output_name))
            if configured_spec is not None:
                configured_dimensions, _ = configured_spec
                _validate_vector_type(output_name, field_value.type)
                fixed_dimensions = _fixed_vector_dimensions(field_value.type)
                if fixed_dimensions is not None and fixed_dimensions != configured_dimensions:
                    raise ValueError(
                        f"TurbopufferSink vector column {output_name!r} Arrow width does not match "
                        "its configured schema dimensions"
                    )
                vector_specs[output_name] = configured_spec
            elif output_name == "vector":
                raise ValueError(
                    "TurbopufferSink schema for 'vector' must be a mapping such as {'type': '[1536]f32', 'ann': True}"
                )
        if len(output_names | set(namespace_schema)) > _MAX_ATTRIBUTE_NAMES:
            raise ValueError(f"TurbopufferSink input and schema exceed the {_MAX_ATTRIBUTE_NAMES}-attribute limit")
        dense_vector_dimensions = [
            dimensions
            for name, value in namespace_schema.items()
            if (dimensions := _dense_vector_dimensions(name, value)) is not None
        ]
        if any(dimensions < 1 or dimensions > _MAX_VECTOR_DIMENSIONS for dimensions in dense_vector_dimensions):
            raise ValueError(f"TurbopufferSink vector dimensions must be between 1 and {_MAX_VECTOR_DIMENSIONS}")
        if len(dense_vector_dimensions) > 2:
            raise ValueError("TurbopufferSink schema must contain at most two dense vector attributes")
        embedded_attribute_count = sum(
            isinstance(value, Mapping) and value.get("embed") is not None for value in namespace_schema.values()
        )
        if embedded_attribute_count > _MAX_EMBEDDED_ATTRIBUTES:
            raise ValueError(
                f"TurbopufferSink schema must contain at most {_MAX_EMBEDDED_ATTRIBUTES} embedded attributes"
            )
        if self.distance_metric is None and (dense_vector_dimensions or embedded_attribute_count):
            raise ValueError(
                "TurbopufferSink distance_metric is required when writing dense vector or embedded attributes"
            )
        id_type = schema.field(self.id_column).type
        if not (
            pa.types.is_integer(id_type)
            or pa.types.is_string(id_type)
            or pa.types.is_large_string(id_type)
            or (
                isinstance(id_type, pa.BaseExtensionType)
                and getattr(id_type, "extension_name", "") in {"arrow.uuid", "uuid"}
            )
        ):
            raise TypeError(
                f"TurbopufferSink ID column {self.id_column!r} must be an integer, string, or UUID, got {id_type}"
            )
        return _BoundTurbopufferSink(
            namespace=self.namespace,
            region=self.region,
            id_column=self.id_column,
            distance_metric=self.distance_metric,
            namespace_schema=namespace_schema,
            disable_backpressure=self.disable_backpressure,
            api_key_env=self.api_key_env,
            max_retries=self.max_retries,
            timeout=self.timeout,
            compression=self.compression,
            batch_size=(
                min(self.batch_size, _MAX_EMBEDDED_ATTRIBUTE_BATCH_ROWS)
                if embedded_attribute_count
                else self.batch_size
            ),
            max_batch_bytes=self.max_batch_bytes,
            vector_specs=vector_specs,
        )


__all__ = ["TurbopufferSink"]
