# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed, full-point Qdrant upserts.

Each input row must provide a stable unsigned 64-bit integer or UUID point ID,
one dense vector (or every configured named dense vector), and explicitly
mapped payload fields. A successful worker batch has been applied by Qdrant,
but worker batches are independent and are not an atomic transaction.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
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

if TYPE_CHECKING:
    from vane import DuckDBPyRelation

_MAX_INT64 = (1 << 63) - 1
_MAX_UINT64 = (1 << 64) - 1
_DEFAULT_MAX_BATCH_ROWS = 1_000
_DEFAULT_MAX_BATCH_BYTES = 16 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 30
_MAX_PAYLOAD_NESTING = 16
_PARTIAL_VISIBILITY_WARNING = (
    "Qdrant applies full-point worker batches independently; a later operation failure can leave this batch visible"
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


def _url(value: object) -> str:
    url = _name("url", value)
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise ValueError("url must be a valid HTTP or HTTPS Qdrant endpoint") from error
    if parsed.scheme not in {"http", "https"} or hostname is None:
        raise ValueError("url must be an absolute HTTP or HTTPS Qdrant endpoint")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError(
            "url must not contain credentials, query parameters, or fragments; use api_key=EnvironmentSecret(...)"
        )
    if parsed.path not in {"", "/"}:
        raise ValueError("url must not contain a path")
    return url


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


def _mapping(name: str, value: object, *, allow_none: bool) -> tuple[tuple[str, str], ...]:
    if value is None and allow_none:
        return ()
    if not isinstance(value, Mapping):
        suffix = " or None" if allow_none else ""
        raise TypeError(f"{name} must be a mapping{suffix}")
    normalized: list[tuple[str, str]] = []
    for source, target in value.items():
        normalized.append((_name(f"{name} source", source), _name(f"{name} target", target)))
    targets = [target for _, target in normalized]
    if len(set(targets)) != len(targets):
        raise ValueError(f"{name} targets must be unique")
    return tuple(normalized)


@dataclass(frozen=True)
class _VectorBinding:
    source_name: str
    target_name: str | None
    data_type: pa.DataType
    fixed_dimension: int | None


@dataclass(frozen=True)
class _PayloadBinding:
    source_name: str
    target_name: str
    data_type: pa.DataType


def _vector_type(field: pa.Field) -> tuple[pa.DataType, int | None]:
    data_type = field.type
    if not (
        pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type)
    ) or not pa.types.is_float32(data_type.value_type):
        raise ValueError(
            f"Qdrant vector source field {field.name!r} must be an Arrow list, large_list, "
            "or fixed_size_list of float32"
        )
    dimension = data_type.list_size if pa.types.is_fixed_size_list(data_type) else None
    if dimension is not None and dimension <= 0:
        raise ValueError(f"Qdrant vector source field {field.name!r} must have a positive dimension")
    return data_type, dimension


def _validate_payload_type(field_name: str, data_type: pa.DataType, *, depth: int = 0) -> None:
    if depth > _MAX_PAYLOAD_NESTING:
        raise ValueError(f"Qdrant payload source field {field_name!r} exceeds the supported nesting depth")
    if (
        pa.types.is_null(data_type)
        or pa.types.is_boolean(data_type)
        or pa.types.is_integer(data_type)
        or pa.types.is_float32(data_type)
        or pa.types.is_float64(data_type)
        or pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
    ):
        return
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type):
        _validate_payload_type(field_name, data_type.value_type, depth=depth + 1)
        return
    if pa.types.is_struct(data_type):
        names = [child.name for child in data_type]
        if len(set(names)) != len(names):
            raise ValueError(f"Qdrant payload struct source field {field_name!r} must have unique child names")
        for child in data_type:
            _name(f"payload struct field in {field_name!r}", child.name)
            _validate_payload_type(field_name, child.type, depth=depth + 1)
        return
    raise ValueError(f"QdrantSink does not support payload field {field_name!r} with Arrow type {data_type}")


def _load_qdrant_sdk() -> tuple[type[Any], Any]:
    try:
        from qdrant_client import QdrantClient, models  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ModuleNotFoundError as error:
        if error.name != "qdrant_client":
            raise
        raise ImportError("QdrantSink requires qdrant-client; install vane-ai[qdrant]") from error
    return QdrantClient, models


class QdrantSink(DataSink):
    """Write relation batches to one Qdrant collection with point upserts.

    ``point_id`` names the source relation column containing a caller-assigned
    Arrow ``uint64`` or UUID string. ``vector_mapping`` is either the source
    column for a collection with one unnamed dense vector, or a mapping from
    source columns to Qdrant named dense vectors. ``payload_mapping`` maps
    source columns to payload keys. Every input column must have exactly one
    role, so callers must project away intentionally unused columns.

    The URL may be a public endpoint string or an ``EnvironmentSecret`` that
    is resolved on each worker. API keys must use ``EnvironmentSecret`` and
    are never stored as plaintext in the serialized sink plan.

    Framework retries are disabled unless ``max_retries`` is positive. A retry
    replays the full input with the same point IDs. Qdrant replaces the complete
    vectors and payload of each point; Vane does not coordinate concurrent
    writers or provide a cross-batch transaction or exactly-once delivery.
    """

    def __init__(
        self,
        collection_name: str,
        *,
        url: str | EnvironmentSecret,
        point_id: str,
        vector_mapping: str | Mapping[str, str],
        payload_mapping: Mapping[str, str] | None = None,
        api_key: EnvironmentSecret | None = None,
        worker_count: int = 1,
        max_batch_rows: int = _DEFAULT_MAX_BATCH_ROWS,
        max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
        max_retries: int = 0,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.collection_name = _name("collection_name", collection_name)
        self.url: str | EnvironmentSecret
        if isinstance(url, EnvironmentSecret):
            self.url = url
        else:
            self.url = _url(url)
        self.point_id = _name("point_id", point_id)
        self._single_vector_source: str | None
        self._named_vector_mapping: tuple[tuple[str, str], ...]
        if isinstance(vector_mapping, str):
            self._single_vector_source = _name("vector_mapping", vector_mapping)
            self._named_vector_mapping = ()
        else:
            named_mapping = _mapping("vector_mapping", vector_mapping, allow_none=False)
            if not named_mapping:
                raise ValueError("vector_mapping must contain at least one named vector")
            self._single_vector_source = None
            self._named_vector_mapping = named_mapping
        self._payload_mapping = _mapping("payload_mapping", payload_mapping, allow_none=True)
        if api_key is not None and not isinstance(api_key, EnvironmentSecret):
            raise TypeError("api_key must be an EnvironmentSecret or None")
        self._api_key = api_key
        self.worker_count = _positive_int("worker_count", worker_count)
        self.max_batch_rows = _positive_int("max_batch_rows", max_batch_rows)
        self.max_batch_bytes = _positive_int("max_batch_bytes", max_batch_bytes)
        self.max_retries = _non_negative_int("max_retries", max_retries)
        self.timeout = _positive_int("timeout", timeout)

    def bind(self, schema: pa.Schema) -> BoundKeyedUpsertSink:
        if not isinstance(schema, pa.Schema):
            raise TypeError("schema must be pyarrow.Schema")
        if len({name.casefold() for name in schema.names}) != len(schema.names):
            raise ValueError("QdrantSink requires case-insensitively unique input column names")

        single_vector_source = self._single_vector_source
        vector_mapping: tuple[tuple[str, str | None], ...] = (
            ((single_vector_source, None),)
            if single_vector_source is not None
            else tuple((source, target) for source, target in self._named_vector_mapping)
        )
        source_roles = [self.point_id]
        source_roles.extend(source for source, _ in vector_mapping)
        source_roles.extend(source for source, _ in self._payload_mapping)
        if len(set(source_roles)) != len(source_roles):
            raise ValueError("point_id, vector_mapping, and payload_mapping source columns must be disjoint")
        unknown_sources = set(source_roles).difference(schema.names)
        if unknown_sources:
            raise ValueError(f"Qdrant mappings contain unknown input columns: {sorted(unknown_sources)!r}")
        unmapped_sources = set(schema.names).difference(source_roles)
        if unmapped_sources:
            raise ValueError(f"QdrantSink requires every input column to be mapped: {sorted(unmapped_sources)!r}")

        point_field = schema.field(self.point_id)
        point_id_is_uuid = pa.types.is_string(point_field.type)
        if not point_id_is_uuid and not pa.types.is_uint64(point_field.type):
            raise ValueError("Qdrant point_id source must be Arrow uint64 or string UUID")

        vectors: list[_VectorBinding] = []
        for source, target in vector_mapping:
            data_type, fixed_dimension = _vector_type(schema.field(source))
            vectors.append(_VectorBinding(source, target, data_type, fixed_dimension))

        payloads: list[_PayloadBinding] = []
        for source, target in self._payload_mapping:
            data_type = schema.field(source).type
            _validate_payload_type(source, data_type)
            payloads.append(_PayloadBinding(source, target, data_type))

        return _BoundQdrantSink(self, schema, point_id_is_uuid, tuple(vectors), tuple(payloads))

    def _resolve_url(self) -> str:
        url = self.url
        if isinstance(url, EnvironmentSecret):
            return _url(url.resolve())
        return url


class _BoundQdrantSink(BoundKeyedUpsertSink):
    def __init__(
        self,
        sink: QdrantSink,
        schema: pa.Schema,
        point_id_is_uuid: bool,
        vectors: tuple[_VectorBinding, ...],
        payloads: tuple[_PayloadBinding, ...],
    ) -> None:
        self._sink = sink
        self._schema = schema
        self._point_id_is_uuid = point_id_is_uuid
        self._vectors = vectors
        self._payloads = payloads

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
        return (self._sink.point_id,)

    def prepare_input(self, relation: DuckDBPyRelation) -> DuckDBPyRelation:
        if not self._point_id_is_uuid:
            return relation
        projections: list[str] = []
        for name in self._schema.names:
            quoted = _quote_identifier(name)
            if name == self._sink.point_id:
                projections.append(f"CAST(TRY_CAST({quoted} AS UUID) AS VARCHAR) AS {quoted}")
            else:
                projections.append(quoted)
        return relation.project(", ".join(projections))

    def open_worker(self, _context: WriteContext) -> DataSinkWorker:
        return _QdrantWorker(self._sink, self._schema, self._point_id_is_uuid, self._vectors, self._payloads)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class _QdrantWorker(DataSinkWorker):
    def __init__(
        self,
        sink: QdrantSink,
        schema: pa.Schema,
        point_id_is_uuid: bool,
        vectors: tuple[_VectorBinding, ...],
        payloads: tuple[_PayloadBinding, ...],
    ) -> None:
        client_type, models = _load_qdrant_sdk()
        client_options: dict[str, Any] = {"url": sink._resolve_url(), "timeout": sink.timeout}
        if sink._api_key is not None:
            client_options["api_key"] = sink._api_key.resolve()
        self._sink = sink
        self._schema = schema
        self._point_id_is_uuid = point_id_is_uuid
        self._vectors = vectors
        self._payloads = payloads
        self._models = models
        self._client: Any | None = client_type(**client_options)
        self._warning_pending = True
        try:
            self._remote_dimensions = self._validate_collection()
            self._result_metadata = {
                "provider": "qdrant",
                "collection": self._sink.collection_name,
                "write_mode": "replace",
                "status": "completed",
            }
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
                        add_note(f"Qdrant client cleanup also failed: {type(close_error).__name__}")
                    except BaseException:
                        pass
            raise

    def _client_or_raise(self) -> Any:
        if self._client is None:
            raise RuntimeError("QdrantSink worker is closed")
        return self._client

    def _validate_vector_params(self, name: str | None, params: object, binding: _VectorBinding) -> None:
        label = "unnamed vector" if name is None else f"vector {name!r}"
        if not isinstance(params, self._models.VectorParams):
            raise ValueError(f"Qdrant collection {label} has invalid parameters")
        dimension = params.size
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Qdrant collection {label} has an invalid dimension")
        if binding.fixed_dimension is not None and binding.fixed_dimension != dimension:
            raise ValueError(
                f"Qdrant collection {label} dimension {dimension} does not match "
                f"Arrow fixed dimension {binding.fixed_dimension}"
            )
        if params.datatype not in {None, self._models.Datatype.FLOAT32}:
            raise ValueError(f"Qdrant collection {label} must use float32 vectors")
        if params.multivector_config is not None:
            raise ValueError("QdrantSink does not support multivector collections")

    def _validate_collection(self) -> dict[str | None, int]:
        info = self._client_or_raise().get_collection(collection_name=self._sink.collection_name)
        if not isinstance(info, self._models.CollectionInfo):
            raise ValueError("Qdrant get_collection returned invalid collection information")
        if not isinstance(info.config, self._models.CollectionConfig) or not isinstance(
            info.config.params, self._models.CollectionParams
        ):
            raise ValueError("Qdrant get_collection returned invalid collection configuration")
        params = info.config.params
        if params.sparse_vectors:
            raise ValueError("QdrantSink does not support collections with sparse vectors")
        remote_vectors = params.vectors
        if self._vectors[0].target_name is None:
            if not isinstance(remote_vectors, self._models.VectorParams):
                raise ValueError("Qdrant collection must define one unnamed dense vector")
            self._validate_vector_params(None, remote_vectors, self._vectors[0])
            return {None: remote_vectors.size}
        if not isinstance(remote_vectors, dict):
            raise ValueError("Qdrant collection must define named dense vectors")
        expected_names: set[str] = set()
        for binding in self._vectors:
            assert binding.target_name is not None
            expected_names.add(binding.target_name)
        if set(remote_vectors) != expected_names:
            raise ValueError(
                "vector_mapping must cover every named dense vector in the Qdrant collection; "
                f"expected {sorted(remote_vectors)!r}, got {sorted(expected_names)!r}"
            )
        for binding in self._vectors:
            assert binding.target_name is not None
            self._validate_vector_params(binding.target_name, remote_vectors[binding.target_name], binding)
        return {binding.target_name: remote_vectors[binding.target_name].size for binding in self._vectors}

    def _point_id(self, value: object) -> int | str:
        if self._point_id_is_uuid:
            if not isinstance(value, str):
                raise ValueError("Qdrant UUID point IDs must be strings")
            try:
                parsed = uuid.UUID(value)
            except (AttributeError, ValueError) as error:
                raise ValueError("Qdrant UUID point IDs must contain valid UUIDs") from error
            if str(parsed) != value:
                raise ValueError("Qdrant UUID point IDs must be normalized canonical UUID strings")
            return value
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > _MAX_UINT64:
            raise ValueError("Qdrant numeric point IDs must be unsigned 64-bit integers")
        return value

    def _vector(self, value: object, *, name: str | None, dimension: int) -> list[float]:
        label = "unnamed vector" if name is None else f"vector {name!r}"
        if not isinstance(value, list) or len(value) != dimension:
            raise ValueError(f"Qdrant {label} has an invalid dimension")
        converted: list[float] = []
        for element in value:
            if isinstance(element, bool) or not isinstance(element, (int, float)):
                raise ValueError(f"Qdrant {label} contains an invalid value")
            normalized = float(element)
            if not math.isfinite(normalized):
                raise ValueError(f"Qdrant {label} contains an invalid value")
            converted.append(normalized)
        return converted

    def _payload_value(self, value: object, *, field: str, depth: int = 0) -> object:
        if depth > _MAX_PAYLOAD_NESTING:
            raise ValueError(f"Qdrant payload field {field!r} exceeds the supported nesting depth")
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value < -_MAX_INT64 - 1 or value > _MAX_INT64:
                raise ValueError(f"Qdrant payload field {field!r} contains an integer outside signed 64-bit range")
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"Qdrant payload field {field!r} contains a non-finite float")
            return value
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(f"Qdrant payload field {field!r} contains invalid UTF-8") from error
            return value
        if isinstance(value, list):
            return [self._payload_value(item, field=field, depth=depth + 1) for item in value]
        if isinstance(value, dict):
            normalized: dict[str, object] = {}
            for key, item in value.items():
                normalized_key = _name(f"nested key in payload field {field!r}", key)
                normalized[normalized_key] = self._payload_value(item, field=field, depth=depth + 1)
            return normalized
        raise ValueError(f"Qdrant payload field {field!r} contains a non-JSON value")

    def _points(self, table: pa.Table) -> tuple[list[Any], int, int]:
        if not isinstance(table, pa.Table):
            raise TypeError(f"QdrantSink expected pyarrow.Table, got {type(table).__name__}")
        if len(table.schema) != len(self._schema) or any(
            batch_field.name != bound_field.name or batch_field.type != bound_field.type
            for batch_field, bound_field in zip(table.schema, self._schema, strict=True)
        ):
            raise ValueError("Qdrant batch schema does not match the bound input schema")
        row_count = table.num_rows
        batch_bytes = table.nbytes
        if row_count > self._sink.max_batch_rows:
            raise ValueError("Qdrant batch exceeds max_batch_rows")
        if batch_bytes > self._sink.max_batch_bytes:
            raise ValueError("Qdrant batch exceeds max_batch_bytes")

        points: list[Any] = []
        for row in table.to_pylist():
            point_id = self._point_id(row[self._sink.point_id])
            if self._vectors[0].target_name is None:
                binding = self._vectors[0]
                vector: list[float] | dict[str, list[float]] = self._vector(
                    row[binding.source_name], name=None, dimension=self._remote_dimensions[None]
                )
            else:
                vector = {}
                for binding in self._vectors:
                    assert binding.target_name is not None
                    vector[binding.target_name] = self._vector(
                        row[binding.source_name],
                        name=binding.target_name,
                        dimension=self._remote_dimensions[binding.target_name],
                    )
            payload = {
                binding.target_name: self._payload_value(row[binding.source_name], field=binding.target_name)
                for binding in self._payloads
            }
            points.append(self._models.PointStruct(id=point_id, vector=vector, payload=payload))
        return points, row_count, batch_bytes

    def write(self, table: pa.Table) -> WriteResult:
        points, row_count, batch_bytes = self._points(table)
        if not points:
            return WriteResult(
                rows_received=0,
                rows_affected=0,
                bytes_received=batch_bytes,
                metadata=self._result_metadata,
            )
        response = self._client_or_raise().upsert(
            collection_name=self._sink.collection_name,
            points=points,
            wait=True,
            timeout=self._sink.timeout,
        )
        if not isinstance(response, self._models.UpdateResult):
            raise RuntimeError("Qdrant upsert returned an invalid update result")
        if response.status != self._models.UpdateStatus.COMPLETED:
            raise RuntimeError(f"Qdrant upsert was not applied: status={response.status.value}")
        warnings = (_PARTIAL_VISIBILITY_WARNING,) if self._warning_pending else ()
        self._warning_pending = False
        return WriteResult(
            rows_received=row_count,
            rows_affected=row_count,
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


__all__ = ["QdrantSink"]
