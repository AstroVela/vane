# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed Apache Doris Arrow Stream Load writes.

Each worker converts an Arrow table directly to an Arrow IPC stream and sends
one synchronous Stream Load request per batch. Doris commits worker batches as
independent transactions; Vane does not provide a transaction spanning the
complete distributed relation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import numpy as np
import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]
import pyarrow.compute as pc  # type: ignore[import-not-found, import-untyped, unused-ignore]

from vane.datasink import (
    BoundDataSink,
    DataSink,
    DataSinkExecutionOptions,
    DataSinkWorker,
    EnvironmentSecret,
    WriteContext,
    WriteResult,
)

if TYPE_CHECKING:
    from types import ModuleType

_MAX_INT32 = (1 << 31) - 1
_MAX_INT64 = (1 << 63) - 1
_DEFAULT_MAX_BATCH_ROWS = 1_000_000
_DEFAULT_MAX_BATCH_BYTES = 128 * 1024 * 1024
_DEFAULT_MAX_REQUEST_BYTES = 160 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 600
_DEFAULT_SEND_BATCH_PARALLELISM = 1
_MAX_TIMEOUT_SECONDS = 259_200
_HTTP_TIMEOUT_GRACE_SECONDS = 30
_HTTP_CONNECT_TIMEOUT_SECONDS = 30
_HTTP_BODY_CHUNK_BYTES = 256 * 1024
_HTTPS_SHUTDOWN_SECONDS = 0.250
_MAX_RESPONSE_BYTES = 1024 * 1024
_LABEL_PREFIX = "vane"
_LABEL_PATTERN = re.compile(r"[-_A-Za-z0-9:]+\Z")
_OBJECT_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
_COLUMN_PATTERN = re.compile(r'[-.A-Za-z0-9_+/?@#$%^&*" ,:]+\Z')
_REDIRECT_STATUS = 307
_SUCCESS_STATUS = "Success"
_PUBLISH_TIMEOUT_STATUS = "Publish Timeout"
_PARTIAL_VISIBILITY_WARNING = (
    "Doris commits worker batches independently; a later operation failure can leave this batch visible"
)
_PUBLISH_TIMEOUT_WARNING = (
    "Doris committed this batch but timed out while publishing it; visibility may be delayed and the batch "
    "must not be retried with a new label"
)


def _name(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    if value != value.strip() or "\x00" in value or "\r" in value or "\n" in value:
        raise ValueError(f"{name} must not contain surrounding whitespace, NUL, CR, or LF characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid UTF-8") from error
    return value


def _doris_object(name: str, value: object) -> str:
    identifier = _name(name, value)
    if len(identifier) > 256:
        raise ValueError(f"{name} must be at most 256 characters")
    if _OBJECT_PATTERN.fullmatch(identifier) is None:
        raise ValueError(f"{name} must be a standard ASCII Doris identifier beginning with a letter")
    return identifier


def _doris_column(name: str, value: object) -> str:
    identifier = _name(name, value)
    if len(identifier) > 256:
        raise ValueError(f"{name} must be at most 256 characters")
    if _COLUMN_PATTERN.fullmatch(identifier) is None:
        raise ValueError(f"{name} must contain only Doris column-name characters that are safe in an ASCII HTTP header")
    return identifier


def _header_identifier(identifier: str) -> str:
    return f"`{identifier}`"


def _endpoint(value: object) -> str:
    endpoint = _name("endpoint", value)
    try:
        parsed = urlsplit(endpoint)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise ValueError("endpoint must be a valid HTTP or HTTPS Doris endpoint") from error
    if parsed.scheme not in {"http", "https"} or hostname is None:
        raise ValueError("endpoint must be an absolute HTTP or HTTPS Doris endpoint")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError(
            "endpoint must not contain credentials, query parameters, or fragments; use password=EnvironmentSecret(...)"
        )
    if parsed.path not in {"", "/"}:
        raise ValueError("endpoint must not contain a path")
    return endpoint.rstrip("/")


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0 or value > _MAX_INT64:
        raise ValueError(f"{name} must be a positive signed 64-bit integer")
    return value


def _field_mapping(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("field_mapping must be a mapping or None")
    normalized: list[tuple[str, str]] = []
    for source, target in value.items():
        normalized.append(
            (
                _name("field_mapping source", source),
                _doris_column("field_mapping target", target),
            )
        )
    return tuple(normalized)


def _vector_dimensions(value: object) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("vector_dimensions must be a mapping or None")
    normalized: list[tuple[str, int]] = []
    for source, dimension in value.items():
        normalized_dimension = _positive_int("vector dimension", dimension)
        if normalized_dimension > _MAX_INT32:
            raise ValueError("vector dimension must fit in a signed 32-bit Arrow ListArray offset")
        normalized.append((_name("vector_dimensions source", source), normalized_dimension))
    return tuple(normalized)


def _trusted_redirect_hosts(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("trusted_redirect_hosts must be a sequence of host names")
    normalized: list[str] = []
    for item in value:
        host = _name("trusted redirect host", item)
        try:
            parsed = urlsplit(f"//{host}")
            parsed_host = parsed.hostname
            parsed_port = parsed.port
        except ValueError as error:
            raise ValueError("trusted redirect hosts must be bare host names or IP addresses") from error
        if parsed_host is None or parsed_port is not None or parsed.username is not None or parsed.password is not None:
            raise ValueError("trusted redirect hosts must be bare host names or IP addresses")
        if parsed.path not in {"", host} or parsed.query or parsed.fragment:
            raise ValueError("trusted redirect hosts must be bare host names or IP addresses")
        normalized.append(parsed_host.casefold())
    if len(set(normalized)) != len(normalized):
        raise ValueError("trusted_redirect_hosts must not contain duplicates")
    return tuple(normalized)


def _load_aiohttp() -> ModuleType:
    try:
        import aiohttp  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ModuleNotFoundError as error:
        if error.name != "aiohttp":
            raise
        raise ImportError("DorisStreamLoadSink requires aiohttp; install vane-ai[doris]") from error
    return aiohttp


@dataclass(frozen=True)
class _FieldBinding:
    source_name: str
    target_name: str
    nullable: bool
    vector_dimension: int | None


@dataclass(frozen=True)
class _HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes


async def _body_chunks(body: pa.Buffer) -> AsyncIterator[memoryview]:
    """Yield bounded zero-copy views for one HTTP request body."""

    view = memoryview(body)
    for offset in range(0, len(view), _HTTP_BODY_CHUNK_BYTES):
        yield view[offset : offset + _HTTP_BODY_CHUNK_BYTES]


class _AioHttpTransport:
    """Synchronous worker facade over aiohttp's real 100-continue support."""

    def __init__(self, user: str, password: str, timeout: int) -> None:
        aiohttp = _load_aiohttp()
        self._aiohttp = aiohttp
        self._loop = asyncio.new_event_loop()
        self._session: Any | None = None
        self._used_https = False
        try:
            self._session = self._loop.run_until_complete(self._open(user, password, timeout))
        except BaseException:
            self._loop.close()
            raise

    async def _open(self, user: str, password: str, timeout: int) -> Any:
        session = self._aiohttp.ClientSession(
            headers={
                "Authorization": self._aiohttp.encode_basic_auth(user, password, encoding="utf-8"),
            },
            auto_decompress=False,
            skip_auto_headers={"Accept-Encoding"},
            timeout=self._aiohttp.ClientTimeout(
                total=float(timeout + _HTTP_TIMEOUT_GRACE_SECONDS),
                sock_connect=float(_HTTP_CONNECT_TIMEOUT_SECONDS),
            ),
            trust_env=False,
        )
        # aiohttp retries PUT once when a persistent connection fails. A
        # Stream Load may already have committed at that point, so even this
        # transport-level retry must remain disabled.
        session._retry_connection = False
        return session

    async def _put(self, url: str, headers: Mapping[str, str], body: pa.Buffer) -> _HttpResponse:
        session = self._session
        if session is None:
            raise RuntimeError("Doris HTTP transport is closed")
        async with session.put(
            url,
            headers=headers,
            # A fresh async iterator makes the in-memory body replayable for
            # the FE and BE requests without enqueueing one giant BytesPayload.
            data=_body_chunks(body),
            allow_redirects=False,
            expect100=True,
        ) as response:
            response_body = bytearray()
            async for chunk in response.content.iter_any():
                response_body.extend(chunk)
                if len(response_body) > _MAX_RESPONSE_BYTES:
                    raise RuntimeError("Doris Stream Load response exceeds 1 MiB")
            return _HttpResponse(response.status, dict(response.headers), bytes(response_body))

    def put(self, url: str, headers: Mapping[str, str], body: pa.Buffer) -> _HttpResponse:
        if urlsplit(url).scheme == "https":
            self._used_https = True
        return self._loop.run_until_complete(self._put(url, headers, body))

    async def _close(self, session: Any) -> None:
        await session.close()
        await asyncio.sleep(_HTTPS_SHUTDOWN_SECONDS if self._used_https else 0)

    def close(self) -> None:
        session = self._session
        if session is None:
            return
        self._loop.run_until_complete(self._close(session))
        self._loop.close()
        self._session = None


def _open_http_transport(user: str, password: str, timeout: int) -> _AioHttpTransport:
    return _AioHttpTransport(user, password, timeout)


class DorisStreamLoadSink(DataSink):
    """Write relation batches with Apache Doris Arrow Stream Load.

    ``field_mapping`` maps input Arrow field names to Doris columns. Unmapped
    input fields retain their names, and every resulting target name must be
    unique. ``vector_dimensions`` maps input fields to the exact dimensions of
    Doris ``ARRAY<FLOAT>`` vector columns. Declared vector columns must contain
    non-null, finite float32 values and are encoded as Arrow ``ListArray``
    columns without converting rows to Python objects.

    The endpoint may address an FE or BE. FE redirects are followed only when
    the destination host is the endpoint host or appears in
    ``trusted_redirect_hosts``; credentials are never forwarded elsewhere.
    HTTPS redirects may not downgrade to HTTP.

    ``max_batch_bytes`` limits input Arrow data and ``max_request_bytes`` is a
    separate hard limit on the encoded IPC request. Peak worker memory includes
    both buffers. Vane full-operation and HTTP-body retries remain disabled
    because the current DataSink batch contract has no replay-stable batch
    identity. If a connection fails after upload, the reported outcome is
    UNKNOWN and its Doris label must be inspected before any manual retry.
    """

    def __init__(
        self,
        database: str,
        table: str,
        *,
        endpoint: str | EnvironmentSecret,
        user: str = "root",
        password: EnvironmentSecret | None = None,
        field_mapping: Mapping[str, str] | None = None,
        vector_dimensions: Mapping[str, int] | None = None,
        trusted_redirect_hosts: Sequence[str] = (),
        worker_count: int = 1,
        max_batch_rows: int = _DEFAULT_MAX_BATCH_ROWS,
        max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
        max_request_bytes: int = _DEFAULT_MAX_REQUEST_BYTES,
        send_batch_parallelism: int = _DEFAULT_SEND_BATCH_PARALLELISM,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.database = _doris_object("database", database)
        self.table = _doris_object("table", table)
        self.endpoint: str | EnvironmentSecret
        if isinstance(endpoint, EnvironmentSecret):
            self.endpoint = endpoint
        else:
            self.endpoint = _endpoint(endpoint)
        self.user = _name("user", user)
        if ":" in self.user:
            raise ValueError("user must not contain a colon because HTTP Basic Auth cannot encode it")
        if password is not None and not isinstance(password, EnvironmentSecret):
            raise TypeError("password must be an EnvironmentSecret or None")
        self._password = password
        self._field_mapping = _field_mapping(field_mapping)
        self._vector_dimensions = _vector_dimensions(vector_dimensions)
        self._trusted_redirect_hosts = _trusted_redirect_hosts(trusted_redirect_hosts)
        self.worker_count = _positive_int("worker_count", worker_count)
        self.max_batch_rows = _positive_int("max_batch_rows", max_batch_rows)
        self.max_batch_bytes = _positive_int("max_batch_bytes", max_batch_bytes)
        self.max_request_bytes = _positive_int("max_request_bytes", max_request_bytes)
        self.send_batch_parallelism = _positive_int("send_batch_parallelism", send_batch_parallelism)
        self.timeout = _positive_int("timeout", timeout)
        if self.timeout > _MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout must be at most {_MAX_TIMEOUT_SECONDS} seconds for Doris Stream Load")

    def bind(self, schema: pa.Schema) -> BoundDataSink:
        if not isinstance(schema, pa.Schema):
            raise TypeError("schema must be pyarrow.Schema")
        if not schema:
            raise ValueError("DorisStreamLoadSink requires at least one input column")
        if len(set(schema.names)) != len(schema.names):
            raise ValueError("DorisStreamLoadSink requires unique input column names")

        mapping = dict(self._field_mapping)
        if len(mapping) != len(self._field_mapping):
            raise ValueError("field_mapping sources must be unique")
        vector_dimensions = dict(self._vector_dimensions)
        if len(vector_dimensions) != len(self._vector_dimensions):
            raise ValueError("vector_dimensions sources must be unique")
        unknown_mapping = set(mapping).difference(schema.names)
        if unknown_mapping:
            raise ValueError(f"field_mapping contains unknown input columns: {sorted(unknown_mapping)!r}")
        unknown_vectors = set(vector_dimensions).difference(schema.names)
        if unknown_vectors:
            raise ValueError(f"vector_dimensions contains unknown input columns: {sorted(unknown_vectors)!r}")

        fields: list[_FieldBinding] = []
        for field in schema:
            target_name = _doris_column("Doris target column", mapping.get(field.name, field.name))
            vector_dimension = vector_dimensions.get(field.name)
            if vector_dimension is not None:
                data_type = field.type
                if not (
                    pa.types.is_list(data_type)
                    or pa.types.is_large_list(data_type)
                    or pa.types.is_fixed_size_list(data_type)
                ) or not pa.types.is_float32(data_type.value_type):
                    raise ValueError(
                        f"Doris vector source field {field.name!r} must be an Arrow list, large_list, "
                        "or fixed_size_list of float32"
                    )
                if pa.types.is_fixed_size_list(data_type) and data_type.list_size != vector_dimension:
                    raise ValueError(
                        f"Doris vector source field {field.name!r} fixed dimension {data_type.list_size} "
                        f"does not match vector_dimensions value {vector_dimension}"
                    )
            fields.append(
                _FieldBinding(
                    source_name=field.name,
                    target_name=target_name,
                    nullable=field.nullable,
                    vector_dimension=vector_dimension,
                )
            )

        target_names = [field.target_name for field in fields]
        if len({name.casefold() for name in target_names}) != len(target_names):
            raise ValueError("field_mapping must produce case-insensitively unique Doris column names")
        return _BoundDorisStreamLoadSink(self, schema, tuple(fields))

    def _resolve_endpoint(self) -> str:
        endpoint = self.endpoint
        if isinstance(endpoint, EnvironmentSecret):
            return _endpoint(endpoint.resolve())
        return endpoint


class _BoundDorisStreamLoadSink(BoundDataSink):
    def __init__(self, sink: DorisStreamLoadSink, schema: pa.Schema, fields: tuple[_FieldBinding, ...]) -> None:
        self._sink = sink
        self._schema = schema
        self._fields = fields

    @property
    def execution_options(self) -> DataSinkExecutionOptions:
        return DataSinkExecutionOptions(
            worker_count=self._sink.worker_count,
            batch_size=self._sink.max_batch_rows,
            target_max_batch_bytes=self._sink.max_batch_bytes,
            max_retries=0,
        )

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _DorisStreamLoadWorker(self._sink, self._schema, self._fields, context)


class _DorisStreamLoadWorker(DataSinkWorker):
    def __init__(
        self,
        sink: DorisStreamLoadSink,
        schema: pa.Schema,
        fields: tuple[_FieldBinding, ...],
        context: WriteContext,
    ) -> None:
        endpoint = sink._resolve_endpoint()
        endpoint_parts = urlsplit(endpoint)
        assert endpoint_parts.hostname is not None
        trusted_hosts = set(sink._trusted_redirect_hosts)
        trusted_hosts.add(endpoint_parts.hostname.casefold())
        password = "" if sink._password is None else sink._password.resolve()

        self._sink = sink
        self._schema = schema
        self._fields = fields
        self._endpoint_scheme = endpoint_parts.scheme
        self._trusted_hosts = frozenset(trusted_hosts)
        self._load_path = f"/api/{quote(sink.database, safe='')}/{quote(sink.table, safe='')}/_stream_load"
        self._url = f"{endpoint}{self._load_path}"
        self._transport: _AioHttpTransport | None = _open_http_transport(sink.user, password, sink.timeout)
        operation_digest = hashlib.sha256(context.operation_id.encode("utf-8")).hexdigest()[:16]
        self._label_stem = f"{_LABEL_PREFIX}_{operation_digest}_{uuid.uuid4().hex[:8]}"
        self._batch_number = 0
        self._warning_pending = True
        self._base_metadata = {
            "provider": "doris",
            "database": sink.database,
            "table": sink.table,
            "format": "arrow",
        }
        try:
            WriteResult(
                rows_received=0,
                rows_affected=0,
                metadata=self._base_metadata,
                warnings=(_PARTIAL_VISIBILITY_WARNING,),
            )
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    try:
                        add_note(f"Doris HTTP client cleanup also failed: {type(close_error).__name__}")
                    except BaseException:
                        pass
            raise

    def _transport_or_raise(self) -> _AioHttpTransport:
        if self._transport is None:
            raise RuntimeError("DorisStreamLoadSink worker is closed")
        return self._transport

    def _next_label(self) -> str:
        self._batch_number += 1
        label = f"{self._label_stem}_{self._batch_number:x}"
        if len(label) > 128 or _LABEL_PATTERN.fullmatch(label) is None:
            raise AssertionError("generated Doris Stream Load label is invalid")
        return label

    def _validate_table(self, table: pa.Table) -> tuple[int, int]:
        if not isinstance(table, pa.Table):
            raise TypeError(f"DorisStreamLoadSink expected pyarrow.Table, got {type(table).__name__}")
        if len(table.schema) != len(self._schema) or any(
            batch_field.name != bound_field.name or batch_field.type != bound_field.type
            for batch_field, bound_field in zip(table.schema, self._schema, strict=True)
        ):
            raise ValueError("Doris Stream Load batch schema does not match the bound input schema")
        row_count = table.num_rows
        batch_bytes = table.nbytes
        if row_count > self._sink.max_batch_rows:
            raise ValueError("Doris Stream Load batch exceeds max_batch_rows")
        if batch_bytes > self._sink.max_batch_bytes:
            raise ValueError("Doris Stream Load batch exceeds max_batch_bytes")
        return row_count, batch_bytes

    def _vector_column(self, column: pa.ChunkedArray, binding: _FieldBinding) -> pa.ChunkedArray:
        dimension = binding.vector_dimension
        assert dimension is not None
        chunks: list[pa.Array] = []
        for chunk in column.chunks:
            if chunk.null_count:
                raise ValueError(f"Doris vector source field {binding.source_name!r} must not contain null vectors")
            if not pa.types.is_fixed_size_list(chunk.type):
                lengths = pc.list_value_length(chunk)
                wrong_length = pc.any(pc.not_equal(lengths, pa.scalar(dimension, type=lengths.type)))
                if wrong_length.as_py() is True:
                    raise ValueError(
                        f"Doris vector source field {binding.source_name!r} contains a vector with an invalid dimension"
                    )
            values = pc.list_flatten(chunk)
            if values.null_count:
                raise ValueError(f"Doris vector source field {binding.source_name!r} must not contain null elements")
            finite = pc.all(pc.is_finite(values))
            if finite.as_py() is False:
                raise ValueError(
                    f"Doris vector source field {binding.source_name!r} must contain only finite float32 values"
                )
            nested_count = len(chunk) * dimension
            if nested_count > _MAX_INT32:
                raise ValueError(f"Doris vector source field {binding.source_name!r} exceeds Arrow ListArray limits")
            offsets = pa.array(
                np.arange(0, nested_count + 1, dimension, dtype=np.int32),
                type=pa.int32(),
            )
            chunks.append(pa.ListArray.from_arrays(offsets, values))
        return pa.chunked_array(chunks, type=pa.list_(pa.float32()))

    def _wire_table(self, table: pa.Table) -> pa.Table:
        arrays: list[pa.ChunkedArray] = []
        fields: list[pa.Field] = []
        for index, binding in enumerate(self._fields):
            column = table.column(index)
            if binding.vector_dimension is not None:
                column = self._vector_column(column, binding)
            arrays.append(column)
            fields.append(
                pa.field(
                    binding.target_name,
                    column.type,
                    nullable=binding.nullable,
                )
            )
        schema = pa.schema(fields)
        return pa.Table.from_arrays(arrays, schema=schema)

    def _arrow_body(self, table: pa.Table) -> pa.Buffer:
        output = pa.BufferOutputStream()
        with pa.ipc.new_stream(output, table.schema) as writer:
            writer.write_table(table)
        body = output.getvalue()
        if body.size > self._sink.max_request_bytes:
            raise ValueError(
                "Doris Arrow IPC request exceeds max_request_bytes: "
                f"wire_bytes={body.size}, max_request_bytes={self._sink.max_request_bytes}"
            )
        return body

    def _headers(self, label: str, body_size: int) -> dict[str, str]:
        return {
            "Expect": "100-continue",
            "Content-Type": "application/vnd.apache.arrow.stream",
            "Content-Length": str(body_size),
            "format": "arrow",
            "columns": ",".join(_header_identifier(field.target_name) for field in self._fields),
            "label": label,
            "strict_mode": "true",
            "max_filter_ratio": "0",
            "send_batch_parallelism": str(self._sink.send_batch_parallelism),
            "timeout": str(self._sink.timeout),
        }

    def _request(self, url: str, headers: Mapping[str, str], body: pa.Buffer) -> _HttpResponse:
        return self._transport_or_raise().put(url, headers, body)

    def _redirect_url(self, source_url: str, response: _HttpResponse) -> str:
        location = next(
            (value for name, value in response.headers.items() if name.casefold() == "location"),
            None,
        )
        if location is None:
            raise RuntimeError("Doris FE redirect did not include a Location header")
        redirect_url = urljoin(source_url, location)
        try:
            parsed = urlsplit(redirect_url)
            hostname = parsed.hostname
            _ = parsed.port
        except ValueError as error:
            raise RuntimeError("Doris FE returned an invalid redirect URL") from error
        if parsed.scheme not in {"http", "https"} or hostname is None:
            raise RuntimeError("Doris FE returned a non-HTTP redirect URL")
        if parsed.query or parsed.fragment:
            raise RuntimeError("Doris FE redirect URL must not contain a query or a fragment")
        if parsed.scheme != self._endpoint_scheme:
            raise RuntimeError("Doris FE redirect must preserve the endpoint scheme")
        if hostname.casefold() not in self._trusted_hosts:
            raise RuntimeError(f"Doris FE redirected to untrusted host {hostname!r}; add it to trusted_redirect_hosts")
        if parsed.path != self._load_path:
            raise RuntimeError("Doris FE redirect changed the Stream Load path")
        # Current Doris FEs copy the Basic Auth userinfo into Location. Never
        # pass those URL credentials to the BE: the session attaches the
        # configured Basic Auth after the destination checks above.
        authority = parsed.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))

    def _put_once(self, headers: Mapping[str, str], body: pa.Buffer) -> _HttpResponse:
        response = self._request(self._url, headers, body)
        if response.status_code != _REDIRECT_STATUS:
            return response
        redirect_url = self._redirect_url(self._url, response)
        redirected = self._request(redirect_url, headers, body)
        if redirected.status_code == _REDIRECT_STATUS:
            raise RuntimeError("Doris Stream Load returned more than one redirect")
        return redirected

    def _response_payload(self, response: _HttpResponse) -> Mapping[str, Any]:
        if len(response.body) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Doris Stream Load response exceeds 1 MiB")
        if response.status_code < 200 or response.status_code >= 300:
            detail = response.body.decode("utf-8", errors="replace").strip()
            if len(detail) > 512:
                detail = f"{detail[:512]}..."
            raise RuntimeError(f"Doris Stream Load HTTP {response.status_code}: {detail or 'empty response'}")
        try:
            payload = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Doris Stream Load returned invalid JSON") from error
        if not isinstance(payload, Mapping):
            raise RuntimeError("Doris Stream Load returned a non-object JSON response")
        return payload

    def _put(self, label: str, body: pa.Buffer) -> Mapping[str, Any]:
        headers = self._headers(label, body.size)
        try:
            return self._response_payload(self._put_once(headers, body))
        except Exception as error:
            raise RuntimeError(
                f"Doris Stream Load batch label {label!r} did not produce an accepted terminal response: "
                f"{type(error).__name__}: {error}; "
                "inspect that label in Doris before retrying with a new label"
            ) from error

    @staticmethod
    def _response_int(payload: Mapping[str, Any], name: str) -> int:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > _MAX_INT64:
            raise RuntimeError(f"Doris Stream Load returned an invalid {name}")
        return value

    def _applied_result(
        self,
        *,
        payload: Mapping[str, Any],
        label: str,
        row_count: int,
        batch_bytes: int,
        body_size: int,
    ) -> WriteResult:
        status = payload.get("Status")
        response_label = payload.get("Label")
        if response_label != label:
            raise RuntimeError("Doris Stream Load returned a different batch label")

        warnings: list[str] = []
        metadata: dict[str, Any] = {
            **self._base_metadata,
            "label": label,
            "status": status,
            "wire_bytes": body_size,
        }
        if status in {_SUCCESS_STATUS, _PUBLISH_TIMEOUT_STATUS}:
            total_rows = self._response_int(payload, "NumberTotalRows")
            loaded_rows = self._response_int(payload, "NumberLoadedRows")
            filtered_rows = self._response_int(payload, "NumberFilteredRows")
            unselected_rows = self._response_int(payload, "NumberUnselectedRows")
            if total_rows != row_count or loaded_rows != row_count or filtered_rows != 0 or unselected_rows != 0:
                raise RuntimeError(
                    "Doris Stream Load row counts do not match the Arrow batch: "
                    f"total={total_rows}, loaded={loaded_rows}, filtered={filtered_rows}, "
                    f"unselected={unselected_rows}, expected={row_count}"
                )
            txn_id = self._response_int(payload, "TxnId")
            load_bytes = self._response_int(payload, "LoadBytes")
            load_time_ms = self._response_int(payload, "LoadTimeMs")
            metadata.update({"txn_id": txn_id, "load_bytes": load_bytes, "load_time_ms": load_time_ms})
            if status == _PUBLISH_TIMEOUT_STATUS:
                warnings.append(_PUBLISH_TIMEOUT_WARNING)
        else:
            message = payload.get("Message")
            detail = message if isinstance(message, str) and message else "no error message"
            if len(detail) > 512:
                detail = f"{detail[:512]}..."
            raise RuntimeError(f"Doris Stream Load was not applied: status={status!r}, message={detail}")

        if self._warning_pending:
            warnings.insert(0, _PARTIAL_VISIBILITY_WARNING)
        result = WriteResult(
            rows_received=row_count,
            rows_affected=row_count,
            bytes_received=batch_bytes,
            metadata=metadata,
            warnings=tuple(warnings),
        )
        self._warning_pending = False
        return result

    def write(self, table: pa.Table) -> WriteResult:
        row_count, batch_bytes = self._validate_table(table)
        if row_count == 0:
            return WriteResult(
                rows_received=0,
                rows_affected=0,
                bytes_received=batch_bytes,
                metadata=self._base_metadata,
            )
        wire_table = self._wire_table(table)
        body = self._arrow_body(wire_table)
        label = self._next_label()
        payload = self._put(label, body)
        try:
            return self._applied_result(
                payload=payload,
                label=label,
                row_count=row_count,
                batch_bytes=batch_bytes,
                body_size=body.size,
            )
        except Exception as error:
            raise RuntimeError(
                f"Doris Stream Load batch label {label!r} returned a terminal response that Vane "
                f"could not accept: {type(error).__name__}: {error}; inspect that label before "
                "submitting new data"
            ) from error

    def abort(self, _error: BaseException) -> None:
        self.close()

    def close(self) -> None:
        transport = self._transport
        if transport is None:
            return
        transport.close()
        self._transport = None


__all__ = ["DorisStreamLoadSink"]
