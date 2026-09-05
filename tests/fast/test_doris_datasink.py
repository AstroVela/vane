# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import json
import os
import uuid
from collections.abc import AsyncIterable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import cloudpickle
import pyarrow as pa
import pytest

import vane
import vane.datasink.doris as doris
from vane import DorisStreamLoadSink, EnvironmentSecret
from vane.datasink import BoundDataSink, WriteContext

_REAL_LOAD_AIOHTTP = doris._load_aiohttp


class _TransportError(Exception):
    pass


@dataclass
class _Response:
    status_code: int
    body: bytes = b""
    headers: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = {}


@dataclass(frozen=True)
class _Call:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes


_ResponseFactory = Callable[[_Call], _Response]


class _Transport:
    responses: list[_Response | _ResponseFactory | BaseException] = []
    instances: list[_Transport] = []
    close_error: BaseException | None = None

    def __init__(self, user: str, password: str, timeout: int) -> None:
        self.options = {"user": user, "password": password, "timeout": timeout}
        self.calls: list[_Call] = []
        self.close_calls = 0
        type(self).instances.append(self)

    def put(self, url: str, headers: Mapping[str, str], body: pa.Buffer) -> _Response:
        call = _Call("PUT", url, dict(headers), body.to_pybytes())
        self.calls.append(call)
        if not type(self).responses:
            raise AssertionError("fake HTTP response queue is empty")
        response = type(self).responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response(call) if callable(response) else response

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _AsyncChunks:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)

    def iter_any(self) -> _AsyncChunks:
        return self

    def __aiter__(self) -> _AsyncChunks:
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._chunks)
        except StopIteration as error:
            raise StopAsyncIteration from error


class _AsyncResponse:
    def __init__(self) -> None:
        self.status = 200
        self.headers = {"content-type": "application/json"}
        self.content = _AsyncChunks([b'{"Status":', b'"Success"}'])


class _AsyncResponseContext:
    def __init__(self, session: _AsyncSession, options: Mapping[str, object]) -> None:
        self._session = session
        self._options = options

    async def __aenter__(self) -> _AsyncResponse:
        chunks: list[bytes] = []
        body = cast(AsyncIterable[bytes | bytearray | memoryview], self._options["data"])
        async for chunk in body:
            chunks.append(bytes(chunk))
        self._session.written_chunks.append(chunks)
        return _AsyncResponse()

    async def __aexit__(self, *_args: object) -> None:
        return None


class _AsyncSession:
    def __init__(self) -> None:
        self.put_options: list[dict[str, object]] = []
        self.written_chunks: list[list[bytes]] = []
        self.closed = False
        self._retry_connection = True

    def put(self, url: str, **options: object) -> _AsyncResponseContext:
        request_options = {"url": url, **options}
        self.put_options.append(request_options)
        return _AsyncResponseContext(self, request_options)

    async def close(self) -> None:
        self.closed = True


class _AioHttpModule:
    def __init__(self) -> None:
        self.session = _AsyncSession()
        self.session_options: dict[str, object] | None = None

    @staticmethod
    def encode_basic_auth(user: str, password: str, *, encoding: str) -> tuple[str, str, str]:
        return user, password, encoding

    @staticmethod
    def ClientTimeout(*, total: float, sock_connect: float) -> tuple[float, float]:
        return total, sock_connect

    def ClientSession(self, **options: object) -> _AsyncSession:
        self.session_options = options
        return self.session


@pytest.fixture(autouse=True)
def _fake_http_transport(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("external_service") is not None:
        return
    _Transport.responses = []
    _Transport.instances = []
    _Transport.close_error = None
    monkeypatch.setattr(doris, "_open_http_transport", _Transport)


def _schema(*, vector_type: pa.DataType | None = None) -> pa.Schema:
    return pa.schema(
        [
            ("source_id", pa.int64()),
            ("embedding", pa.list_(pa.float32(), 3) if vector_type is None else vector_type),
            ("source_title", pa.string()),
        ]
    )


def _destination_schema(
    *,
    id_type: pa.DataType | None = None,
    vector_type: pa.DataType | None = None,
    title_type: pa.DataType | None = None,
) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int32() if id_type is None else id_type, nullable=False),
            pa.field(
                "embedding",
                pa.list_(pa.float32()) if vector_type is None else vector_type,
                nullable=False,
            ),
            pa.field("title", pa.string() if title_type is None else title_type, nullable=False),
        ]
    )


def _table(
    *,
    vectors: list[list[float] | None] | None = None,
    schema: pa.Schema | None = None,
) -> pa.Table:
    values = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] if vectors is None else vectors
    if schema is None:
        return pa.table(
            {
                "source_id": [1, 2],
                "embedding": pa.array(values, type=pa.list_(pa.float32(), 3)),
                "source_title": ["one", "two"],
            }
        )
    return pa.table(
        {
            "source_id": [1, 2],
            "embedding": values,
            "source_title": ["one", "two"],
        },
        schema=schema,
    )


def _sink(**overrides: object) -> DorisStreamLoadSink:
    options: dict[str, object] = {
        "endpoint": "http://fe.example:8030",
        "destination_schema": _destination_schema(),
        "field_mapping": {"source_id": "id", "source_title": "title"},
        "vector_dimensions": {"embedding": 3},
        "trusted_redirect_hosts": ("be.example",),
        "timeout": 5,
    }
    options.update(overrides)
    return DorisStreamLoadSink("analytics", "items", **options)  # type: ignore[arg-type]


def _bound(*, schema: pa.Schema | None = None, **overrides: object) -> BoundDataSink:
    return _sink(**overrides).bind(_schema() if schema is None else schema)


def _worker(*, schema: pa.Schema | None = None, **overrides: object) -> Any:
    return _bound(schema=schema, **overrides).open_worker(WriteContext("doris-test-operation"))


def _success(call: _Call, *, status: str = "Success", **overrides: object) -> _Response:
    table = pa.ipc.open_stream(call.body).read_all()
    payload: dict[str, object] = {
        "TxnId": 17,
        "Label": call.headers["label"],
        "Status": status,
        "Message": "OK",
        "NumberTotalRows": table.num_rows,
        "NumberLoadedRows": table.num_rows,
        "NumberFilteredRows": 0,
        "NumberUnselectedRows": 0,
        "LoadBytes": len(call.body),
        "LoadTimeMs": 12,
    }
    payload.update(overrides)
    return _Response(200, json.dumps(payload).encode())


def test_doris_sink_is_public_without_importing_aiohttp() -> None:
    assert vane.DorisStreamLoadSink is DorisStreamLoadSink
    assert doris.__all__ == ["DorisStreamLoadSink"]


def test_doris_sink_reports_the_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_aiohttp(name: str, *args: object, **kwargs: object) -> Any:
        if name == "aiohttp":
            raise ModuleNotFoundError("No module named 'aiohttp'", name="aiohttp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_aiohttp)
    with pytest.raises(ImportError, match=r"vane-ai\[doris\]"):
        _REAL_LOAD_AIOHTTP()


def test_doris_transport_streams_replayable_chunks_with_real_expect_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aiohttp = _AioHttpModule()
    monkeypatch.setattr(doris, "_load_aiohttp", lambda: aiohttp)
    transport = doris._AioHttpTransport("alice", "secret", 17)
    body = b"a" * (2 * doris._HTTP_BODY_CHUNK_BYTES + 7)

    first_response = transport.put(
        "http://fe.example:8030/api/db/table/_stream_load",
        {"Content-Length": str(len(body))},
        pa.py_buffer(body),
    )
    second_response = transport.put(
        "http://be.example:8040/api/db/table/_stream_load",
        {"Content-Length": str(len(body))},
        pa.py_buffer(body),
    )
    transport.close()

    assert aiohttp.session_options == {
        "headers": {"Authorization": ("alice", "secret", "utf-8")},
        "auto_decompress": False,
        "skip_auto_headers": {"Accept-Encoding"},
        "timeout": (47.0, 30.0),
        "trust_env": False,
    }
    assert aiohttp.session._retry_connection is False
    assert len(aiohttp.session.put_options) == 2
    assert all(options["allow_redirects"] is False for options in aiohttp.session.put_options)
    assert all(options["expect100"] is True for options in aiohttp.session.put_options)
    assert [b"".join(chunks) for chunks in aiohttp.session.written_chunks] == [body, body]
    assert [[len(chunk) for chunk in chunks] for chunks in aiohttp.session.written_chunks] == [
        [doris._HTTP_BODY_CHUNK_BYTES, doris._HTTP_BODY_CHUNK_BYTES, 7],
        [doris._HTTP_BODY_CHUNK_BYTES, doris._HTTP_BODY_CHUNK_BYTES, 7],
    ]
    assert first_response.status_code == second_response.status_code == 200
    assert first_response.body == second_response.body == b'{"Status":"Success"}'
    assert aiohttp.session.closed is True


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"database": ""}, ValueError, "database"),
        ({"table": "bad`table"}, ValueError, "ASCII Doris identifier"),
        ({"database": ".."}, ValueError, "beginning with a letter"),
        ({"endpoint": "doris.example:8030"}, ValueError, "HTTP or HTTPS"),
        ({"endpoint": "http://doris.example/path"}, ValueError, "path"),
        ({"endpoint": "http://user:secret@doris.example"}, ValueError, "credentials"),
        ({"endpoint": "http://doris.example?token=secret"}, ValueError, "query parameters"),
        ({"user": ""}, ValueError, "user"),
        ({"user": "domain:alice"}, ValueError, "colon"),
        ({"password": "plain-text"}, TypeError, "EnvironmentSecret"),
        ({"destination_schema": object()}, TypeError, "pyarrow.Schema"),
        ({"destination_schema": pa.schema([])}, ValueError, "at least one"),
        (
            {"destination_schema": pa.schema([("value", pa.int32()), ("VALUE", pa.int32())])},
            ValueError,
            "case-insensitively unique",
        ),
        ({"destination_schema": pa.schema([("created_at", pa.timestamp("us"))])}, ValueError, "temporal"),
        (
            {"destination_schema": pa.schema([("events", pa.list_(pa.timestamp("us")))])},
            ValueError,
            "temporal",
        ),
        ({"destination_schema": pa.schema([("payload", pa.binary())])}, ValueError, "unsupported Arrow type"),
        ({"field_mapping": []}, TypeError, "field_mapping"),
        ({"vector_dimensions": []}, TypeError, "vector_dimensions"),
        ({"vector_dimensions": {"embedding": 1 << 31}}, ValueError, "signed 32-bit"),
        ({"trusted_redirect_hosts": "be.example"}, TypeError, "sequence"),
        ({"trusted_redirect_hosts": ("http://be.example",)}, ValueError, "bare host"),
        ({"trusted_redirect_hosts": ("be.example:8040",)}, ValueError, "bare host"),
        ({"trusted_redirect_hosts": ("be.example", "BE.EXAMPLE")}, ValueError, "duplicates"),
        ({"worker_count": True}, TypeError, "worker_count"),
        ({"worker_count": 0}, ValueError, "worker_count"),
        ({"max_batch_rows": -1}, ValueError, "max_batch_rows"),
        ({"max_batch_bytes": 1.5}, TypeError, "max_batch_bytes"),
        ({"max_request_bytes": 1.5}, TypeError, "max_request_bytes"),
        ({"send_batch_parallelism": 0}, ValueError, "send_batch_parallelism"),
        ({"timeout": 0}, ValueError, "timeout"),
        ({"timeout": 259_201}, ValueError, "259200"),
    ],
)
def test_doris_sink_validates_constructor(overrides: dict[str, object], error: type[Exception], message: str) -> None:
    options: dict[str, object] = {
        "database": "analytics",
        "table": "items",
        "endpoint": "http://doris.example:8030",
        "destination_schema": _destination_schema(),
    }
    options.update(overrides)
    with pytest.raises(error, match=message):
        DorisStreamLoadSink(**options)  # type: ignore[arg-type]


def test_doris_sink_binds_mapping_vectors_and_execution_options() -> None:
    sink = _sink(worker_count=3, max_batch_rows=17, max_batch_bytes=2_048, send_batch_parallelism=4)
    assert sink.destination_schema == _destination_schema()
    with pytest.raises(AttributeError):
        setattr(sink, "destination_schema", pa.schema([("other", pa.int32())]))
    bound = sink.bind(_schema())

    assert isinstance(bound, BoundDataSink)
    assert bound.execution_options.worker_count == 3
    assert bound.execution_options.batch_size == 17
    assert bound.execution_options.target_max_batch_bytes == 2_048
    assert bound.execution_options.max_retries == 0


def test_doris_sink_rejects_invalid_bound_schema_and_mappings() -> None:
    with pytest.raises(TypeError, match="pyarrow.Schema"):
        _sink().bind(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        _sink().bind(pa.schema([]))
    with pytest.raises(ValueError, match="unique input"):
        _sink(field_mapping=None, vector_dimensions=None).bind(pa.schema([("id", pa.int64()), ("id", pa.int64())]))
    with pytest.raises(ValueError, match="unknown input"):
        _sink(field_mapping={"missing": "id"}).bind(_schema())
    with pytest.raises(ValueError, match="unknown input"):
        _sink(vector_dimensions={"missing": 3}).bind(_schema())
    with pytest.raises(ValueError, match="case-insensitively unique"):
        _sink(field_mapping={"source_id": "VALUE", "source_title": "value"}).bind(_schema())
    with pytest.raises(ValueError, match="ASCII"):
        _sink(field_mapping={"source_title": "标题"}).bind(_schema())
    with pytest.raises(ValueError, match="ASCII"):
        _sink(field_mapping={"source_title": "title=value"}).bind(_schema())
    with pytest.raises(ValueError, match="exactly match destination_schema"):
        _sink(
            destination_schema=pa.schema(
                [("title", pa.string()), ("embedding", pa.list_(pa.float32())), ("id", pa.int32())]
            )
        ).bind(_schema())
    with pytest.raises(ValueError, match="float32"):
        _sink().bind(_schema(vector_type=pa.list_(pa.float64())))
    with pytest.raises(ValueError, match="fixed dimension"):
        _sink().bind(_schema(vector_type=pa.list_(pa.float32(), 2)))
    with pytest.raises(ValueError, match=r"destination field 'embedding'.*list<float32>"):
        _sink(destination_schema=_destination_schema(vector_type=pa.list_(pa.float64()))).bind(_schema())


def test_doris_sink_rejects_temporal_source_types_during_bind() -> None:
    schema = pa.schema(
        [
            ("source_id", pa.int64()),
            ("embedding", pa.list_(pa.float32(), 3)),
            ("source_title", pa.list_(pa.timestamp("us"))),
        ]
    )
    with pytest.raises(ValueError, match=r"source field 'source_title'.*temporal"):
        _sink().bind(schema)


def test_doris_sink_writes_arrow_stream_without_row_materialization() -> None:
    _Transport.responses = [_success, _success]
    worker = _worker()
    table = _table()

    first = worker.write(table)
    second = worker.write(table)

    transport = _Transport.instances[0]
    assert transport.options == {"user": "root", "password": "", "timeout": 5}
    assert len(transport.calls) == 2
    call = transport.calls[0]
    assert call.method == "PUT"
    assert call.url == "http://fe.example:8030/api/analytics/items/_stream_load"
    assert call.headers["Expect"] == "100-continue"
    assert call.headers["Content-Type"] == "application/vnd.apache.arrow.stream"
    assert call.headers["Content-Length"] == str(len(call.body))
    assert call.headers["format"] == "arrow"
    assert call.headers["columns"] == "`id`,`embedding`,`title`"
    assert call.headers["strict_mode"] == "true"
    assert call.headers["max_filter_ratio"] == "0"
    assert call.headers["send_batch_parallelism"] == "1"
    assert call.headers["timeout"] == "5"
    assert call.headers["label"].startswith("vane_")
    assert transport.calls[1].headers["label"] != call.headers["label"]

    decoded = pa.ipc.open_stream(call.body).read_all()
    assert decoded.schema == _destination_schema()
    assert decoded.schema.field("id").type == pa.int32()
    assert decoded.schema.field("embedding").type == pa.list_(pa.float32())
    assert decoded.column("id").to_pylist() == [1, 2]
    vectors = decoded.column("embedding").to_pylist()
    assert vectors[0] == pytest.approx([0.1, 0.2, 0.3])
    assert vectors[1] == pytest.approx([0.4, 0.5, 0.6])
    assert decoded.column("title").to_pylist() == ["one", "two"]

    assert first.rows_received == first.rows_affected == 2
    assert first.bytes_received == table.nbytes
    assert first.metadata["provider"] == "doris"
    assert first.metadata["database"] == "analytics"
    assert first.metadata["table"] == "items"
    assert first.metadata["format"] == "arrow"
    assert first.metadata["status"] == "Success"
    assert first.metadata["txn_id"] == 17
    assert first.metadata["wire_bytes"] == len(call.body)
    assert len(first.warnings) == 1
    assert second.warnings == ()


def test_doris_sink_safely_casts_nested_destination_values() -> None:
    source_schema = pa.schema([("values", pa.list_(pa.int64())), ("score", pa.float64())])
    destination_schema = pa.schema(
        [
            pa.field("values", pa.list_(pa.field("item", pa.int32(), nullable=False)), nullable=False),
            pa.field("score", pa.float32(), nullable=False),
        ]
    )
    sink = DorisStreamLoadSink(
        "analytics",
        "items",
        endpoint="http://fe.example:8030",
        destination_schema=destination_schema,
        timeout=5,
    )
    worker = sink.bind(source_schema).open_worker(WriteContext("nested-cast"))
    _Transport.responses = [_success]

    worker.write(pa.table({"values": [[1, 2], [3]], "score": [1.25, 2.5]}, schema=source_schema))

    decoded = pa.ipc.open_stream(_Transport.instances[0].calls[0].body).read_all()
    assert decoded.schema == destination_schema
    assert decoded.column("values").to_pylist() == [[1, 2], [3]]
    assert decoded.column("score").to_pylist() == pytest.approx([1.25, 2.5])


@pytest.mark.parametrize(
    ("column", "target_type", "path"),
    [
        (
            pa.chunked_array([[0.1], [None, float("inf"), 1e40]], type=pa.float64()),
            pa.float32(),
            "score",
        ),
        (
            pa.chunked_array([[float("-inf"), -1e40]], type=pa.float64()),
            pa.float32(),
            "score",
        ),
        (
            pa.chunked_array([[[0.1, None]], [None, [1e40]]], type=pa.list_(pa.float64())),
            pa.list_(pa.float32()),
            r"score\[\]",
        ),
        (
            pa.chunked_array([[[0.1, -1e40]]], type=pa.large_list(pa.float64())),
            pa.list_(pa.float32()),
            r"score\[\]",
        ),
        (
            pa.chunked_array([[[0.1, 1e40]]], type=pa.list_(pa.float64(), 2)),
            pa.list_(pa.float32()),
            r"score\[\]",
        ),
        (
            pa.chunked_array([[[[0.1], None, [-1e40]]]], type=pa.list_(pa.list_(pa.float64()))),
            pa.list_(pa.list_(pa.float32())),
            r"score\[\]\[\]",
        ),
        (
            pa.chunked_array([pa.array([0.1, None, 1e40]).dictionary_encode()]),
            pa.float32(),
            "score",
        ),
    ],
    ids=[
        "scalar-positive",
        "scalar-negative",
        "list",
        "large-list",
        "fixed-list",
        "nested-list",
        "dictionary",
    ],
)
def test_doris_sink_rejects_float_narrowing_overflow_before_http(
    column: pa.ChunkedArray, target_type: pa.DataType, path: str
) -> None:
    table = pa.table({"score": column})
    sink = DorisStreamLoadSink(
        "analytics",
        "items",
        endpoint="http://fe.example:8030",
        destination_schema=pa.schema([("score", target_type)]),
    )
    worker = sink.bind(table.schema).open_worker(WriteContext("float-overflow"))
    _Transport.responses = [_success]

    with pytest.raises(ValueError, match=f"destination field '{path}'.*floating-point overflow"):
        worker.write(table)

    assert not _Transport.instances[0].calls
    worker.close()


def test_doris_sink_float_narrowing_preserves_rounding_and_existing_nonfinite_values() -> None:
    max_float32 = float.fromhex("0x1.fffffep+127")
    source = pa.array(
        [1e40, 0.1, None, float("nan"), float("inf"), float("-inf"), max_float32 + 1e30, -max_float32 - 1e30, -1e40],
        type=pa.float64(),
    ).slice(1, 7)
    table = pa.table({"score": pa.chunked_array([source.slice(0, 3), source.slice(3)])})
    sink = DorisStreamLoadSink(
        "analytics",
        "items",
        endpoint="http://fe.example:8030",
        destination_schema=pa.schema([("score", pa.float32())]),
    )
    worker = sink.bind(table.schema).open_worker(WriteContext("float-rounding"))
    _Transport.responses = [_success]

    result = worker.write(table)

    decoded = pa.ipc.open_stream(_Transport.instances[0].calls[0].body).read_all()
    assert result.rows_affected == 7
    assert decoded.column("score").to_pylist() == pytest.approx(
        [0.1, None, float("nan"), float("inf"), float("-inf"), max_float32, -max_float32], nan_ok=True
    )
    worker.close()


def test_doris_sink_float_narrowing_ignores_hidden_list_values() -> None:
    source = pa.ListArray.from_arrays(
        pa.array([0, 1, 2, 4, 6, 7], type=pa.int32()),
        pa.array([1e40, -1e40, 0.1, None, float("inf"), float("-inf"), 1e40], type=pa.float64()),
        mask=pa.array([False, True, False, False, False]),
    ).slice(1, 3)
    table = pa.table({"score": source})
    sink = DorisStreamLoadSink(
        "analytics",
        "items",
        endpoint="http://fe.example:8030",
        destination_schema=pa.schema([("score", pa.list_(pa.float32()))]),
    )
    worker = sink.bind(table.schema).open_worker(WriteContext("float-list-slice"))
    _Transport.responses = [_success]

    worker.write(table)

    decoded = pa.ipc.open_stream(_Transport.instances[0].calls[0].body).read_all()
    rows = decoded.column("score").to_pylist()
    assert rows[0] is None
    assert rows[1] == pytest.approx([0.1, None])
    assert rows[2] == [float("inf"), float("-inf")]
    worker.close()


def test_doris_sink_rejects_unsafe_casts_and_destination_nulls_before_http() -> None:
    worker = _worker()
    overflow = _table().set_column(
        0,
        "source_id",
        pa.chunked_array([[1 << 40, 2]], type=pa.int64()),
    )
    with pytest.raises(ValueError, match=r"source_id.*safely cast.*int32"):
        worker.write(overflow)
    assert not _Transport.instances[0].calls

    worker = _worker()
    null_title = _table().set_column(
        2,
        "source_title",
        pa.chunked_array([["one", None]], type=pa.string()),
    )
    with pytest.raises(ValueError, match=r"destination field 'title'.*non-nullable"):
        worker.write(null_title)
    assert not _Transport.instances[1].calls

    source_schema = pa.schema([("values", pa.list_(pa.int64()))])
    destination_schema = pa.schema(
        [pa.field("values", pa.list_(pa.field("item", pa.int32(), nullable=False)), nullable=False)]
    )
    sink = DorisStreamLoadSink(
        "analytics",
        "items",
        endpoint="http://fe.example:8030",
        destination_schema=destination_schema,
        timeout=5,
    )
    worker = sink.bind(source_schema).open_worker(WriteContext("nested-null"))
    with pytest.raises(ValueError, match=r"destination field 'values\[\]'.*non-nullable"):
        worker.write(pa.table({"values": [[1, None]]}, schema=source_schema))
    assert not _Transport.instances[2].calls

    dictionary_list = pa.DictionaryArray.from_arrays(
        pa.array([1, None, 0], type=pa.int8()),
        pa.array([[1e40], [0.1, None]], type=pa.list_(pa.float64())),
    )
    table = pa.table({"values": dictionary_list})
    sink = DorisStreamLoadSink(
        "analytics",
        "items",
        endpoint="http://fe.example:8030",
        destination_schema=pa.schema([("values", pa.list_(pa.float32()))]),
    )
    worker = sink.bind(table.schema).open_worker(WriteContext("unsupported-dictionary-list"))
    with pytest.raises(ValueError, match=r"values.*cannot be safely cast.*Unsupported cast"):
        worker.write(table)
    assert not _Transport.instances[3].calls
    worker.close()


def test_doris_sink_defers_endpoint_and_password_resolution_to_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VANE_TEST_DORIS_ENDPOINT", "https://private-doris.example:8030")
    monkeypatch.setenv("VANE_TEST_DORIS_PASSWORD", "worker-only-password")
    sink = _sink(
        endpoint=EnvironmentSecret("VANE_TEST_DORIS_ENDPOINT"),
        password=EnvironmentSecret("VANE_TEST_DORIS_PASSWORD"),
    )
    serialized = cloudpickle.dumps(sink)
    assert b"private-doris.example" not in serialized
    assert b"worker-only-password" not in serialized

    bound = sink.bind(_schema())
    assert b"private-doris.example" not in cloudpickle.dumps(bound)
    assert b"worker-only-password" not in cloudpickle.dumps(bound)
    worker = bound.open_worker(WriteContext("secret-test"))

    assert _Transport.instances[0].options["password"] == "worker-only-password"
    worker.close()


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[0.1, 0.2, 0.3], None], "null vectors"),
        ([[0.1, 0.2], [0.3, 0.4, 0.5]], "invalid dimension"),
        ([[0.1, None, 0.3], [0.4, 0.5, 0.6]], "null elements"),
        ([[0.1, float("nan"), 0.3], [0.4, 0.5, 0.6]], "finite float32"),
        ([[0.1, float("inf"), 0.3], [0.4, 0.5, 0.6]], "finite float32"),
    ],
)
def test_doris_sink_validates_vector_values_before_http(vectors: list[list[float] | None], message: str) -> None:
    schema = _schema(vector_type=pa.list_(pa.float32()))
    worker = _worker(schema=schema)
    with pytest.raises(ValueError, match=message):
        worker.write(_table(vectors=vectors, schema=schema))
    assert not _Transport.instances[0].calls


def test_doris_sink_accepts_large_list_vectors_and_encodes_plain_list() -> None:
    schema = _schema(vector_type=pa.large_list(pa.float32()))
    table = _table(schema=schema)
    _Transport.responses = [_success]

    _worker(schema=schema).write(table)

    decoded = pa.ipc.open_stream(_Transport.instances[0].calls[0].body).read_all()
    assert decoded.schema.field("embedding").type == pa.list_(pa.float32())


def test_doris_sink_normalizes_sliced_vector_offsets() -> None:
    table = _table().slice(1, 1)
    _Transport.responses = [_success]

    result = _worker().write(table)

    decoded = pa.ipc.open_stream(_Transport.instances[0].calls[0].body).read_all()
    assert result.rows_affected == 1
    assert decoded.column("embedding").to_pylist()[0] == pytest.approx([0.4, 0.5, 0.6])


def test_doris_sink_enforces_schema_and_batch_limits_before_http() -> None:
    worker = _worker(max_batch_rows=1)
    with pytest.raises(ValueError, match="max_batch_rows"):
        worker.write(_table())
    assert not _Transport.instances[0].calls

    worker = _worker(max_batch_bytes=1)
    with pytest.raises(ValueError, match="max_batch_bytes"):
        worker.write(_table())
    assert not _Transport.instances[1].calls

    worker = _worker()
    bad_schema = pa.schema(
        [("source_id", pa.int64()), ("embedding", pa.list_(pa.float32(), 3)), ("other", pa.string())]
    )
    bad_table = pa.table(
        {"source_id": [1], "embedding": [[0.1, 0.2, 0.3]], "other": ["bad"]},
        schema=bad_schema,
    )
    with pytest.raises(ValueError, match="bound input schema"):
        worker.write(bad_table)
    assert not _Transport.instances[2].calls

    table = _table()
    worker = _worker(max_request_bytes=table.nbytes)
    with pytest.raises(ValueError, match="max_request_bytes"):
        worker.write(table)
    assert not _Transport.instances[3].calls


def test_doris_sink_skips_http_for_an_empty_batch() -> None:
    worker = _worker()
    result = worker.write(_table().slice(0, 0))
    assert result.rows_received == result.rows_affected == 0
    assert not _Transport.instances[0].calls


def test_doris_sink_follows_one_trusted_redirect_and_replays_identical_body() -> None:
    _Transport.responses = [
        _Response(307, headers={"Location": "http://root:@be.example:8040/api/analytics/items/_stream_load"}),
        _success,
    ]

    result = _worker().write(_table())

    calls = _Transport.instances[0].calls
    assert result.rows_affected == 2
    assert [call.url for call in calls] == [
        "http://fe.example:8030/api/analytics/items/_stream_load",
        "http://be.example:8040/api/analytics/items/_stream_load",
    ]
    assert calls[0].body == calls[1].body
    assert calls[0].headers["label"] == calls[1].headers["label"]


@pytest.mark.parametrize(
    ("location", "message"),
    [
        ("http://evil.example:8040/api/analytics/items/_stream_load", "untrusted host"),
        ("https://be.example:8040/api/analytics/items/_stream_load", "preserve the endpoint scheme"),
        ("http://be.example:8040/api/other/items/_stream_load", "changed the Stream Load path"),
        ("http://be.example:8040/api/analytics/items/_stream_load?token=secret", "must not contain a query"),
    ],
)
def test_doris_sink_rejects_unsafe_redirects(location: str, message: str) -> None:
    _Transport.responses = [_Response(307, headers={"Location": location})]
    with pytest.raises(RuntimeError, match=message):
        _worker().write(_table())
    assert len(_Transport.instances[0].calls) == 1


@pytest.mark.parametrize(
    "response",
    [_TransportError("connection reset"), _Response(503, b"temporarily unavailable")],
)
def test_doris_sink_never_retries_an_unknown_batch(response: _Response | BaseException) -> None:
    _Transport.responses = [response, _success]

    with pytest.raises(RuntimeError, match=r"label 'vane_.+'.*inspect that label"):
        _worker().write(_table())

    assert len(_Transport.instances[0].calls) == 1


def test_doris_sink_treats_publish_timeout_as_applied_with_warning() -> None:
    _Transport.responses = [lambda call: _success(call, status="Publish Timeout")]

    result = _worker().write(_table())

    assert result.rows_affected == 2
    assert result.metadata["status"] == "Publish Timeout"
    assert len(result.warnings) == 2
    assert "visibility may be delayed" in result.warnings[1]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_Response(401, b"unauthorized"), "HTTP 401"),
        (_Response(200, b"not json"), "invalid JSON"),
        (_Response(200, b"[]"), "non-object"),
        (
            lambda call: _Response(
                200,
                json.dumps({"Label": call.headers["label"], "Status": "Fail", "Message": "bad schema"}).encode(),
            ),
            "bad schema",
        ),
        (lambda call: _success(call, NumberLoadedRows=1), "row counts do not match"),
        (lambda call: _success(call, Label="different"), "different batch label"),
        (lambda call: _success(call, TxnId=True), "invalid TxnId"),
        (_Response(200, b"x" * (1024 * 1024 + 1)), "exceeds 1 MiB"),
    ],
)
def test_doris_sink_rejects_invalid_http_and_stream_load_responses(
    response: _Response | _ResponseFactory,
    message: str,
) -> None:
    _Transport.responses = [response]
    with pytest.raises(RuntimeError, match=message):
        _worker().write(_table())


def test_doris_sink_propagates_transport_failure_and_closes_on_abort() -> None:
    _Transport.responses = [_TransportError("connection reset")]
    worker = _worker()
    with pytest.raises(RuntimeError, match="connection reset"):
        worker.write(_table())
    worker.abort(_TransportError("connection reset"))
    assert _Transport.instances[0].close_calls == 1
    worker.close()
    assert _Transport.instances[0].close_calls == 1


def test_doris_sink_retains_transport_ownership_when_close_fails() -> None:
    worker = _worker()
    _Transport.close_error = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        worker.close()
    assert _Transport.instances[0].close_calls == 1

    _Transport.close_error = None
    worker.close()
    assert _Transport.instances[0].close_calls == 2


@pytest.mark.external_service
def test_doris_sink_live_arrow_stream_load() -> None:
    endpoint = os.environ.get("VANE_TEST_DORIS_ENDPOINT")
    if not endpoint:
        pytest.skip("VANE_TEST_DORIS_ENDPOINT is required for the external Doris test")
    pymysql = pytest.importorskip("pymysql")
    redirect_host = os.environ.get("VANE_TEST_DORIS_REDIRECT_HOST")
    database = f"vane_sink_{uuid.uuid4().hex}"
    connection = pymysql.connect(
        host=os.environ.get("VANE_TEST_DORIS_MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("VANE_TEST_DORIS_MYSQL_PORT", "9030")),
        user=os.environ.get("VANE_TEST_DORIS_USER", "root"),
        password=os.environ.get("VANE_TEST_DORIS_PASSWORD", ""),
        autocommit=True,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database}`")
            cursor.execute(
                f"""
                CREATE TABLE `{database}`.`items` (
                    `id` INT NOT NULL,
                    `embedding` ARRAY<FLOAT> NOT NULL,
                    `title` VARCHAR(64) NOT NULL
                ) ENGINE=OLAP
                DUPLICATE KEY(`id`)
                DISTRIBUTED BY HASH(`id`) BUCKETS 1
                PROPERTIES ("replication_num" = "1")
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE `{database}`.`temporal_items` (
                    `id` INT NOT NULL,
                    `happened_at` DATETIME NOT NULL
                ) ENGINE=OLAP
                DUPLICATE KEY(`id`)
                DISTRIBUTED BY HASH(`id`) BUCKETS 1
                PROPERTIES ("replication_num" = "1")
                """
            )

        input_table = _table()
        assert input_table.schema.field("source_id").type == pa.int64()
        relation = vane.from_arrow(input_table)
        summary = relation.write_datasink(
            DorisStreamLoadSink(
                database,
                "items",
                endpoint=endpoint,
                destination_schema=_destination_schema(),
                user=os.environ.get("VANE_TEST_DORIS_USER", "root"),
                password=(
                    EnvironmentSecret("VANE_TEST_DORIS_PASSWORD") if "VANE_TEST_DORIS_PASSWORD" in os.environ else None
                ),
                field_mapping={"source_id": "id", "source_title": "title"},
                vector_dimensions={"embedding": 3},
                trusted_redirect_hosts=(() if redirect_host is None else (redirect_host,)),
                timeout=60,
            ),
            operation_id=f"doris-live-{uuid.uuid4()}",
        )
        assert summary.rows_received == summary.rows_affected == 2
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT `id`, `title`, ARRAY_SIZE(`embedding`), `embedding`[1], `embedding`[3] "
                f"FROM `{database}`.`items` ORDER BY `id`"
            )
            rows = cursor.fetchall()
            assert [(row[0], row[1], row[2]) for row in rows] == [(1, "one", 3), (2, "two", 3)]
            assert rows[0][3:] == pytest.approx((0.1, 0.3))
            assert rows[1][3:] == pytest.approx((0.4, 0.6))

        temporal_relation = vane.from_arrow(
            pa.table(
                {
                    "id": pa.array([1], type=pa.int32()),
                    "happened_at": pa.array([datetime(2026, 9, 5)], type=pa.timestamp("us")),
                }
            )
        )
        with pytest.raises(ValueError, match="unsupported temporal Arrow type"):
            temporal_relation.write_datasink(
                DorisStreamLoadSink(
                    database,
                    "temporal_items",
                    endpoint=endpoint,
                    destination_schema=pa.schema(
                        [
                            pa.field("id", pa.int32(), nullable=False),
                            pa.field("happened_at", pa.timestamp("us"), nullable=False),
                        ]
                    ),
                    user=os.environ.get("VANE_TEST_DORIS_USER", "root"),
                    password=(
                        EnvironmentSecret("VANE_TEST_DORIS_PASSWORD")
                        if "VANE_TEST_DORIS_PASSWORD" in os.environ
                        else None
                    ),
                    trusted_redirect_hosts=(() if redirect_host is None else (redirect_host,)),
                    timeout=60,
                ),
                operation_id=f"doris-temporal-live-{uuid.uuid4()}",
            )
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM `{database}`.`temporal_items`")
            assert cursor.fetchone() == (0,)
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS `{database}` FORCE")
        finally:
            connection.close()
