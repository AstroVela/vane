# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cloudpickle
import pyarrow as pa
import pytest

import vane
import vane.datasink.qdrant as qdrant
from vane import EnvironmentSecret, QdrantSink
from vane.datasink import (
    BoundKeyedUpsertSink,
    DataSink,
    DataSinkWriteError,
    WriteContext,
    WriteOutcome,
)

_REAL_LOAD_QDRANT_SDK = qdrant._load_qdrant_sdk


class _UpdateStatus(str, Enum):
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    WAIT_TIMEOUT = "wait_timeout"


class _Datatype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    UINT8 = "uint8"


@dataclass
class _VectorParams:
    size: object
    datatype: _Datatype | None = None
    multivector_config: object | None = None


@dataclass
class _CollectionParams:
    vectors: object
    sparse_vectors: object | None = None


@dataclass
class _CollectionConfig:
    params: object


@dataclass
class _CollectionInfo:
    config: object


@dataclass
class _PointStruct:
    id: int | str
    vector: list[float] | dict[str, list[float]]
    payload: dict[str, object]


@dataclass
class _UpdateResult:
    status: _UpdateStatus
    operation_id: int | None = 1


class _Models:
    CollectionConfig = _CollectionConfig
    CollectionInfo = _CollectionInfo
    CollectionParams = _CollectionParams
    Datatype = _Datatype
    PointStruct = _PointStruct
    UpdateResult = _UpdateResult
    UpdateStatus = _UpdateStatus
    VectorParams = _VectorParams


def _collection_info(
    *,
    vectors: object | None = None,
    sparse_vectors: object | None = None,
    params: object | None = None,
) -> _CollectionInfo:
    if params is None:
        params = _CollectionParams(
            vectors=_VectorParams(size=3) if vectors is None else vectors,
            sparse_vectors=sparse_vectors,
        )
    return _CollectionInfo(config=_CollectionConfig(params=params))


_DEFAULT_RESPONSE = object()


class _Client:
    collection_info: object = _collection_info()
    get_error: BaseException | None = None
    upsert_error: BaseException | None = None
    close_error: BaseException | None = None
    response: object = _DEFAULT_RESPONSE
    instances: list[_Client] = []
    store: dict[int | str, _PointStruct] = {}

    def __init__(self, **kwargs: object) -> None:
        self.options = kwargs
        self.get_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.close_calls = 0
        type(self).instances.append(self)

    def get_collection(self, **kwargs: object) -> object:
        self.get_calls.append(kwargs)
        if self.get_error is not None:
            raise self.get_error
        return deepcopy(self.collection_info)

    def upsert(self, **kwargs: object) -> object:
        self.upsert_calls.append(kwargs)
        if self.upsert_error is not None:
            raise self.upsert_error
        points = kwargs["points"]
        assert isinstance(points, list)
        for point in points:
            assert isinstance(point, _PointStruct)
            type(self).store[point.id] = deepcopy(point)
        if self.response is _DEFAULT_RESPONSE:
            return _UpdateResult(_UpdateStatus.COMPLETED)
        return self.response

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture(autouse=True)
def _fake_sdk(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("external_service") is not None:
        return
    _Client.collection_info = _collection_info()
    _Client.get_error = None
    _Client.upsert_error = None
    _Client.close_error = None
    _Client.response = _DEFAULT_RESPONSE
    _Client.instances = []
    _Client.store = {}
    monkeypatch.setattr(qdrant, "_load_qdrant_sdk", lambda: (_Client, _Models))


def _arrow_schema(
    *,
    id_type: pa.DataType | None = None,
    vector_type: pa.DataType | None = None,
    payload_type: pa.DataType | None = None,
) -> pa.Schema:
    return pa.schema(
        [
            ("id", pa.uint64() if id_type is None else id_type),
            ("embedding", pa.list_(pa.float32()) if vector_type is None else vector_type),
            ("title", pa.string() if payload_type is None else payload_type),
        ]
    )


def _sink(**kwargs: object) -> QdrantSink:
    options: dict[str, object] = {
        "url": "https://qdrant.example:6333",
        "point_id": "id",
        "vector_mapping": "embedding",
        "payload_mapping": {"title": "title"},
        "timeout": 5,
    }
    options.update(kwargs)
    return QdrantSink("items", **options)  # type: ignore[arg-type]


def _bound(*, schema: pa.Schema | None = None, **kwargs: object) -> BoundKeyedUpsertSink:
    return _sink(**kwargs).bind(_arrow_schema() if schema is None else schema)


def _worker(*, schema: pa.Schema | None = None, **kwargs: object) -> Any:
    return _bound(schema=schema, **kwargs).open_worker(WriteContext("qdrant-test"))


def _table(
    *,
    ids: list[int] | None = None,
    vectors: list[list[float] | None] | None = None,
    titles: list[object] | None = None,
    schema: pa.Schema | None = None,
) -> pa.Table:
    return pa.table(
        {
            "id": [1, 2] if ids is None else ids,
            "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] if vectors is None else vectors,
            "title": ["one", "two"] if titles is None else titles,
        },
        schema=_arrow_schema() if schema is None else schema,
    )


def test_qdrant_sink_is_public_without_importing_qdrant_client() -> None:
    assert vane.QdrantSink is QdrantSink
    assert qdrant.__all__ == ["QdrantSink"]


def test_qdrant_sink_reports_the_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_qdrant_client(name: str, *args: object, **kwargs: object) -> Any:
        if name == "qdrant_client":
            raise ModuleNotFoundError("No module named 'qdrant_client'", name="qdrant_client")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_qdrant_client)
    with pytest.raises(ImportError, match=r"vane-ai\[qdrant\]"):
        _REAL_LOAD_QDRANT_SDK()


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"collection_name": ""}, ValueError, "collection_name"),
        ({"collection_name": " items"}, ValueError, "whitespace"),
        ({"url": "qdrant.example:6333"}, ValueError, "HTTP or HTTPS"),
        ({"url": "https://qdrant.example/path"}, ValueError, "path"),
        ({"url": "https://qdrant.example:not-a-port"}, ValueError, "valid HTTP or HTTPS"),
        ({"url": "https://user:secret@qdrant.example"}, ValueError, "credentials"),
        ({"url": "https://qdrant.example?api_key=secret"}, ValueError, "query parameters"),
        ({"point_id": ""}, ValueError, "point_id"),
        ({"vector_mapping": []}, TypeError, "vector_mapping"),
        ({"vector_mapping": {}}, ValueError, "at least one"),
        ({"vector_mapping": {"embedding": "v", "other": "v"}}, ValueError, "targets"),
        ({"payload_mapping": []}, TypeError, "payload_mapping"),
        ({"payload_mapping": {"title": "x", "other": "x"}}, ValueError, "targets"),
        ({"api_key": "plain-text"}, TypeError, "EnvironmentSecret"),
        ({"worker_count": True}, TypeError, "worker_count"),
        ({"worker_count": 0}, ValueError, "worker_count"),
        ({"max_batch_rows": -1}, ValueError, "max_batch_rows"),
        ({"max_batch_bytes": 1.5}, TypeError, "max_batch_bytes"),
        ({"max_retries": True}, TypeError, "max_retries"),
        ({"max_retries": -1}, ValueError, "max_retries"),
        ({"timeout": 1.5}, TypeError, "timeout"),
        ({"timeout": 0}, ValueError, "timeout"),
    ],
)
def test_qdrant_sink_validates_constructor(overrides: dict[str, object], error: type[Exception], message: str) -> None:
    options: dict[str, object] = {
        "collection_name": "items",
        "url": "https://qdrant.example:6333",
        "point_id": "id",
        "vector_mapping": "embedding",
    }
    options.update(overrides)
    with pytest.raises(error, match=message):
        QdrantSink(**options)  # type: ignore[arg-type]


def test_qdrant_sink_binds_key_mappings_and_execution_options() -> None:
    sink = _sink(worker_count=3, max_batch_rows=17, max_batch_bytes=2_048, max_retries=2)
    bound = sink.bind(_arrow_schema())

    assert isinstance(bound, BoundKeyedUpsertSink)
    assert tuple(bound.key_columns) == ("id",)
    assert bound.execution_options.worker_count == 3
    assert bound.execution_options.batch_size == 17
    assert bound.execution_options.target_max_batch_bytes == 2_048
    assert bound.execution_options.max_retries == 2


def test_qdrant_sink_rejects_invalid_bound_schema_and_mappings() -> None:
    with pytest.raises(TypeError, match="pyarrow.Schema"):
        _sink().bind(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="case-insensitively unique"):
        _sink().bind(pa.schema([("id", pa.uint64()), ("ID", pa.uint64())]))
    with pytest.raises(ValueError, match="uint64 or string UUID"):
        _sink().bind(_arrow_schema(id_type=pa.int64()))
    with pytest.raises(ValueError, match="float32"):
        _sink().bind(_arrow_schema(vector_type=pa.list_(pa.float64())))
    with pytest.raises(ValueError, match="source columns must be disjoint"):
        _sink(payload_mapping={"embedding": "payload"}).bind(
            pa.schema([("id", pa.uint64()), ("embedding", pa.list_(pa.float32()))])
        )
    with pytest.raises(ValueError, match="unknown input"):
        _sink(vector_mapping="missing").bind(_arrow_schema())
    with pytest.raises(ValueError, match="every input column"):
        _sink(payload_mapping=None).bind(_arrow_schema())
    with pytest.raises(ValueError, match="does not support payload"):
        _sink().bind(_arrow_schema(payload_type=pa.binary()))


def test_qdrant_sink_writes_single_vector_full_points_and_bounded_results() -> None:
    worker = _worker()
    table = _table()

    first = worker.write(table)
    second = worker.write(table)

    assert first.rows_received == first.rows_affected == 2
    assert first.bytes_received == table.nbytes
    assert first.metadata == {
        "provider": "qdrant",
        "collection": "items",
        "write_mode": "replace",
        "status": "completed",
    }
    assert len(first.warnings) == 1
    assert second.warnings == ()
    client = _Client.instances[0]
    assert client.options == {"url": "https://qdrant.example:6333", "timeout": 5}
    assert client.get_calls == [{"collection_name": "items"}]
    assert client.upsert_calls[0]["collection_name"] == "items"
    assert client.upsert_calls[0]["wait"] is True
    assert client.upsert_calls[0]["timeout"] == 5
    assert client.upsert_calls[0]["points"] == [
        _PointStruct(1, pytest.approx([0.1, 0.2, 0.3]), {"title": "one"}),
        _PointStruct(2, pytest.approx([0.4, 0.5, 0.6]), {"title": "two"}),
    ]


def test_qdrant_sink_writes_every_named_vector_and_mapped_payload() -> None:
    _Client.collection_info = _collection_info(
        vectors={"dense": _VectorParams(size=2), "summary": _VectorParams(size=1)}
    )
    schema = pa.schema(
        [
            ("point", pa.uint64()),
            ("dense_source", pa.list_(pa.float32(), 2)),
            ("summary_source", pa.list_(pa.float32(), 1)),
            ("source_title", pa.string()),
            ("attributes", pa.struct([("rank", pa.int32()), ("tags", pa.list_(pa.string()))])),
        ]
    )
    sink = QdrantSink(
        "items",
        url="https://qdrant.example:6333",
        point_id="point",
        vector_mapping={"dense_source": "dense", "summary_source": "summary"},
        payload_mapping={"source_title": "title", "attributes": "attributes"},
    )
    table = pa.table(
        {
            "point": [7],
            "dense_source": [[0.1, 0.2]],
            "summary_source": [[0.3]],
            "source_title": ["seven"],
            "attributes": [{"rank": 1, "tags": ["a", "b"]}],
        },
        schema=schema,
    )

    sink.bind(schema).open_worker(WriteContext("named")).write(table)

    point = _Client.instances[0].upsert_calls[0]["points"][0]
    assert point == _PointStruct(
        7,
        {"dense": pytest.approx([0.1, 0.2]), "summary": pytest.approx([0.3])},
        {"title": "seven", "attributes": {"rank": 1, "tags": ["a", "b"]}},
    )


def test_qdrant_sink_defers_endpoint_and_api_key_resolution_to_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VANE_TEST_QDRANT_URL", "https://private-qdrant.example:6333")
    monkeypatch.setenv("VANE_TEST_QDRANT_API_KEY", "worker-only-api-key")
    sink = _sink(
        url=EnvironmentSecret("VANE_TEST_QDRANT_URL"),
        api_key=EnvironmentSecret("VANE_TEST_QDRANT_API_KEY"),
    )
    serialized = cloudpickle.dumps(sink)
    assert b"private-qdrant.example" not in serialized
    assert b"worker-only-api-key" not in serialized

    bound = sink.bind(_arrow_schema())
    assert b"private-qdrant.example" not in cloudpickle.dumps(bound)
    assert b"worker-only-api-key" not in cloudpickle.dumps(bound)
    assert not _Client.instances
    worker = bound.open_worker(WriteContext("secret-test"))

    assert _Client.instances[0].options == {
        "url": "https://private-qdrant.example:6333",
        "timeout": 5,
        "api_key": "worker-only-api-key",
    }
    worker.close()


class _ProjectionRelation:
    def __init__(self) -> None:
        self.projection: str | None = None

    def project(self, projection: str) -> _ProjectionRelation:
        self.projection = projection
        return self


class _RejectingQdrantBound(BoundKeyedUpsertSink):
    def __init__(self, delegate: BoundKeyedUpsertSink) -> None:
        self._delegate = delegate

    @property
    def execution_options(self) -> Any:
        return self._delegate.execution_options

    @property
    def key_columns(self) -> Any:
        return self._delegate.key_columns

    def prepare_input(self, relation: Any) -> Any:
        return self._delegate.prepare_input(relation)

    def open_worker(self, _context: WriteContext) -> Any:
        raise AssertionError("Qdrant worker must not open for rejected point IDs")


class _RejectingQdrantSink(DataSink):
    def __init__(self, sink: QdrantSink) -> None:
        self._sink = sink

    def bind(self, schema: pa.Schema) -> BoundKeyedUpsertSink:
        return _RejectingQdrantBound(self._sink.bind(schema))


def test_qdrant_sink_normalizes_uuid_ids_before_global_key_validation() -> None:
    schema = _arrow_schema(id_type=pa.string())
    bound = _bound(schema=schema)
    relation = _ProjectionRelation()

    prepared = bound.prepare_input(relation)  # type: ignore[arg-type]

    assert prepared is relation
    assert relation.projection == ('CAST(TRY_CAST("id" AS UUID) AS VARCHAR) AS "id", "embedding", "title"')
    assert tuple(bound.key_columns) == ("id",)


@pytest.mark.parametrize(
    ("ids", "operation_id"),
    [
        (["not-a-uuid"], "invalid-uuid"),
        (
            ["123e4567-e89b-12d3-a456-426614174000", "123E4567-E89B-12D3-A456-426614174000"],
            "duplicate-normalized-uuid",
        ),
    ],
)
def test_qdrant_sink_rejects_invalid_or_duplicate_uuid_ids_before_worker_open(
    monkeypatch: pytest.MonkeyPatch,
    ids: list[str],
    operation_id: str,
) -> None:
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.from_arrow(
        pa.table(
            {
                "id": ids,
                "embedding": pa.array([[0.1, 0.2, 0.3]] * len(ids), type=pa.list_(pa.float32())),
                "title": ["title"] * len(ids),
            },
            schema=_arrow_schema(id_type=pa.string()),
        )
    )

    with pytest.raises(DataSinkWriteError) as exc_info:
        relation.write_datasink(_RejectingQdrantSink(_sink()), operation_id=operation_id)

    assert exc_info.value.outcome is WriteOutcome.ABORTED


@pytest.mark.parametrize(
    ("collection_info", "message"),
    [
        (None, "invalid collection information"),
        (_CollectionInfo(config=None), "invalid collection configuration"),
        (_CollectionInfo(config=_CollectionConfig(params=None)), "invalid collection configuration"),
        (_collection_info(vectors={"dense": _VectorParams(3)}), "one unnamed"),
        (_collection_info(sparse_vectors={"sparse": object()}), "sparse vectors"),
        (_collection_info(vectors=_VectorParams(3, datatype=_Datatype.FLOAT16)), "float32"),
        (_collection_info(vectors=_VectorParams(3, multivector_config=object())), "multivector"),
        (_collection_info(vectors=_VectorParams(True)), "invalid dimension"),
    ],
)
def test_qdrant_sink_rejects_unsupported_collection_contract(collection_info: object, message: str) -> None:
    _Client.collection_info = collection_info
    with pytest.raises(ValueError, match=message):
        _worker()
    assert _Client.instances[0].close_calls == 1


def test_qdrant_sink_validates_vector_dimensions_and_named_collection_schema() -> None:
    _Client.collection_info = _collection_info(vectors=_VectorParams(3))
    with pytest.raises(ValueError, match="fixed dimension"):
        _worker(schema=_arrow_schema(vector_type=pa.list_(pa.float32(), 2)))

    _Client.collection_info = _collection_info(vectors=_VectorParams(3))
    named_sink = _sink(vector_mapping={"embedding": "dense"})
    with pytest.raises(ValueError, match="named dense vectors"):
        named_sink.bind(_arrow_schema()).open_worker(WriteContext("named-required"))

    _Client.collection_info = _collection_info(vectors={"dense": _VectorParams(3), "other": _VectorParams(3)})
    with pytest.raises(ValueError, match="cover every named"):
        named_sink.bind(_arrow_schema()).open_worker(WriteContext("missing-named"))


def test_qdrant_sink_validates_all_values_before_upsert() -> None:
    worker = _worker()
    with pytest.raises(ValueError, match="point IDs"):
        worker.write(_table(ids=[None, 2]))  # type: ignore[list-item]
    with pytest.raises(ValueError, match="invalid dimension"):
        worker.write(_table(vectors=[[0.1, 0.2], [0.3, 0.4]]))
    with pytest.raises(ValueError, match="invalid value"):
        worker.write(_table(vectors=[[0.1, float("nan"), 0.3], [0.4, 0.5, 0.6]]))
    assert not _Client.instances[0].upsert_calls

    float_schema = _arrow_schema(payload_type=pa.float64())
    float_worker = _worker(schema=float_schema)
    with pytest.raises(ValueError, match="non-finite"):
        float_worker.write(_table(titles=[float("inf"), 1.0], schema=float_schema))
    assert not _Client.instances[1].upsert_calls

    uint_schema = _arrow_schema(payload_type=pa.uint64())
    uint_worker = _worker(schema=uint_schema)
    with pytest.raises(ValueError, match="signed 64-bit"):
        uint_worker.write(_table(titles=[1 << 63, 1], schema=uint_schema))
    assert not _Client.instances[2].upsert_calls


def test_qdrant_sink_rejects_noncanonical_uuid_values_before_upsert() -> None:
    schema = _arrow_schema(id_type=pa.string())
    worker = _worker(schema=schema)
    canonical = "123e4567-e89b-12d3-a456-426614174000"
    invalid_tables = [
        pa.table(
            {"id": ["not-a-uuid"], "embedding": [[0.1, 0.2, 0.3]], "title": ["bad"]},
            schema=schema,
        ),
        pa.table(
            {"id": [canonical.upper()], "embedding": [[0.1, 0.2, 0.3]], "title": ["bad"]},
            schema=schema,
        ),
    ]
    for table in invalid_tables:
        with pytest.raises(ValueError, match="UUID point IDs"):
            worker.write(table)
    assert not _Client.instances[0].upsert_calls


def test_qdrant_sink_enforces_batch_and_schema_limits_before_upsert() -> None:
    worker = _worker(max_batch_rows=1)
    with pytest.raises(ValueError, match="max_batch_rows"):
        worker.write(_table())
    assert not _Client.instances[0].upsert_calls

    worker = _worker(max_batch_bytes=1)
    with pytest.raises(ValueError, match="max_batch_bytes"):
        worker.write(_table())
    assert not _Client.instances[1].upsert_calls

    worker = _worker()
    bad_schema = pa.schema([("id", pa.uint64()), ("embedding", pa.list_(pa.float32())), ("other", pa.string())])
    with pytest.raises(ValueError, match="bound input schema"):
        worker.write(pa.table({"id": [1], "embedding": [[0.1, 0.2, 0.3]], "other": ["x"]}, schema=bad_schema))
    assert not _Client.instances[2].upsert_calls


def test_qdrant_sink_validates_static_result_payload_before_upsert() -> None:
    collection = "c" * (65 * 1024)
    sink = QdrantSink(
        collection,
        url="https://qdrant.example:6333",
        point_id="id",
        vector_mapping="embedding",
        payload_mapping={"title": "title"},
    )

    with pytest.raises(ValueError, match="64 KiB"):
        sink.bind(_arrow_schema()).open_worker(WriteContext("bounded-result"))

    assert _Client.instances[0].close_calls == 1
    assert not _Client.instances[0].upsert_calls


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (None, "invalid update result"),
        ({"status": "completed"}, "invalid update result"),
        (_UpdateResult(_UpdateStatus.ACKNOWLEDGED), "status=acknowledged"),
        (_UpdateResult(_UpdateStatus.WAIT_TIMEOUT), "status=wait_timeout"),
    ],
)
def test_qdrant_sink_requires_completed_application_acknowledgement(response: object, message: str) -> None:
    _Client.response = response
    worker = _worker()
    with pytest.raises(RuntimeError, match=message):
        worker.write(_table())


def test_qdrant_sink_propagates_provider_failures_and_closes_on_abort() -> None:
    _Client.get_error = RuntimeError("get failed")
    with pytest.raises(RuntimeError, match="get failed"):
        _worker()
    assert _Client.instances[0].close_calls == 1

    _Client.get_error = None
    worker = _worker()
    _Client.upsert_error = TimeoutError("upsert timed out")
    with pytest.raises(TimeoutError, match="upsert timed out"):
        worker.write(_table())
    worker.abort(TimeoutError("upsert timed out"))
    assert _Client.instances[1].close_calls == 1
    worker.close()
    assert _Client.instances[1].close_calls == 1


def test_qdrant_sink_retains_client_ownership_when_close_fails() -> None:
    worker = _worker()
    _Client.close_error = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        worker.close()
    assert _Client.instances[0].close_calls == 1

    _Client.close_error = None
    worker.close()
    assert _Client.instances[0].close_calls == 2


def test_qdrant_sink_replay_replaces_the_same_points() -> None:
    table = _table()
    first_worker = _bound(max_retries=1).open_worker(WriteContext("stable-operation"))
    second_worker = _bound(max_retries=1).open_worker(WriteContext("stable-operation"))

    first_worker.write(table)
    snapshot = deepcopy(_Client.store)
    second_worker.write(table)

    assert _Client.store == snapshot
    assert set(_Client.store) == {1, 2}
    assert sum(len(client.upsert_calls) for client in _Client.instances) == 2


@pytest.mark.external_service
def test_qdrant_sink_live_full_point_upsert() -> None:
    url = os.environ.get("VANE_TEST_QDRANT_URL")
    if not url:
        pytest.skip("VANE_TEST_QDRANT_URL is required for the external Qdrant test")
    qdrant_client = pytest.importorskip("qdrant_client")
    models = qdrant_client.models
    api_key_value = os.environ.get("VANE_TEST_QDRANT_API_KEY")
    options: dict[str, object] = {"url": url, "timeout": 30}
    if api_key_value is not None:
        options["api_key"] = api_key_value
    client = qdrant_client.QdrantClient(**options)
    collection = f"vane_sink_e2e_{uuid.uuid4().hex}"
    collection_created = False
    try:
        assert client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
            timeout=30,
        )
        collection_created = True
        seed_result = client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(id=1, vector=[0.9, 0.9], payload={"title": "old", "obsolete": True}),
                models.PointStruct(id=2, vector=[0.8, 0.8], payload={"title": "old", "obsolete": True}),
            ],
            wait=True,
            timeout=30,
        )
        assert seed_result.status is models.UpdateStatus.COMPLETED
        relation = vane.from_arrow(
            pa.table(
                {
                    "id": pa.array([1, 2], type=pa.uint64()),
                    "embedding": pa.array([[0.1, 0.2], [0.3, 0.4]], type=pa.list_(pa.float32(), 2)),
                    "title": ["one", "two"],
                }
            )
        )
        api_key = EnvironmentSecret("VANE_TEST_QDRANT_API_KEY") if api_key_value is not None else None
        summary = relation.write_datasink(
            QdrantSink(
                collection,
                url=url,
                point_id="id",
                vector_mapping="embedding",
                payload_mapping={"title": "title"},
                api_key=api_key,
                timeout=30,
            ),
            operation_id=f"qdrant-live-{uuid.uuid4()}",
        )
        assert summary.rows_affected == 2
        points = client.retrieve(
            collection_name=collection,
            ids=[1, 2],
            with_payload=True,
            with_vectors=True,
        )
        assert {point.id: point.payload for point in points} == {1: {"title": "one"}, 2: {"title": "two"}}
        assert {point.id: point.vector for point in points} == {
            1: pytest.approx([0.1, 0.2]),
            2: pytest.approx([0.3, 0.4]),
        }
    finally:
        try:
            if collection_created:
                client.delete_collection(collection_name=collection, timeout=30)
        finally:
            client.close()
