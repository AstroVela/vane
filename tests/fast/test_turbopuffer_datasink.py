# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import json
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import cloudpickle
import pyarrow as pa
import pytest

import vane
import vane.datasink.turbopuffer as turbopuffer
from vane import EnvironmentSecret, TurbopufferSink
from vane.datasink import BoundKeyedUpsertSink, WriteContext

_REAL_LOAD_TURBOPUFFER_SDK = turbopuffer._load_turbopuffer_sdk
_DEFAULT_RESPONSE = object()


@dataclass
class _Response:
    rows_affected: object
    rows_upserted: object
    status: object = "OK"


class _Namespace:
    def __init__(self, client: _Client, name: str) -> None:
        self._client = client
        self.name = name

    def write(self, **kwargs: object) -> object:
        self._client.write_calls.append(kwargs)
        if _Client.write_error is not None:
            raise _Client.write_error
        columns = kwargs["upsert_columns"]
        assert isinstance(columns, dict)
        ids = columns["id"]
        assert isinstance(ids, list)
        for index, document_id in enumerate(ids):
            _Client.store[document_id] = {
                name: deepcopy(values[index]) for name, values in columns.items() if isinstance(values, list)
            }
        if _Client.response is _DEFAULT_RESPONSE:
            return _Response(len(ids), len(ids))
        return _Client.response


class _Client:
    instances: list[_Client] = []
    store: dict[object, dict[str, object]] = {}
    response: object = _DEFAULT_RESPONSE
    namespace_error: BaseException | None = None
    write_error: BaseException | None = None
    close_error: BaseException | None = None

    def __init__(self, **kwargs: object) -> None:
        self.options = kwargs
        self.namespace_calls: list[str] = []
        self.write_calls: list[dict[str, object]] = []
        self.close_calls = 0
        type(self).instances.append(self)

    def namespace(self, name: str) -> _Namespace:
        self.namespace_calls.append(name)
        if type(self).namespace_error is not None:
            raise type(self).namespace_error
        return _Namespace(self, name)

    def close(self) -> None:
        self.close_calls += 1
        if type(self).close_error is not None:
            raise type(self).close_error


@pytest.fixture(autouse=True)
def _fake_sdk(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("external_service") is not None:
        return
    _Client.instances = []
    _Client.store = {}
    _Client.response = _DEFAULT_RESPONSE
    _Client.namespace_error = None
    _Client.write_error = None
    _Client.close_error = None
    monkeypatch.setenv("VANE_TEST_TURBOPUFFER_API_KEY", "worker-only-test-secret")
    monkeypatch.setattr(turbopuffer, "_load_turbopuffer_sdk", lambda: (_Client, _Response))


def _schema(
    *,
    id_type: pa.DataType | None = None,
    vector_type: pa.DataType | None = None,
    title_type: pa.DataType | None = None,
    include_ignored: bool = True,
) -> pa.Schema:
    fields: list[tuple[str, pa.DataType]] = [
        ("document_id", pa.uint64() if id_type is None else id_type),
        ("embedding", pa.list_(pa.float32(), 3) if vector_type is None else vector_type),
        ("title", pa.string() if title_type is None else title_type),
        ("score", pa.float64()),
        ("tags", pa.list_(pa.string())),
    ]
    if include_ignored:
        fields.append(("ignored", pa.string()))
    return pa.schema(fields)


def _projected_schema(**kwargs: object) -> pa.Schema:
    return _schema(include_ignored=False, **kwargs)  # type: ignore[arg-type]


def _sink(**overrides: object) -> TurbopufferSink:
    options: dict[str, object] = {
        "namespace": "documents",
        "region": "gcp-us-central1",
        "api_key": EnvironmentSecret("VANE_TEST_TURBOPUFFER_API_KEY"),
        "id_column": "document_id",
        "vector_column": "embedding",
        "distance_metric": "cosine_distance",
        "attribute_mapping": {"title": "title", "score": "ranking", "tags": "labels"},
        "timeout": 5,
    }
    options.update(overrides)
    return TurbopufferSink(**options)  # type: ignore[arg-type]


def _bound(schema: pa.Schema | None = None, **overrides: object) -> BoundKeyedUpsertSink:
    return _sink(**overrides).bind(_schema() if schema is None else schema)


def _worker(schema: pa.Schema | None = None, **overrides: object) -> Any:
    return _bound(schema, **overrides).open_worker(WriteContext("turbopuffer-test"))


def _table(
    *,
    ids: list[object] | None = None,
    vectors: list[object] | None = None,
    titles: list[object] | None = None,
    scores: list[object] | None = None,
    tags: list[object] | None = None,
    schema: pa.Schema | None = None,
) -> pa.Table:
    return pa.table(
        {
            "document_id": [1, 2] if ids is None else ids,
            "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] if vectors is None else vectors,
            "title": ["one", "two"] if titles is None else titles,
            "score": [1.25, 2.5] if scores is None else scores,
            "tags": [["a"], ["b", "c"]] if tags is None else tags,
        },
        schema=_projected_schema() if schema is None else schema,
    )


def test_turbopuffer_sink_is_public_without_importing_the_optional_sdk() -> None:
    assert vane.TurbopufferSink is TurbopufferSink
    assert turbopuffer.__all__ == ["TurbopufferSink"]


def test_turbopuffer_sink_reports_the_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_turbopuffer(name: str, *args: object, **kwargs: object) -> Any:
        if name == "turbopuffer":
            raise ModuleNotFoundError("No module named 'turbopuffer'", name="turbopuffer")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_turbopuffer)
    with pytest.raises(ImportError, match=r"vane-ai\[turbopuffer\]"):
        _REAL_LOAD_TURBOPUFFER_SDK()


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"namespace": ""}, ValueError, "namespace"),
        ({"namespace": "bad/name"}, ValueError, "must match"),
        ({"namespace": "n" * 129}, ValueError, "must match"),
        ({"region": " gcp-us-central1"}, ValueError, "whitespace"),
        ({"region": "GCP-US-CENTRAL1"}, ValueError, "lowercase DNS label"),
        ({"region": "bad.region"}, ValueError, "lowercase DNS label"),
        ({"region": "r" * 64}, ValueError, "at most 63"),
        ({"api_key": "plaintext"}, TypeError, "EnvironmentSecret"),
        ({"id_column": ""}, ValueError, "id_column"),
        ({"vector_column": "document_id"}, ValueError, "different"),
        ({"distance_metric": "dot_product"}, ValueError, "distance_metric"),
        ({"attribute_mapping": []}, TypeError, "mapping"),
        ({"attribute_mapping": {"title": "id"}}, ValueError, "reserved"),
        ({"attribute_mapping": {"title": "vector"}}, ValueError, "reserved"),
        ({"attribute_mapping": {"title": "$dist"}}, ValueError, "reserved"),
        ({"attribute_mapping": {"title": "你" * 43}}, ValueError, "128 UTF-8 bytes"),
        ({"attribute_mapping": {"title": "same", "score": "same"}}, ValueError, "targets"),
        ({"attribute_mapping": {"document_id": "source"}}, ValueError, "must not reuse"),
        ({"worker_count": True}, TypeError, "worker_count"),
        ({"worker_count": 0}, ValueError, "worker_count"),
        ({"max_batch_rows": -1}, ValueError, "max_batch_rows"),
        ({"max_batch_bytes": 1.5}, TypeError, "max_batch_bytes"),
        ({"max_request_bytes": 512 * 1024 * 1024 + 1}, ValueError, "max_request_bytes"),
        ({"max_request_bytes": 100, "max_batch_bytes": 101}, ValueError, "must not exceed"),
        ({"max_retries": True}, TypeError, "max_retries"),
        ({"max_retries": -1}, ValueError, "max_retries"),
        ({"timeout": None}, TypeError, "timeout"),
        ({"timeout": float("inf")}, ValueError, "timeout"),
    ],
)
def test_turbopuffer_sink_validates_constructor(
    overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _sink(**overrides)


def test_turbopuffer_sink_enforces_the_attribute_name_count_limit() -> None:
    mapping = {f"source_{index}": f"target_{index}" for index in range(1_023)}
    with pytest.raises(ValueError, match="1024 attribute-name limit"):
        _sink(attribute_mapping=mapping)


def test_turbopuffer_sink_validates_bound_schema_and_sources() -> None:
    with pytest.raises(TypeError, match="pyarrow.Schema"):
        _sink().bind(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique ignoring case"):
        _sink().bind(pa.schema([("document_id", pa.uint64()), ("DOCUMENT_ID", pa.uint64())]))
    with pytest.raises(ValueError, match="id_column"):
        _sink(id_column="missing").bind(_schema())
    with pytest.raises(ValueError, match="attribute_mapping source"):
        _sink(attribute_mapping={"missing": "target"}).bind(_schema())
    with pytest.raises(ValueError, match="uint64 or string"):
        _sink().bind(_schema(id_type=pa.int64()))
    with pytest.raises(ValueError, match="fixed-size list"):
        _sink().bind(_schema(vector_type=pa.list_(pa.float32())))
    with pytest.raises(ValueError, match="fixed-size list"):
        _sink().bind(_schema(vector_type=pa.list_(pa.float64(), 3)))
    with pytest.raises(ValueError, match="between 1 and 10752"):
        _sink().bind(_schema(vector_type=pa.list_(pa.float32(), 10_753)))
    with pytest.raises(ValueError, match="does not support"):
        _sink().bind(_schema(title_type=pa.struct([("nested", pa.string())])))


def test_turbopuffer_sink_projects_explicit_columns_and_binds_execution_options() -> None:
    bound = _bound(worker_count=3, max_batch_rows=17, max_batch_bytes=2_048, max_retries=2)
    relation = vane.from_arrow(
        pa.table(
            {
                "document_id": pa.array([1], type=pa.uint64()),
                "embedding": pa.array([[0.1, 0.2, 0.3]], type=pa.list_(pa.float32(), 3)),
                "title": ["one"],
                "score": [1.25],
                "tags": [["a"]],
                "ignored": ["not-written"],
            }
        )
    )

    prepared = bound.prepare_input(relation)

    assert isinstance(bound, BoundKeyedUpsertSink)
    assert tuple(bound.key_columns) == ("document_id",)
    assert prepared.columns == ["document_id", "embedding", "title", "score", "tags"]
    assert bound.execution_options.worker_count == 3
    assert bound.execution_options.batch_size == 17
    assert bound.execution_options.target_max_batch_bytes == 2_048
    assert bound.execution_options.max_retries == 2


@pytest.mark.parametrize(
    ("arrow_type", "value", "expected_schema"),
    [
        (pa.bool_(), True, "bool"),
        (pa.int8(), -8, "int"),
        (pa.int64(), -64, "int"),
        (pa.uint8(), 8, "uint"),
        (pa.uint64(), 64, "uint"),
        (pa.float32(), 1.25, "float"),
        (pa.float64(), 2.5, "float"),
        (pa.string(), "text", "string"),
        (pa.list_(pa.bool_()), [True, False], "[]bool"),
        (pa.list_(pa.int64()), [-1, 2], "[]int"),
        (pa.list_(pa.uint64()), [1, 2], "[]uint"),
        (pa.list_(pa.float64()), [1.25, 2.5], "[]float"),
        (pa.list_(pa.string()), ["a", "b"], "[]string"),
    ],
)
def test_turbopuffer_sink_maps_supported_arrow_attributes(
    arrow_type: pa.DataType, value: object, expected_schema: str
) -> None:
    schema = pa.schema(
        [
            ("document_id", pa.uint64()),
            ("embedding", pa.list_(pa.float32(), 2)),
            ("attribute", arrow_type),
        ]
    )
    worker = _worker(schema, attribute_mapping={"attribute": "stored"})
    table = pa.table(
        {
            "document_id": pa.array([1], type=pa.uint64()),
            "embedding": pa.array([[0.1, 0.2]], type=pa.list_(pa.float32(), 2)),
            "attribute": pa.array([value], type=arrow_type),
        },
        schema=schema,
    )

    worker.write(table)

    call = _Client.instances[0].write_calls[0]
    assert call["schema"] == {
        "id": "uint",
        "vector": {"type": "[2]f32", "ann": True},
        "stored": expected_schema,
    }


def test_turbopuffer_sink_defers_secret_resolution_and_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VANE_TEST_TURBOPUFFER_API_KEY", "actual-worker-secret")
    sink = _sink()
    serialized_sink = cloudpickle.dumps(sink)
    assert b"actual-worker-secret" not in serialized_sink

    bound = sink.bind(_schema())
    assert b"actual-worker-secret" not in cloudpickle.dumps(bound)
    assert not _Client.instances
    worker = bound.open_worker(WriteContext("secret-test"))

    assert _Client.instances[0].options == {
        "api_key": "actual-worker-secret",
        "region": "gcp-us-central1",
        "base_url": "https://{region}.turbopuffer.com",
        "timeout": 5.0,
        "max_retries": 0,
        "compression": False,
    }
    assert _Client.instances[0].namespace_calls == ["documents"]
    worker.close()


def test_turbopuffer_sink_rejects_an_empty_worker_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VANE_TEST_TURBOPUFFER_API_KEY", "")
    with pytest.raises(RuntimeError, match="non-empty credential"):
        _worker()
    assert not _Client.instances


def test_turbopuffer_sink_uses_columnar_overwrite_and_returns_bounded_results() -> None:
    worker = _worker()
    table = _table()

    first = worker.write(table)
    second = worker.write(table)

    client = _Client.instances[0]
    call = client.write_calls[0]
    columns = call["upsert_columns"]
    assert isinstance(columns, dict)
    assert set(columns) == {"id", "vector", "title", "ranking", "labels"}
    assert columns["id"] == [1, 2]
    assert columns["title"] == ["one", "two"]
    assert columns["ranking"] == [1.25, 2.5]
    assert columns["labels"] == [["a"], ["b", "c"]]
    assert columns["vector"][0] == pytest.approx([0.1, 0.2, 0.3])
    assert call["distance_metric"] == "cosine_distance"
    assert call["schema"] == {
        "id": "uint",
        "vector": {"type": "[3]f32", "ann": True},
        "title": "string",
        "ranking": "float",
        "labels": "[]string",
    }
    assert call["timeout"] == 5.0
    assert "upsert_rows" not in call
    request = {
        "distance_metric": call["distance_metric"],
        "schema": call["schema"],
        "upsert_columns": call["upsert_columns"],
    }
    request_bytes = len(json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())
    assert first.rows_received == first.rows_affected == 2
    assert first.bytes_received == table.nbytes
    assert first.metadata == {
        "provider": "turbopuffer",
        "namespace": "documents",
        "write_mode": "overwrite",
        "request_bytes": request_bytes,
    }
    assert len(first.warnings) == 1
    assert second.warnings == ()


def test_turbopuffer_sink_supports_string_ids_and_enforces_the_64_byte_limit() -> None:
    schema = _projected_schema(id_type=pa.string())
    worker = _worker(_schema(id_type=pa.string()))
    worker.write(_table(ids=["first", "second"], schema=schema))
    assert set(_Client.store) == {"first", "second"}
    assert _Client.instances[0].write_calls[0]["schema"]["id"] == "string"

    with pytest.raises(ValueError, match="64 UTF-8 bytes"):
        worker.write(_table(ids=["你" * 22, "second"], schema=schema))
    assert len(_Client.instances[0].write_calls) == 1


def test_turbopuffer_sink_validates_values_before_the_provider_call() -> None:
    worker = _worker()
    invalid_tables = [
        (_table(ids=[None, 2]), "must not be null"),
        (_table(ids=[1, 1]), "duplicate"),
        (_table(vectors=[None, [0.4, 0.5, 0.6]]), "null or invalid dimension"),
        (_table(vectors=[[0.1, float("nan"), 0.3], [0.4, 0.5, 0.6]]), "non-finite"),
        (_table(scores=[float("inf"), 2.5]), "ranking.*invalid"),
        (_table(tags=[["a", None], ["b"]]), "array attribute.*invalid"),
    ]
    for table, message in invalid_tables:
        with pytest.raises(ValueError, match=message):
            worker.write(table)
    assert not _Client.instances[0].write_calls


def test_turbopuffer_sink_enforces_batch_schema_and_request_limits_before_write() -> None:
    worker = _worker(max_batch_rows=1)
    with pytest.raises(ValueError, match="max_batch_rows"):
        worker.write(_table())
    assert not _Client.instances[0].write_calls

    worker = _worker(max_batch_bytes=1)
    with pytest.raises(ValueError, match="max_batch_bytes"):
        worker.write(_table())
    assert not _Client.instances[1].write_calls

    worker = _worker()
    bad_schema = pa.schema(
        [
            ("document_id", pa.uint64()),
            ("embedding", pa.list_(pa.float32(), 3)),
            ("different", pa.string()),
            ("score", pa.float64()),
            ("tags", pa.list_(pa.string())),
        ]
    )
    with pytest.raises(ValueError, match="bound input schema"):
        worker.write(
            pa.table(
                {
                    "document_id": pa.array([1], type=pa.uint64()),
                    "embedding": pa.array([[0.1, 0.2, 0.3]], type=pa.list_(pa.float32(), 3)),
                    "different": ["one"],
                    "score": [1.25],
                    "tags": [["a"]],
                },
                schema=bad_schema,
            )
        )
    assert not _Client.instances[2].write_calls

    table = _table(ids=[1], vectors=[[0.1, 0.2, 0.3]], titles=["one"], scores=[1.25], tags=[["a"]])
    limit = table.nbytes + 1
    worker = _worker(max_batch_bytes=limit, max_request_bytes=limit)
    with pytest.raises(ValueError, match="max_request_bytes"):
        worker.write(table)
    assert not _Client.instances[3].write_calls


@pytest.mark.parametrize(
    "response",
    [
        None,
        object(),
        _Response(2, 2, status="NOT_OK"),
        _Response(True, 2),
        _Response(1, 2),
        _Response(2, None),
        _Response(2, 1),
    ],
)
def test_turbopuffer_sink_rejects_invalid_provider_results(response: object) -> None:
    _Client.response = response
    worker = _worker()
    with pytest.raises(RuntimeError, match="invalid response|affected-row count"):
        worker.write(_table())


def test_turbopuffer_sink_propagates_provider_failures_and_closes_on_abort() -> None:
    _Client.namespace_error = RuntimeError("namespace failed")
    with pytest.raises(RuntimeError, match="namespace failed"):
        _worker()
    assert _Client.instances[0].close_calls == 1

    _Client.namespace_error = None
    worker = _worker()
    _Client.write_error = TimeoutError("write timed out")
    with pytest.raises(TimeoutError, match="write timed out"):
        worker.write(_table())
    worker.abort(TimeoutError("write timed out"))
    assert _Client.instances[1].close_calls == 1
    worker.close()
    assert _Client.instances[1].close_calls == 1


def test_turbopuffer_sink_retains_client_ownership_when_close_fails() -> None:
    worker = _worker()
    _Client.close_error = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        worker.close()
    assert _Client.instances[0].close_calls == 1

    _Client.close_error = None
    worker.close()
    assert _Client.instances[0].close_calls == 2


def test_turbopuffer_sink_replay_overwrites_the_same_document_ids() -> None:
    table = _table()
    first_worker = _bound(max_retries=1).open_worker(WriteContext("stable-operation"))
    second_worker = _bound(max_retries=1).open_worker(WriteContext("stable-operation"))

    first_worker.write(table)
    snapshot = deepcopy(_Client.store)
    second_worker.write(table)

    assert _Client.store == snapshot
    assert set(_Client.store) == {1, 2}
    assert sum(len(client.write_calls) for client in _Client.instances) == 2
    assert all(client.options["max_retries"] == 0 for client in _Client.instances)


@pytest.mark.external_service
def test_turbopuffer_sink_live_whole_document_upsert() -> None:
    api_key = os.environ.get("VANE_TEST_TURBOPUFFER_API_KEY")
    region = os.environ.get("VANE_TEST_TURBOPUFFER_REGION")
    if not api_key or not region:
        pytest.skip("VANE_TEST_TURBOPUFFER_API_KEY and VANE_TEST_TURBOPUFFER_REGION are required")
    sdk = pytest.importorskip("turbopuffer")
    namespace = f"vane-sink-e2e-{uuid.uuid4().hex}"
    client = sdk.Turbopuffer(
        api_key=api_key,
        region=region,
        base_url="https://{region}.turbopuffer.com",
        timeout=30.0,
        max_retries=0,
        compression=False,
    )
    resource = client.namespace(namespace)
    worker = None
    namespace_created = False
    try:
        schema = pa.schema(
            [
                ("document_id", pa.uint64()),
                ("embedding", pa.list_(pa.float32(), 2)),
                ("title", pa.string()),
            ]
        )
        sink = TurbopufferSink(
            namespace,
            region=region,
            api_key=EnvironmentSecret("VANE_TEST_TURBOPUFFER_API_KEY"),
            id_column="document_id",
            vector_column="embedding",
            distance_metric="cosine_distance",
            attribute_mapping={"title": "title"},
            timeout=30.0,
        )
        worker = sink.bind(schema).open_worker(WriteContext(f"turbopuffer-live-{uuid.uuid4()}"))
        first = pa.table(
            {
                "document_id": pa.array([1, 2], type=pa.uint64()),
                "embedding": pa.array([[0.1, 0.2], [0.3, 0.4]], type=pa.list_(pa.float32(), 2)),
                "title": ["one", "two"],
            },
            schema=schema,
        )
        updated = pa.table(
            {
                "document_id": pa.array([1, 2], type=pa.uint64()),
                "embedding": pa.array([[0.1, 0.2], [0.3, 0.4]], type=pa.list_(pa.float32(), 2)),
                "title": ["updated", "two"],
            },
            schema=schema,
        )

        assert worker.write(first).rows_affected == 2
        namespace_created = True
        assert worker.write(updated).rows_affected == 2
        response = resource.query(
            rank_by=("id", "asc"),
            limit=10,
            include_attributes=True,
            consistency={"level": "strong"},
            timeout=30.0,
        )
        rows = response.rows or []
        assert {row.id: row["title"] for row in rows} == {1: "updated", 2: "two"}
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            try:
                if namespace_created:
                    resource.delete_all(timeout=30.0)
            finally:
                client.close()
