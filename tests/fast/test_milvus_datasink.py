# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import os
import uuid
from collections.abc import Mapping
from copy import deepcopy
from enum import IntEnum
from typing import Any

import cloudpickle
import pyarrow as pa
import pytest

import vane
import vane.datasink.milvus as milvus
from vane import EnvironmentSecret, MilvusSink
from vane.datasink import BoundKeyedUpsertSink, WriteContext

_REAL_LOAD_MILVUS_SDK = milvus._load_milvus_sdk


class _DataType(IntEnum):
    BOOL = 1
    INT8 = 2
    INT16 = 3
    INT32 = 4
    INT64 = 5
    FLOAT = 10
    DOUBLE = 11
    VARCHAR = 21
    TEXT = 25
    FLOAT_VECTOR = 101


def _description(
    *,
    collection_name: str = "items",
    auto_id: bool = False,
    dynamic: bool = False,
    functions: list[object] | None = None,
    primary_type: _DataType = _DataType.INT64,
    title_type: _DataType = _DataType.VARCHAR,
) -> dict[str, object]:
    title_params: dict[str, int] = {"max_length": 16} if title_type is _DataType.VARCHAR else {}
    return {
        "collection_name": collection_name,
        "auto_id": auto_id,
        "enable_dynamic_field": dynamic,
        "functions": [] if functions is None else functions,
        "fields": [
            {
                "name": "id",
                "type": primary_type,
                "is_primary": True,
                "nullable": False,
                "params": {"max_length": 32} if primary_type is _DataType.VARCHAR else {},
            },
            {
                "name": "embedding",
                "type": _DataType.FLOAT_VECTOR,
                "nullable": False,
                "params": {"dim": 3},
            },
            {
                "name": "title",
                "type": title_type,
                "nullable": False,
                "params": title_params,
            },
        ],
    }


_DEFAULT_RESPONSE = object()


class _Client:
    description: object = _description()
    describe_error: BaseException | None = None
    upsert_error: BaseException | None = None
    close_error: BaseException | None = None
    response: object = _DEFAULT_RESPONSE
    instances: list[_Client] = []
    store: dict[object, dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        self.options = kwargs
        self.describe_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.close_calls = 0
        type(self).instances.append(self)

    def describe_collection(self, **kwargs: object) -> object:
        self.describe_calls.append(kwargs)
        if self.describe_error is not None:
            raise self.describe_error
        return deepcopy(self.description)

    def upsert(self, **kwargs: object) -> object:
        self.upsert_calls.append(kwargs)
        if self.upsert_error is not None:
            raise self.upsert_error
        records = kwargs["data"]
        assert isinstance(records, list)
        description = self.description
        assert isinstance(description, Mapping)
        fields = description["fields"]
        assert isinstance(fields, list)
        primary_names = [field["name"] for field in fields if isinstance(field, Mapping) and field.get("is_primary")]
        assert len(primary_names) == 1
        primary_name = primary_names[0]
        for record in records:
            assert isinstance(record, dict)
            type(self).store[record[primary_name]] = deepcopy(record)
        if self.response is _DEFAULT_RESPONSE:
            return {"upsert_count": len(records), "ids": [record[primary_name] for record in records]}
        return self.response

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture(autouse=True)
def _fake_sdk(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("external_service") is not None:
        return
    _Client.description = _description()
    _Client.describe_error = None
    _Client.upsert_error = None
    _Client.close_error = None
    _Client.response = _DEFAULT_RESPONSE
    _Client.instances = []
    _Client.store = {}
    monkeypatch.setattr(milvus, "_load_milvus_sdk", lambda: (_Client, _DataType))


def _arrow_schema(
    *,
    id_type: pa.DataType | None = None,
    vector_type: pa.DataType | None = None,
    title_type: pa.DataType | None = None,
) -> pa.Schema:
    return pa.schema(
        [
            ("id", pa.int64() if id_type is None else id_type),
            ("embedding", pa.list_(pa.float32()) if vector_type is None else vector_type),
            ("title", pa.string() if title_type is None else title_type),
        ]
    )


def _sink(**kwargs: object) -> MilvusSink:
    options: dict[str, object] = {
        "uri": "https://milvus.example:19530",
        "primary_key": "id",
        "timeout": 5,
    }
    options.update(kwargs)
    return MilvusSink("items", **options)  # type: ignore[arg-type]


def _bound(**kwargs: object) -> BoundKeyedUpsertSink:
    return _sink(**kwargs).bind(_arrow_schema())


def _worker(**kwargs: object) -> Any:
    return _bound(**kwargs).open_worker(WriteContext("milvus-test"))


def _table(
    *,
    ids: list[int] | None = None,
    vectors: list[list[float]] | None = None,
    titles: list[str | None] | None = None,
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


def test_milvus_sink_is_public_without_importing_pymilvus() -> None:
    assert vane.MilvusSink is MilvusSink
    assert milvus.__all__ == ["MilvusSink"]


def test_milvus_sink_reports_the_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_pymilvus(name: str, *args: object, **kwargs: object) -> Any:
        if name == "pymilvus":
            raise ModuleNotFoundError("No module named 'pymilvus'", name="pymilvus")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_pymilvus)
    with pytest.raises(ImportError, match=r"vane-ai\[milvus\]"):
        _REAL_LOAD_MILVUS_SDK()


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"collection_name": ""}, ValueError, "collection_name"),
        ({"collection_name": " items"}, ValueError, "whitespace"),
        ({"uri": "milvus.example:19530"}, ValueError, "HTTP or HTTPS"),
        ({"uri": "https://milvus.example/database"}, ValueError, "database path"),
        ({"uri": "https://milvus.example:not-a-port"}, ValueError, "valid HTTP or HTTPS"),
        ({"uri": "https://user:secret@milvus.example"}, ValueError, "credentials"),
        ({"uri": "https://milvus.example?token=secret"}, ValueError, "query parameters"),
        ({"primary_key": ""}, ValueError, "primary_key"),
        ({"token": "plain-text"}, TypeError, "EnvironmentSecret"),
        ({"database": " "}, ValueError, "database"),
        ({"worker_count": True}, TypeError, "worker_count"),
        ({"worker_count": 0}, ValueError, "worker_count"),
        ({"max_batch_rows": -1}, ValueError, "max_batch_rows"),
        ({"max_batch_bytes": 1.5}, TypeError, "max_batch_bytes"),
        ({"max_retries": True}, TypeError, "max_retries"),
        ({"max_retries": -1}, ValueError, "max_retries"),
        ({"timeout": None}, TypeError, "timeout"),
        ({"timeout": float("inf")}, ValueError, "timeout"),
        ({"field_mapping": []}, TypeError, "field_mapping"),
        ({"field_mapping": {"id": "pk", "title": "pk"}}, ValueError, "targets"),
    ],
)
def test_milvus_sink_validates_constructor(overrides: dict[str, object], error: type[Exception], message: str) -> None:
    options: dict[str, object] = {
        "collection_name": "items",
        "uri": "https://milvus.example:19530",
        "primary_key": "id",
    }
    options.update(overrides)
    with pytest.raises(error, match=message):
        MilvusSink(**options)  # type: ignore[arg-type]


def test_milvus_sink_binds_key_mapping_and_execution_options() -> None:
    sink = _sink(
        primary_key="remote_id",
        field_mapping={"id": "remote_id"},
        worker_count=3,
        max_batch_rows=17,
        max_batch_bytes=2_048,
        max_retries=2,
    )
    bound = sink.bind(_arrow_schema())

    assert isinstance(bound, BoundKeyedUpsertSink)
    assert tuple(bound.key_columns) == ("id",)
    assert bound.execution_options.worker_count == 3
    assert bound.execution_options.batch_size == 17
    assert bound.execution_options.target_max_batch_bytes == 2_048
    assert bound.execution_options.max_retries == 2


def test_milvus_sink_writes_mapped_fields_with_a_varchar_primary_key() -> None:
    description = _description(primary_type=_DataType.VARCHAR)
    fields = description["fields"]
    assert isinstance(fields, list)
    for field in fields:
        assert isinstance(field, dict)
        field["name"] = {"id": "remote_id", "embedding": "vector", "title": "text"}[field["name"]]
    _Client.description = description
    schema = _arrow_schema(id_type=pa.string())
    sink = _sink(
        primary_key="remote_id",
        field_mapping={"id": "remote_id", "embedding": "vector", "title": "text"},
    )
    worker = sink.bind(schema).open_worker(WriteContext("mapped-varchar"))
    table = pa.table(
        {"id": ["a", "b"], "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "title": ["one", "two"]},
        schema=schema,
    )

    worker.write(table)

    assert _Client.instances[0].upsert_calls[0]["data"] == [
        {"remote_id": "a", "vector": pytest.approx([0.1, 0.2, 0.3]), "text": "one"},
        {"remote_id": "b", "vector": pytest.approx([0.4, 0.5, 0.6]), "text": "two"},
    ]


@pytest.mark.parametrize(
    ("arrow_type", "remote_type", "value"),
    [
        (pa.bool_(), _DataType.BOOL, True),
        (pa.int8(), _DataType.INT8, 8),
        (pa.int16(), _DataType.INT16, 16),
        (pa.int32(), _DataType.INT32, 32),
        (pa.int64(), _DataType.INT64, 64),
        (pa.float32(), _DataType.FLOAT, 1.25),
        (pa.float64(), _DataType.DOUBLE, 2.5),
        (pa.string(), _DataType.VARCHAR, "varchar"),
        (pa.large_string(), _DataType.TEXT, "text"),
    ],
)
def test_milvus_sink_supports_scalar_arrow_mappings(
    arrow_type: pa.DataType,
    remote_type: _DataType,
    value: object,
) -> None:
    _Client.description = _description(title_type=remote_type)
    schema = _arrow_schema(title_type=arrow_type)
    worker = _sink().bind(schema).open_worker(WriteContext("scalar-mapping"))
    table = pa.table(
        {"id": [1], "embedding": [[0.1, 0.2, 0.3]], "title": [value]},
        schema=schema,
    )

    result = worker.write(table)

    assert result.rows_affected == 1


def test_milvus_sink_rejects_invalid_bound_schema_and_mapping() -> None:
    with pytest.raises(TypeError, match="pyarrow.Schema"):
        _sink().bind(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique input column"):
        _sink().bind(pa.schema([("id", pa.int64()), ("id", pa.int64())]))
    with pytest.raises(ValueError, match="unknown input"):
        _sink(field_mapping={"missing": "remote"}).bind(_arrow_schema())
    with pytest.raises(ValueError, match="unique Milvus"):
        _sink(field_mapping={"id": "title"}).bind(_arrow_schema())
    with pytest.raises(ValueError, match="exactly one"):
        _sink(primary_key="missing").bind(_arrow_schema())
    with pytest.raises(ValueError, match="int64 or string"):
        _sink(primary_key="embedding").bind(_arrow_schema())
    with pytest.raises(ValueError, match="does not support"):
        _sink().bind(_arrow_schema(vector_type=pa.list_(pa.float64())))


def test_milvus_sink_defers_secret_resolution_and_sdk_import_to_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VANE_TEST_MILVUS_TOKEN", "worker-only-secret")
    sink = _sink(token=EnvironmentSecret("VANE_TEST_MILVUS_TOKEN"), database="analytics")
    serialized = cloudpickle.dumps(sink)
    assert b"worker-only-secret" not in serialized

    bound = sink.bind(_arrow_schema())
    assert b"worker-only-secret" not in cloudpickle.dumps(bound)
    assert not _Client.instances
    worker = bound.open_worker(WriteContext("secret-test"))

    assert _Client.instances[0].options == {
        "uri": "https://milvus.example:19530",
        "timeout": 5.0,
        "dedicated": True,
        "token": "worker-only-secret",
        "db_name": "analytics",
    }
    worker.close()


@pytest.mark.parametrize(
    ("description", "message"),
    [
        (None, "invalid description"),
        ({**_description(), "collection_name": "other"}, "different collection"),
        (_description(auto_id=True), "AutoID"),
        (_description(dynamic=True), "dynamic fields"),
        (_description(functions=[{"name": "embed"}]), "collections with functions"),
        ({**_description(), "functions": None}, "function metadata"),
        ({**_description(), "fields": []}, "no fields"),
    ],
)
def test_milvus_sink_rejects_unsupported_collection_contract(description: object, message: str) -> None:
    _Client.description = description
    with pytest.raises(ValueError, match=message):
        _worker()
    assert _Client.instances[0].close_calls == 1


def test_milvus_sink_rejects_invalid_collection_fields_and_primary_key() -> None:
    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    fields.append(deepcopy(fields[0]))
    _Client.description = description
    with pytest.raises(ValueError, match="duplicate field"):
        _worker()

    _Client.description = _description(primary_type=_DataType.INT32)
    with pytest.raises(ValueError, match="INT64 or VARCHAR"):
        _worker()

    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    assert isinstance(fields[0], dict)
    fields[0]["auto_id"] = True
    _Client.description = description
    with pytest.raises(ValueError, match="AutoID"):
        _worker()

    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    assert isinstance(fields[0], dict)
    fields[0]["is_primary"] = False
    _Client.description = description
    with pytest.raises(ValueError, match="primary key does not match"):
        _worker()


def test_milvus_sink_validates_required_unknown_and_function_output_fields() -> None:
    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    fields.append({"name": "required", "type": _DataType.INT64, "nullable": False, "params": {}})
    _Client.description = description
    with pytest.raises(ValueError, match="missing required"):
        _worker()

    _Client.description = _description()
    with pytest.raises(ValueError, match="absent from"):
        _sink(field_mapping={"title": "missing"}).bind(_arrow_schema()).open_worker(WriteContext("unknown"))

    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    assert isinstance(fields[1], dict)
    fields[1]["is_function_output"] = True
    _Client.description = description
    with pytest.raises(ValueError, match="function output"):
        _worker()


def test_milvus_sink_allows_omitted_nullable_and_defaulted_fields() -> None:
    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    fields.extend(
        [
            {"name": "optional", "type": _DataType.INT64, "nullable": True, "params": {}},
            {"name": "defaulted", "type": _DataType.INT64, "default_value": 7, "params": {}},
        ]
    )
    _Client.description = description
    worker = _worker()
    worker.write(_table())
    assert len(_Client.instances[0].upsert_calls) == 1


def test_milvus_sink_validates_remote_types_and_parameters() -> None:
    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    assert isinstance(fields[2], dict)
    fields[2]["type"] = _DataType.INT64
    _Client.description = description
    with pytest.raises(ValueError, match="does not match Arrow"):
        _worker()

    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    assert isinstance(fields[2], dict)
    fields[2]["params"] = {"max_length": 0}
    _Client.description = description
    with pytest.raises(ValueError, match="invalid max_length"):
        _worker()

    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    assert isinstance(fields[1], dict)
    fields[1]["params"] = {"dim": "3"}
    _Client.description = description
    with pytest.raises(ValueError, match="invalid dim"):
        _worker()


def test_milvus_sink_accepts_text_fields_and_fixed_vector_dimensions() -> None:
    _Client.description = _description(title_type=_DataType.TEXT)
    worker = _sink().bind(_arrow_schema(vector_type=pa.list_(pa.float32(), 3))).open_worker(WriteContext("fixed"))
    result = worker.write(_table(schema=_arrow_schema(vector_type=pa.list_(pa.float32(), 3))))
    assert result.rows_affected == 2

    with pytest.raises(ValueError, match="fixed dimension"):
        _sink().bind(_arrow_schema(vector_type=pa.list_(pa.float32(), 2))).open_worker(WriteContext("bad-fixed"))


def test_milvus_sink_validates_values_before_upsert() -> None:
    worker = _worker()
    with pytest.raises(ValueError, match="invalid dimension"):
        worker.write(_table(vectors=[[0.1, 0.2], [0.3, 0.4]]))
    with pytest.raises(ValueError, match="invalid value"):
        worker.write(_table(vectors=[[0.1, float("nan"), 0.3], [0.4, 0.5, 0.6]]))
    with pytest.raises(ValueError, match="max_length"):
        worker.write(_table(titles=["你" * 6, "two"]))
    with pytest.raises(ValueError, match="null"):
        worker.write(_table(titles=[None, "two"]))
    with pytest.raises(ValueError, match="null"):
        worker.write(_table(ids=[None, 2]))  # type: ignore[list-item]
    assert not _Client.instances[0].upsert_calls


def test_milvus_sink_allows_null_for_nullable_or_defaulted_mapped_field() -> None:
    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    assert isinstance(fields[2], dict)
    fields[2]["nullable"] = True
    _Client.description = description
    worker = _worker()
    worker.write(_table(titles=[None, "two"]))

    description = _description()
    fields = description["fields"]
    assert isinstance(fields, list)
    assert isinstance(fields[2], dict)
    fields[2]["default_value"] = "untitled"
    _Client.description = description
    worker = _worker()
    worker.write(_table(titles=[None, "two"]))


def test_milvus_sink_enforces_batch_and_schema_limits_before_upsert() -> None:
    worker = _worker(max_batch_rows=1)
    with pytest.raises(ValueError, match="max_batch_rows"):
        worker.write(_table())
    assert not _Client.instances[0].upsert_calls

    worker = _worker(max_batch_bytes=1)
    with pytest.raises(ValueError, match="max_batch_bytes"):
        worker.write(_table())
    assert not _Client.instances[1].upsert_calls

    worker = _worker()
    bad_schema = pa.schema([("id", pa.int64()), ("embedding", pa.list_(pa.float32())), ("other", pa.string())])
    with pytest.raises(ValueError, match="bound input schema"):
        worker.write(pa.table({"id": [1], "embedding": [[0.1, 0.2, 0.3]], "other": ["x"]}, schema=bad_schema))
    assert not _Client.instances[2].upsert_calls


def test_milvus_sink_writes_override_upserts_and_returns_bounded_results() -> None:
    worker = _worker()
    table = _table()

    first = worker.write(table)
    second = worker.write(table)

    assert first.rows_received == first.rows_affected == 2
    assert first.bytes_received == table.nbytes
    assert first.metadata == {"provider": "milvus", "collection": "items", "write_mode": "override"}
    assert len(first.warnings) == 1
    assert second.warnings == ()
    client = _Client.instances[0]
    assert client.describe_calls == [{"collection_name": "items", "timeout": 5.0}]
    assert client.upsert_calls[0]["collection_name"] == "items"
    assert client.upsert_calls[0]["timeout"] == 5.0
    assert client.upsert_calls[0]["partial_update"] is False
    assert client.upsert_calls[0]["data"] == [
        {"id": 1, "embedding": pytest.approx([0.1, 0.2, 0.3]), "title": "one"},
        {"id": 2, "embedding": pytest.approx([0.4, 0.5, 0.6]), "title": "two"},
    ]


def test_milvus_sink_validates_static_result_payload_before_upsert() -> None:
    collection = "c" * (65 * 1024)
    _Client.description = _description(collection_name=collection)
    sink = MilvusSink(collection, uri="https://milvus.example:19530", primary_key="id")

    with pytest.raises(ValueError, match="64 KiB"):
        sink.bind(_arrow_schema()).open_worker(WriteContext("bounded-result"))

    assert _Client.instances[0].close_calls == 1
    assert not _Client.instances[0].upsert_calls


@pytest.mark.parametrize(
    "response",
    [None, {}, {"upsert_count": True}, {"upsert_count": 1}, {"upsert_count": "2"}],
)
def test_milvus_sink_rejects_invalid_provider_results(response: object) -> None:
    _Client.response = response
    worker = _worker()
    with pytest.raises(RuntimeError, match="affected-row count"):
        worker.write(_table())


def test_milvus_sink_propagates_provider_failures_and_closes_on_abort() -> None:
    _Client.describe_error = RuntimeError("describe failed")
    with pytest.raises(RuntimeError, match="describe failed"):
        _worker()
    assert _Client.instances[0].close_calls == 1

    _Client.describe_error = None
    worker = _worker()
    _Client.upsert_error = TimeoutError("upsert timed out")
    with pytest.raises(TimeoutError, match="upsert timed out"):
        worker.write(_table())
    worker.abort(TimeoutError("upsert timed out"))
    assert _Client.instances[1].close_calls == 1
    worker.close()
    assert _Client.instances[1].close_calls == 1


def test_milvus_sink_retains_client_ownership_when_close_fails() -> None:
    worker = _worker()
    _Client.close_error = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        worker.close()
    assert _Client.instances[0].close_calls == 1

    _Client.close_error = None
    worker.close()
    assert _Client.instances[0].close_calls == 2


def test_milvus_sink_replay_replaces_the_same_primary_keys() -> None:
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
def test_milvus_sink_live_full_row_upsert() -> None:
    uri = os.environ.get("VANE_TEST_MILVUS_URI")
    if not uri:
        pytest.skip("VANE_TEST_MILVUS_URI is required for the external Milvus test")
    pymilvus = pytest.importorskip("pymilvus")
    token_value = os.environ.get("VANE_TEST_MILVUS_TOKEN")
    options: dict[str, object] = {"uri": uri, "timeout": 30.0, "dedicated": True}
    if token_value is not None:
        options["token"] = token_value
    client = pymilvus.MilvusClient(**options)
    collection = f"vane_sink_e2e_{uuid.uuid4().hex}"
    collection_created = False
    try:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", pymilvus.DataType.INT64, is_primary=True)
        schema.add_field("embedding", pymilvus.DataType.FLOAT_VECTOR, dim=2)
        schema.add_field("title", pymilvus.DataType.VARCHAR, max_length=64)
        index = client.prepare_index_params()
        index.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(
            collection_name=collection,
            schema=schema,
            index_params=index,
            consistency_level="Strong",
            timeout=30.0,
        )
        collection_created = True
        relation = vane.from_arrow(
            pa.table(
                {
                    "id": [1, 2],
                    "embedding": pa.array([[0.1, 0.2], [0.3, 0.4]], type=pa.list_(pa.float32(), 2)),
                    "title": ["one", "two"],
                }
            )
        )
        token = EnvironmentSecret("VANE_TEST_MILVUS_TOKEN") if token_value is not None else None
        summary = relation.write_datasink(
            MilvusSink(collection, uri=uri, primary_key="id", token=token, timeout=30.0),
            operation_id=f"milvus-live-{uuid.uuid4()}",
        )
        assert summary.rows_affected == 2
        rows = client.query(
            collection_name=collection, ids=[1, 2], output_fields=["id", "title"], consistency_level="Strong"
        )
        assert {row["id"]: row["title"] for row in rows} == {1: "one", 2: "two"}
    finally:
        try:
            if collection_created:
                client.drop_collection(collection_name=collection, timeout=30.0)
        finally:
            client.close()
