# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa
import pytest

import vane.datasink.milvus as milvus
from vane import EnvironmentSecret, MilvusSink
from vane.datasink import WriteContext


def _schema(*, auto_id: bool = False, dynamic: bool = False, primary_type: int = 5) -> dict[str, object]:
    return {
        "auto_id": auto_id,
        "enable_dynamic_field": dynamic,
        "fields": [
            {"name": "id", "type": primary_type, "is_primary": True, "nullable": False, "params": {}},
            {"name": "embedding", "type": 101, "nullable": False, "params": {"dim": 3}},
            {"name": "title", "type": 21, "nullable": False, "params": {"max_length": 16}},
        ],
    }


class _Client:
    schema: Mapping[str, object] = _schema()
    instances: list["_Client"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.upserts: list[dict[str, object]] = []
        self.closed = False
        type(self).instances.append(self)

    def describe_collection(self, **kwargs: object) -> Mapping[str, object]:
        assert kwargs == {"collection_name": "items", "timeout": 5.0}
        return self.schema

    def upsert(self, **kwargs: object) -> Mapping[str, int]:
        self.upserts.append(kwargs)
        return {"upsert_count": len(kwargs["data"])}

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _Client.schema = _schema()
    _Client.instances = []
    monkeypatch.setattr(milvus, "_load_milvus_client", lambda: _Client)


def _arrow_schema() -> pa.Schema:
    return pa.schema([("id", pa.int64()), ("embedding", pa.list_(pa.float32())), ("title", pa.string())])


def _sink(**kwargs: object) -> MilvusSink:
    options: dict[str, object] = {
        "uri": "http://milvus:19530",
        "primary_key": "id",
        "timeout": 5,
    }
    options.update(kwargs)
    return MilvusSink("items", **options)  # type: ignore[arg-type]


def _worker(**kwargs: object):
    return _sink(**kwargs).bind(_arrow_schema()).open_worker(WriteContext("test-op"))


def _table(
    *, ids: list[int] | None = None, vectors: list[list[float]] | None = None, titles: list[str] | None = None
) -> pa.Table:
    return pa.table(
        {
            "id": ids or [1, 2],
            "embedding": vectors or [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "title": titles or ["one", "two"],
        },
        schema=_arrow_schema(),
    )


def test_milvus_sink_binds_explicit_primary_key_and_writes_override_upserts() -> None:
    worker = _worker()

    result = worker.write(_table())

    assert result.rows_received == result.rows_affected == 2
    assert result.metadata == {"provider": "milvus", "collection": "items"}
    client = _Client.instances[0]
    assert client.kwargs == {"uri": "http://milvus:19530"}
    assert len(client.upserts) == 1
    assert client.upserts[0]["collection_name"] == "items"
    assert client.upserts[0]["timeout"] == 5.0
    assert client.upserts[0]["data"] == [
        {"id": 1, "embedding": pytest.approx([0.1, 0.2, 0.3]), "title": "one"},
        {"id": 2, "embedding": pytest.approx([0.4, 0.5, 0.6]), "title": "two"},
    ]
    worker.close()
    assert client.closed


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (_schema(auto_id=True), "auto_id disabled"),
        (_schema(dynamic=True), "dynamic fields"),
        (_schema(primary_type=4), "INT64 or VARCHAR"),
        ({**_schema(), "fields": _schema()["fields"][1:]}, "primary key does not match"),
    ],
)
def test_milvus_sink_rejects_unsafe_or_invalid_collection_schema(schema: Mapping[str, object], message: str) -> None:
    _Client.schema = schema

    with pytest.raises(ValueError, match=message):
        _worker()
    assert _Client.instances[0].closed


def test_milvus_sink_rejects_missing_required_and_unknown_fields() -> None:
    schema = _schema()
    schema["fields"] = [*schema["fields"], {"name": "required", "type": 5, "nullable": False, "params": {}}]
    _Client.schema = schema
    with pytest.raises(ValueError, match="missing required"):
        _worker()

    _Client.schema = _schema()
    sink = _sink(field_mapping={"title": "missing"})
    with pytest.raises(ValueError, match="absent"):
        sink.bind(_arrow_schema()).open_worker(WriteContext("test-op"))


def test_milvus_sink_validates_arrow_schema_mapping_and_primary_key() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _sink(primary_key="missing").bind(_arrow_schema())
    with pytest.raises(ValueError, match="int64 or string"):
        _sink(primary_key="embedding").bind(_arrow_schema())
    with pytest.raises(ValueError, match="unique"):
        _sink(field_mapping={"id": "id", "title": "id"}).bind(_arrow_schema())
    with pytest.raises(ValueError, match="does not support"):
        _sink().bind(pa.schema([("id", pa.int64()), ("embedding", pa.list_(pa.float64())), ("title", pa.string())]))
    with pytest.raises(ValueError, match="must not contain credentials"):
        MilvusSink("items", uri="https://user:secret@milvus.example", primary_key="id")


def test_milvus_sink_validates_collection_field_types_and_vector_dimensions() -> None:
    schema = _schema()
    schema["fields"][1]["type"] = 100
    _Client.schema = schema
    with pytest.raises(ValueError, match="does not match Arrow"):
        _worker()

    _Client.schema = _schema()
    worker = _worker()
    with pytest.raises(ValueError, match="invalid dimension"):
        worker.write(_table(vectors=[[0.1, 0.2], [0.3, 0.4]]))


def test_milvus_sink_enforces_batch_and_value_limits_before_sdk_submission() -> None:
    worker = _worker(max_batch_rows=1)
    with pytest.raises(ValueError, match="max_batch_rows"):
        worker.write(_table())
    assert not _Client.instances[0].upserts

    worker = _worker(max_batch_bytes=1)
    with pytest.raises(ValueError, match="max_batch_bytes"):
        worker.write(_table())
    assert not _Client.instances[1].upserts

    worker = _worker()
    with pytest.raises(ValueError, match="max_length"):
        worker.write(_table(titles=["x" * 17, "two"]))
    assert not _Client.instances[2].upserts

    worker = _worker()
    with pytest.raises(ValueError, match="max_length"):
        worker.write(_table(titles=["你" * 6, "two"]))
    assert not _Client.instances[3].upserts

    worker = _worker()
    with pytest.raises(ValueError, match="invalid value"):
        worker.write(_table(vectors=[[0.1, float("nan"), 0.3], [0.4, 0.5, 0.6]]))
    assert not _Client.instances[4].upserts


def test_milvus_sink_resolves_token_only_on_worker_and_redacts_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILVUS_TOKEN", "super-secret")
    sink = _sink(token=EnvironmentSecret("MILVUS_TOKEN"), database="default")
    bound = sink.bind(_arrow_schema())
    assert "super-secret" not in repr(sink)
    worker = bound.open_worker(WriteContext("test-op"))
    assert _Client.instances[0].kwargs == {
        "uri": "http://milvus:19530",
        "token": "super-secret",
        "db_name": "default",
    }
    worker.abort(RuntimeError("planned"))
    assert _Client.instances[0].closed


def test_milvus_sink_rejects_plaintext_tokens_and_invalid_provider_result() -> None:
    with pytest.raises(TypeError, match="EnvironmentSecret"):
        _sink(token="secret")  # type: ignore[arg-type]

    worker = _worker()
    _Client.instances[0].upsert = lambda **kwargs: {"upsert_count": 1}  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="affected-row count"):
        worker.write(_table())


def test_milvus_sink_rejects_nonfinite_timeout_and_mismatched_batches() -> None:
    with pytest.raises(ValueError, match="timeout"):
        _sink(timeout=float("nan"))

    worker = _worker()
    with pytest.raises(ValueError, match="bound input schema"):
        worker.write(pa.table({"id": [1], "embedding": [[0.1, 0.2, 0.3]], "other": ["x"]}))
