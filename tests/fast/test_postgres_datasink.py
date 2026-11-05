# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import builtins
import datetime as dt
import os
import uuid
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import cloudpickle
import pyarrow as pa
import pytest

import vane
import vane.datasink as datasink
import vane.datasink.postgres as postgres
from vane import EnvironmentSecret, PostgresSink
from vane.datasink import BoundDataSink, DataSinkExecutionOptions, DataSinkWriteError, WriteContext, WriteOutcome

_REAL_LOAD_ADBC = postgres._load_adbc_postgresql

_TARGET_TYPES = {"id": "bigint", "title": "text", "embedding": "vector"}


class _DriverError(Exception):
    pass


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self._rows: list[tuple[Any, ...]] = []
        self.rowcount = -1

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, parameters: tuple[Any, ...] | None = None) -> None:
        connection = self._connection
        connection.statements.append((sql, parameters))
        if "pg_attribute" in sql:
            self._rows = list(connection.target_types.items())
        elif sql.startswith("INSERT INTO"):
            failure = connection.merge_errors.pop(0) if connection.merge_errors else None
            if failure is not None:
                raise failure
            self.rowcount = connection.ingests[-1][1].num_rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def adbc_ingest(self, table_name: str, data: pa.Table, *, mode: str, temporary: bool) -> None:
        self._connection.ingests.append((table_name, data, mode, temporary))


class _Connection:
    instances: list[_Connection] = []
    target_types: dict[str, str] = _TARGET_TYPES
    close_error: BaseException | None = None

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.statements: list[tuple[str, tuple[Any, ...] | None]] = []
        self.ingests: list[tuple[str, pa.Table, str, bool]] = []
        self.merge_errors: list[BaseException | None] = []
        self.commit_errors: list[BaseException] = []
        self.rollback_error: BaseException | None = None
        self.commits = 0
        self.rollbacks = 0
        self.close_calls = 0
        type(self).instances.append(self)

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        if self.commit_errors:
            raise self.commit_errors.pop(0)
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1
        if self.rollback_error is not None:
            raise self.rollback_error

    def close(self) -> None:
        self.close_calls += 1
        if type(self).close_error is not None:
            raise type(self).close_error

    @property
    def merge_sql(self) -> str:
        return next(sql for sql, _ in self.statements if sql.startswith("INSERT INTO"))


@pytest.fixture(autouse=True)
def _fake_connection(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("external_service") is not None:
        return
    _Connection.instances = []
    _Connection.target_types = dict(_TARGET_TYPES)
    _Connection.close_error = None
    monkeypatch.setattr(postgres, "_connect", _Connection)


def _schema(*, vector_type: pa.DataType | None = None) -> pa.Schema:
    return pa.schema(
        [
            ("source_id", pa.int64()),
            ("source_title", pa.string()),
            ("embedding", pa.list_(pa.float32(), 3) if vector_type is None else vector_type),
        ]
    )


def _table(schema: pa.Schema | None = None) -> pa.Table:
    return pa.table(
        {
            "source_id": [1, 2],
            "source_title": ["one", "two"],
            "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        },
        schema=_schema() if schema is None else schema,
    )


def _sink(**overrides: object) -> PostgresSink:
    options: dict[str, object] = {
        "dsn": "postgresql://writer@db.example:5432/app",
        "key_columns": "source_id",
        "schema": "public",
        "field_mapping": {"source_id": "id", "source_title": "title"},
        "timeout": 5,
    }
    options.update(overrides)
    return PostgresSink("items", **options)  # type: ignore[arg-type]


def _bound(*, schema: pa.Schema | None = None, **overrides: object) -> BoundDataSink:
    return _sink(**overrides).bind(_schema() if schema is None else schema)


def _worker(*, schema: pa.Schema | None = None, **overrides: object) -> Any:
    return _bound(schema=schema, **overrides).open_worker(WriteContext("postgres-test-operation"))


def _single_column_worker(source_type: pa.DataType, target_type: str = "text") -> Any:
    _Connection.target_types = {"id": "bigint", "value": target_type}
    sink = PostgresSink("items", dsn="postgresql://db.example/app", key_columns="id")
    return sink.bind(pa.schema([("id", pa.int64()), ("value", source_type)])).open_worker(WriteContext("types"))


def test_postgres_sink_is_public_without_importing_adbc() -> None:
    assert vane.PostgresSink is PostgresSink
    assert postgres.__all__ == ["PostgresSink"]


def test_postgres_sink_reports_the_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_driver(name: str, *args: object, **kwargs: object) -> Any:
        if name.startswith("adbc_driver_postgresql"):
            raise ModuleNotFoundError("No module named 'adbc_driver_postgresql'", name="adbc_driver_postgresql")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_driver)
    with pytest.raises(ImportError, match=r"vane-ai\[postgres\]"):
        _REAL_LOAD_ADBC()


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"dsn": "postgresql://writer:secret@db.example/app"}, ValueError, "must not contain a password"),
        ({"dsn": "postgres://db.example/app?password=secret"}, ValueError, "must not contain a password"),
        ({"dsn": "host=db.example password=secret"}, ValueError, "must not contain a password"),
        ({"dsn": "host=db.example\nuser=x"}, ValueError, "NUL, CR, or LF"),
        ({"dsn": "postgresql://writer#:secret@db.example/app"}, ValueError, "must not contain '#'"),
        ({"dsn": "postgresql://writer@db.example/app?password="}, ValueError, "must not contain a password"),
        ({"dsn": "host=db.example user='writer'password=secret"}, ValueError, "must not contain a password"),
        ({"dsn": "host=db.example\tpassword = secret"}, ValueError, "must not contain a password"),
        ({"dsn": "host=db.example user='unterminated"}, ValueError, "not a valid libpq connection string"),
        ({"dsn": "host=db.example user"}, ValueError, "not a valid libpq connection string"),
        ({"password": "secret"}, TypeError, "EnvironmentSecret or None"),
        ({"key_columns": []}, ValueError, "at least one column"),
        ({"key_columns": ["source_id", "source_id"]}, ValueError, "duplicates"),
        ({"key_columns": 1}, TypeError, "column name"),
        ({"schema": "s" * 64}, ValueError, "at most 63 UTF-8 bytes"),
        ({"field_mapping": {"source_title": "名" * 22}}, ValueError, "at most 63 UTF-8 bytes"),
        ({"field_mapping": ["source_id"]}, TypeError, "mapping"),
        ({"worker_count": 0}, ValueError, "positive"),
        ({"max_retries": -1}, ValueError, "non-negative"),
        ({"timeout": True}, TypeError, "positive integer"),
    ],
)
def test_postgres_sink_validates_constructor(
    overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _sink(**overrides)


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://writer@db.example/app",
        "host=db.example user=writer dbname=app",
        "host=db.example application_name='a password=b'",
        "host=db.example user=a\\ password=b",
    ],
)
def test_postgres_sink_accepts_dsns_without_passwords(dsn: str) -> None:
    assert _sink(dsn=dsn).dsn == dsn


def test_postgres_sink_binds_keys_mapping_and_execution_options() -> None:
    sink = _sink(worker_count=3, max_batch_rows=10, max_batch_bytes=1024, max_retries=2)
    bound = sink.bind(_schema())

    assert bound.execution_options == DataSinkExecutionOptions(
        worker_count=3, batch_size=10, target_max_batch_bytes=1024, max_retries=2
    )
    assert isinstance(bound, datasink.BoundKeyedUpsertSink)
    assert tuple(bound.key_columns) == ("source_id",)
    cloudpickle.loads(cloudpickle.dumps(bound))


def test_postgres_sink_rejects_invalid_bound_schema_and_mappings() -> None:
    with pytest.raises(ValueError, match="unknown input columns"):
        _sink(field_mapping={"missing": "id"}).bind(_schema())
    with pytest.raises(ValueError, match="key_columns contains unknown"):
        _sink(key_columns="id").bind(_schema())
    with pytest.raises(ValueError, match="unique PostgreSQL column names"):
        _sink(field_mapping={"source_id": "title", "source_title": "title"}).bind(_schema())
    with pytest.raises(ValueError, match="unique input column names"):
        _sink().bind(pa.schema([("a", pa.int32()), ("a", pa.int32())]))


@pytest.mark.parametrize(
    "source_type",
    [
        pa.struct([("a", pa.int32())]),
        pa.map_(pa.string(), pa.int64()),
        pa.list_(pa.list_(pa.int64())),
        pa.month_day_nano_interval(),
        pa.decimal256(40, 2),
        pa.binary_view(),
    ],
)
def test_postgres_sink_rejects_unsupported_sources_before_opening_a_connection(source_type: pa.DataType) -> None:
    with pytest.raises(ValueError, match="unsupported Arrow type"):
        _sink().bind(pa.schema([("source_id", pa.int64()), ("value", source_type)]))
    assert _Connection.instances == []


def test_postgres_sink_prepares_a_parameterized_typed_upsert() -> None:
    worker = _worker()
    connection = _Connection.instances[0]

    assert connection.dsn == "postgresql://writer@db.example:5432/app"
    assert connection.statements[0] == ("SELECT set_config('statement_timeout', $1, false)", ("5s",))
    assert connection.statements[1][1] == ('"public"."items"',)
    assert connection.commits == 1
    _Connection.instances[0].statements.clear()
    worker.write(_table())
    stage = connection.ingests[0][0]
    assert stage.startswith("vane_stage_")
    assert connection.merge_sql == (
        'INSERT INTO "public"."items" AS "vane_target" ("id", "title", "embedding") '
        'SELECT CAST("c0" AS bigint), CAST("c1" AS text), CAST("c2" AS vector) '
        f'FROM "{stage}" ON CONFLICT ("id") DO UPDATE SET "title" = EXCLUDED."title", '
        '"embedding" = EXCLUDED."embedding"'
    )
    assert connection.statements[-1] == (f'TRUNCATE "{stage}"', None)


def test_postgres_sink_inserts_only_new_keys_when_every_column_is_a_key() -> None:
    _Connection.target_types = {"tenant": "text", "item": "uuid"}
    sink = PostgresSink("links", dsn="postgresql://db.example/app", key_columns=["tenant", "item"])
    worker = sink.bind(pa.schema([("tenant", pa.string()), ("item", pa.string())])).open_worker(WriteContext("keys"))
    worker.write(pa.table({"tenant": ["a"], "item": ["a1a2a3a4-b1b2-c1c2-d1d2-d3d4d5d6d7d8"]}))

    merge = _Connection.instances[0].merge_sql
    assert merge.startswith(
        'INSERT INTO "links" AS "vane_target" ("tenant", "item") SELECT CAST("c0" AS text), CAST("c1" AS uuid)'
    )
    assert merge.endswith('ON CONFLICT ("tenant", "item") DO NOTHING')


def test_postgres_sink_quotes_identifiers() -> None:
    _Connection.target_types = {'we"ird': "bigint", "值": "text"}
    sink = PostgresSink('ta"ble', dsn="postgresql://db.example/app", schema="Mixed Case", key_columns='we"ird')
    worker = sink.bind(pa.schema([('we"ird', pa.int64()), ("值", pa.string())])).open_worker(WriteContext("quote"))
    worker.write(pa.table({'we"ird': [1], "值": ["v"]}))

    connection = _Connection.instances[0]
    assert connection.statements[1][1] == ('"Mixed Case"."ta""ble"',)
    assert connection.merge_sql.startswith('INSERT INTO "Mixed Case"."ta""ble" AS "vane_target" ("we""ird", "值")')
    assert 'ON CONFLICT ("we""ird") DO UPDATE SET "值" = EXCLUDED."值"' in connection.merge_sql
    assert connection.ingests[0][1].column_names == ["c0", "c1"]


@pytest.mark.parametrize(
    ("target_types", "message"),
    [({}, "does not exist"), ({"id": "bigint", "title": "text"}, "no columns named \\['embedding'\\]")],
)
def test_postgres_sink_checks_the_target_table_and_closes_on_failure(
    target_types: dict[str, str], message: str
) -> None:
    _Connection.target_types = target_types
    with pytest.raises(ValueError, match=message):
        _worker()
    assert _Connection.instances[0].close_calls == 1


def test_postgres_sink_stages_batches_in_one_transaction_each() -> None:
    worker = _worker()
    connection = _Connection.instances[0]
    first = worker.write(_table())
    second = worker.write(_table())
    worker.close()

    assert [(mode, temporary) for _, _, mode, temporary in connection.ingests] == [("create", True), ("append", True)]
    assert len({name for name, *_ in connection.ingests}) == 1
    assert connection.commits == 3
    assert connection.rollbacks == 0
    assert connection.close_calls == 1
    staged = connection.ingests[0][1]
    assert staged.schema == pa.schema([("c0", pa.int64()), ("c1", pa.string()), ("c2", pa.list_(pa.float32()))])
    assert staged.column("c2").to_pylist()[0] == pytest.approx([0.1, 0.2, 0.3])
    assert first.rows_received == first.rows_affected == 2
    assert first.metadata == {"provider": "postgres", "table": "items", "write_mode": "upsert", "schema": "public"}
    assert len(first.warnings) == 1 and "independently" in first.warnings[0]
    assert second.warnings == ()


@pytest.mark.parametrize(
    ("source_type", "values", "wire_type", "expected"),
    [
        (pa.uint64(), [2**63 - 1], pa.int64(), [2**63 - 1]),
        (pa.date64(), [dt.date(2026, 10, 30)], pa.date32(), [dt.date(2026, 10, 30)]),
        (
            pa.timestamp("ns", tz="UTC"),
            [dt.datetime(2026, 10, 30, 8, 0, 0, 123456, tzinfo=dt.timezone.utc)],
            pa.timestamp("us", tz="UTC"),
            [dt.datetime(2026, 10, 30, 8, 0, 0, 123456, tzinfo=dt.timezone.utc)],
        ),
        (pa.time32("s"), [dt.time(8, 0)], pa.time64("us"), [dt.time(8, 0)]),
        (pa.duration("ns"), [dt.timedelta(seconds=5)], pa.duration("us"), [dt.timedelta(seconds=5)]),
        (pa.list_(pa.uint64()), [[1, 2]], pa.list_(pa.int64()), [[1, 2]]),
    ],
)
def test_postgres_sink_sends_types_that_adbc_stores_exactly(
    source_type: pa.DataType, values: list[object], wire_type: pa.DataType, expected: list[object]
) -> None:
    worker = _single_column_worker(source_type)
    worker.write(pa.table({"id": [1], "value": pa.array(values, type=source_type)}))

    staged = _Connection.instances[0].ingests[0][1].column("c1")
    assert staged.type == wire_type
    if pa.types.is_timestamp(wire_type) and wire_type.tz is not None:
        # Compare UTC values without a time zone database, which Windows CI does not install.
        staged = staged.combine_chunks().view(pa.timestamp(wire_type.unit))
        expected = [
            value.astimezone(dt.timezone.utc).replace(tzinfo=None)
            for value in expected
            if isinstance(value, dt.datetime)
        ]
    assert staged.to_pylist() == expected


def test_postgres_sink_decodes_dictionaries() -> None:
    worker = _single_column_worker(pa.dictionary(pa.int32(), pa.string()))
    worker.write(pa.table({"id": [1, 2], "value": pa.array(["a", "b"]).dictionary_encode()}))
    assert _Connection.instances[0].ingests[0][1].column("c1").to_pylist() == ["a", "b"]


def test_postgres_sink_sends_fixed_size_lists_as_lists() -> None:
    _worker().write(_table())
    assert _Connection.instances[0].ingests[0][1].column("c2").type == pa.list_(pa.float32())


@pytest.mark.parametrize(
    ("source_type", "values"),
    [
        (pa.uint64(), [2**63 + 5]),
        (pa.timestamp("ns"), [1_793_347_200_123_456_789]),
        (pa.time64("ns"), [1]),
        (pa.list_(pa.uint64()), [[2**64 - 1]]),
    ],
)
def test_postgres_sink_rejects_values_adbc_would_change_before_sending(
    source_type: pa.DataType, values: list[object]
) -> None:
    worker = _single_column_worker(source_type)
    with pytest.raises(ValueError, match="cannot be sent without changing its values"):
        worker.write(pa.table({"id": [1], "value": pa.array(values, type=source_type)}))
    assert _Connection.instances[0].ingests == []


def test_postgres_sink_enforces_schema_and_batch_limits_before_sending() -> None:
    with pytest.raises(ValueError, match="max_batch_rows"):
        _worker(max_batch_rows=1).write(_table())
    with pytest.raises(ValueError, match="max_batch_bytes"):
        _worker(max_batch_bytes=8).write(_table())
    with pytest.raises(ValueError, match="does not match the bound input schema"):
        _worker().write(pa.table({"source_id": [1]}))
    assert all(connection.ingests == [] for connection in _Connection.instances)


def test_postgres_sink_skips_an_empty_batch() -> None:
    result = _worker().write(_table().slice(0, 0))
    assert result.rows_received == result.rows_affected == 0
    assert _Connection.instances[0].ingests == []


_WORKER_PASSWORD = "worker-only p@ss:/?&#%='\\"


@pytest.mark.parametrize(
    ("dsn", "expected"),
    [
        (
            "postgresql://writer@private-db.example/app",
            "postgresql://writer@private-db.example/app?password=worker-only%20p%40ss%3A%2F%3F%26%23%25%3D%27%5C",
        ),
        (
            "postgres://writer@private-db.example/app?sslmode=require",
            "postgres://writer@private-db.example/app?sslmode=require"
            "&password=worker-only%20p%40ss%3A%2F%3F%26%23%25%3D%27%5C",
        ),
        (
            "host=private-db.example user=writer dbname=app",
            "password='worker-only p@ss:/?&#%=\\'\\\\' host=private-db.example user=writer dbname=app",
        ),
    ],
)
def test_postgres_sink_resolves_the_dsn_and_password_on_the_worker(
    monkeypatch: pytest.MonkeyPatch, dsn: str, expected: str
) -> None:
    monkeypatch.setenv("VANE_TEST_POSTGRES_DSN", dsn)
    monkeypatch.setenv("VANE_TEST_POSTGRES_PASSWORD", _WORKER_PASSWORD)
    sink = _sink(
        dsn=EnvironmentSecret("VANE_TEST_POSTGRES_DSN"), password=EnvironmentSecret("VANE_TEST_POSTGRES_PASSWORD")
    )
    bound = sink.bind(_schema())
    for serialized in (cloudpickle.dumps(sink), cloudpickle.dumps(bound)):
        assert b"private-db.example" not in serialized
        assert b"worker-only" not in serialized
    assert "worker-only" not in repr(sink)

    bound.open_worker(WriteContext("secret"))
    assert _Connection.instances[0].dsn == expected


@pytest.mark.parametrize(
    ("dsn", "password", "message"),
    [
        ("postgresql://writer:inline@db.example/app", None, "must not contain a password"),
        ("host=db.example password=inline", None, "must not contain a password"),
        ("postgresql://writer@db.example/app", "", "password must not be empty"),
    ],
)
def test_postgres_sink_checks_worker_secrets_before_connecting(
    monkeypatch: pytest.MonkeyPatch, dsn: str, password: str | None, message: str
) -> None:
    monkeypatch.setenv("VANE_TEST_POSTGRES_DSN", dsn)
    options: dict[str, object] = {"dsn": EnvironmentSecret("VANE_TEST_POSTGRES_DSN")}
    if password is not None:
        monkeypatch.setenv("VANE_TEST_POSTGRES_PASSWORD", password)
        options["password"] = EnvironmentSecret("VANE_TEST_POSTGRES_PASSWORD")
    bound = _bound(**options)

    with pytest.raises(ValueError, match=message) as error:
        bound.open_worker(WriteContext("secret"))
    assert "inline" not in str(error.value)
    assert _Connection.instances == []


@pytest.mark.parametrize("dsn", ["postgresql://writer@db.example/app", "host=db.example user=writer dbname=app"])
def test_postgres_sink_redacts_the_password_from_connection_errors(monkeypatch: pytest.MonkeyPatch, dsn: str) -> None:
    def echoing_connect(connection_string: str) -> Any:
        # libpq echoes some malformed connection strings in its errors.
        raise _DriverError(f'invalid connection string: "{connection_string}"')

    monkeypatch.setenv("VANE_TEST_POSTGRES_PASSWORD", _WORKER_PASSWORD)
    monkeypatch.setattr(postgres, "_connect", echoing_connect)
    bound = _bound(dsn=dsn, password=EnvironmentSecret("VANE_TEST_POSTGRES_PASSWORD"))

    with pytest.raises(RuntimeError, match="PostgreSQL connection failed: _DriverError: invalid connection") as error:
        bound.open_worker(WriteContext("connect"))
    message = str(error.value)
    escaped = _WORKER_PASSWORD.replace("\\", "\\\\").replace("'", "\\'")
    for form in ("worker-only", quote(_WORKER_PASSWORD, safe=""), escaped):
        assert form not in message
    assert "***" in message
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.parametrize(("driver_version", "rejected"), [((1, 12), True), ((1, 13), False), (None, False)])
def test_postgres_sink_rejects_null_list_elements_with_drivers_before_1_13(
    monkeypatch: pytest.MonkeyPatch, driver_version: tuple[int, int] | None, rejected: bool
) -> None:
    monkeypatch.setattr(postgres, "_adbc_postgresql_version", lambda: driver_version)
    worker = _single_column_worker(pa.list_(pa.float64()), "double precision[]")
    connection = _Connection.instances[0]
    # A NULL list is written correctly by every driver version.
    worker.write(pa.table({"id": [1, 2], "value": pa.array([[1.5], None], type=pa.list_(pa.float64()))}))
    assert len(connection.ingests) == 1

    with_null_element = pa.table({"id": [3], "value": pa.array([[1.5, None]], type=pa.list_(pa.float64()))})
    if rejected:
        with pytest.raises(ValueError, match="NULL list elements"):
            worker.write(with_null_element)
        assert len(connection.ingests) == 1
    else:
        worker.write(with_null_element)
        assert connection.ingests[-1][1].column("c1").to_pylist() == [[1.5, None]]


def test_postgres_sink_rolls_back_a_failed_batch_and_recreates_the_stage() -> None:
    worker = _worker()
    connection = _Connection.instances[0]
    connection.merge_errors = [_DriverError("expected 3 dimensions, not 2")]

    with pytest.raises(RuntimeError, match="batch was rolled back: _DriverError: expected 3 dimensions"):
        worker.write(_table())
    assert connection.rollbacks == 1

    worker = _worker()
    connection = _Connection.instances[1]
    connection.merge_errors = [None, _DriverError("constraint")]
    worker.write(_table())
    with pytest.raises(RuntimeError, match="rolled back"):
        worker.write(_table())
    assert [mode for _, _, mode, _ in connection.ingests] == ["create", "append"]


def test_postgres_sink_keeps_the_batch_error_when_rollback_fails() -> None:
    worker = _worker()
    connection = _Connection.instances[0]
    connection.merge_errors = [_DriverError("bad vector")]
    connection.rollback_error = _DriverError("connection lost")

    with pytest.raises(RuntimeError, match="rolled back: _DriverError: bad vector") as exc_info:
        worker.write(_table())
    if hasattr(BaseException, "add_note"):  # Python 3.11+
        notes = getattr(exc_info.value.__cause__, "__notes__", [])
        assert any("rollback also failed: _DriverError" in note for note in notes)


def test_postgres_sink_reports_an_unknown_commit_outcome_without_rollback() -> None:
    worker = _worker()
    connection = _Connection.instances[0]
    connection.commit_errors = [_DriverError("server closed the connection")]

    with pytest.raises(RuntimeError, match="commit did not report its outcome.*replaying it upserts the same keys"):
        worker.write(_table())
    assert connection.rollbacks == 0


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, asyncio.CancelledError, SystemExit, GeneratorExit])
def test_postgres_sink_interruption_keeps_its_identity(error_type: type[BaseException]) -> None:
    interruption = error_type("planned write interruption")
    worker = _worker()
    connection = _Connection.instances[0]
    connection.merge_errors = [interruption]
    try:
        with pytest.raises(error_type) as exc_info:
            worker.write(_table())
        assert exc_info.value is interruption
        assert datasink._is_execution_interruption(exc_info.value)
        assert "rolled back" in exc_info.value.args[0]
        assert "planned write interruption" in exc_info.value.args[0]
        worker.abort(interruption)
    finally:
        worker.close()
    assert connection.rollbacks == 1
    assert connection.close_calls == 1


def test_postgres_sink_retains_connection_ownership_when_close_fails() -> None:
    worker = _worker()
    _Connection.close_error = _DriverError("close failed")
    with pytest.raises(_DriverError):
        worker.close()
    _Connection.close_error = None
    worker.close()
    worker.close()
    assert _Connection.instances[0].close_calls == 2


@pytest.fixture(params=["local-fast", "local", pytest.param("ray", marks=pytest.mark.real_ray)])
def _postgres_runner(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    runner_type = request.param
    runner = None
    if runner_type == "ray":
        request.getfixturevalue("ray_local")
        from vane.runners.ray.runner import RayRunner

        runner = RayRunner(address=None, max_task_backlog=None)
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    if runner is not None:
        monkeypatch.setattr(vane._native, "set_runner_ray", lambda *_args, **_kwargs: runner)
    try:
        yield runner_type
    finally:
        if runner_type == "ray":
            assert runner is not None
            runner.close()


@pytest.fixture
def _live_schema() -> Iterator[tuple[Any, str]]:
    dsn = os.environ.get("VANE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("VANE_TEST_POSTGRES_DSN is required for the external PostgreSQL test")
    adbc = pytest.importorskip("adbc_driver_postgresql.dbapi")
    password = os.environ.get("VANE_TEST_POSTGRES_PASSWORD")
    connection = adbc.connect(dsn if password is None else postgres._with_password(dsn, password))
    connection.adbc_connection.set_autocommit(True)
    schema = f"vane_sink_{uuid.uuid4().hex}"
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(f'CREATE SCHEMA "{schema}"')
    try:
        yield connection, schema
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            connection.close()


def _live_rows(connection: Any, sql: str) -> list[tuple[Any, ...]]:
    with connection.cursor() as cursor:
        cursor.execute(sql)
        return cursor.fetchall()


def _live_sink(schema: str, table: str, **options: Any) -> PostgresSink:
    password = EnvironmentSecret("VANE_TEST_POSTGRES_PASSWORD") if "VANE_TEST_POSTGRES_PASSWORD" in os.environ else None
    return PostgresSink(
        table,
        dsn=EnvironmentSecret("VANE_TEST_POSTGRES_DSN"),
        password=password,
        schema=schema,
        timeout=60,
        **options,
    )


_LIVE_ITEMS = """
    CREATE TABLE "{schema}"."items" (
        id uuid PRIMARY KEY,
        title varchar(32) NOT NULL,
        embedding vector(3) NOT NULL,
        doc jsonb,
        seen_at timestamptz NOT NULL,
        qty integer
    )
"""

_LIVE_INPUT = """
    SELECT
        ('00000000-0000-0000-0000-00000000000' || i)::UUID AS item_id,
        '{prefix}' || i AS title,
        [i / 10.0, 0.5, 1.0]::FLOAT[] AS embedding,
        ('{{"rank": ' || i || '}}')::JSON AS doc,
        TIMESTAMPTZ '2026-10-30 08:00:00.123456+00' AS seen_at,
        (i * 10)::BIGINT AS qty
    FROM range(1, 4) AS t(i)
"""


@pytest.mark.external_service
def test_postgres_sink_live_upsert(_live_schema: tuple[Any, str], _postgres_runner: str) -> None:
    connection, schema = _live_schema
    with connection.cursor() as cursor:
        cursor.execute(_LIVE_ITEMS.format(schema=schema))

    for prefix in ("first-", "second-"):
        with vane.connect() as source:
            summary = source.sql(_LIVE_INPUT.format(prefix=prefix)).write_datasink(
                _live_sink(
                    schema,
                    "items",
                    key_columns="item_id",
                    field_mapping={"item_id": "id"},
                    worker_count=2,
                    max_batch_rows=1,
                ),
                operation_id=f"postgres-live-{uuid.uuid4()}",
            )
        assert summary.outcome is WriteOutcome.APPLIED
        assert summary.rows_received == summary.rows_affected == 3

    rows = _live_rows(
        connection,
        f'SELECT id::text, title, embedding::text, doc ->> \'rank\', seen_at, qty FROM "{schema}"."items" ORDER BY id',
    )
    assert [(row[0][-1], row[1], row[2], row[3], row[5]) for row in rows] == [
        ("1", "second-1", "[0.1,0.5,1]", "1", 10),
        ("2", "second-2", "[0.2,0.5,1]", "2", 20),
        ("3", "second-3", "[0.3,0.5,1]", "3", 30),
    ]
    assert {row[4] for row in rows} == {dt.datetime(2026, 10, 30, 8, 0, 0, 123456, tzinfo=dt.timezone.utc)}


@pytest.mark.external_service
def test_postgres_sink_live_rolls_back_a_batch_with_an_invalid_vector(
    _live_schema: tuple[Any, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VANE_RUNNER", "local")
    connection, schema = _live_schema
    with connection.cursor() as cursor:
        cursor.execute(f'CREATE TABLE "{schema}"."items" (id bigint PRIMARY KEY, embedding vector(3) NOT NULL)')
    relation = vane.sql(
        "SELECT i::BIGINT AS id, (CASE WHEN i = 2 THEN [1.0, 2.0] ELSE [1.0, 2.0, 3.0] END)::FLOAT[] AS embedding "
        "FROM range(1, 4) AS t(i)"
    )
    with pytest.raises(DataSinkWriteError) as exc_info:
        relation.write_datasink(_live_sink(schema, "items", key_columns="id"))

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert "rolled back" in str(exc_info.value)
    assert _live_rows(connection, f'SELECT count(*) FROM "{schema}"."items"') == [(0,)]


@pytest.mark.external_service
@pytest.mark.parametrize(
    ("value_sql", "column_type"),
    [
        ("18446744073709551615::UBIGINT", "bigint"),
        ("TIMESTAMP_NS '2026-10-30 08:00:00.123456789'", "timestamp"),
    ],
)
def test_postgres_sink_live_rejects_values_postgres_would_change(
    _live_schema: tuple[Any, str], monkeypatch: pytest.MonkeyPatch, value_sql: str, column_type: str
) -> None:
    monkeypatch.setenv("VANE_RUNNER", "local")
    connection, schema = _live_schema
    with connection.cursor() as cursor:
        cursor.execute(f'CREATE TABLE "{schema}"."items" (id bigint PRIMARY KEY, value {column_type})')
    relation = vane.sql(f"SELECT 1::BIGINT AS id, {value_sql} AS value")
    with pytest.raises(DataSinkWriteError, match="cannot be sent without changing its values"):
        relation.write_datasink(_live_sink(schema, "items", key_columns="id"))
    assert _live_rows(connection, f'SELECT count(*) FROM "{schema}"."items"') == [(0,)]


@pytest.mark.external_service
def test_postgres_sink_live_applies_the_target_column_length(
    _live_schema: tuple[Any, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VANE_RUNNER", "local")
    connection, schema = _live_schema
    with connection.cursor() as cursor:
        cursor.execute(f'CREATE TABLE "{schema}"."items" (k varchar(3) PRIMARY KEY, v varchar(5))')
        cursor.execute(f"""INSERT INTO "{schema}"."items" VALUES ('abc', 'old')""")

    for query in ("SELECT 'abcX' AS k, 'new' AS v", "SELECT 'abd' AS k, 'too long' AS v"):
        with pytest.raises(DataSinkWriteError, match="value too long"):
            vane.sql(query).write_datasink(_live_sink(schema, "items", key_columns="k"))
    # A truncated key would have replaced the existing row instead of failing.
    assert _live_rows(connection, f'SELECT k, v FROM "{schema}"."items"') == [("abc", "old")]


@pytest.mark.external_service
def test_postgres_sink_live_upserts_into_a_table_named_excluded(
    _live_schema: tuple[Any, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VANE_RUNNER", "local")
    connection, schema = _live_schema
    with connection.cursor() as cursor:
        cursor.execute(f'CREATE TABLE "{schema}"."excluded" (id bigint PRIMARY KEY, v text)')
    for value in ("first", "second"):
        vane.sql(f"SELECT 1::BIGINT AS id, '{value}' AS v").write_datasink(
            _live_sink(schema, "excluded", key_columns="id")
        )
    assert _live_rows(connection, f'SELECT id, v FROM "{schema}"."excluded"') == [(1, "second")]


@pytest.mark.external_service
def test_postgres_sink_live_keeps_null_list_elements(
    _live_schema: tuple[Any, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VANE_RUNNER", "local")
    connection, schema = _live_schema
    with connection.cursor() as cursor:
        cursor.execute(f'CREATE TABLE "{schema}"."items" (id bigint PRIMARY KEY, f float8[], s text[], b boolean[])')
    relation = vane.sql(
        "SELECT 1::BIGINT AS id, [1.5, NULL]::DOUBLE[] AS f, ['a', NULL]::VARCHAR[] AS s, [true, NULL]::BOOLEAN[] AS b"
    )
    sink = _live_sink(schema, "items", key_columns="id")
    driver_version = postgres._adbc_postgresql_version()
    if driver_version is not None and driver_version < (1, 13):
        with pytest.raises(DataSinkWriteError, match="NULL list elements"):
            relation.write_datasink(sink)
        assert _live_rows(connection, f'SELECT count(*) FROM "{schema}"."items"') == [(0,)]
    else:
        relation.write_datasink(sink)
        assert _live_rows(connection, f'SELECT f::text, s::text, b::text FROM "{schema}"."items"') == [
            ("{1.5,NULL}", "{a,NULL}", "{t,NULL}")
        ]


@pytest.mark.external_service
def test_postgres_sink_live_rejects_duplicate_keys_before_writing(
    _live_schema: tuple[Any, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VANE_RUNNER", "local")
    connection, schema = _live_schema
    with connection.cursor() as cursor:
        cursor.execute(f'CREATE TABLE "{schema}"."items" (id bigint PRIMARY KEY, title text)')
    relation = vane.sql("SELECT 1::BIGINT AS id, 'a' AS title UNION ALL SELECT 1::BIGINT, 'b'")
    with pytest.raises(DataSinkWriteError) as exc_info:
        relation.write_datasink(_live_sink(schema, "items", key_columns="id"))

    assert exc_info.value.outcome is WriteOutcome.ABORTED
    assert _live_rows(connection, f'SELECT count(*) FROM "{schema}"."items"') == [(0,)]
