# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import shutil
from collections import deque
from pathlib import Path

import pytest

import vane


def _configured_artifact_path(environment_variable: str) -> Path:
    configured_path = os.environ.get(environment_variable)
    if configured_path is None:
        pytest.skip(f"set {environment_variable} to test the staged artifact")

    path = Path(configured_path).resolve()
    assert path.is_file(), f"loadable extension artifact does not exist: {path}"
    return path


@pytest.fixture(scope="module")
def loadable_extension_path() -> Path:
    return _configured_artifact_path("VANE_TEST_LOADABLE_EXTENSION_PATH")


@pytest.fixture(scope="module")
def loadable_httpfs_extension_path() -> Path:
    return _configured_artifact_path("VANE_TEST_LOADABLE_HTTPFS_EXTENSION_PATH")


def _replanned_batch(batch):
    return batch


class _ReplannedActor:
    def __call__(self, batch):
        return batch


@pytest.fixture(params=["local", "local-fast"])
def private_extension_connection(monkeypatch, request):
    artifact = _configured_artifact_path("VANE_TEST_LOADABLE_BATCH_DEMO_PATH")
    monkeypatch.setenv("VANE_RUNNER", request.param)

    import ray

    def forbid_ray(*_args, **_kwargs):
        raise AssertionError("local extension execution must not initialize Ray")

    monkeypatch.setattr(ray, "init", forbid_ray)
    with vane.connect(config={"autoinstall_known_extensions": "false", "autoload_known_extensions": "false"}) as con:
        con.load_extension(str(artifact))
        yield con


def test_private_extension_loads_from_a_long_path(tmp_path):
    from vane._native_runtime_format import filesystem_path

    artifact = _configured_artifact_path("VANE_TEST_LOADABLE_BATCH_DEMO_PATH")
    directory = tmp_path
    while len(str(directory)) < 280:
        directory /= "long-extension-path-component"
    filesystem_path(directory).mkdir(parents=True)
    target = directory / artifact.name
    shutil.copyfile(artifact, filesystem_path(target))
    with vane.connect(config={"autoinstall_known_extensions": False, "autoload_known_extensions": False}) as con:
        # Pass the ordinary absolute path to the engine: the native loader,
        # rather than its caller, must handle Windows MAX_PATH conversion.
        con.load_extension(str(target))
        assert con.execute("SELECT * FROM batch_demo_query('SELECT 42')").fetchall() == [(42,)]


def _copy_query(sql):
    return f"COPY ({sql}) TO 'unused' (FORMAT batch_demo_replan)"


def _execute_private(con, sql, entry):
    if entry == "copy":
        return con.execute(_copy_query(sql))
    if entry == "prepared_connection":
        return con.execute("SELECT * FROM batch_demo_query(?, prepared := true)", [sql])
    return con.execute("SELECT * FROM batch_demo_query(?)", [sql])


@pytest.mark.parametrize("entry", ["copy", "connection", "prepared_connection"])
@pytest.mark.parametrize("backend", ["task", "actor"])
@pytest.mark.parametrize("rows", [0, 4101])
def test_private_extension_preserves_local_udf_execution(private_extension_connection, entry, backend, rows):
    con = private_extension_connection
    source_sql = (
        "SELECT i::BIGINT AS id, CASE WHEN i % 3 = 0 THEN NULL ELSE 'row-' || i END AS payload, "
        f"[i::FLOAT, NULL::FLOAT, (i + 1)::FLOAT]::FLOAT[3] AS vector FROM range({rows}) t(i)"
    )
    expected = con.sql(source_sql + " ORDER BY id").fetchall()
    assert _execute_private(con, source_sql + " ORDER BY id", entry).fetchall() == expected
    options = {
        "schema": {"id": vane.sqltypes.BIGINT, "payload": vane.sqltypes.VARCHAR, "vector": "FLOAT[3]"},
        "batch_size": 137,
        "cpus": 1,
        "gpus": 0,
        "execution_backend": f"subprocess_{backend}",
    }
    if backend == "actor":
        options["actor_number"] = 1
    con.sql(source_sql).map_batches(_ReplannedActor if backend == "actor" else _replanned_batch, **options).create_view(
        "copy_input"
    )
    assert _execute_private(con, "SELECT * FROM copy_input ORDER BY id", entry).fetchall() == expected
    assert con.execute("SELECT 1").fetchone() == (1,)


@pytest.mark.parametrize("entry", ["connection", "prepared_connection"])
def test_private_extension_cached_connections_share_runtime(monkeypatch, tmp_path, entry):
    artifact = _configured_artifact_path("VANE_TEST_LOADABLE_BATCH_DEMO_PATH")
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    config = {"autoinstall_known_extensions": False, "autoload_known_extensions": False}
    path = str(tmp_path / "runtime.duckdb")
    with vane.connect(path, config=config) as first:
        first.load_extension(str(artifact))
        first.sql("SELECT 42::BIGINT AS id").map_batches(
            _replanned_batch, schema={"id": "BIGINT"}, execution_backend="subprocess_task", cpus=1, gpus=0
        ).create_view("runtime_input")
        with vane.connect(path, config=config) as cached, cached.cursor() as sibling:
            for con in (first, cached, sibling):
                assert _execute_private(con, "SELECT * FROM runtime_input", entry).fetchall() == [(42,)]


def test_private_extension_prepared_execution_renews_actor_resources(private_extension_connection, monkeypatch):
    from vane.execution import udf_subprocess

    pools = []
    create_pool = udf_subprocess.LocalSubprocessActorPool

    def track_pools(*args, **kwargs):
        pool = create_pool(*args, **kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr(udf_subprocess, "LocalSubprocessActorPool", track_pools)
    con = private_extension_connection
    con.sql("SELECT i::BIGINT AS id FROM range(3) t(i)").map_batches(
        _ReplannedActor,
        schema={"id": "BIGINT"},
        execution_backend="subprocess_actor",
        actor_number=1,
        cpus=1,
        gpus=0,
    ).create_view("prepared_input")
    relation = con.sql(
        "SELECT * FROM batch_demo_query(?, prepared := true, executions := 2)",
        params=["SELECT * FROM prepared_input ORDER BY id"],
    )
    assert relation.columns == ["id"]
    assert pools == [], "schema-only native preparation must not create actor pools"
    assert relation.fetchall() == [(0,), (1,), (2,)] * 2
    assert len(pools) == 2, "each execution must prepare fresh query-owned resources"
    assert all(not pool.cleanup_pending() for pool in pools)
    assert all(pool.worker_pids() == [] for pool in pools)
    assert con.execute("SELECT 1").fetchone() == (1,)


class _ExtensionInferenceExecutor:
    def __init__(self):
        self.ready = deque()
        self.finished = False
        self.wakeups = []

    def _notify(self):
        callbacks, self.wakeups = self.wakeups, []
        for callback in callbacks:
            callback()

    def submit(self, prefix, prompts, rows):
        self.ready.append(([f"generated:{prefix or ''}{prompt}" for prompt in prompts], rows))
        self._notify()

    def take_ready_result(self):
        return self.ready.popleft() if self.ready else None

    def finished_submitting(self):
        self.finished = True
        self._notify()

    def all_tasks_finished(self):
        return self.finished and not self.ready

    def register_wakeup_callback(self, callback):
        if self.ready or self.all_tasks_finished():
            return False
        self.wakeups.append(callback)
        return True

    def wait_for_result(self):
        assert self.ready or self.finished

    def shutdown(self):
        self.finished = True


@pytest.mark.parametrize("entry", ["copy", "connection", "prepared_connection"])
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_private_extension_executes_host_inference_factory(private_extension_connection, monkeypatch, entry, engine):
    module = importlib.import_module(f"vane.execution.{engine}")
    calls = []

    def build_host_executor(model, _options):
        calls.append(model)
        return _ExtensionInferenceExecutor()

    monkeypatch.setattr(module, "build_executor", build_host_executor)
    sql = f"""SELECT vllm('hello', 'extension-fixture-no-model', struct_pack(
        __vane_vllm_payload_version := 1,
        __vane_vllm_public_options_json := '{{}}',
        __vane_vllm_secret_payload := encode('{{"payload_version":1,"values":[]}}'),
        engine := '{engine}')) AS generated"""
    assert _execute_private(private_extension_connection, sql, entry).fetchall() == [("generated:hello",)]
    assert calls == ["extension-fixture-no-model"]
    assert private_extension_connection.execute("SELECT 1").fetchone() == (1,)


@pytest.mark.parametrize("entry", ["copy", "connection", "prepared_connection"])
def test_private_extension_restores_datasource_callbacks(private_extension_connection, entry):
    import pyarrow as pa

    from vane.datasource import DataSource, DataSourceTask

    class SourceTask(DataSourceTask):
        def execute(self):
            yield pa.record_batch({"id": pa.array([1, None, 3], type=pa.int64())})

    class Source(DataSource):
        @property
        def schema(self):
            return {"id": "BIGINT"}

        def get_tasks(self):
            yield SourceTask()

    con = private_extension_connection
    before = dict(vane._native._datasource_factory_registry_state_for_test())
    con.from_datasource(Source()).create_view("source_input")
    sql = "SELECT * FROM source_input ORDER BY id"
    expected = [(1,), (3,), (None,)]
    assert con.execute(sql).fetchall() == expected
    # The COPY fixture restores bind data and scan callbacks inside its DSO.
    # Also exercise this from a connection created by that private engine.
    if entry in ("connection", "prepared_connection"):
        sql = _copy_query(sql)
    assert _execute_private(con, sql, entry).fetchall() == expected
    after = dict(vane._native._datasource_factory_registry_state_for_test())
    for key in ("registry_size", "factory_count", "owner_count"):
        assert after[key] == before[key]
    assert con.execute("SELECT 1").fetchone() == (1,)


def test_staged_tpch_extension_loads_without_static_linkage(loadable_extension_path: Path):
    connection = vane.connect(config={"allow_unsigned_extensions": "true"})
    try:
        initial_state = connection.execute(
            """
            SELECT loaded, installed, install_mode
            FROM duckdb_extensions()
            WHERE extension_name = 'tpch'
            """
        ).fetchone()
        assert initial_state == (False, False, "NOT_INSTALLED")

        connection.load_extension(str(loadable_extension_path))

        loaded_state = connection.execute(
            """
            SELECT loaded, installed, install_mode
            FROM duckdb_extensions()
            WHERE extension_name = 'tpch'
            """
        ).fetchone()
        assert loaded_state == (True, False, "NOT_INSTALLED")
        assert connection.execute("SELECT count(*) FROM tpch_queries()").fetchone() == (22,)
    finally:
        connection.close()


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize("transaction", [False, True])
def test_direct_tpch_pragma_keeps_native_data_execution(
    loadable_extension_path, monkeypatch, tmp_path, entry, transaction
):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    database = str(tmp_path / "tpch.duckdb")
    config = {"allow_unsigned_extensions": "true"}
    with vane.connect(database, config=config) as setup:
        setup.load_extension(str(loadable_extension_path))
        setup.execute("CALL dbgen(sf=0)")

    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("an unchanged native pragma must not initialize Ray")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect(database, config=config) as connection:
        connection.load_extension(str(loadable_extension_path))
        if transaction:
            connection.begin()
        assert getattr(connection, entry)("PRAGMA tpch(1)").fetchall() == []
        if transaction:
            connection.commit()
        # Completed rows pass source admission, but this unsigned fixture was
        # loaded directly and cannot enter a distributed connection snapshot.
        result = connection.sql("PRAGMA tpch(1)")
        with pytest.raises(ValueError, match="not loaded through vane.DynamicExtensionResolver: tpch"):
            vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(result.project("*"), None)


def test_staged_httpfs_extension_loads_without_static_linkage(loadable_httpfs_extension_path: Path):
    connection = vane.connect(config={"allow_unsigned_extensions": "true"})
    try:
        initial_state = connection.execute(
            """
            SELECT loaded, installed, install_mode
            FROM duckdb_extensions()
            WHERE extension_name = 'httpfs'
            """
        ).fetchone()
        assert initial_state == (False, False, "NOT_INSTALLED")

        connection.load_extension(str(loadable_httpfs_extension_path))

        loaded_state = connection.execute(
            """
            SELECT loaded, installed, install_mode
            FROM duckdb_extensions()
            WHERE extension_name = 'httpfs'
            """
        ).fetchone()
        assert loaded_state == (True, False, "NOT_INSTALLED")
    finally:
        connection.close()
