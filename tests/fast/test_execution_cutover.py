# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public execution boundaries after removal of the global runners."""

import gc
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import vane
from vane.execution.request_admission import RequestAdmissionLimits


@pytest.mark.parametrize("environment", ["ray", "local", "local-fast", "invalid"])
def test_default_connection_is_local_and_query_has_managed_results(monkeypatch, environment):
    monkeypatch.setenv("VANE_RUNNER", environment)
    with vane.connect() as connection:
        assert connection.backend == "local"
        assert connection.sql("select 7").backend == "local"
        assert connection.execute("select 3").fetchall() == [(3,)]
        with connection.query("select $x::bigint as value", {"x": 42}) as result:
            assert isinstance(result, vane.QueryResult)
            assert result.collect().to_pylist() == [{"value": 42}]
        assert connection.query_runtime.backend == "local"


def test_local_import_and_execution_do_not_require_ray_or_removed_modules():
    script = """
import sys
sys.modules['ray'] = None
sys.modules['vane.runners'] = None
sys.modules['vane._ray_cxx'] = None
import vane
for name in ('runners', 'ray_cxx', 'set_runner_local', 'set_runner_ray', 'get_runner',
             'get_or_create_runner', 'get_or_infer_runner_type', 'teardown_runner'):
    assert not hasattr(vane, name), name
assert not hasattr(vane._native, '_connect_with_runner')
with vane.connect() as connection:
    connection.execute('create table t as select range x from range(4)')
    assert connection.sql('select sum(x) from t').fetchone() == (6,)
    with connection.query('select x from t order by x') as result:
        assert result.collect().column(0).to_pylist() == [0, 1, 2, 3]
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


def test_default_runtime_is_published_once_across_concurrent_cursors(monkeypatch):
    from vane.execution import query_runtime

    original = query_runtime.QueryRuntime
    entered = threading.Barrier(2)

    def create(*args, **kwargs):
        runtime = original(*args, **kwargs)
        entered.wait(10)
        return runtime

    monkeypatch.setattr(query_runtime, "QueryRuntime", create)
    with vane.connect() as connection:
        cursors = [connection.cursor(), connection.cursor()]

        def run(cursor):
            with cursor.query("select 42") as result:
                assert result.collect().column(0).to_pylist() == [42]
            return cursor.query_runtime

        with ThreadPoolExecutor(2) as threads:
            runtimes = list(threads.map(run, cursors))
        assert runtimes[0] is runtimes[1] is connection.query_runtime
        assert connection.query_runtime.resource_snapshot()["request_admission"]["active_requests"] == 0


def test_resources_without_backend_select_local():
    resources = replace(vane.QueryResources(), max_active_queries=1)
    with vane.connect(resources=resources) as connection:
        assert connection.backend == "local"
        assert connection.query_runtime.resources == resources
        with connection.query("select 7") as result:
            assert result.collect().column(0).to_pylist() == [7]


@pytest.mark.parametrize("entry", ["execute", "executemany", "sql", "from_query"])
def test_ray_native_entries_fail_before_starting_workers(entry):
    with vane.Runtime() as application, application.connect() as connection:
        with pytest.raises(vane.NotImplementedException, match="query\\(\\)"):
            if entry == "executemany":
                connection.executemany("select ?", [[1]])
            else:
                getattr(connection, entry)("select 1")
        assert connection.query_runtime.pool.workers == []


def test_model_runtime_and_query_runtime_cannot_replace_each_other():
    with vane.connect() as connection:
        connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))
        with pytest.raises(vane.InvalidInputException, match="model runtime"):
            connection.query("select 1")
        assert connection.execute("select 7").fetchone() == (7,)
    with vane.connect() as connection:
        with connection.query("select 1") as result:
            result.collect()
        with pytest.raises(vane.InvalidInputException, match="must run once"):
            connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 1))


def test_native_model_metadata_collection_never_executes_callable():
    def forbidden(table):
        raise AssertionError("metadata executed user code")

    with vane.connect() as connection:
        relation = connection.sql("select 1 as value").map_batches(
            forbidden, schema={"value": "INTEGER"}, execution_backend="subprocess_task"
        )
        nodes = relation._collect_udf_metadata()
        assert len(nodes) == 1
        assert nodes[0]["execution_backend"] == "subprocess_task"
        assert relation._collect_udf_metadata() == nodes
    gc.collect()
