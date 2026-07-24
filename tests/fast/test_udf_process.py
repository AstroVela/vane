# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa

import duckdb


def test_unregister_timeout_detaches_stale_dispatcher_work():
    """A timed-out slot must not retain its context or poison later UDFs."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        from __future__ import annotations

        import gc
        import threading
        import time

        import _duckdb
        import duckdb
        import duckdb.execution.udf as udf_exec
        import pyarrow as pa

        dispatcher_blocked = threading.Event()
        release_dispatcher = threading.Event()
        stale_executor_closed = threading.Event()
        build_count = 0


        class FakeExecutor:
            def __init__(self, *, block_result):
                self._block_result = block_result
                self._output = None
                self._finished = False
                self._wakeup = None
                self._admission_state = "idle"
                self._retained_input_bytes = 0

            def register_wakeup(self, callback):
                self._wakeup = callback

            def request_task_admission(self, retained_input_bytes):
                self._retained_input_bytes = int(retained_input_bytes)
                self._admission_state = "ready"
                return True

            def task_admission_state(self):
                return {
                    "state": self._admission_state,
                    "available": self._admission_state == "ready",
                    "retained_input_bytes": self._retained_input_bytes,
                }

            def submit_with_id(self, submit_id, table):
                self._admission_state = "idle"
                values = table.column(0).to_pylist()
                self._output = (
                    "__vane_submit_result__",
                    int(submit_id),
                    pa.table({"y": [value + 1 for value in values]}),
                )
                if self._wakeup is not None:
                    self._wakeup()

            def take_ready_result(self):
                if self._output is None:
                    return None
                if self._block_result:
                    self._block_result = False
                    dispatcher_blocked.set()
                    if not release_dispatcher.wait(timeout=10):
                        raise RuntimeError("test did not release blocked dispatcher")
                result = self._output
                self._output = None
                return result

            def finished_submitting(self):
                self._finished = True

            def all_tasks_finished(self):
                return self._finished and self._output is None

            def close(self):
                stale_executor_closed.set()


        def build_executor(_payload, options=None):
            del options
            global build_count
            build_count += 1
            return FakeExecutor(block_result=build_count == 1)


        def add_one(table):
            values = table.column(0).to_pylist()
            return pa.table({"y": [value + 1 for value in values]})


        def make_relation(connection):
            return connection.sql("select i::BIGINT as x from range(2) t(i)").map_batches(
                add_one,
                schema={"y": duckdb.sqltypes.BIGINT},
                execution_backend="subprocess_task",
                streaming_breaker=False,
            )


        udf_exec.build_executor = build_executor
        connection = duckdb.connect()
        relation = make_relation(connection)
        query_errors = []


        def run_blocked_query():
            try:
                relation.fetchall()
            except BaseException as exc:
                query_errors.append(exc)


        query_thread = threading.Thread(target=run_blocked_query)
        query_thread.start()
        try:
            assert dispatcher_blocked.wait(timeout=5), "dispatcher never entered the blocking result callback"
            connection.interrupt()
            teardown_deadline = time.monotonic() + 5
            while query_thread.is_alive() and time.monotonic() < teardown_deadline:
                _duckdb._wake_udf_executor_slots_for_testing()
                query_thread.join(timeout=0.01)
            assert not query_thread.is_alive(), "query teardown did not honor the unregister deadline"
            assert query_errors, "the interrupted query unexpectedly succeeded"

            relation = None
            query_errors.clear()
            connection.close()
            del connection
            gc.collect()

            release_dispatcher.set()
            assert stale_executor_closed.wait(timeout=5), "detached slot was not eventually cleaned"

            healthy_connection = duckdb.connect()
            try:
                assert make_relation(healthy_connection).fetchall() == [(1,), (2,)]
            finally:
                healthy_connection.close()
        finally:
            release_dispatcher.set()
            query_thread.join(timeout=5)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env={
            **os.environ,
            "VANE_ENABLE_UDF_TEST_HOOKS": "1",
            "VANE_UDF_UNREGISTER_TIMEOUT_MS": "50",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_create_function_rejects_removed_process_and_ray_args():
    con = duckdb.connect()

    def add_one(value):
        return value + 1

    with pytest.raises(TypeError):
        con.create_function(
            "bad_process_arg",
            add_one,
            ["BIGINT"],
            "BIGINT",
            type="native",
            use_process=True,
        )

    with pytest.raises(TypeError):
        con.create_function(
            "bad_ray_arg",
            add_one,
            ["BIGINT"],
            "BIGINT",
            type="native",
            ray=True,
        )


def test_map_batches_rejects_removed_process_and_actor_count_args():
    con = duckdb.connect()

    def add_one(table):
        values = table.column(0).to_pylist()
        return pa.table({"out": [value + 1 for value in values]})

    rel = con.sql("select i from range(0, 4) t(i)")

    with pytest.raises(TypeError):
        rel.map_batches(
            add_one,
            schema={"out": duckdb.sqltypes.BIGINT},
            use_process=True,
        )

    with pytest.raises(TypeError):
        rel.map_batches(
            add_one,
            schema={"out": duckdb.sqltypes.BIGINT},
            actor_count=1,
        )


def test_ray_task_map_batches_local_execution_is_rejected(monkeypatch):
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    con = duckdb.connect()

    def add_ten(table):
        values = table.column(0).to_pylist()
        return pa.table({"out": [value + 10 for value in values]})

    relation = con.sql("select i from range(0, 5) t(i)").map_batches(
        add_ten,
        schema={"out": duckdb.sqltypes.BIGINT},
        execution_backend="ray_task",
        batch_size=2,
    )

    with pytest.raises(Exception, match="distributed Ray UDF payload requires query_id"):
        relation.fetchall()


def test_flat_map_rejects_removed_actor_count_arg():
    con = duckdb.connect()

    def expand(row):
        return [{"out": row["i"]}, {"out": row["i"] + 10}]

    with pytest.raises(TypeError):
        con.sql("select i from range(0, 2) t(i)").flat_map(
            expand,
            schema={"out": duckdb.sqltypes.BIGINT},
            actor_count=1,
        )
