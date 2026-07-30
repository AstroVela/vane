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
        from duckdb.execution.ref_bundle import make_local_shm_ref_bundle_result

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
                    make_local_shm_ref_bundle_result(pa.table({"y": [value + 1 for value in values]})),
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


def test_unregister_timeout_keeps_context_alive_during_input_conversion():
    """Input Arrow conversion must not outlive its ClientContext lease."""
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
        from duckdb.execution.ref_bundle import make_local_shm_ref_bundle_result

        conversion_blocked = threading.Event()
        release_conversion = threading.Event()
        stale_executor_closed = threading.Event()
        stale_submit_called = threading.Event()
        build_count = 0
        original_pyarrow_lib = pa.lib


        class BlockingPyArrowLib:
            def __init__(self, target):
                self._target = target
                self._blocked = False

            def __getattr__(self, name):
                if name == "Table" and not self._blocked:
                    self._blocked = True
                    conversion_blocked.set()
                    if not release_conversion.wait(timeout=10):
                        raise RuntimeError("test did not release input Arrow conversion")
                return getattr(self._target, name)


        class FakeExecutor:
            def __init__(self, *, stale):
                self._stale = stale
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
                if self._stale:
                    stale_submit_called.set()
                self._admission_state = "idle"
                values = table.column(0).to_pylist()
                self._output = (
                    "__vane_submit_result__",
                    int(submit_id),
                    make_local_shm_ref_bundle_result(pa.table({"y": [value + 1 for value in values]})),
                )
                if self._wakeup is not None:
                    self._wakeup()

            def take_ready_result(self):
                if self._output is None:
                    return None
                result = self._output
                self._output = None
                return result

            def finished_submitting(self):
                self._finished = True

            def all_tasks_finished(self):
                return self._finished and self._output is None

            def close(self):
                if self._stale:
                    stale_executor_closed.set()


        def build_executor(_payload, options=None):
            del options
            global build_count
            build_count += 1
            return FakeExecutor(stale=build_count == 1)


        def add_one(table):
            values = table.column(0).to_pylist()
            return pa.table({"y": [value + 1 for value in values]})


        def make_relation(connection, sql_type, output_type):
            return connection.sql(f"select i::{sql_type} as x from range(2) t(i)").map_batches(
                add_one,
                schema={"y": output_type},
                execution_backend="subprocess_task",
            )


        udf_exec.build_executor = build_executor
        connection = duckdb.connect()
        connection.execute("SET arrow_lossless_conversion = true")
        relation = make_relation(connection, "HUGEINT", duckdb.sqltypes.HUGEINT)
        query_errors = []


        def run_blocked_query():
            try:
                relation.fetchall()
            except BaseException as exc:
                query_errors.append(exc)


        pa.lib = BlockingPyArrowLib(original_pyarrow_lib)
        query_thread = threading.Thread(target=run_blocked_query)
        query_thread.start()
        try:
            assert conversion_blocked.wait(timeout=5), "dispatcher never entered input Arrow conversion"
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

            pa.lib = original_pyarrow_lib
            release_conversion.set()
            assert stale_executor_closed.wait(timeout=5), "detached slot was not eventually cleaned"
            assert not stale_submit_called.is_set(), "retired input was submitted after Arrow conversion resumed"

            healthy_connection = duckdb.connect()
            try:
                assert make_relation(healthy_connection, "BIGINT", duckdb.sqltypes.BIGINT).fetchall() == [(1,), (2,)]
            finally:
                healthy_connection.close()
        finally:
            pa.lib = original_pyarrow_lib
            release_conversion.set()
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


def test_mixed_streaming_inputs_preserve_task_admission_owner():
    """A lazy input must not consume a materialized input's ready grant."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        from __future__ import annotations

        import threading
        import time
        import uuid

        import duckdb
        import duckdb.execution.udf as udf_exec
        import pyarrow as pa
        from duckdb.execution.ref_bundle import (
            make_local_shm_ref_bundle_result,
            materialize_ref_bundle,
        )

        materialized_request_started = threading.Event()
        lazy_output_taken = threading.Event()
        async_errors = []
        admission_requests = []
        downstream_submits = []


        class Stage1:
            def __call__(self, table):
                return pa.table({"y": table.column(0).to_pylist()})


        class Stage2:
            def __call__(self, table):
                return pa.table({"z": table.column(0).to_pylist()})


        class FakeExecutor:
            def __init__(self, name):
                self._name = name
                self._output = []
                self._finished = False
                self._wakeup = None
                self._admission_state = "idle"
                self._retained_input_bytes = 0
                self._pending_outputs = 0
                self._error = ""
                self._lock = threading.Lock()

            def _notify(self):
                wakeup = self._wakeup
                if wakeup is not None:
                    wakeup()

            def _set_ready(self):
                with self._lock:
                    if self._admission_state != "requested":
                        return
                    self._admission_state = "ready"
                self._notify()

            def _set_failed(self, error):
                with self._lock:
                    self._error = str(error)
                    self._admission_state = "failed"
                    self._pending_outputs = 0
                async_errors.append(str(error))
                self._notify()

            def supports_async_wakeup(self):
                return True

            def register_wakeup(self, callback):
                self._wakeup = callback

            def request_task_admission(self, retained_input_bytes):
                retained = int(retained_input_bytes)
                with self._lock:
                    if self._admission_state != "idle":
                        return False
                    self._retained_input_bytes = retained
                    self._admission_state = "requested"
                admission_requests.append((self._name, retained))
                if self._name != "Stage2" or retained == 0:
                    self._set_ready()
                    return True

                materialized_request_started.set()

                def grant_after_lazy_arrives():
                    if not lazy_output_taken.wait(timeout=10):
                        self._set_failed("lazy output never reached the dispatcher")
                        return
                    # The dispatcher has converted the upstream lazy result.
                    # Leave the materialized grant parked while the awakened
                    # UNION pipeline moves that result into the downstream sink.
                    time.sleep(1)
                    self._set_ready()

                threading.Thread(target=grant_after_lazy_arrives, daemon=True).start()
                return True

            def task_admission_state(self):
                with self._lock:
                    state = {
                        "state": self._admission_state,
                        "available": self._admission_state == "ready",
                        "retained_input_bytes": self._retained_input_bytes,
                    }
                    if self._error:
                        state["error"] = self._error
                    return state

            def _consume_admission(self):
                with self._lock:
                    self._admission_state = "idle"
                    self._retained_input_bytes = 0
                    self._pending_outputs += 1

            def _publish(self, submit_id, table):
                result_name = "y" if self._name == "Stage1" else "z"
                result = pa.table({result_name: table.column(0).to_pylist()})
                output = (
                    "__vane_submit_result__",
                    int(submit_id),
                    make_local_shm_ref_bundle_result(result),
                )
                with self._lock:
                    self._output.append(output)
                    self._pending_outputs -= 1
                self._notify()

            def submit_with_id(self, submit_id, table):
                self._consume_admission()
                if self._name == "Stage2":
                    downstream_submits.append("materialized")
                    self._publish(submit_id, table)
                    return None

                def publish_after_materialized_request():
                    if not materialized_request_started.wait(timeout=10):
                        self._set_failed("materialized request was not registered first")
                        return
                    self._publish(submit_id, table)

                threading.Thread(target=publish_after_materialized_request, daemon=True).start()
                return None

            def submit_ref_bundle_with_id(self, submit_id, refs, slices, metadata, names):
                self._consume_admission()
                if self._name != "Stage2":
                    raise RuntimeError("only the downstream UDF accepts lazy input")
                downstream_submits.append("lazy")
                table = materialize_ref_bundle(refs, slices, metadata, names)
                self._publish(submit_id, table)
                return None

            def take_ready_result(self):
                with self._lock:
                    if not self._output:
                        return None
                    result = self._output.pop(0)
                if self._name == "Stage1":
                    lazy_output_taken.set()
                return result

            def finished_submitting(self):
                self._finished = True

            def all_tasks_finished(self):
                with self._lock:
                    return self._finished and self._pending_outputs == 0 and not self._output


        def build_executor(payload, options=None):
            del options
            return FakeExecutor(str(payload["udf_name"]))


        udf_exec.build_executor = build_executor
        connection = duckdb.connect()
        cursor = connection.cursor()
        lazy = connection.sql("SELECT i::BIGINT AS x FROM range(4) t(i)").map_batches(
            Stage1,
            schema={"y": duckdb.sqltypes.BIGINT},
            execution_backend="ray_actor",
            actor_number=1,
            gpus=0.0,
            batch_size=4,
            task_input_max_bytes=1024 * 1024,
        )
        materialized = connection.sql("SELECT (100 + i)::BIGINT AS y FROM range(4) t(i)")
        relation = materialized.union(lazy).map_batches(
            Stage2,
            schema={"z": duckdb.sqltypes.BIGINT},
            execution_backend="ray_actor",
            actor_number=1,
            gpus=0.0,
            batch_size=4,
            task_input_max_bytes=1024 * 1024,
        )
        plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"mixed-task-admission-{uuid.uuid4().hex[:8]}",
        ).to_physical_plan(connection)
        handles = {
            str(node["node_id"]): {
                "actor_handles": [f"actor-{node['node_id']}"],
                "actor_node_ids": ["node-a"],
                "query_driver_handle": object(),
                "session_config": {},
            }
            for node in plan.collect_udf_nodes(conn=connection)
        }
        plan.set_udf_actor_handles(handles, conn=connection)

        try:
            result = duckdb.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
                cursor,
                plan,
                None,
                None,
            )
            values = sorted(
                value
                for partition in result.partition_payloads
                for value in partition.column(0).to_pylist()
            )
            assert values == [0, 1, 2, 3, 100, 101, 102, 103], values
            assert not async_errors, async_errors
            downstream_requests = [
                retained
                for name, retained in admission_requests
                if name == "Stage2"
            ]
            assert len(downstream_requests) == 2, downstream_requests
            assert downstream_requests[0] > 0, downstream_requests
            assert downstream_requests[1] == 0, downstream_requests
            assert downstream_submits == ["materialized", "lazy"], downstream_submits
        finally:
            cursor.close()
            connection.close()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=os.environ.copy(),
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
