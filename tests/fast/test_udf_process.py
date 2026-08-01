# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa

import duckdb


def test_native_dispatcher_shutdown_is_terminal():
    """Process-owner shutdown must reject every later slot registration."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import _duckdb
        import duckdb
        import pyarrow as pa


        def passthrough(table):
            return pa.table({"y": table.column(0)})


        _duckdb._shutdown_udf_executor_dispatcher()
        _duckdb._shutdown_udf_executor_dispatcher()

        connection = duckdb.connect()
        try:
            relation = connection.sql("select i::BIGINT as x from range(2) t(i)").map_batches(
                passthrough,
                schema={"y": duckdb.sqltypes.BIGINT},
                execution_backend="subprocess_task",
            )
            try:
                relation.fetchall()
            except BaseException as exc:
                assert "udf dispatcher has been shut down" in str(exc), str(exc)
            else:
                raise AssertionError("dispatcher accepted work after terminal shutdown")
        finally:
            connection.close()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env={**os.environ},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_native_dispatcher_concurrent_shutdown_never_reopens():
    """Register either linearizes before shutdown or observes its terminal fence."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import threading

        import _duckdb
        import duckdb
        import pyarrow as pa


        def passthrough(table):
            return pa.table({"y": table.column(0)})


        connection = duckdb.connect()
        relation = connection.sql("select i::BIGINT as x from range(2) t(i)").map_batches(
            passthrough,
            schema={"y": duckdb.sqltypes.BIGINT},
            execution_backend="subprocess_task",
        )
        barrier = threading.Barrier(3)
        query_errors = []
        shutdown_errors = []


        def run_query():
            barrier.wait()
            try:
                relation.fetchall()
            except BaseException as exc:
                query_errors.append(exc)


        def run_shutdown():
            barrier.wait()
            try:
                _duckdb._shutdown_udf_executor_dispatcher()
            except BaseException as exc:
                shutdown_errors.append(exc)


        query_thread = threading.Thread(target=run_query)
        shutdown_thread = threading.Thread(target=run_shutdown)
        query_thread.start()
        shutdown_thread.start()
        barrier.wait()
        query_thread.join(timeout=10)
        shutdown_thread.join(timeout=10)
        assert not query_thread.is_alive(), "query raced shutdown without terminating"
        assert not shutdown_thread.is_alive(), "dispatcher shutdown deadlocked with Register"
        assert not shutdown_errors, shutdown_errors
        connection.close()

        final_connection = duckdb.connect()
        try:
            final_relation = final_connection.sql("select i::BIGINT as x from range(2) t(i)").map_batches(
                passthrough,
                schema={"y": duckdb.sqltypes.BIGINT},
                execution_backend="subprocess_task",
            )
            try:
                final_relation.fetchall()
            except BaseException as exc:
                assert "udf dispatcher has been shut down" in str(exc), str(exc)
            else:
                raise AssertionError("concurrent Register reopened the dispatcher after shutdown")
        finally:
            final_connection.close()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_native_dispatcher_isolates_async_task_admission_failure():
    """An asynchronous admission failure must fail one slot, not the process."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import duckdb
        import duckdb.execution.udf as udf_exec
        import pyarrow as pa
        from duckdb.execution.ref_bundle import make_local_shm_ref_bundle_result

        build_count = 0


        class FakeExecutor:
            def __init__(self, *, fail_admission):
                self._fail_admission = fail_admission
                self._wakeup = None
                self._state = "idle"
                self._retained_input_bytes = 0
                self._output = None
                self._finished = False

            def register_wakeup(self, callback):
                self._wakeup = callback

            def request_task_admission(self, retained_input_bytes):
                self._retained_input_bytes = int(retained_input_bytes)
                self._state = "failed" if self._fail_admission else "ready"
                if self._wakeup is not None:
                    self._wakeup()
                return True

            def task_admission_state(self):
                state = {
                    "state": self._state,
                    "available": self._state == "ready",
                    "retained_input_bytes": self._retained_input_bytes,
                }
                if self._state == "failed":
                    state["error"] = "injected asynchronous admission failure"
                return state

            def submit_with_id(self, submit_id, table):
                if self._fail_admission:
                    raise AssertionError("failed admission must not submit")
                self._state = "idle"
                values = table.column(0).to_pylist()
                self._output = (
                    "__vane_submit_result__",
                    int(submit_id),
                    make_local_shm_ref_bundle_result(
                        pa.table({"y": [value + 1 for value in values]})
                    ),
                )
                if self._wakeup is not None:
                    self._wakeup()

            def take_ready_result(self):
                result = self._output
                self._output = None
                return result

            def finished_submitting(self):
                self._finished = True

            def all_tasks_finished(self):
                return self._finished and self._output is None

            def close(self):
                self._state = "closed"


        def build_executor(_payload, options=None):
            del options
            global build_count
            build_count += 1
            return FakeExecutor(fail_admission=build_count == 1)


        def add_one(table):
            return pa.table({"y": [value + 1 for value in table.column(0).to_pylist()]})


        def make_relation(connection):
            return connection.sql("select 1::BIGINT as x").map_batches(
                add_one,
                schema={"y": duckdb.sqltypes.BIGINT},
                execution_backend="subprocess_task",
            )


        udf_exec.build_executor = build_executor
        failed_connection = duckdb.connect()
        try:
            try:
                make_relation(failed_connection).fetchall()
            except BaseException as exc:
                assert "injected asynchronous admission failure" in str(exc), exc
            else:
                raise AssertionError("failed task admission unexpectedly succeeded")
        finally:
            failed_connection.close()

        healthy_connection = duckdb.connect()
        try:
            assert make_relation(healthy_connection).fetchall() == [(2,)]
        finally:
            healthy_connection.close()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env={**os.environ, "VANE_RUNNER": "local-fast"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_streaming_task_wakeup_epoch_handles_early_and_duplicate_callbacks():
    """Early and duplicate callbacks must each schedule a blocked task at most once."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import vane


        @vane.func(return_dtype="INTEGER")
        def add_one(value):
            return value + 1


        @vane.func(return_dtype="INTEGER")
        def times_two(value):
            return value * 2


        connection = vane.connect()
        try:
            relation = connection.sql("select i::INTEGER as x from range(3) t(i)")
            result = relation.select(
                add_one(vane.col("x")).alias("a"),
                times_two(vane.col("x")).alias("b"),
                times_two(add_one(vane.col("x"))).alias("nested"),
            )
            assert result.fetchall() == [(1, 0, 2), (2, 2, 4), (3, 4, 6)]
            assert connection.sql("select 42").fetchall() == [(42,)]
        finally:
            connection.close()
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
            "VANE_TEST_DUPLICATE_UDF_WAKEUPS": "1",
            "VANE_RUNNER": "local-fast",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_native_dispatcher_shutdown_closes_active_executor():
    """Terminal shutdown must close Python ownership before it returns."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import threading

        import _duckdb
        import duckdb
        import duckdb.execution.udf as udf_exec
        import pyarrow as pa

        submitted = threading.Event()
        closed = threading.Event()


        class FakeExecutor:
            def __init__(self):
                self._wakeup = None
                self._retained_input_bytes = 0
                self._admission_state = "idle"

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

            def submit_with_id(self, _submit_id, _table):
                self._admission_state = "idle"
                submitted.set()

            def take_ready_result(self):
                return None

            def finished_submitting(self):
                return None

            def all_tasks_finished(self):
                return False

            def close(self):
                closed.set()


        def build_executor(_payload, options=None):
            del options
            return FakeExecutor()


        def passthrough(table):
            return pa.table({"y": table.column(0)})


        udf_exec.build_executor = build_executor
        connection = duckdb.connect()
        relation = connection.sql("select i::BIGINT as x from range(2) t(i)").map_batches(
            passthrough,
            schema={"y": duckdb.sqltypes.BIGINT},
            execution_backend="subprocess_task",
        )
        query_errors = []


        def run_query():
            try:
                relation.fetchall()
            except BaseException as exc:
                query_errors.append(exc)


        query_thread = threading.Thread(target=run_query)
        query_thread.start()
        try:
            assert submitted.wait(timeout=5), "query never submitted to the fake executor"
            _duckdb._shutdown_udf_executor_dispatcher()
            assert closed.is_set(), "shutdown returned before closing the active executor"
            query_thread.join(timeout=5)
            assert not query_thread.is_alive(), "active query did not observe terminal shutdown"
            assert query_errors, "active query unexpectedly succeeded during terminal shutdown"
            assert "shutdown interrupted active execution" in str(query_errors[0]), query_errors[0]
        finally:
            query_thread.join(timeout=5)
            connection.close()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env={**os.environ},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_native_dispatcher_terminal_shutdown_uses_one_aggregate_collector_deadline():
    """Terminal Ray cleanup must not apply one collector timeout per slot."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import threading
        import time

        import _duckdb
        import duckdb
        import duckdb.execution.udf as udf_exec
        import duckdb.execution.udf_stream_result_collector as collector_mod
        import pyarrow as pa

        slot_count = 4
        state_lock = threading.Lock()
        all_tracked = threading.Event()
        aggregate_shutdown = threading.Event()
        tracked_count = 0
        cancel_calls = 0
        shutdown_calls = 0


        class FakeCollector:
            def __init__(self):
                self._wakeup = None

            def set_wakeup_callback(self, callback):
                self._wakeup = callback

            def track_generator_ref(self, _slot_id, _submit_id, _source, _error_context):
                global tracked_count
                with state_lock:
                    tracked_count += 1
                    if tracked_count == slot_count:
                        all_tracked.set()

            def discard_generator_ref(self, _slot_id, _source):
                return None

            def drain_results(self, _capacities):
                return []

            def cancel_slot(self, _slot_id):
                global cancel_calls
                with state_lock:
                    cancel_calls += 1
                time.sleep(0.1)

            def slot_has_pending(self, _slot_id):
                return False

            def retire_slot(self, _slot_id):
                return None

            def shutdown(self):
                global shutdown_calls
                with state_lock:
                    shutdown_calls += 1
                time.sleep(0.1)
                aggregate_shutdown.set()


        class FakeExecutor:
            def __init__(self):
                self._wakeup = None
                self._retained_input_bytes = 0
                self._admission_state = "idle"

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

            def submit_with_id(self, _submit_id, _table):
                self._admission_state = "idle"
                return object()

            def take_ready_result(self):
                return None

            def finished_submitting(self):
                return None

            def all_tasks_finished(self):
                return False

            def close(self):
                return None


        def build_executor(_payload, options=None):
            del options
            return FakeExecutor()


        def passthrough(table):
            return pa.table({"y": table.column(0)})


        udf_exec.build_executor = build_executor
        collector_mod.AsyncResultCollector = FakeCollector
        connections = [duckdb.connect() for _ in range(slot_count)]
        relations = [
            connection.sql("select 1::BIGINT as x").map_batches(
                passthrough,
                schema={"y": duckdb.sqltypes.BIGINT},
                execution_backend="ray_task",
            )
            for connection in connections
        ]
        query_errors = []


        def run_query(relation):
            try:
                relation.fetchall()
            except BaseException as exc:
                query_errors.append(exc)


        query_threads = [
            threading.Thread(target=run_query, args=(relation,))
            for relation in relations
        ]
        for thread in query_threads:
            thread.start()
        try:
            assert all_tracked.wait(timeout=5), "not every Ray slot reached the collector"
            assert cancel_calls == 0, cancel_calls
            started_at = time.monotonic()
            _duckdb._shutdown_udf_executor_dispatcher()
            elapsed = time.monotonic() - started_at
            assert aggregate_shutdown.is_set(), "aggregate collector shutdown was not called"
            assert shutdown_calls == 1, shutdown_calls
            assert cancel_calls == 0, cancel_calls
            assert elapsed < 1.0, elapsed
            for thread in query_threads:
                thread.join(timeout=5)
            assert not any(thread.is_alive() for thread in query_threads)
            assert len(query_errors) == slot_count, query_errors
            assert all(
                "shutdown interrupted active execution" in str(error)
                for error in query_errors
            ), query_errors
        finally:
            for thread in query_threads:
                thread.join(timeout=5)
            for connection in connections:
                connection.close()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env={**os.environ},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_native_dispatcher_terminal_shutdown_closes_executor_after_pending_collector_handoff():
    """Aggregate shutdown must close local ownership after per-slot cleanup stalls."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import threading

        import _duckdb
        import duckdb
        import duckdb.execution.udf as udf_exec
        import duckdb.execution.udf_stream_result_collector as collector_mod
        import pyarrow as pa

        tracked = threading.Event()
        cancel_started = threading.Event()
        executor_closed = threading.Event()


        class FakeCollector:
            def __init__(self):
                self._wakeup = None
                self._events = []
                self._pending_slots = set()

            def set_wakeup_callback(self, callback):
                self._wakeup = callback

            def track_generator_ref(self, slot_id, submit_id, _source, _error_context):
                self._events.append((int(slot_id), int(submit_id), "complete", None))
                tracked.set()
                if self._wakeup is not None:
                    self._wakeup()

            def discard_generator_ref(self, _slot_id, _source):
                return None

            def drain_results(self, _capacities):
                events, self._events = self._events, []
                return events

            def cancel_slot(self, slot_id):
                self._pending_slots.add(int(slot_id))
                cancel_started.set()

            def slot_has_pending(self, slot_id):
                return int(slot_id) in self._pending_slots

            def retire_slot(self, _slot_id):
                raise AssertionError("pending slot must not retire before terminal shutdown")

            def shutdown(self):
                return None


        class FakeExecutor:
            def __init__(self):
                self._wakeup = None
                self._retained_input_bytes = 0
                self._admission_state = "idle"

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

            def submit_with_id(self, _submit_id, _table):
                self._admission_state = "idle"
                return object()

            def take_ready_result(self):
                return None

            def finished_submitting(self):
                return None

            def all_tasks_finished(self):
                return False

            def close(self):
                executor_closed.set()


        def build_executor(_payload, options=None):
            del options
            return FakeExecutor()


        def passthrough(table):
            return pa.table({"y": table.column(0)})


        udf_exec.build_executor = build_executor
        collector_mod.AsyncResultCollector = FakeCollector
        connection = duckdb.connect()
        relation = connection.sql("select 1::BIGINT as x").map_batches(
            passthrough,
            schema={"y": duckdb.sqltypes.BIGINT},
            execution_backend="ray_task",
        )
        query_errors = []


        def run_query():
            try:
                relation.fetchall()
            except BaseException as exc:
                query_errors.append(exc)


        query_thread = threading.Thread(target=run_query)
        query_thread.start()
        try:
            assert tracked.wait(timeout=5), "Ray stream was not tracked"
            query_thread.join(timeout=5)
            assert not query_thread.is_alive(), "completed query did not reach unregister"
            assert query_errors == [], query_errors
            assert cancel_started.is_set(), "slot cleanup never transferred to the collector"
            assert not executor_closed.is_set(), "pending per-slot cleanup unexpectedly closed the executor"

            _duckdb._shutdown_udf_executor_dispatcher()
            assert executor_closed.is_set(), "terminal shutdown skipped the pending slot's executor close"
        finally:
            query_thread.join(timeout=5)
            connection.close()
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
            "VANE_UDF_UNREGISTER_TIMEOUT_MS": "50",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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


@pytest.mark.parametrize("failure_phase", ["cancel", "pending", "retire"])
def test_retired_ray_submit_is_discarded_before_python_ref_is_dropped(failure_phase):
    """A Ray stream returned after slot retirement must reach collector cleanup."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        from __future__ import annotations

        import os
        import threading
        import time

        import _duckdb
        import duckdb
        import duckdb.execution.udf as udf_exec
        import duckdb.execution.udf_stream_result_collector as collector_mod
        import pyarrow as pa

        submit_entered = threading.Event()
        release_submit = threading.Event()
        discarded = threading.Event()
        executor_closed = threading.Event()
        collector_retired = threading.Event()
        submitted_ref = object()
        failure_phase = os.environ["VANE_TEST_COLLECTOR_FAILURE_PHASE"]
        cancel_calls = 0
        pending_checks = 0
        retire_calls = 0


        class FakeCollector:
            def __init__(self):
                self._wakeup = None

            def set_wakeup_callback(self, callback):
                self._wakeup = callback

            def discard_generator_ref(self, _slot_id, ref):
                assert ref is submitted_ref
                discarded.set()

            def cancel_slot(self, _slot_id):
                global cancel_calls
                cancel_calls += 1
                if failure_phase == "cancel" and cancel_calls <= 3:
                    raise RuntimeError("planned transient collector cancellation failure")
                return None

            def slot_has_pending(self, _slot_id):
                global pending_checks
                pending_checks += 1
                pending = failure_phase == "pending" and pending_checks == 1
                if pending and self._wakeup is not None:
                    self._wakeup()
                return pending

            def retire_slot(self, _slot_id):
                global retire_calls
                retire_calls += 1
                if failure_phase == "retire" and retire_calls <= 3:
                    raise RuntimeError("planned transient collector retirement failure")
                collector_retired.set()

            def drain_results(self, _capacities):
                return []

            def shutdown(self):
                return None


        class FakeExecutor:
            def __init__(self):
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

            def submit_with_id(self, _submit_id, _table):
                self._admission_state = "idle"
                submit_entered.set()
                if not release_submit.wait(timeout=10):
                    raise RuntimeError("test did not release blocked Ray submit")
                return submitted_ref

            def finished_submitting(self):
                return None

            def all_tasks_finished(self):
                return False

            def close(self):
                executor_closed.set()


        def build_executor(_payload, options=None):
            del options
            return FakeExecutor()


        def passthrough(table):
            return pa.table({"y": table.column(0)})


        udf_exec.build_executor = build_executor
        collector_mod.AsyncResultCollector = FakeCollector
        connection = duckdb.connect()
        relation = connection.sql("select i::BIGINT as x from range(2) t(i)").map_batches(
            passthrough,
            schema={"y": duckdb.sqltypes.BIGINT},
            execution_backend="ray_task",
        )
        query_errors = []


        def run_query():
            try:
                relation.fetchall()
            except BaseException as exc:
                query_errors.append(exc)


        query_thread = threading.Thread(target=run_query)
        query_thread.start()
        try:
            assert submit_entered.wait(timeout=5), "dispatcher never entered the Ray submit"
            connection.interrupt()
            teardown_deadline = time.monotonic() + 5
            while query_thread.is_alive() and time.monotonic() < teardown_deadline:
                _duckdb._wake_udf_executor_slots_for_testing()
                query_thread.join(timeout=0.01)
            assert not query_thread.is_alive(), "query teardown did not honor the unregister deadline"
            assert query_errors, "the interrupted query unexpectedly succeeded"

            release_submit.set()
            assert discarded.wait(timeout=5), "stale Ray stream was dropped without collector cleanup"
            assert executor_closed.wait(timeout=5), "retired executor was not eventually closed"
            assert collector_retired.wait(timeout=5), "collector cancellation was not retried through retirement"
            assert cancel_calls == (4 if failure_phase in {"cancel", "retire"} else 2)
            assert pending_checks == (1 if failure_phase == "cancel" else 4 if failure_phase == "retire" else 2)
            assert retire_calls == (4 if failure_phase == "retire" else 1)
        finally:
            release_submit.set()
            query_thread.join(timeout=5)
            connection.close()
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
            "VANE_TEST_COLLECTOR_FAILURE_PHASE": failure_phase,
            "VANE_UDF_UNREGISTER_TIMEOUT_MS": "50",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_pending_ray_slot_cleanup_does_not_spin_or_block_healthy_slot():
    """A remote cleanup ACK may delay its slot, never the process dispatcher."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        from __future__ import annotations

        import threading
        import time

        import _duckdb
        import duckdb
        import duckdb.execution.udf as udf_exec
        import duckdb.execution.udf_stream_result_collector as collector_mod
        import pyarrow as pa
        from duckdb.execution.ref_bundle import make_local_shm_ref_bundle_result

        state_lock = threading.Lock()
        slow_tracked = threading.Event()
        slow_cancel_started = threading.Event()
        slow_retired = threading.Event()
        collector_instance = None


        class FakeSource:
            def __init__(self, value):
                self.value = int(value)


        class FakeCollector:
            def __init__(self):
                global collector_instance
                self._wakeup = None
                self._slow_slot = None
                self._pending = set()
                self._cancelled = set()
                self._events = {}
                self.pending_checks = 0
                collector_instance = self

            def set_wakeup_callback(self, callback):
                self._wakeup = callback

            def track_generator_ref(self, slot_id, submit_id, source, _error_context):
                slot_id = int(slot_id)
                submit_id = int(submit_id)
                if source.value == 1:
                    with state_lock:
                        self._slow_slot = slot_id
                    slow_tracked.set()
                    return
                payload = make_local_shm_ref_bundle_result(pa.table({"y": [source.value]}))
                with state_lock:
                    self._events[slot_id] = [
                        (
                            slot_id,
                            submit_id,
                            "data",
                            payload,
                            "output-request:healthy",
                            "output-lease:healthy",
                        ),
                        (slot_id, submit_id, "complete", None),
                    ]
                if self._wakeup is not None:
                    self._wakeup()

            def discard_generator_ref(self, _slot_id, _source):
                return None

            def drain_results(self, capacities):
                results = []
                with state_lock:
                    for slot_id in list(self._events):
                        if int(capacities.get(slot_id, {}).get("rows", 0)) <= 0:
                            continue
                        results.extend(self._events.pop(slot_id))
                return results

            def handoff_output_block_lease(self, _request_id, _lease_id):
                return True

            def release_output_block_lease(self, _request_id, _lease_id):
                return True

            def cancel_slot(self, slot_id):
                slot_id = int(slot_id)
                with state_lock:
                    self._events.pop(slot_id, None)
                    if slot_id == self._slow_slot and slot_id not in self._cancelled:
                        self._cancelled.add(slot_id)
                        self._pending.add(slot_id)
                        slow_cancel_started.set()

            def slot_has_pending(self, slot_id):
                with state_lock:
                    self.pending_checks += 1
                    return int(slot_id) in self._pending

            def retire_slot(self, slot_id):
                if int(slot_id) == self._slow_slot:
                    slow_retired.set()
                return None

            def release_slow_cleanup(self):
                with state_lock:
                    self._pending.discard(self._slow_slot)
                    wakeup = self._wakeup
                if wakeup is not None:
                    wakeup()

            def shutdown(self):
                return None


        class FakeExecutor:
            def __init__(self):
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

            def submit_with_id(self, _submit_id, table):
                self._admission_state = "idle"
                return FakeSource(table.column(0).to_pylist()[0])

            def take_ready_result(self):
                return None

            def finished_submitting(self):
                return None

            def all_tasks_finished(self):
                return False

            def close(self):
                return None


        def build_executor(_payload, options=None):
            del options
            return FakeExecutor()


        def passthrough(table):
            return pa.table({"y": table.column(0)})


        udf_exec.build_executor = build_executor
        collector_mod.AsyncResultCollector = FakeCollector
        slow_connection = duckdb.connect()
        healthy_connection = duckdb.connect()
        slow_relation = slow_connection.sql("select 1::BIGINT as x").map_batches(
            passthrough,
            schema={"y": duckdb.sqltypes.BIGINT},
            execution_backend="ray_task",
        )
        healthy_relation = healthy_connection.sql("select 2::BIGINT as x").map_batches(
            passthrough,
            schema={"y": duckdb.sqltypes.BIGINT},
            execution_backend="ray_task",
        )
        slow_errors = []


        def run_slow():
            try:
                slow_relation.fetchall()
            except BaseException as exc:
                slow_errors.append(exc)


        slow_thread = threading.Thread(target=run_slow)
        slow_thread.start()
        try:
            assert slow_tracked.wait(timeout=5), "slow Ray slot was not tracked"
            slow_connection.interrupt()
            deadline = time.monotonic() + 5
            while not slow_cancel_started.is_set() and time.monotonic() < deadline:
                _duckdb._wake_udf_executor_slots_for_testing()
                time.sleep(0.01)
            assert slow_cancel_started.is_set(), "slow Ray slot never entered cleanup"

            with state_lock:
                checks_before_idle = collector_instance.pending_checks
            time.sleep(0.1)
            with state_lock:
                idle_checks = collector_instance.pending_checks - checks_before_idle
            assert idle_checks <= 1, f"pending cleanup busy-spun dispatcher: {idle_checks} checks"

            healthy_result = []
            healthy_errors = []

            def run_healthy():
                try:
                    healthy_result.extend(healthy_relation.fetchall())
                except BaseException as exc:
                    healthy_errors.append(exc)

            healthy_thread = threading.Thread(target=run_healthy)
            healthy_thread.start()
            healthy_thread.join(timeout=5)
            assert not healthy_thread.is_alive(), "pending cleanup blocked the healthy Ray slot"
            assert healthy_errors == [], healthy_errors
            assert healthy_result == [(2,)], healthy_result
            assert not slow_retired.is_set(), "slow slot retired before its cleanup ACK"

            collector_instance.release_slow_cleanup()
            assert slow_retired.wait(timeout=5), "cleanup ACK did not wake slot retirement"
            slow_thread.join(timeout=5)
            assert not slow_thread.is_alive(), "slow query did not finish after cleanup ACK"
            assert slow_errors, "interrupted slow query unexpectedly succeeded"
        finally:
            if collector_instance is not None:
                collector_instance.release_slow_cleanup()
            slow_thread.join(timeout=5)
            slow_connection.close()
            healthy_connection.close()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={
            **os.environ,
            "VANE_ENABLE_UDF_TEST_HOOKS": "1",
            "VANE_UDF_UNREGISTER_TIMEOUT_MS": "5000",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_output_lease_callback_failure_isolated_to_owning_ray_slot():
    """One descriptor callback must not drop or fail another slot's callback."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        from __future__ import annotations

        import threading

        import duckdb
        import duckdb.execution.udf as udf_exec
        import duckdb.execution.udf_stream_result_collector as collector_mod
        import pyarrow as pa
        from duckdb.execution.ref_bundle import make_local_shm_ref_bundle_result

        both_tracked = threading.Event()
        good_handoff = threading.Event()


        class FakeSource:
            def __init__(self, value):
                self.value = int(value)


        class FakeCollector:
            def __init__(self):
                self._lock = threading.Lock()
                self._wakeup = None
                self._events = {}
                self._tracked = set()
                self._cancelled = set()

            def set_wakeup_callback(self, callback):
                self._wakeup = callback

            def track_generator_ref(self, slot_id, submit_id, source, _error_context):
                slot_id = int(slot_id)
                submit_id = int(submit_id)
                value = int(source.value)
                request_id = "output-request:bad" if value == 1 else "output-request:good"
                payload = make_local_shm_ref_bundle_result(pa.table({"y": [value]}))
                with self._lock:
                    self._tracked.add(slot_id)
                    self._events[slot_id] = [
                        (
                            slot_id,
                            submit_id,
                            "data",
                            payload,
                            request_id,
                            f"output-lease:{value}",
                        ),
                        (slot_id, submit_id, "complete", None),
                    ]
                    if len(self._tracked) == 2:
                        both_tracked.set()
                if self._wakeup is not None:
                    self._wakeup()

            def discard_generator_ref(self, _slot_id, _source):
                return None

            def drain_results(self, capacities):
                if not both_tracked.is_set():
                    return []
                results = []
                with self._lock:
                    # Force the failing callback ahead of the healthy callback
                    # in the dispatcher's same release batch.
                    ordered = sorted(
                        self._events,
                        key=lambda slot_id: self._events[slot_id][0][4] != "output-request:bad",
                    )
                    for slot_id in ordered:
                        if slot_id in self._cancelled:
                            self._events.pop(slot_id, None)
                            continue
                        capacity = capacities.get(slot_id, {})
                        if int(capacity.get("rows", 0)) <= 0:
                            continue
                        results.extend(self._events.pop(slot_id, ()))
                return results

            def handoff_output_block_lease(self, request_id, _lease_id):
                if str(request_id) == "output-request:bad":
                    raise RuntimeError("planned output lease handoff failure")
                good_handoff.set()
                return True

            def release_output_block_lease(self, _request_id, _lease_id):
                return True

            def cancel_slot(self, slot_id):
                with self._lock:
                    self._cancelled.add(int(slot_id))
                    self._events.pop(int(slot_id), None)

            def slot_has_pending(self, _slot_id):
                return False

            def retire_slot(self, _slot_id):
                return None

            def shutdown(self):
                return None


        class FakeExecutor:
            def __init__(self):
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

            def submit_with_id(self, _submit_id, table):
                self._admission_state = "idle"
                return FakeSource(table.column(0).to_pylist()[0])

            def take_ready_result(self):
                return None

            def finished_submitting(self):
                return None

            def all_tasks_finished(self):
                return False

            def close(self):
                return None


        def build_executor(_payload, options=None):
            del options
            return FakeExecutor()


        def passthrough(table):
            return pa.table({"y": table.column(0)})


        udf_exec.build_executor = build_executor
        collector_mod.AsyncResultCollector = FakeCollector
        bad_connection = duckdb.connect()
        good_connection = duckdb.connect()
        bad_relation = bad_connection.sql("select 1::BIGINT as x").map_batches(
            passthrough,
            schema={"y": duckdb.sqltypes.BIGINT},
            execution_backend="ray_task",
        )
        good_relation = good_connection.sql("select 2::BIGINT as x").map_batches(
            passthrough,
            schema={"y": duckdb.sqltypes.BIGINT},
            execution_backend="ray_task",
        )
        barrier = threading.Barrier(3)
        bad_errors = []
        good_results = []
        good_errors = []


        def run_bad():
            barrier.wait()
            try:
                bad_relation.fetchall()
            except BaseException as exc:
                bad_errors.append(exc)


        def run_good():
            barrier.wait()
            try:
                good_results.extend(good_relation.fetchall())
            except BaseException as exc:
                good_errors.append(exc)


        bad_thread = threading.Thread(target=run_bad)
        good_thread = threading.Thread(target=run_good)
        bad_thread.start()
        good_thread.start()
        barrier.wait()
        bad_thread.join(timeout=10)
        good_thread.join(timeout=10)
        try:
            assert not bad_thread.is_alive(), "failing Ray slot did not terminate"
            assert not good_thread.is_alive(), "healthy Ray slot did not terminate"
            assert bad_errors, (bad_errors, good_errors, good_results)
            assert "planned output lease handoff failure" in str(bad_errors[0]), bad_errors[0]
            assert good_errors == [], (bad_errors, good_errors, good_results)
            assert good_results == [(2,)], (bad_errors, good_errors, good_results)
            assert good_handoff.is_set(), "healthy output lease callback was dropped"
        finally:
            bad_connection.close()
            good_connection.close()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ},
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
