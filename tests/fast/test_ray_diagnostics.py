# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import vane
from tests.ray_diagnostic_helpers import raise_diagnostic_error


class _Worker:
    def __init__(self, error, handles):
        self.error = error
        self.handles = handles
        self.status_calls = 0
        self.pop_calls = 0

    def fte_query_status(self, _query_id):
        self.status_calls += 1
        if self.status_calls > bool(self.handles):
            raise_diagnostic_error(self.error, long_traceback=True)
        return {"failed": False, "finished": False, "selected_attempt_task_ids": []}

    def pop_fte_result_handles(self, _query_id):
        self.pop_calls += 1
        return self.handles if self.pop_calls == 1 else []

    def task_input_stream_exhausted(self, _query_id, _source_node_ids):
        return self.handles

    def register_query_owner(self, _query_id, _owner_query_id):
        pass

    def worker_snapshots(self):
        return []

    def stats_fragments(self):
        return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

    def fte_prepare_drop_query(self, _query_id):
        return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

    def fte_cleanup_query(self, _query_id):
        return {}

    def drop_query(self, _query_id):
        pass

    def prepare_shutdown(self):
        pass

    def finish_shutdown(self):
        pass

    def abort_shutdown(self):
        pass

    def shutdown(self):
        pass


@contextmanager
def _diagnostic_runner(monkeypatch, backend_kind, error, handles=()):
    import vane.runners.ray.worker_handle as worker_handle

    worker = _Worker(error, handles)
    if backend_kind == "native":
        monkeypatch.setattr(
            worker_handle,
            "start_ray_workers",
            lambda _ids, _manager_id: [vane.ray_cxx.RayWorkerRuntime("diagnostic-worker", worker, 1.0, 0.0, 1024)],
        )
        monkeypatch.setattr(worker_handle, "try_autoscale", lambda _bundles: None)
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    else:
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner(worker)
    try:
        runner.warm_up()
        runner._register_query_owner_for_test("diagnostic-query", "diagnostic-query")
        yield runner, worker
    finally:
        runner.shutdown()


@pytest.mark.parametrize("backend_kind", ["native", "python"])
@pytest.mark.parametrize("message", ["status exploded", "主错误🙂" + "界" * 10_000, "primary\x00after-nul\ud800"])
def test_fte_diagnostics_keep_primary_message_independent_of_traceback(monkeypatch, backend_kind, message):
    with _diagnostic_runner(monkeypatch, backend_kind, RuntimeError(message)) as (runner, worker):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "RuntimeError: " in diagnostic
    assert message.split("\x00")[0][:6] in diagnostic
    assert "error detail exceeds" not in diagnostic
    assert "/_pytest/" not in diagnostic
    assert "/pluggy/" not in diagnostic
    assert len(diagnostic.encode("utf-8")) <= 4096
    assert worker.status_calls == 1
    if "\x00" in message:
        assert "after-nul" in diagnostic
        assert "\\ud800" in diagnostic


@pytest.mark.parametrize("backend_kind", ["native", "python"])
@pytest.mark.parametrize("padding", ["\x00" * 2000, "\x00" * 10_000, "\ud800" * 2000])
def test_fte_status_diagnostics_preserve_tail_after_text_normalization(monkeypatch, backend_kind, padding):
    with _diagnostic_runner(monkeypatch, backend_kind, RuntimeError("unused")) as (runner, worker):
        worker.fte_query_status = lambda _query_id: {
            "failed": True,
            "finished": False,
            "selected_attempt_task_ids": [],
            "message": "status-head:" + padding + ":reason-tail",
        }
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "status-head:" in diagnostic
    assert ":reason-tail" in diagnostic
    assert len(diagnostic.encode("utf-8")) <= 4096


@pytest.mark.parametrize("padding", ["x" * 2000, "界" * 100_000])
def test_native_exception_diagnostics_preserve_oversized_message_tail(padding):
    diagnostic = vane.ray_cxx._native_error_diagnostic_for_test("native-head:" + padding + ":reason-tail")
    assert "native-head:" in diagnostic
    assert ":reason-tail" in diagnostic
    assert len(diagnostic.encode("utf-8")) <= 4096


@pytest.mark.parametrize("backend_kind", ["native", "python"])
def test_fte_diagnostics_do_not_execute_exception_formatters_or_descriptors(monkeypatch, backend_kind):
    calls = []

    class HostileError(RuntimeError):
        def __str__(self):
            calls.append("str")
            raise AssertionError("exception formatting must not execute provider code")

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("exception repr must not execute provider code")

        @property
        def args(self):
            calls.append("args")
            raise AssertionError("exception args descriptor must not execute provider code")

    error = HostileError("original primary failure")
    with _diagnostic_runner(monkeypatch, backend_kind, error) as (runner, _worker):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "HostileError: original primary failure" in diagnostic
    assert calls == []


@pytest.mark.parametrize("backend_kind", ["native", "python"])
def test_fte_diagnostics_do_not_compare_custom_exception_dictionary_keys(monkeypatch, backend_kind):
    calls = []

    class OpaqueKey:
        def __hash__(self):
            return hash("cause")

        def __eq__(self, other):
            calls.append(other)
            raise AssertionError("diagnostics must not execute dictionary key comparison")

    error = RuntimeError("original primary failure")
    error.__dict__[OpaqueKey()] = None
    with _diagnostic_runner(monkeypatch, backend_kind, error) as (runner, _):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "RuntimeError: original primary failure" in diagnostic
    assert calls == []


@pytest.mark.parametrize("backend_kind", ["native", "python"])
def test_fte_diagnostics_bound_type_name_without_spending_message_budget(monkeypatch, backend_kind):
    error_type = type("LargeError" + "界" * 10_000, (RuntimeError,), {})
    with _diagnostic_runner(monkeypatch, backend_kind, error_type("primary survives type truncation")) as (runner, _):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "primary survives type truncation" in diagnostic
    assert "LargeError" in diagnostic
    assert len(diagnostic.encode("utf-8")) <= 4096


@pytest.mark.parametrize("backend_kind", ["native", "python"])
def test_fte_diagnostics_preserve_ray_task_error_cause(monkeypatch, backend_kind):
    import ray

    error = ray.exceptions.RayTaskError(
        "remote_status",
        "remote-traceback\n" * 10_000,
        ValueError("remote primary cause"),
        proctitle="diagnostic-worker",
        pid=123,
        ip="127.0.0.1",
    )
    with _diagnostic_runner(monkeypatch, backend_kind, error) as (runner, _worker):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "ValueError: remote primary cause" in diagnostic
    assert len(diagnostic.encode("utf-8")) <= 4096


_RAY_MESSAGE_FIELDS = {
    "RayActorError": "error_msg",
    "ActorDiedError": "error_msg",
    "ActorUnavailableError": "error_msg",
    "TaskCancelledError": "error_message",
    "RuntimeEnvSetupError": "error_message",
    "TaskUnschedulableError": "error_message",
    "ActorUnschedulableError": "error_message",
    "OutOfMemoryError": "message",
    "NodeDiedError": "message",
    "RpcError": "message",
}


def _ray_message_error(error_name, message):
    from ray import exceptions

    error_type = getattr(exceptions, error_name)
    if error_name == "ActorDiedError":
        error = error_type()
        error.error_msg = message
    elif error_name == "ActorUnavailableError":
        error = error_type(error_message=message, actor_id=None)
    else:
        error = error_type(**{_RAY_MESSAGE_FIELDS[error_name]: message})
    assert error.args == ()
    return error


@pytest.mark.parametrize("backend_kind", ["native", "python"])
@pytest.mark.parametrize("error_name", _RAY_MESSAGE_FIELDS)
def test_fte_diagnostics_preserve_ray_canonical_messages_with_empty_args(monkeypatch, backend_kind, error_name):
    error = _ray_message_error(error_name, "canonical Ray failure reason")
    with _diagnostic_runner(monkeypatch, backend_kind, error) as (runner, _):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert error_name + ": " in diagnostic
    assert "canonical Ray failure reason" in diagnostic
    assert len(diagnostic.encode("utf-8")) <= 4096


@pytest.mark.parametrize("backend_kind", ["native", "python"])
@pytest.mark.parametrize("error_name", ["ActorDiedError", "TaskCancelledError"])
@pytest.mark.parametrize(
    "message",
    [
        "reason-head:" + "界" * 10_000 + ":reason-tail",
        "reason-head:" + "\x00" * 1000 + ":reason-tail",
        "before\x00after\ud800",
    ],
)
def test_fte_diagnostics_bound_ray_canonical_message_edges(monkeypatch, backend_kind, error_name, message):
    error = _ray_message_error(error_name, message)
    with _diagnostic_runner(monkeypatch, backend_kind, error) as (runner, _):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert len(diagnostic.encode("utf-8")) <= 4096
    if "reason-tail" in message:
        assert "reason-head:" in diagnostic
        assert ":reason-tail" in diagnostic
    else:
        assert "before\\x00after\\ud800" in diagnostic


@pytest.mark.parametrize("backend_kind", ["native", "python"])
@pytest.mark.parametrize("error_name", ["ActorDiedError", "TaskCancelledError"])
def test_fte_diagnostics_read_inherited_ray_fields_without_provider_code(monkeypatch, backend_kind, error_name):
    from ray import exceptions

    calls = []

    def forbidden(*_args):
        calls.append("provider formatting or attribute access")
        raise AssertionError(calls[-1])

    field = _RAY_MESSAGE_FIELDS[error_name]
    error_type = type(
        "HostileRayError",
        (getattr(exceptions, error_name),),
        {field: property(forbidden), "__str__": forbidden, "__repr__": forbidden},
    )
    error = BaseException.__new__(error_type)
    BaseException.__init__(error)

    class OpaqueKey:
        active = False

        def __hash__(self):
            return hash(field)

        def __eq__(self, other):
            if self.active:
                forbidden(other)
            return False

    key = OpaqueKey()
    error.__dict__[key] = None
    error.__dict__[field] = "canonical inherited reason"
    key.active = True
    with _diagnostic_runner(monkeypatch, backend_kind, error) as (runner, _):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "HostileRayError: canonical inherited reason" in diagnostic
    assert calls == []


@pytest.mark.parametrize("backend_kind", ["native", "python"])
def test_fte_diagnostics_keep_ray_message_fields_in_exception_chains(monkeypatch, backend_kind):
    error = RuntimeError("outer failure")
    error.__cause__ = _ray_message_error("TaskCancelledError", "canonical cancellation reason")
    with _diagnostic_runner(monkeypatch, backend_kind, error) as (runner, _):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "TaskCancelledError: canonical cancellation reason" in diagnostic


@pytest.mark.parametrize("backend_kind", ["native", "python"])
def test_fte_diagnostics_preserve_ray_task_wrapper_and_canonical_cause(monkeypatch, backend_kind):
    from ray import exceptions

    error = exceptions.RayTaskError(
        "remote_cancel_operation",
        "remote traceback\n" * 10_000,
        exceptions.TaskCancelledError(error_message="remote cancellation reason"),
        proctitle="diagnostic-worker",
        pid=123,
        ip="127.0.0.1",
    ).as_instanceof_cause()
    with _diagnostic_runner(monkeypatch, backend_kind, error) as (runner, _):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "RayTaskError(TaskCancelledError): <TaskCancelledError>" in diagnostic
    assert "TaskCancelledError: remote cancellation reason" in diagnostic
    assert len(diagnostic.encode("utf-8")) <= 4096


@pytest.mark.parametrize("backend_kind", ["native", "python"])
def test_fte_diagnostics_ignore_ray_field_names_on_unrelated_exceptions(monkeypatch, backend_kind):
    error_type = type("TaskCancelledError", (RuntimeError,), {})
    error = error_type("original primary reason")
    error.error_message = "unrelated metadata"
    with _diagnostic_runner(monkeypatch, backend_kind, error) as (runner, _):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "original primary reason" in diagnostic
    assert "unrelated metadata" not in diagnostic


@pytest.mark.real_ray
@pytest.mark.parametrize("failure_kind", ["actor-death", "task-cancellation"])
def test_ray_diagnostics_preserve_real_failures_during_materialization(ray_local, failure_kind):
    import ray

    if failure_kind == "actor-death":

        @ray.remote(num_cpus=0)
        class Actor:
            def ping(self):
                return 1

        actor = Actor.remote()
        try:
            assert ray.get(actor.ping.remote(), timeout=20) == 1
        finally:
            ray.kill(actor, no_restart=True)
        # ray.kill() submits an asynchronous request. Wait for an actual
        # failed ObjectRef before exercising its materialization path.
        deadline = time.monotonic() + 20
        while True:
            ref = actor.ping.remote()
            try:
                ray.get(ref, timeout=max(0.1, deadline - time.monotonic()))
            except ray.exceptions.ActorDiedError:
                break
            if time.monotonic() >= deadline:
                pytest.fail("actor did not die after ray.kill()")
            time.sleep(0.05)
        error_type = ray.exceptions.ActorDiedError
        expected = "ray.kill"
    else:

        @ray.remote(num_cpus=0)
        def pending():
            time.sleep(60)

        ref = pending.remote()
        ray.cancel(ref, force=False)
        error_type = ray.exceptions.TaskCancelledError
        expected = "TaskCancelledError"

    with pytest.raises(error_type) as original:
        ray.get(ref, timeout=20)
    if failure_kind == "actor-death":
        assert original.value.args == ()
        assert expected in original.value.error_msg

    partition = vane.ray_cxx._RayBackedResultPartitionForTest(ref)
    with pytest.raises(RuntimeError) as captured:
        partition.materialize()
    diagnostic = str(captured.value)
    assert expected in diagnostic
    assert "[no message]" not in diagnostic
    assert len(diagnostic.encode("utf-8")) <= 4096


@pytest.mark.parametrize("backend_kind", ["native", "python"])
def test_fte_diagnostics_bound_non_string_arguments_without_repr(monkeypatch, backend_kind):
    class Opaque:
        def __repr__(self):
            raise AssertionError("diagnostics must not call argument repr")

    error = RuntimeError(17, "primary argument", Opaque(), 10**10_000, "omitted argument")
    with _diagnostic_runner(monkeypatch, backend_kind, error) as (runner, _worker):
        diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
    assert "17, primary argument, <" in diagnostic
    assert "<int>" in diagnostic
    assert "additional arguments omitted" in diagnostic
    assert len(diagnostic.encode("utf-8")) <= 4096


@pytest.mark.parametrize("backend_kind", ["native", "python"])
@pytest.mark.parametrize("chain_kind", ["cause", "context", "suppressed", "cycle", "long"])
def test_fte_diagnostics_bound_exception_chains(monkeypatch, backend_kind, chain_kind):
    primary = RuntimeError("primary failure")
    cause = ValueError("underlying failure")
    if chain_kind in {"context", "suppressed"}:
        primary.__context__ = cause
        primary.__suppress_context__ = chain_kind == "suppressed"
    else:
        primary.__cause__ = cause
        if chain_kind == "cycle":
            cause.__cause__ = primary
        elif chain_kind == "long":
            current = cause
            for i in range(100):
                current.__cause__ = ValueError(f"nested failure {i}")
                current = current.__cause__
    try:
        with _diagnostic_runner(monkeypatch, backend_kind, primary) as (runner, _worker):
            diagnostic = runner._wait_fte_query_diagnostic_for_test("diagnostic-query")
        assert "RuntimeError: primary failure" in diagnostic
        assert ("ValueError: underlying failure" in diagnostic) == (chain_kind != "suppressed")
        assert len(diagnostic.encode("utf-8")) <= 4096
        if chain_kind == "cycle":
            assert "exception chain cycle" in diagnostic
        elif chain_kind == "long":
            assert "exception chain limit" in diagnostic
            assert "nested failure 99" not in diagnostic
    finally:
        cause.__cause__ = None
        primary.__cause__ = None
        primary.__context__ = None


class _ReleaseHandle:
    worker_id = "diagnostic-worker"

    def __init__(self, index):
        self.index = index
        self.release_calls = 0
        self.task_id = SimpleNamespace(
            query_id="diagnostic-query", fragment_execution_id=0, partition_id=index, attempt_id=0
        )
        self.task_context_info = {"query_idx": 1, "last_node_id": 2, "task_id": index, "node_ids": [2]}

    def done(self):
        return False

    def release_result_payload(self):
        self.release_calls += 1
        if self.release_calls == 1:
            raise_diagnostic_error(RuntimeError(f"cleanup-{self.index}:" + "清" * 5000), long_traceback=True)


@pytest.mark.parametrize("backend_kind", ["native", "python"])
def test_fte_diagnostics_keep_primary_when_many_cleanup_errors_are_merged(monkeypatch, backend_kind):
    handles = [_ReleaseHandle(i) for i in range(20)]
    with _diagnostic_runner(monkeypatch, backend_kind, RuntimeError("primary status failure"), handles) as (
        runner,
        worker,
    ):
        # Python backends publish handles through task/input submission; native
        # worker managers collect them while polling query status.
        diagnostic = runner._wait_fte_query_diagnostic_for_test(
            "diagnostic-query", exhaust_input=backend_kind == "python"
        )
        assert "RuntimeError: primary status failure" in diagnostic
        assert "cleanup-0:" in diagnostic
        assert "additional 5 error(s) omitted" in diagnostic
        assert diagnostic.index("primary status failure") < diagnostic.index("cleanup-0:")
        assert diagnostic.index("cleanup-0:") < diagnostic.index("Traceback [")
        assert len(diagnostic.encode("utf-8")) <= 65536
        assert [handle.release_calls for handle in handles] == [1] * 20
        runner.drop_query_fragments("diagnostic-query")
        assert [handle.release_calls for handle in handles] == [2] * 20
        assert worker.status_calls == 2


@pytest.mark.parametrize("backend_kind", ["native", "python"])
def test_fte_diagnostics_are_identical_for_overlapping_teardown_callers(monkeypatch, backend_kind):
    thread_count = 6
    barrier = threading.Barrier(thread_count)
    condition = threading.Condition()
    calls = []
    errors = []

    def abort_query(_worker, query_id):
        with condition:
            calls.append(query_id)
            call_number = len(calls)
            condition.notify_all()
            if call_number == 1:
                # Give every caller time to join the same teardown generation.
                condition.wait_for(lambda: len(calls) == thread_count, timeout=0.5)
        if call_number == 1:
            raise_diagnostic_error(RuntimeError("shared teardown primary failure"), long_traceback=True)
        return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

    method = "fte_prepare_drop_query" if backend_kind == "native" else "drop_query"
    monkeypatch.setattr(_Worker, method, abort_query)
    with _diagnostic_runner(monkeypatch, backend_kind, RuntimeError("unused status error")) as (runner, _):

        def drop_query():
            try:
                barrier.wait(timeout=5)
                runner.drop_query_fragments("diagnostic-query")
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=drop_query) for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert not any(thread.is_alive() for thread in threads)
        assert len(errors) == thread_count
        messages = {str(error) for error in errors}
        assert len(messages) == 1
        message = messages.pop()
        assert "RuntimeError: shared teardown primary failure" in message
        assert len(message.encode("utf-8")) <= 4096
        assert calls == ["diagnostic-query"]
        runner.drop_query_fragments("diagnostic-query")
        assert calls == ["diagnostic-query"] * 2
