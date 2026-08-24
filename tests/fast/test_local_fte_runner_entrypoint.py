# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys

import pytest


def test_set_runner_local_entrypoint_in_subprocess():
    script = """
import os
import vane.runners as runners
from vane.runners.local import LocalRunner

os.environ.pop("VANE_RUNNER", None)
runner = runners.set_runner_local(num_workers=1, max_running_tasks=1)
assert isinstance(runner, LocalRunner)
assert runner.name == "local"
assert os.environ["VANE_RUNNER"] == "local"
assert os.environ["VANE_LOCAL_FTE_WORKERS"] == "1"
assert runner.max_running_tasks == 1
assert runners.get_or_infer_runner_type() == "local"
try:
    runner.run_iter(None)
except NotImplementedError as exc:
    assert "local FTE run_iter" in str(exc)
else:
    raise AssertionError("local FTE run_iter should be hidden until streaming is wired")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_root_set_runner_local_configures_the_execution_mode_in_subprocess():
    script = """
import os
import vane
import vane.runners as runners
from typing import Any, get_type_hints
from vane.runners.local import set_runner_local as set_runner_local_from_package
from vane.runners.ray import set_runner_ray as set_runner_ray_from_package
from vane.runners.runner import Runner

os.environ.pop("VANE_RUNNER", None)
runner = vane.set_runner_local(num_workers=1, max_running_tasks=1)
assert runner.name == "local"
assert os.environ["VANE_RUNNER"] == "local"
assert vane.sql("SELECT 42").fetchall() == [(42,)]
assert get_type_hints(vane.set_runner_local)["return"] is Any
assert get_type_hints(vane.set_runner_ray)["return"] is Any
assert get_type_hints(runners.get_or_create_runner)["return"] is Runner
assert get_type_hints(set_runner_local_from_package)["return"] is Runner
assert get_type_hints(set_runner_ray_from_package)["return"] is Runner
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_get_or_create_runner_accepts_local_env_in_subprocess():
    script = """
import os
import vane.runners as runners

os.environ["VANE_RUNNER"] = "local"
runner = runners.get_or_create_runner()
assert runner.name == "local"
assert runners.get_or_infer_runner_type() == "local"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_get_or_create_runner_rejects_native_fte_env_in_subprocess():
    script = """
import os
import vane
import vane.runners as runners

os.environ["VANE_RUNNER"] = "native-fte"
try:
    runners.get_or_create_runner()
except vane.InvalidInputException as exc:
    assert "Please use 'local-fast', 'local', or 'ray'" in str(exc)
else:
    raise AssertionError("native-fte should no longer be a public runner")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_local_runner_preloads_arrow_dataset_imports():
    from vane.runners.local.runner import _preload_arrow_dataset_imports

    _preload_arrow_dataset_imports()
    _preload_arrow_dataset_imports()


def test_local_runner_cleanup_note_cannot_mask_primary_failure():
    from vane.runners.local.runner import _add_exception_note

    class _UnnotableError(RuntimeError):
        @property
        def add_note(self):
            raise RuntimeError("planned add_note lookup failure")

    primary_error = _UnnotableError("planned primary failure")

    _add_exception_note(primary_error, "secondary cleanup failure")

    assert str(primary_error) == "planned primary failure"


def test_local_runner_rejects_unknown_native_copy_outcome():
    from vane.runners import CopyOutcomeUnknownError
    from vane.runners.local.runner import _require_known_copy_outcome

    result = {
        "copy_output_committed": False,
        "copy_output_outcome_unknown": True,
        "copy_output_outcome_error": "marker readback unavailable",
        "copy_output_base_path": "s3://bucket/out",
        "copy_output_run_id": "run-local-unknown",
        "copy_output_manifest_path": "s3://bucket/out.duckdb_commit/run-local-unknown/manifest.txt",
        "copy_output_committed_marker_path": "s3://bucket/out.duckdb_commit/run-local-unknown/committed",
    }

    try:
        _require_known_copy_outcome("local-copy", result)
    except CopyOutcomeUnknownError as error:
        assert error.operation_id == "local-copy"
        assert error.run_id == "run-local-unknown"
        assert error.safe_to_retry is False
    else:
        raise AssertionError("local runner must reject an unknown COPY outcome")


def test_local_runner_records_cleanup_failures_on_unknown_copy_outcome():
    from vane.runners import CopyOutcomeUnknownError
    from vane.runners.local.runner import _record_unknown_copy_cleanup_errors

    error = CopyOutcomeUnknownError(
        "local-copy",
        "s3://bucket/out",
        "run-local-unknown",
        cleanup_warnings=("native cleanup warning",),
    )

    recorded = _record_unknown_copy_cleanup_errors(
        error,
        "local write resource shutdown",
        [RuntimeError("backend join timed out"), ValueError("fragment close failed")],
    )

    assert recorded is True
    assert error.cleanup_warnings == (
        "native cleanup warning",
        "local write resource shutdown failed: RuntimeError: backend join timed out",
        "local write resource shutdown failed: ValueError: fragment close failed",
    )
    assert "backend join timed out" in str(error)
    assert error.safe_to_retry is False


def test_local_runner_uses_write_specific_logical_plan_factory(monkeypatch):
    from vane.runners.local import runner as runner_module

    relation = object()

    class FakeLogicalPlan:
        @staticmethod
        def from_duckdb_write_relation(actual_relation, _query_id):
            assert actual_relation is relation
            raise RuntimeError("write transaction validation reached")

    monkeypatch.setattr(runner_module, "_preload_arrow_dataset_imports", lambda: None)
    monkeypatch.setattr(
        runner_module,
        "require_ray_cxx_attr",
        lambda name: FakeLogicalPlan if name == "PyLogicalPlan" else object,
    )

    with pytest.raises(RuntimeError, match="write transaction validation reached"):
        runner_module.LocalRunner().run_write(relation)


def test_local_fragment_executor_passes_authoritative_task_attempt_to_native(monkeypatch):
    from vane.runners.fte import FteTaskAttemptId, FteTaskId
    from vane.runners.local import runner as runner_module

    attempt_id = FteTaskAttemptId(FteTaskId("query-id", 7, 3), 2)
    native_calls = []

    class FakeCursor:
        def close(self):
            pass

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    class FakePlanRunner:
        def execute_native(self, *args):
            native_calls.append(args)
            return "ok"

    monkeypatch.setattr(
        runner_module.NativeFteWorkerManagerBackend,
        "materialize_task_context",
        staticmethod(lambda _request, *, merge_scan_split_batches: {}),
    )
    monkeypatch.setattr(runner_module, "require_ray_cxx_attr", lambda _name: object())

    executor = runner_module._InProcessFragmentExecutor()
    monkeypatch.setattr(executor, "_get_conn", lambda: FakeConnection())
    monkeypatch.setattr(executor, "_get_plan_runner", lambda: FakePlanRunner())

    result = executor(
        {
            "task_id": attempt_id.to_dict(),
            "fragment_plan": object(),
        }
    )

    assert result == "ok"
    assert len(native_calls) == 1
    assert native_calls[0][10] == {"task_id": str(attempt_id)}


def test_local_runner_rejects_invalid_num_workers():
    from vane.runners.local import _normalize_num_workers
    from vane.runners.local.runner import _normalize_num_workers as normalize_runner

    for normalize in (_normalize_num_workers, normalize_runner):
        for value in (0, -1, 1.5, True, "2"):
            try:
                normalize(value)
            except ValueError as exc:
                assert "num_workers must be a positive integer" in str(exc)
            else:
                raise AssertionError(f"expected invalid num_workers for {value!r}")


def test_local_runner_smoke_writes_parquet_in_subprocess():
    script = """
import pathlib
import tempfile

import vane
from vane.runners.local import set_runner_local

tmp = pathlib.Path(tempfile.mkdtemp())
src = tmp / "input.parquet"
dst = tmp / "output.parquet"

setup_conn = vane.connect()
setup_conn.execute(f"COPY (SELECT i::integer as x FROM range(3) tbl(i)) TO '{src}' (FORMAT PARQUET)")

set_runner_local(num_workers=1, max_running_tasks=1)
conn = vane.connect()
conn.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))

assert dst.exists()
assert sorted(row[0] for row in conn.read_parquet(str(dst)).fetchall()) == [0, 1, 2]
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_local_runner_repartition_write_uses_local_exchange_node_in_subprocess():
    script = """
import pathlib
import tempfile

import vane
from vane.runners.local import set_runner_local

tmp = pathlib.Path(tempfile.mkdtemp())
src = tmp / "input.parquet"
dst = tmp / "output.parquet"

setup_conn = vane.connect()
setup_conn.execute(
    f"COPY (SELECT i::integer as x, (i % 3)::integer as k FROM range(20) tbl(i)) TO '{src}' (FORMAT PARQUET)"
)

set_runner_local(num_workers=1, max_running_tasks=1)
conn = vane.connect()
conn.read_parquet(str(src)).repartition(4).write_parquet(str(dst))

rows = conn.sql(f"select count(*), sum(x) from read_parquet('{dst}')").fetchone()
assert rows == (20, 190)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_local_runner_collects_udf_actor_shutdown_errors_after_attempting_every_pool():
    from vane.runners.local.runner import _shutdown_udf_actor_pools

    calls = []

    class FakePool:
        def __init__(self, name, *, fail=False):
            self.name = name
            self.fail = fail

        def shutdown(self, *, kill):
            calls.append((self.name, kill))
            if self.fail:
                raise RuntimeError(f"{self.name} cleanup failed")

    pools = [FakePool("first"), FakePool("second", fail=True)]
    graceful_errors = _shutdown_udf_actor_pools(pools, kill=False)
    forced_errors = _shutdown_udf_actor_pools(pools, kill=True)

    assert calls == [
        ("second", False),
        ("first", False),
        ("second", True),
        ("first", True),
    ]
    assert len(graceful_errors) == 1
    assert "second cleanup failed" in str(graceful_errors[0])
    assert len(forced_errors) == 1
    assert "second cleanup failed" in str(forced_errors[0])


def test_local_runner_teardown_releases_actor_pools_after_execution_resources():
    from vane.runners.local.runner import _shutdown_local_write_resources

    events = []
    backend_timeouts = []
    fragment_timeouts = []

    class FakeBackend:
        def request_shutdown(self):
            events.append("backend-request")

        def shutdown(self, *, timeout_s):
            backend_timeouts.append(timeout_s)
            events.append("backend-join")

    class FakeFragmentExecutor:
        def request_shutdown(self):
            events.append("fragments-request")

        def close(self, *, timeout_s):
            fragment_timeouts.append(timeout_s)
            events.append("fragments-close")

    class FakeConn:
        def close(self):
            events.append("connection")

    class FakePool:
        def __init__(self, name):
            self.name = name

        def shutdown(self, *, kill):
            events.append(f"pool:{self.name}:{kill}")

    errors = _shutdown_local_write_resources(
        FakeBackend(),
        FakeFragmentExecutor(),
        FakeConn(),
        [FakePool("first"), FakePool("second")],
        timeout_s=7.0,
    )

    assert errors == []
    assert 0 < fragment_timeouts[0] <= backend_timeouts[0] <= 7.0
    assert events == [
        "backend-request",
        "fragments-request",
        "backend-join",
        "fragments-close",
        "pool:second:False",
        "pool:first:False",
        "connection",
    ]


def test_local_runner_teardown_forces_actors_and_keeps_other_dependencies_when_backend_does_not_stop():
    from vane.runners.local.runner import _shutdown_local_write_resources

    events = []

    class FakeBackend:
        def request_shutdown(self):
            events.append("backend-request")

        def shutdown(self, *, timeout_s):
            events.append("backend-join")
            raise RuntimeError("backend join timed out")

    class FakeFragmentExecutor:
        def request_shutdown(self):
            events.append("fragments-request")

        def close(self):
            raise AssertionError("fragment resources must remain alive")

    class UnexpectedConn:
        def close(self):
            raise AssertionError("driver connection must remain alive")

    class FakePool:
        def shutdown(self, *, kill):
            events.append(f"pool:{kill}")

    errors = _shutdown_local_write_resources(
        FakeBackend(),
        FakeFragmentExecutor(),
        UnexpectedConn(),
        [FakePool()],
        timeout_s=7.0,
    )

    assert events == ["backend-request", "fragments-request", "backend-join", "pool:True"]
    assert len(errors) == 1
    assert "backend join timed out" in str(errors[0])


def test_local_runner_teardown_forces_actors_and_keeps_connection_when_fragment_close_fails():
    from vane.runners.local.runner import _shutdown_local_write_resources

    events = []

    class FakeBackend:
        def request_shutdown(self):
            events.append("backend-request")

        def shutdown(self, *, timeout_s):
            events.append("backend-join")

    class FakeFragmentExecutor:
        def request_shutdown(self):
            events.append("fragments-request")

        def close(self, *, timeout_s):
            events.append("fragments-close")
            raise RuntimeError("fragment close failed")

    class UnexpectedConn:
        def close(self):
            raise AssertionError("driver connection must remain alive")

    class FakePool:
        def shutdown(self, *, kill):
            events.append(f"pool:{kill}")

    errors = _shutdown_local_write_resources(
        FakeBackend(),
        FakeFragmentExecutor(),
        UnexpectedConn(),
        [FakePool()],
        timeout_s=7.0,
    )

    assert len(errors) == 1
    assert "fragment close failed" in str(errors[0])
    assert events == [
        "backend-request",
        "fragments-request",
        "backend-join",
        "fragments-close",
        "pool:True",
    ]
