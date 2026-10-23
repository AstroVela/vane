# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ownership, alignment and failure contracts for composed streaming UDFs."""

from collections import deque
from types import SimpleNamespace

import pyarrow as pa
import pytest

from vane.execution import udf_zip
from vane.execution.ref_bundle import REF_BUNDLE_RESULT_MARKER


def branch(batch):
    return batch


@pytest.fixture
def branches(monkeypatch):
    instances = []

    class Ref:
        def __init__(self, key):
            self.key = key
            self.released = False

        def release(self):
            assert not self.released
            self.released = True

    class Executor:
        def __init__(self, payload):
            self.values = deque(sequences[len(instances)])
            self.refs = []
            self.permits = 0
            self.cancelled = False
            self.closed = None
            instances.append(self)

        def register_wakeup(self, wakeup):
            self.wakeup = wakeup

        def request_task_admission(self, size):
            assert size > 0

        def task_admission_state(self):
            return {"available": True}

        def submit(self, batch):
            self.wakeup()

        def finished_submitting(self):
            pass

        def _take_ready_result_with_release(self):
            assert self.permits == 0
            if not self.values or self.values[0] == "wait":
                return None
            key = self.values.popleft()
            if isinstance(key, BaseException):
                return key, lambda: None
            ref = Ref(key)
            self.refs.append(ref)
            self.permits += 1

            def consumed():
                assert self.permits == 1
                self.permits -= 1

            return (REF_BUNDLE_RESULT_MARKER, [ref], None, None), consumed

        def all_tasks_finished(self):
            return not self.values

        def _cancel_execution_scopes(self, reason):
            self.cancelled = True

        def close(self, *, kill):
            self.closed = kill

    def materialize(refs, **kwargs):
        return pa.table({"key": [refs[0].key], "value": ["value"]})

    sequences = []
    monkeypatch.setattr("vane.execution.udf_subprocess.UDFExecutor", Executor)
    monkeypatch.setattr("vane.execution.ref_bundle.materialize_ref_bundle", materialize)

    def start(left, right, *, merge=lambda left, right: left, budget=4096, timeout=1):
        sequences[:] = [left, right]
        return udf_zip._zip_outputs(
            pa.table({"input": [1]}),
            (branch, branch),
            ([], []),
            ("key",),
            merge,
            (1, 1),
            (1024**3, 1024**3),
            budget,
            timeout,
        )

    start.instances = instances
    return start


def test_zip_holds_both_output_permits_until_consumer_advances(branches):
    stream = branches([0, 1], [0, 1])
    assert next(stream)["key"].to_pylist() == [0]
    assert [item.permits for item in branches.instances] == [1, 1]
    assert all(not ref.released for item in branches.instances for ref in item.refs)
    assert next(stream)["key"].to_pylist() == [1]
    assert all(item.refs[0].released for item in branches.instances)
    with pytest.raises(StopIteration):
        next(stream)
    assert all(item.closed is False and not item.cancelled for item in branches.instances)
    assert all(ref.released for item in branches.instances for ref in item.refs)


@pytest.mark.parametrize("left,right", [([0], [1]), ([0], []), ([], [0]), ([0, 0], [0, 0]), ([0, 2], [0])])
def test_zip_rejects_missing_extra_mismatched_and_duplicate_keys(branches, left, right):
    with pytest.raises(ValueError, match="branches"):
        list(branches(left, right))
    assert all(item.closed and item.cancelled for item in branches.instances)
    assert all(ref.released for item in branches.instances for ref in item.refs)
    assert all(item.permits == 0 for item in branches.instances)


def test_zip_resource_failure_remains_fatal_and_cancels_sibling(branches):
    error = MemoryError("branch allocation failed")
    with pytest.raises(MemoryError) as raised:
        list(branches([0], [error]))
    assert raised.value is error
    assert all(item.cancelled and item.closed for item in branches.instances)
    assert all(ref.released for item in branches.instances for ref in item.refs)


def test_zip_consumer_close_returns_permits_and_cancels_both(branches):
    stream = branches([0, 1], [0, 1])
    next(stream)
    stream.close()
    assert all(item.cancelled and item.closed and item.permits == 0 for item in branches.instances)
    assert all(ref.released for item in branches.instances for ref in item.refs)


def test_zip_waiting_sibling_cannot_cause_read_ahead(branches):
    with pytest.raises(TimeoutError, match="aligned outputs"):
        list(branches([0, 1, 2], ["wait"], timeout=0.02))
    assert list(branches.instances[0].values) == [1, 2]
    assert all(item.cancelled and item.closed for item in branches.instances)


def test_zip_checks_bytes_before_merge(branches):
    def forbidden(*args):
        pytest.fail("oversized inputs must not reach merge")

    with pytest.raises(ValueError, match="byte budget"):
        list(branches([0], [0], merge=forbidden, budget=1))
    assert all(ref.released for item in branches.instances for ref in item.refs)


def test_zip_checks_combined_bytes_before_merge(branches):
    one = pa.table({"key": [0], "value": ["value"]}).get_total_buffer_size()
    with pytest.raises(ValueError, match="combined byte budget"):
        list(branches([0], [0], budget=one))
    assert all(ref.released for item in branches.instances for ref in item.refs)


@pytest.mark.parametrize("key", [None, float("nan"), float("inf")])
def test_zip_rejects_null_and_nonfinite_keys_before_merge(branches, key):
    with pytest.raises(ValueError, match="non-null and finite"):
        list(branches([key], [key]))
    assert all(item.cancelled and item.closed for item in branches.instances)


@pytest.mark.parametrize(
    "options",
    [
        dict(cpus=(0, 1)),
        dict(memory_bytes=(1, 0)),
        dict(timeout=0),
        dict(on=[]),
        dict(execution_backend="subprocess_actor"),
        dict(max_buffer_bytes=0),
    ],
)
def test_zip_rejects_invalid_configuration_before_binding(options):
    def function(batch):
        return batch

    with pytest.raises(ValueError):
        udf_zip.zip_map_batches(
            None,
            function,
            function,
            left_schema={"key": "BIGINT"},
            right_schema={"key": "BIGINT"},
            schema={"key": "BIGINT"},
            merge=function,
            **({"on": "key"} | options),
        )


@pytest.mark.parametrize("dtype", ["FILE", "STRUCT(asset FILE)", "FILE[]"])
@pytest.mark.parametrize("position", ["input", "output"])
def test_zip_rejects_descriptor_bound_handles_before_worker_binding(dtype, position):
    relation = SimpleNamespace(types=[dtype] if position == "input" else [])
    schema = {"key": "BIGINT", "payload": dtype} if position == "output" else {"key": "BIGINT"}
    with pytest.raises(ValueError, match="descriptor-bound"):
        udf_zip.zip_map_batches(
            relation,
            branch,
            branch,
            left_schema=schema,
            right_schema={"key": "BIGINT"},
            schema={"key": "BIGINT"},
            on="key",
            merge=branch,
        )
