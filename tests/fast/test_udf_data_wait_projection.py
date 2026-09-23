# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import vane
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits


@pytest.fixture
def native_environment(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    gc.collect()
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 420_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    with monkeypatch.context() as cpus:
        cpus.setattr(udf_subprocess.os, "cpu_count", lambda: 1)
        task_runtime = udf_subprocess._GlobalSubprocessTaskRuntime()
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", task_runtime)
    yield manager
    task_runtime.close(kill=True)
    gc.collect()
    assert manager.snapshot()["usage_bytes"] == 0


def _plan(connection, *, projection, minimum=2, actor=False):
    def expand(table):
        return pa.table({"blob": [bytes([65 + x]) * (65_536 + x) for x in table.column(0).to_pylist()]})

    class Expand:
        def __call__(self, table):
            return expand(table)

    def consume(table):
        sizes = []
        for value in table.column(0).to_pylist():
            if isinstance(value, dict):
                size = value["sizes"][0]
                assert value["text"] == chr(65 + size - 65_536) * size
            elif isinstance(value, bytes):
                size = len(value)
                assert value == bytes([65 + size - 65_536]) * size
            else:
                size = value
            sizes.append(size)
        return pa.table({"size": sizes})

    relation = connection.sql("SELECT unnest([0, 1, 2, 3])::BIGINT AS x").map_batches(
        Expand if actor else expand,
        schema={"blob": vane.sqltypes.BLOB},
        execution_backend="subprocess_actor" if actor else "subprocess_task",
        actor_number=1 if actor else None,
        batch_size=1,
        min_task_batch_size=1,
        task_input_max_bytes=8,
    )
    if projection == "cast":
        relation = relation.project("CAST(blob AS VARCHAR) AS text").project("CAST(text AS BLOB) AS blob")
    elif projection == "nested":
        relation = relation.project(
            "struct_pack(text := CAST(blob AS VARCHAR), sizes := [octet_length(blob)::BIGINT]) AS packet"
        )
    elif projection == "unnest":
        relation = relation.project("blob, unnest(range(2049)) AS copy").project(
            "octet_length(blob)::BIGINT + copy AS size"
        )
    else:
        if projection == "filter":
            # Rejected inputs must release the filter's expression references too.
            relation = relation.filter("octet_length(blob) % 2 = 0")
        relation = relation.project("octet_length(blob)::BIGINT AS size")
    relation = relation.map_batches(
        consume,
        schema={"size": vane.sqltypes.BIGINT},
        execution_backend="subprocess_task",
        batch_size=minimum,
        min_task_batch_size=minimum,
        task_input_max_bytes=140_000,
    )
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)


def _runtime(plan, *, limited, wait=True):
    return LocalModelRuntime(
        session_id=plan.session_id(),
        session_config=plan.session_config(),
        request_limit=RequestAdmissionLimits(1, 1),
        task_limit=TaskAdmissionLimits(1, 8) if limited else None,
        data_limit=DataAdmissionLimits(420_000, 140_000, 70_000, wait=DataAdmissionWaitLimits(8, 15) if wait else None),
    )


def _sizes(result):
    return sorted(value for table in result.partition_payloads for value in table.column(0).to_pylist())


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("minimum", [1, 2])
@pytest.mark.parametrize("projection", ["length", "cast", "nested", "filter"])
def test_consumed_native_projection_releases_upstream_bytes(native_environment, limited, minimum, projection):
    with vane.connect() as connection:
        plan = _plan(connection, projection=projection, minimum=minimum)
        with _runtime(plan, limited=limited) as runtime:
            result = runtime.request().execute(plan, {}, conn=connection)
            assert _sizes(result) == ([65_536, 65_538] if projection == "filter" else list(range(65_536, 65_540)))
            del result
            assert runtime.resource_snapshot()["data"]["queries"] == 0
        # The plan may remain alive after execution; no consumed native view may remain charged.
        assert native_environment.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("actor,wait", [(True, True), (False, False)])
@pytest.mark.parametrize("projection", ["length", "cast"])
def test_projection_waits_support_actors_and_preserve_fail_fast_execution(native_environment, actor, wait, projection):
    with vane.connect() as connection:
        plan = _plan(connection, projection=projection, actor=actor)
        with _runtime(plan, limited=True, wait=wait) as runtime:
            result = runtime.request().execute(plan, {}, conn=connection)
            assert _sizes(result) == list(range(65_536, 65_540))
            del result


@pytest.mark.parametrize("limited", [False, True])
def test_projection_input_survives_an_operator_with_more_output(native_environment, limited):
    with vane.connect() as connection:
        # Each input expands past STANDARD_VECTOR_SIZE, requiring a continuation
        # with the same input while the downstream UDF can block the pipeline.
        plan = _plan(connection, projection="unnest", minimum=2048)
        with _runtime(plan, limited=limited) as runtime:
            result = runtime.request().execute(plan, {}, conn=connection)
            assert _sizes(result) == sorted(65_536 + x + copy for x in range(4) for copy in range(2049))
            del result


@pytest.mark.parametrize("limited", [False, True])
def test_projection_cleanup_keeps_external_views_charged(native_environment, monkeypatch, limited):
    views = []
    completions = []
    take_result = udf_subprocess.UDFExecutor.take_ready_result

    def retain_first_output(executor):
        result = take_result(executor)
        if result is not None and not isinstance(result[2], BaseException):
            if not completions:
                views.extend(ref.to_table().column(0) for ref in result[2][1])
            completions.append(result[0])
        return result

    monkeypatch.setattr(udf_subprocess.UDFExecutor, "take_ready_result", retain_first_output)
    with vane.connect() as connection:
        plan = _plan(connection, projection="length", minimum=1)
        with _runtime(plan, limited=limited) as runtime:
            request = runtime.request()
            with ThreadPoolExecutor(max_workers=1) as threads:
                future = threads.submit(request.execute, plan, {}, conn=connection)
                try:
                    deadline = time.monotonic() + 10
                    while True:
                        snapshot = runtime.resource_snapshot()["data"]
                        if len(completions) >= 2 and snapshot["queued_byte_admissions"] and not snapshot["tasks"]:
                            break
                        if future.done():
                            future.result()
                            pytest.fail("producer advanced while its output view was retained")
                        assert time.monotonic() < deadline
                        time.sleep(0.01)
                    assert 65_536 < snapshot["retained_bytes"] < 70_000
                    assert not future.done()
                    assert views[0][0].as_py() == b"A" * 65_536
                    views.clear()
                    gc.collect()
                    result = future.result(timeout=20)
                    assert _sizes(result) == list(range(65_536, 65_540))
                    del result
                finally:
                    views.clear()
                    request.shutdown(kill=True)
            assert native_environment.snapshot()["usage_bytes"] == 0
