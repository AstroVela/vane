# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real CUDA acceptance; no external models or downloads required."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from local_gpu_helpers import (
    assert_idle,
    configure,
    execution_demand,
    gpu_pool_snapshot,
    model_definition,
    register,
    run,
    wait_for,
)
from local_gpu_helpers import query_gpu_environment as query_gpu_environment

import vane
from vane.execution.request_admission import RequestExecutionTimeout
from vane.execution.result_delivery import ResultDeliveryLimits

pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("query_gpu_environment")]


@pytest.fixture(scope="module")
def cuda_devices():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("real CUDA acceptance requires an available CUDA device")
    # An explicit override supports provisioned CI; otherwise select a visible
    # physical UUID without depending on a machine's ordinal ordering.
    device = os.environ.get("VANE_TEST_CUDA_DEVICE")
    if device is None:
        device = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    return (device,)


@pytest.mark.parametrize("batch", [False, True])
def test_cuda_model_reuse_across_sql_relations_and_managed_results(tmp_path, cuda_devices, batch):
    with vane.connect(config={"threads": 2}) as connection:
        runtime = configure(connection, cuda_devices, result_limit=ResultDeliveryLimits(1, 262_144))
        model = register(runtime, model_definition(tmp_path, batch=batch, cuda=True), cuda_devices)
        model.prewarm()
        worker = gpu_pool_snapshot(runtime)["workers"][0]
        vane.attach_function(model, connection=connection)
        rows = []
        with connection.cursor() as cursor:
            for value in (1, 2):
                rows.append(run(cursor, value))
                relation = cursor.sql("SELECT 3 AS x").project(model(vane.col("x")))
                rows.append(json.loads(relation.fetchone()[0]))
        with connection.execute_result("SELECT gpu_encode(4)") as result:
            table = result.take()
            rows.append(json.loads(table.column(0)[0].as_py()))
            del table
            assert execution_demand(runtime) == 0
            assert runtime.resource_snapshot()["reserved_resources"]["gpu"] == 1
        assert [row["value"] for row in rows] == [36, 52, 44, 52, 60]
        assert {row["device"] for row in rows} == set(cuda_devices)
        assert {row["pid"] for row in rows} == {worker["pid"]}
        assert {row["device_count"] for row in rows} == {1}
        assert len((tmp_path / "initialized").read_text().splitlines()) == 1
        assert_idle(runtime)
    assert_idle(runtime, resident=0)
    with pytest.raises(ProcessLookupError):
        os.kill(worker["pid"], 0)


@pytest.mark.parametrize("limited", [False, True])
def test_cuda_device_admission_serializes_cursors_and_drain_allows_active_work(tmp_path, cuda_devices, limited):
    with vane.connect(config={"threads": 2}) as connection:
        runtime = configure(connection, cuda_devices, limited=limited)
        model = register(runtime, model_definition(tmp_path, cuda=True), cuda_devices)
        model.prewarm()
        vane.attach_function(model, connection=connection)
        with connection.cursor() as first, connection.cursor() as second, ThreadPoolExecutor(2) as clients:
            blocked = clients.submit(run, first, -1)
            queued = None
            try:
                wait_for(lambda: (tmp_path / "entered-1").exists(), blocked)
                queued = clients.submit(run, second, 2)
                wait_for(lambda: runtime.resource_snapshot()["active_borrows"] == 2, queued)
                assert execution_demand(runtime) == 1
                assert not queued.done()
                runtime.drain()
                with pytest.raises(RuntimeError, match="drain"):
                    model.prewarm()
                assert runtime.resource_snapshot()["reserved_resources"]["gpu"] == 1
            finally:
                (tmp_path / "release-1").touch()
            first_row = blocked.result(timeout=20)
            assert queued is not None
            second_row = queued.result(timeout=20)
            assert first_row["value"] == 20 and second_row["value"] == 44
            assert first_row["pid"] == second_row["pid"]
        assert_idle(runtime)
        runtime.close(timeout=20)
    assert_idle(runtime, resident=0)


@pytest.mark.parametrize("action", ["cancel", "deadline"])
def test_cuda_cancellation_and_deadline_retire_worker_before_reuse(tmp_path, cuda_devices, action):
    with vane.connect(config={"threads": 2}) as connection:
        runtime = configure(connection, cuda_devices, execution_timeout=2 if action == "deadline" else 45)
        model = register(runtime, model_definition(tmp_path, cuda=True), cuda_devices)
        model.prewarm()
        original = gpu_pool_snapshot(runtime)["workers"][0]
        vane.attach_function(model, connection=connection)
        with connection.cursor() as cursor, ThreadPoolExecutor(1) as clients:
            future = clients.submit(run, cursor, -1)
            try:
                wait_for(lambda: (tmp_path / "entered-1").exists(), future)
                assert execution_demand(runtime) == 1
                if action == "cancel":
                    cursor.interrupt()
                with pytest.raises(
                    RequestExecutionTimeout if action == "deadline" else Exception,
                    match="timeout|cancel|interrupt|deadline",
                ):
                    future.result(timeout=25)
            finally:
                (tmp_path / "release-1").touch()
                if not future.done():
                    cursor.interrupt()
        assert_idle(runtime)
        with pytest.raises(ProcessLookupError):
            os.kill(original["pid"], 0)
        # Replacement initializes CUDA before a subsequent request's deadline.
        model.prewarm()
        replacement = gpu_pool_snapshot(runtime)["workers"][0]
        assert replacement["pid"] != original["pid"]
        assert replacement["device"] == original["device"] == cuda_devices[0]
        assert replacement["generation"] == original["generation"] + 1
        assert run(connection, 5)["value"] == 68
        assert_idle(runtime)


@pytest.mark.parametrize("failure", [-3, -4])
def test_cuda_model_failure_replaces_worker_without_replaying_query(tmp_path, cuda_devices, failure):
    with vane.connect() as connection:
        runtime = configure(connection, cuda_devices)
        model = register(runtime, model_definition(tmp_path, cuda=True), cuda_devices)
        model.prewarm()
        original = gpu_pool_snapshot(runtime)["workers"][0]
        vane.attach_function(model, connection=connection)
        with pytest.raises(Exception):
            run(connection, failure)
        assert_idle(runtime)
        result = run(connection, 6)
        assert result["value"] == 76 and result["pid"] != original["pid"]
        assert result["device"] == cuda_devices[0]
        assert len((tmp_path / "initialized").read_text().splitlines()) == 2
        field = "worker_losses" if failure == -3 else "execution_errors"
        assert runtime.resource_snapshot()["worker_failures"][field] == 1
        assert_idle(runtime)


def test_cuda_initialization_failure_returns_resident_device_for_another_model(tmp_path, cuda_devices):
    with vane.connect() as connection:
        runtime = configure(connection, cuda_devices)
        failed = register(runtime, model_definition(tmp_path, cuda=True, fail_init=True), cuda_devices)
        with pytest.raises(Exception, match="GPU fixture initialization failure"):
            failed.prewarm()
        assert runtime.resource_snapshot()["reserved_resources"]["gpu"] == 0
        assert runtime.resource_snapshot()["exclusive_resources"] == {}
        model = register(runtime, model_definition(tmp_path, cuda=True), cuda_devices, name="working")
        vane.attach_function(model, connection=connection)
        assert run(connection, 7)["value"] == 84
        assert_idle(runtime)


def test_cuda_output_still_obeys_runtime_byte_limit(tmp_path, cuda_devices):
    with vane.connect() as connection:
        runtime = configure(connection, cuda_devices)
        model = register(runtime, model_definition(tmp_path, cuda=True), cuda_devices)
        vane.attach_function(model, connection=connection)
        with pytest.raises(Exception, match="output|bytes|capacity"):
            run(connection, -5)
        assert_idle(runtime)


def test_cuda_output_can_feed_cpu_udf_under_one_task_and_byte_budget(tmp_path, cuda_devices):
    @vane.func(return_dtype="BIGINT")
    def host_value(value):
        return json.loads(value)["value"]

    with vane.connect(config={"threads": 2}) as connection:
        runtime = configure(connection, cuda_devices)
        model = register(runtime, model_definition(tmp_path, batch=True, cuda=True), cuda_devices)
        vane.attach_function(model, connection=connection)
        vane.attach_function(host_value, "host_value", parameters=["VARCHAR"], connection=connection)
        rows = connection.execute("SELECT host_value(gpu_encode(i)) FROM range(4) r(i) ORDER BY i").fetchall()
        assert rows == [(28,), (36,), (44,), (52,)]
        assert_idle(runtime)
