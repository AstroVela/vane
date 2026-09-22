# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import uuid

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


def _plan(connection, shape):
    # Keep the complete nested Arrow batch within the 140,000-byte input cap.
    # Even one retained producer row prevents its next protected envelope.
    blob_size = 32_768 if shape == "sort_nested" else 65_536

    def expand(table):
        ids = table.column(0).to_pylist()
        blobs = [bytes([65 + x]) * (blob_size + x) for x in ids]
        if shape == "unnest_many":
            # The input-backed list crosses STANDARD_VECTOR_SIZE, with a NULL
            # at the boundary. Its serialized output still fits 70,000 bytes.
            lists = [
                [None if i == 2048 else bytes([65 + x]) + i.to_bytes(4, "big") + b"z" * (19 + x) for i in range(2051)]
                for x in ids
            ]
            return pa.table({"blobs": pa.array(lists, type=pa.list_(pa.binary()))})
        if shape == "unnest_empty":
            return pa.table(
                {"blobs": pa.array([None if x % 2 else [] for x in ids], type=pa.list_(pa.binary())), "padding": blobs}
            )
        if shape.startswith("unnest"):
            return pa.table({"blobs": [[blob] for blob in blobs]})
        if shape == "sort_nested":
            return pa.table({"packet": [{"blob": blob, "ids": [x, None, x + 1]} for x, blob in zip(ids, blobs)]})
        return pa.table({"blob": blobs})

    def consume(table):
        values = []
        for value in table.column(0).to_pylist():
            if shape == "unnest_many":
                if value is None:
                    values.append(-1)
                    continue
                x = value[0] - 65
                i = int.from_bytes(value[1:5], "big")
                assert value == bytes([65 + x]) + i.to_bytes(4, "big") + b"z" * (19 + x)
                values.append(x * 10_000 + i)
                continue
            if isinstance(value, dict):
                x = len(value["blob"]) - blob_size
                assert value["ids"] == [x, None, x + 1]
                value = value["blob"]
            x = len(value) - blob_size
            assert value == bytes([65 + x]) * (blob_size + x)
            values.append(x)
        return pa.table({"value": pa.array(values, type=pa.int64())})

    if shape.startswith("unnest"):
        schema = {"blobs": vane.list_type(vane.sqltypes.BLOB)}
        if shape == "unnest_empty":
            schema["padding"] = vane.sqltypes.BLOB
    elif shape == "sort_nested":
        schema = {"packet": vane.struct_type({"blob": vane.sqltypes.BLOB, "ids": vane.list_type(vane.sqltypes.BIGINT)})}
    else:
        schema = {"blob": vane.sqltypes.BLOB}
    relation = connection.sql("SELECT unnest([3, 0, 2, 1])::BIGINT AS x").map_batches(
        expand,
        schema=schema,
        execution_backend="subprocess_task",
        batch_size=1,
        min_task_batch_size=1,
        task_input_max_bytes=8,
    )
    if shape == "unnest_expression":
        relation = relation.project("unnest(list_reverse(blobs)) AS blob")
    elif shape.startswith("unnest"):
        relation = relation.project("unnest(blobs) AS blob")
    elif shape == "sort_key":
        # A direct key has no payload; the sort-key expression can retain input.
        relation = relation.order("blob DESC")
    elif shape == "sort_nested":
        relation = relation.order("octet_length(packet.blob) DESC")
    else:
        # A computed key plus a borrowed payload exercises both temporary owners.
        relation = relation.order("octet_length(blob) DESC")
    minimum = 2048 if shape == "unnest_many" else 1
    relation = relation.map_batches(
        consume,
        schema={"value": vane.sqltypes.BIGINT},
        execution_backend="subprocess_task",
        batch_size=minimum,
        min_task_batch_size=minimum,
        task_input_max_bytes=140_000,
    )
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)


def _execute(connection, manager, *, shape, limited, waiting):
    plan = _plan(connection, shape)
    with LocalModelRuntime(
        session_id=plan.session_id(),
        session_config=plan.session_config(),
        request_limit=RequestAdmissionLimits(1, 1),
        task_limit=TaskAdmissionLimits(1, 8) if limited else None,
        data_limit=DataAdmissionLimits(
            420_000, 140_000, 70_000, wait=DataAdmissionWaitLimits(8, 15) if waiting else None
        ),
    ) as runtime:
        result = runtime.request().execute(plan, {}, conn=connection)
        values = [value for table in result.partition_payloads for value in table.column(0).to_pylist()]
        if shape == "unnest_many":
            assert sorted(values) == sorted(-1 if i == 2048 else x * 10_000 + i for x in range(4) for i in range(2051))
        elif shape == "unnest_empty":
            assert values == []
        elif shape.startswith("unnest"):
            assert sorted(values) == [0, 1, 2, 3]
        else:
            assert values == [3, 2, 1, 0]
        del result
        # Keeping the plan alive must not keep consumed native input charged.
        assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0
        assert manager.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("waiting", [False, True])
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize(
    "shape", ["unnest", "unnest_expression", "unnest_many", "unnest_empty", "sort_payload", "sort_key", "sort_nested"]
)
def test_consumed_native_inputs_release_byte_capacity(native_environment, shape, limited, waiting):
    with vane.connect() as connection:
        _execute(connection, native_environment, shape=shape, limited=limited, waiting=waiting)


@pytest.mark.parametrize("limited", [False, True])
def test_sort_cleanup_preserves_external_runs(native_environment, limited):
    with vane.connect() as connection:
        connection.execute("PRAGMA debug_force_external=true")
        _execute(connection, native_environment, shape="sort_nested", limited=limited, waiting=True)
