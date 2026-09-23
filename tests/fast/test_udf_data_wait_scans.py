# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid

import pytest

import vane
from vane.execution.local_resource_graph import LocalResourceGraphAdapter
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits
from vane.execution.udf_local_model import LocalModelRuntime


def _plan(connection, tmp_path, source, *, udf):
    connection.execute("CREATE TABLE data AS SELECT range AS x FROM range(4)")
    if source == "parquet":
        filename = str(tmp_path / "input.parquet")
        connection.execute("COPY data TO ? (FORMAT PARQUET)", [filename])
        relation = connection.read_parquet(filename)
    elif source == "table":
        relation = connection.table("data")
    else:
        relation = connection.sql("SELECT range AS x FROM range(4)")
    if udf:
        relation = relation.map_batches(
            lambda batch: batch,
            schema={"x": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=1,
            min_task_batch_size=1,
        )
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(connection)


@pytest.mark.parametrize("waiting", [False, True])
@pytest.mark.parametrize("source", ["range", "table", "parquet"])
@pytest.mark.parametrize("udf", [False, True])
def test_native_scans_execute_with_byte_wait(monkeypatch, tmp_path, source, waiting, udf):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        plan = _plan(connection, tmp_path, source, udf=udf)
        with LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
            data_limit=DataAdmissionLimits(
                420_000, 140_000, 70_000, wait=DataAdmissionWaitLimits(8, 15) if waiting else None
            ),
        ) as runtime:
            result = runtime.request().execute(plan, {}, conn=connection)
            assert [value for table in result.partition_payloads for value in table.column(0).to_pylist()] == [
                0,
                1,
                2,
                3,
            ]


@pytest.mark.parametrize(
    "source,distributed", [("range", False), ("table", False), ("parquet", False), ("range", True), ("parquet", True)]
)
@pytest.mark.parametrize("failure", [False, True])
def test_local_graph_collection_preserves_scan_state(monkeypatch, tmp_path, source, distributed, failure):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        plan = _plan(connection, tmp_path, source, udf=True)
        original = plan.__getstate__()[1]
        if distributed:
            plan.collect_query_resource_graph_metadata(conn=connection)
            assert plan.__getstate__()[1] != original
        before = plan.__getstate__()[1]
        adapter = LocalResourceGraphAdapter(plan)
        for _ in range(2):
            if failure:
                # Conversion fails after DAG construction has annotated the scans.
                with pytest.raises(vane.InternalException, match="Connection object"):
                    adapter.collect_resource_graph_metadata(conn=object())
            else:
                metadata = adapter.collect_resource_graph_metadata(conn=connection)
                assert metadata["udf_node_ids"]
            assert plan.__getstate__()[1] == before


def test_ray_graph_collection_still_requires_distributable_scans(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        plan = _plan(connection, tmp_path, "table", udf=False)
        with pytest.raises(vane.InvalidInputException, match="does not provide a distributable file list"):
            plan.collect_query_resource_graph_metadata(conn=connection)
