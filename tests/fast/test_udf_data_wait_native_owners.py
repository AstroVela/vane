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
            if isinstance(value, int):
                values.append(value)
                continue
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
    elif shape in {"topn", "topn_rejected"}:
        relation = relation.order("octet_length(blob) DESC").limit(1 if shape == "topn_rejected" else 3)
    elif shape in {"ungrouped_min", "filtered_min"}:
        aggregate = "min(blob)" if shape == "ungrouped_min" else "min(blob) FILTER (WHERE octet_length(blob) % 2 = 0)"
        relation = relation.aggregate(f"{aggregate} AS blob")
    elif shape in {"hash_min", "filtered_hash_min"}:
        aggregate = "min(blob)" if shape == "hash_min" else "min(blob) FILTER (WHERE octet_length(blob) != 65539)"
        relation = relation.aggregate(
            f"{aggregate} AS blob, octet_length(blob) % 2 AS k", "octet_length(blob) % 2"
        ).project("blob")
    elif shape == "grouping_sets":
        relation = relation.query(
            "producer", "SELECT min(blob) AS blob FROM producer GROUP BY GROUPING SETS ((), (octet_length(blob) % 2))"
        )
    elif shape == "perfect_hash_min":
        relation = relation.aggregate(
            "min(blob) FILTER (WHERE octet_length(blob) != 65539) AS blob, (octet_length(blob) % 2)::UTINYINT AS k",
            "(octet_length(blob) % 2)::UTINYINT",
        ).project("blob")
        assert "PERFECT_HASH_GROUP_BY" in relation.explain()
    elif shape == "count_distinct":
        relation = relation.aggregate("count(DISTINCT blob) FILTER (WHERE octet_length(blob) % 2 = 0) AS blob")
    elif shape == "distinct":
        relation = relation.distinct()
    elif shape in {"window", "partitioned_window"}:
        partition = "PARTITION BY octet_length(blob) % 2 " if shape == "partitioned_window" else ""
        relation = (
            relation.project(f"blob, row_number() OVER ({partition}ORDER BY blob) AS rn")
            .filter("rn <= 3")
            .project("blob")
        )
    elif shape in {"lag", "lead"}:
        relation = relation.project(f"{shape}(blob) OVER () AS blob").filter("blob IS NOT NULL")
    elif shape == "running_min":
        relation = relation.project("min(blob) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS blob")
    elif shape == "join_build":
        probe = connection.sql("SELECT unnest(range(65536, 65636))::BIGINT AS k").set_alias("p")
        relation = probe.join(relation.set_alias("b"), "k = octet_length(blob)").project("blob")
    elif shape == "join_probe":
        build = connection.sql("SELECT unnest(range(65536, 65540))::BIGINT AS k").set_alias("b")
        relation = relation.set_alias("p").join(build, "octet_length(blob) = k").project("blob")
    elif shape.startswith(("nested_join", "merge_join", "ie_join")):
        connection.execute("SET disabled_optimizers='join_order,build_side_probe_side'")
        if shape.startswith(("merge_join", "ie_join")):
            connection.execute("SET nested_loop_join_threshold=0")
            connection.execute("SET merge_join_threshold=0")
        if shape.startswith("ie_join"):
            connection.execute("SET prefer_range_joins=true")
            other_sql = "SELECT * FROM (VALUES ('0'::BLOB, 'Z'::BLOB), ('X'::BLOB, 'Y'::BLOB)) t(lo, hi)"
            condition = "lo < blob AND hi > blob"
            expected_operator = "IE_JOIN"
        else:
            other_sql = "SELECT unnest(['0'::BLOB, 'Z'::BLOB]) AS key"
            condition = "key < blob"
            expected_operator = "PIECEWISE_MERGE_JOIN" if shape.startswith("merge_join") else "NESTED_LOOP_JOIN"
        other = connection.sql(other_sql).set_alias("other")
        relation = relation.set_alias("producer")
        if shape.endswith("build"):
            relation = other.join(relation, condition).project("blob")
        else:
            relation = relation.join(other, condition).project("blob")
        assert expected_operator in relation.explain()
    elif shape.startswith(("blockwise", "asof")) or shape in {"left_delim", "right_delim"}:
        connection.execute("SET disabled_optimizers='deliminator,join_order,build_side_probe_side'")
        relation = relation.set_alias("producer")
        if shape.startswith("blockwise"):
            other = connection.sql("SELECT unnest([1, 2, 3, 4])::BIGINT AS key").set_alias("other")
            relation = relation.join(
                other, "octet_length(blob) + key = 65540", how="semi" if shape.endswith("semi") else "inner"
            ).project("blob")
            expected_operator = "BLOCKWISE_NL_JOIN"
        elif shape.startswith("asof"):
            connection.execute("SET asof_loop_join_threshold=0")
            other_sql = "(SELECT unnest([65536, 65537, 65538, 65539])::BIGINT AS k) other"
            if shape.endswith("build"):
                query = f"SELECT blob FROM {other_sql} ASOF JOIN producer ON k >= octet_length(blob)"
            else:
                query = f"SELECT blob FROM producer ASOF JOIN {other_sql} ON octet_length(blob) >= k"
            relation = relation.query("producer", query)
            expected_operator = "ASOF_JOIN"
        else:
            if shape == "right_delim":
                connection.execute("SET disabled_optimizers='deliminator'")
            relation = relation.query(
                "producer",
                "SELECT blob FROM producer WHERE EXISTS "
                "(SELECT 1 FROM (SELECT range + 65536 AS k FROM range(1000)) other "
                "WHERE k = octet_length(producer.blob))",
            )
            expected_operator = "RIGHT_DELIM_JOIN" if shape == "right_delim" else "LEFT_DELIM_JOIN"
        assert expected_operator in relation.explain()
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
        elif shape.startswith("sort"):
            assert values == [3, 2, 1, 0]
        else:
            expected = {
                "topn": [1, 2, 3],
                "topn_rejected": [3],
                "ungrouped_min": [0],
                "filtered_min": [0],
                "hash_min": [0, 1],
                "filtered_hash_min": [0, 1],
                "grouping_sets": [0, 0, 1],
                "perfect_hash_min": [0, 1],
                "count_distinct": [2],
                "distinct": [0, 1, 2, 3],
                "window": [0, 1, 2],
                "partitioned_window": [0, 1, 2, 3],
                "lag": [0, 2, 3],
                "lead": [0, 1, 2],
                "running_min": [0, 0, 0, 3],
                "join_build": [0, 1, 2, 3],
                "join_probe": [0, 1, 2, 3],
                "nested_join_build": [0, 1, 2, 3],
                "nested_join_probe": [0, 1, 2, 3],
                "merge_join_build": [0, 1, 2, 3],
                "merge_join_probe": [0, 1, 2, 3],
                "ie_join_build": [0, 1, 2, 3],
                "ie_join_probe": [0, 1, 2, 3],
                "blockwise_inner": [0, 1, 2, 3],
                "blockwise_semi": [0, 1, 2, 3],
                "asof_build": [0, 1, 2, 3],
                "asof_probe": [0, 1, 2, 3],
                "left_delim": [0, 1, 2, 3],
                "right_delim": [0, 1, 2, 3],
            }
            assert sorted(values) == expected[shape]
        del result
        # Keeping the plan alive must not keep consumed native input charged.
        assert runtime.resource_snapshot()["data"]["usage_bytes"] == 0
        assert manager.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("waiting", [False, True])
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        "unnest",
        "unnest_expression",
        "unnest_many",
        "unnest_empty",
        "sort_payload",
        "sort_key",
        "sort_nested",
        "topn",
        "topn_rejected",
        "ungrouped_min",
        "filtered_min",
        "hash_min",
        "filtered_hash_min",
        "grouping_sets",
        "perfect_hash_min",
        "count_distinct",
        "distinct",
        "window",
        "partitioned_window",
        "lag",
        "lead",
        "running_min",
        "join_build",
        "join_probe",
        "nested_join_build",
        "nested_join_probe",
        "merge_join_build",
        "merge_join_probe",
        "ie_join_build",
        "ie_join_probe",
        "blockwise_inner",
        "blockwise_semi",
        "asof_build",
        "asof_probe",
        "left_delim",
        "right_delim",
    ],
)
def test_consumed_native_inputs_release_byte_capacity(native_environment, shape, limited, waiting):
    with vane.connect() as connection:
        _execute(connection, native_environment, shape=shape, limited=limited, waiting=waiting)


@pytest.mark.parametrize("limited", [False, True])
def test_sort_cleanup_preserves_external_runs(native_environment, limited):
    with vane.connect() as connection:
        connection.execute("PRAGMA debug_force_external=true")
        _execute(connection, native_environment, shape="sort_nested", limited=limited, waiting=True)
