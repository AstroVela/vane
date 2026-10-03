# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Submission preparation uses real native plans, without a task scheduler."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

import vane
from vane._native import execution_plan as native
from vane.execution.compiler import FragmentCompileOptions
from vane.execution.plan import Distribution
from vane.execution.query_options import (
    DistributedMode,
    FteOptions,
    LocalExecution,
    QueryExecutionOptions,
    RayExecution,
)
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import (
    FragmentSourceSnapshot,
    RayQuerySpec,
    check_plan_capabilities,
    native_plan_capabilities,
    prepare_ray_query,
    prepare_worker_plan,
)


@pytest.fixture
def connection(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(config={"threads": 1}) as connection:
        yield connection


def options(mode=DistributedMode.PIPELINED):
    fte = FteOptions("test-store", 3, 0.01) if mode is DistributedMode.FTE else None
    return QueryExecutionOptions(RayExecution(mode, fte), 10, 60, 30)


def resources(contexts=32):
    return ResourceDemand(4, contexts, MemoryDemand(2**26, 2**22, 2**22, 2**22), 4)


def prepare(connection, sql="select range + 1 as value from range(7)", **kwargs):
    arguments = dict(query_id="submission-test", options=options(), resources=resources())
    arguments.update(kwargs)
    return prepare_ray_query(connection, sql, **arguments)


def execute_single_fragment(connection, spec):
    """Finite materialized oracle; not the future TaskRuntime or scheduler."""
    prepare_worker_plan(connection, spec)
    assert len(spec.graph.fragments) == 1
    fragment = spec.graph.fragments[0]
    assignments = {source.source_id: [split.split_id for split in source.splits] for source in fragment.sources}
    return native._execute_fragment_for_test(connection, fragment.native_plan, {}, assignments)


@pytest.mark.parametrize("mode", list(DistributedMode))
@pytest.mark.parametrize(
    "sql", ["select 42 as answer, NULL::BIGINT as missing", "select range / 2 as half from range(7)"]
)
def test_submission_transport_restores_session_and_executes_real_sql(connection, mode, sql):
    connection.execute("SET integer_division=true")
    connection.execute("SET TimeZone='UTC'")
    expected = connection.execute(sql).fetchall()
    spec = prepare(connection, sql, options=options(mode))
    transported = RayQuerySpec.from_dict(
        json.loads(json.dumps(spec.to_dict())), expected_engine_identity=native.engine_identity()
    )
    connection.execute("SET integer_division=false")
    connection.execute("SET TimeZone='Asia/Shanghai'")
    assert transported == spec
    assert transported.result_schema == transported.graph.fragments[-1].outputs[0].schema
    with vane.connect(config={"threads": 1}) as worker:
        assert execute_single_fragment(worker, transported) == expected
        assert worker.execute("select current_setting('integer_division'), current_setting('TimeZone')").fetchone() == (
            True,
            "UTC",
        )


@pytest.mark.parametrize("mode", list(DistributedMode))
@pytest.mark.parametrize("disabled_optimizers", ["", "expression_rewriter"])
@pytest.mark.parametrize(
    "sql",
    [
        "select current_timestamp::varchar from range(3)",
        "select current_date::varchar from range(3)",
        "select current_time::varchar from range(3)",
        "select localtimestamp::varchar from range(3)",
        "select localtime::varchar from range(3)",
        "select range from range(3) where current_timestamp::varchar <> ''",
        "select case when range = 0 then current_timestamp::varchar else 'later' end from range(3)",
    ],
)
def test_submission_rejects_query_stable_expressions_before_constant_folding(
    connection, mode, disabled_optimizers, sql
):
    connection.execute(f"SET disabled_optimizers='{disabled_optimizers}'")
    assert len(connection.execute(sql).fetchall()) == 3
    with pytest.raises(vane.NotImplementedException, match="consistent across tasks and attempts"):
        prepare(connection, sql, options=options(mode), compile_options=FragmentCompileOptions(3))


def test_worker_preparation_is_session_scoped_on_shared_database(connection):
    with vane.connect(config={"threads": 1}) as planning:
        planning.execute("SET ieee_floating_point_ops=false")
        planning.execute("SET TimeZone='UTC'")
        spec = prepare(planning, "select 1::DOUBLE / (range - 1)::DOUBLE as quotient from range(3)")
        expected = planning.execute("select 1::DOUBLE / (range - 1)::DOUBLE from range(3)").fetchall()
    with connection.cursor() as worker, connection.cursor() as sibling:
        before = sibling.execute(
            "select current_setting('ieee_floating_point_ops'), current_setting('TimeZone')"
        ).fetchone()
        assert execute_single_fragment(worker, spec) == expected
        after = sibling.execute(
            "select current_setting('ieee_floating_point_ops'), current_setting('TimeZone')"
        ).fetchone()
        assert after == before
        worker.execute("SET ieee_floating_point_ops=true")
        # Each attempt must restore the submitted semantics again.
        assert execute_single_fragment(worker, spec) == expected


def test_submission_survives_planning_connection_close_and_process_boundary(connection):
    with vane.connect(config={"threads": 1}) as planning:
        planning.execute("SET integer_division=true")
        spec = prepare(planning, "select range / 2 as half from range(9)", options=options(DistributedMode.FTE))
    program = """
import json
import sys
import vane
from vane._native import execution_plan as native
from vane.execution.submission import RayQuerySpec, prepare_worker_plan
spec = RayQuerySpec.from_dict(json.load(sys.stdin), expected_engine_identity=native.engine_identity())
with vane.connect(config={"threads": 1}) as worker:
    prepare_worker_plan(worker, spec)
    fragment = spec.graph.fragments[0]
    assignments = {source.source_id: [split.split_id for split in source.splits] for source in fragment.sources}
    rows = native._execute_fragment_for_test(worker, fragment.native_plan, {}, assignments)
    print(json.dumps(rows))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        input=json.dumps(spec.to_dict()),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == [[value // 2] for value in range(9)]


@pytest.mark.parametrize("mode,contexts", [(DistributedMode.PIPELINED, 9), (DistributedMode.FTE, 4)])
def test_context_declaration_covers_active_graph_or_largest_materialized_stage(connection, mode, contexts):
    spec = prepare(
        connection,
        options=options(mode),
        resources=resources(contexts),
        compile_options=FragmentCompileOptions(4, (0,)),
    )
    assert [fragment.partition_count for fragment in spec.graph.fragments] == [4, 4, 1]
    with vane.connect(config={"threads": 1}) as worker:
        prepare_worker_plan(worker, spec)
    with pytest.raises(ValueError, match=f"at least {contexts} task contexts"):
        replace(spec, resources=resources(contexts - 1))


def test_parquet_submission_freezes_file_membership_and_validates_visible_files(connection, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"value": [1, 2, 3]}), tmp_path / "original.parquet")
    spec = prepare(connection, f"select value from read_parquet('{tmp_path}/*.parquet')")
    pq.write_table(pa.table({"value": [99]}), tmp_path / "late.parquet")
    with vane.connect(config={"threads": 1}) as worker:
        assert execute_single_fragment(worker, spec) == [(1,), (2,), (3,)]


@pytest.mark.parametrize("change", ["size", "mtime", "missing"])
def test_worker_rejects_changed_or_missing_source_before_loading_tasks(connection, tmp_path, change):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"value": [1, 2, 3]}), path)
    spec = prepare(connection, f"select value from read_parquet('{path}')")
    if change == "missing":
        path.unlink()
    elif change == "mtime":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    else:
        with path.open("ab") as output:
            output.write(b"changed")
    with vane.connect(config={"threads": 1}) as worker:
        with pytest.raises((vane.InvalidInputException, vane.IOException), match="snapshot changed|Cannot open file"):
            prepare_worker_plan(worker, spec)


def test_parquet_does_not_claim_immutable_fte_replay(connection, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"value": [1]}), path)
    sql = f"select value from read_parquet('{path}')"
    with pytest.raises(vane.NotImplementedException, match="immutable source version"):
        prepare(connection, sql, options=options(DistributedMode.FTE))
    spec = prepare(connection, sql)
    with pytest.raises(ValueError, match="immutable source versions"):
        replace(spec, options=options(DistributedMode.FTE))


def test_relative_source_paths_cannot_depend_on_worker_working_directory(connection, tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"value": [1]}), tmp_path / "data.parquet")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(vane.NotImplementedException, match="absolute local paths"):
        prepare(connection, "select value from read_parquet('data.parquet')")


def test_custom_collation_is_outside_initial_connection_profile(connection):
    connection.execute("SET default_collation='nocase'")
    with pytest.raises(vane.NotImplementedException, match="custom collations"):
        prepare(connection, "select 'a' = 'A'")


def test_global_only_settings_must_match_without_mutating_worker_database(connection):
    connection.execute("SET disabled_optimizers='filter_pushdown'")
    spec = prepare(connection)
    with vane.connect(config={"threads": 1}) as worker:
        with pytest.raises(vane.InvalidInputException, match="database setting disabled_optimizers"):
            prepare_worker_plan(worker, spec)
        assert worker.execute("select current_setting('disabled_optimizers')").fetchone() == ("",)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"engine_identity": "another-engine"}, "engine identity"),
        ({"protocol_version": 2}, "protocol"),
        ({"type_profile": "unknown-types"}, "type or connection"),
        ({"connection_profile": "unknown-session"}, "type or connection"),
        ({"distributions": (Distribution.GATHER,)}, "distribution"),
        ({"scans": ()}, "source capability or split codec"),
    ],
)
def test_incompatible_worker_capabilities_are_rejected(connection, changes, match):
    spec = prepare(connection, compile_options=FragmentCompileOptions(2, (0,)))
    capabilities = native_plan_capabilities(connection)
    check_plan_capabilities(spec, capabilities)
    with pytest.raises(ValueError, match=match):
        check_plan_capabilities(spec, replace(capabilities, **changes))


@pytest.mark.parametrize("kind", ["connection", "source"])
@pytest.mark.parametrize("damage", ["truncated", "trailing", "foreign_engine"])
def test_worker_rejects_damaged_snapshot_envelopes(connection, kind, damage):
    spec = prepare(connection)
    payload = spec.connection_snapshot if kind == "connection" else spec.source_snapshots[0].payload
    if damage == "truncated":
        payload = payload[:-1]
    elif damage == "trailing":
        payload += b"trailing"
    else:
        identity = native.engine_identity().encode()
        assert identity in payload
        payload = payload.replace(identity, b"x" * len(identity))
    if kind == "connection":
        damaged = replace(spec, connection_snapshot=payload)
    else:
        damaged = replace(spec, source_snapshots=(replace(spec.source_snapshots[0], payload=payload),))
    with vane.connect(config={"threads": 1}) as worker:
        with pytest.raises(vane.SerializationException):
            prepare_worker_plan(worker, damaged)


def test_snapshot_from_another_query_cannot_validate_different_scan_splits(connection):
    first = prepare(connection, "select range from range(4)")
    second = prepare(connection, "select range from range(9)")
    with vane.connect(config={"threads": 1}) as worker:
        with pytest.raises(vane.InvalidInputException, match="source snapshot changed"):
            prepare_worker_plan(worker, replace(second, source_snapshots=first.source_snapshots))


@pytest.mark.parametrize("damage", ["names", "schema", "hash"])
def test_native_loading_rejects_forged_submission_metadata(connection, damage):
    spec = prepare(connection, compile_options=FragmentCompileOptions(2, (0,)))
    if damage == "names":
        damaged = replace(spec, result_names=("wrong",))
    elif damage == "hash":
        edge, *rest = spec.graph.exchanges
        graph = replace(spec.graph, exchanges=(replace(edge, partitioning=edge.partitioning + b"trailing"), *rest))
        damaged = replace(spec, graph=graph)
    else:
        # Change every connected port so graph topology remains valid while
        # its declared schema no longer matches the native plans.
        fragments = tuple(
            replace(
                fragment,
                inputs=tuple(replace(port, schema=b"forged") for port in fragment.inputs),
                outputs=tuple(replace(port, schema=b"forged") for port in fragment.outputs),
            )
            for fragment in spec.graph.fragments
        )
        damaged = replace(spec, graph=replace(spec.graph, fragments=fragments))
    with vane.connect(config={"threads": 1}) as worker:
        with pytest.raises((ValueError, vane.SerializationException), match="disagree|trailing"):
            prepare_worker_plan(worker, damaged)


def test_blueprint_cache_identity_excludes_query_id_and_tracks_semantic_inputs(connection):
    first = prepare(connection, query_id="first")
    second = prepare(connection, query_id="second")
    assert first.cache_key() == second.cache_key()
    assert first.query_id != second.query_id
    variants = [
        replace(first, options=options(DistributedMode.FTE)),
        replace(first, options=replace(first.options, execution_timeout=61)),
        replace(first, resources=replace(first.resources, cpu_share=2)),
        prepare(connection, "select range + 2 as value from range(7)"),
        prepare(connection, compile_options=FragmentCompileOptions(2, (0,))),
    ]
    connection.execute("SET TimeZone='UTC'")
    utc = prepare(connection)
    connection.execute("SET TimeZone='Asia/Shanghai'")
    shanghai = prepare(connection)
    assert utc.connection_snapshot != shanghai.connection_snapshot
    assert utc.cache_key() != shanghai.cache_key()
    assert all(variant.cache_key() != first.cache_key() for variant in variants)


def test_equivalent_setting_aliases_share_snapshot_and_cache_identity(connection):
    implicit = prepare(connection)
    connection.execute("SET default_order='asc'")
    connection.execute("SET default_null_order='last'")
    explicit = prepare(connection)
    assert implicit.connection_snapshot == explicit.connection_snapshot
    assert implicit.cache_key() == explicit.cache_key()


def test_file_version_participates_in_blueprint_cache_identity(connection, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"value": [1]}), path)
    sql = f"select value from read_parquet('{path}')"
    first = prepare(connection, sql)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    second = prepare(connection, sql)
    assert first.source_snapshots != second.source_snapshots
    assert first.cache_key() != second.cache_key()


def test_submission_copies_caller_buffers_and_rejects_extra_transport_fields(connection):
    spec = prepare(connection)
    connection_bytes = bytearray(spec.connection_snapshot)
    source_bytes = bytearray(spec.source_snapshots[0].payload)
    names = list(spec.result_names)
    snapshots = [FragmentSourceSnapshot(spec.source_snapshots[0].fragment_id, source_bytes)]
    frozen = replace(spec, connection_snapshot=connection_bytes, source_snapshots=snapshots, result_names=names)
    connection_bytes.clear()
    source_bytes.clear()
    names.clear()
    snapshots.clear()
    assert frozen == spec
    with pytest.raises(FrozenInstanceError):
        frozen.result_names = ("changed",)
    for field in (None, "resources", "options"):
        payload = spec.to_dict()
        target = payload if field is None else payload[field]
        target["unknown"] = True
        with pytest.raises(ValueError, match="exactly these fields"):
            RayQuerySpec.from_dict(payload, expected_engine_identity=native.engine_identity())
    with pytest.raises(ValueError, match="exactly one source snapshot"):
        replace(spec, source_snapshots=())


def test_local_execution_never_enters_submission_protocol(connection, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("local execution entered Ray submission")

    monkeypatch.setattr(native, "compile_submission", forbidden)
    with pytest.raises(ValueError, match="RayExecution"):
        prepare(connection, options=QueryExecutionOptions(LocalExecution(), 10, 60, 30))
    assert connection.execute("select 17").fetchall() == [(17,)]


def test_preparation_does_not_invoke_legacy_runner_or_environment_configuration(connection, monkeypatch):
    from vane import _ray_cxx, runners

    def forbidden(*args, **kwargs):
        raise AssertionError("submission entered legacy execution")

    monkeypatch.setattr(runners, "get_or_create_runner", forbidden)
    monkeypatch.setattr(_ray_cxx, "validate_plan_serialization_for_submission", forbidden)
    original = prepare(connection)
    monkeypatch.setenv("VANE_RUNNER", "invalid-runner")
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "invalid-exchange")
    assert prepare(connection) == original
    prepare_worker_plan(connection, original)


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), "2"])
def test_invalid_cpu_declarations_fail(value):
    with pytest.raises(ValueError, match="cpu_share"):
        replace(resources(), cpu_share=value)


@pytest.mark.parametrize("field", ["operator_bytes", "result_bytes", "exchange_bytes", "staging_bytes"])
@pytest.mark.parametrize("value", [True, -1, 0.5, 2**63])
def test_invalid_byte_declarations_fail(field, value):
    with pytest.raises(ValueError, match=field):
        replace(resources().memory, **{field: value})


@pytest.mark.parametrize("field", ["task_contexts", "io_concurrency"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, 2**63])
def test_invalid_concurrency_declarations_fail(field, value):
    with pytest.raises(ValueError, match=field):
        replace(resources(), **{field: value})


@pytest.mark.parametrize("field", ["exchange_bytes", "staging_bytes"])
def test_ray_submission_requires_explicit_transport_memory(connection, field):
    demand = resources()
    with pytest.raises(ValueError, match="exchange and staging memory"):
        prepare(connection, resources=replace(demand, memory=replace(demand.memory, **{field: 0})))
