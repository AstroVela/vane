# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real native plans executed by a finite test harness, without a Ray scheduler."""

import json
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import replace

import pytest

import vane
from vane._native import execution_plan as native
from vane.execution.compiler import FragmentCompileOptions, compile_fragment_graph, validate_native_graph
from vane.execution.plan import Distribution, FragmentGraph


@pytest.fixture
def connection(monkeypatch):
    # The existing API still calls native local execution "local-fast". The new
    # compiler neither reads this setting nor goes through that query runner.
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(config={"threads": 1}) as connection:
        yield connection


def compile_sql(connection, sql, partitions=3, hash_columns=()):
    return compile_fragment_graph(
        connection,
        sql,
        query_id="native-compiler-test",
        options=FragmentCompileOptions(partitions, hash_columns),
    )


def execute_graph(connection, graph):
    """Materialized reference harness only; this does not test pipelined execution."""
    validate_native_graph(connection, graph)
    fragments = {fragment.fragment_id: fragment for fragment in graph.fragments}
    outputs = {}
    for fragment_id in graph.topological_fragment_ids():
        fragment = fragments[fragment_id]
        inputs = [dict() for _ in range(fragment.partition_count)]
        for edge in graph.exchanges:
            if edge.consumer_fragment_id != fragment_id:
                continue
            producer = fragments[edge.producer_fragment_id]
            rows = [row for task_rows in outputs[edge.producer_fragment_id] for row in task_rows]
            if edge.distribution is Distribution.GATHER:
                inputs[0][edge.consumer_port] = rows
            elif edge.distribution is Distribution.HASH:
                ids = native._hash_rows_for_test(
                    connection, producer.outputs[0].schema, edge.partitioning, rows, fragment.partition_count
                )
                for target in inputs:
                    target[edge.consumer_port] = []
                for row, partition in zip(rows, ids, strict=True):
                    inputs[partition][edge.consumer_port].append(row)
            else:
                raise AssertionError(f"unexpected exchange {edge.distribution}")
        outputs[fragment_id] = []
        for partition in range(fragment.partition_count):
            assignments = {
                source.source_id: [
                    split.split_id
                    for index, split in enumerate(source.splits)
                    if index % fragment.partition_count == partition
                ]
                for source in fragment.sources
            }
            outputs[fragment_id].append(
                native._execute_fragment_for_test(connection, fragment.native_plan, inputs[partition], assignments)
            )
    return outputs[graph.result.fragment_id][0]


@pytest.mark.parametrize("target", ["same", "sibling"])
@pytest.mark.parametrize(
    "entry",
    [
        "compile_fragment_graph",
        "prepare_ray_query",
        "native_plan_capabilities",
        "prepare_worker_plan",
        "inspect_submitted_fragment",
        "inspect_fragment",
        "validate_hash",
        "execute_fragment",
        "hash_rows",
    ],
)
def test_fragment_connection_entries_reject_python_input_callback_reentry(entry, target):
    pytest.importorskip("fsspec")
    script = textwrap.dedent(
        """
        import faulthandler
        import io
        import os
        import sys
        from datetime import datetime, timezone
        import fsspec
        import pyarrow as pa
        import pyarrow.parquet as pq
        import vane
        from vane._native import execution_plan as native
        from vane.execution.compiler import FragmentCompileOptions, compile_fragment_graph
        from vane.execution.plan import Distribution
        from vane.execution.query_options import QueryExecutionOptions, RayExecution
        from vane.execution.resource_demand import MemoryDemand, ResourceDemand
        from vane.execution.submission import native_plan_capabilities, prepare_ray_query, prepare_worker_plan

        os.environ["VANE_RUNNER"] = "local-fast"
        entry, target = sys.argv[1:]
        output = pa.BufferOutputStream()
        pq.write_table(pa.table({"x": [1, 2, 3]}), output)
        payload = output.getvalue().to_pybytes()
        armed, attempts = False, []

        def prepare(connection):
            return prepare_ray_query(
                connection, "SELECT 7", query_id="callback-entry",
                options=QueryExecutionOptions(RayExecution(), 10, 60, 30),
                resources=ResourceDemand(1, 2, MemoryDemand(4096, 4096, 4096, 4096), 1),
            )

        class Filesystem(fsspec.AbstractFileSystem):
            protocol = "fragmentcallback"

            def _open(self, path, mode="rb", **kwargs):
                if armed and not attempts:
                    attempts.append(entry)
                    try:
                        invoke()
                    except vane.InvalidInputException as error:
                        assert "Python input callback" in str(error), str(error)
                    else:
                        raise AssertionError("input callback entered " + entry)
                return io.BytesIO(payload)

            def info(self, path, **kwargs):
                return {"name": path, "size": len(payload), "type": "file"}

            def modified(self, path):
                return datetime(2026, 1, 1, tzinfo=timezone.utc)

        with vane.connect(config={"threads": 1}) as connection, connection.cursor() as sibling:
            selected = connection if target == "same" else sibling
            spec = prepare(connection)
            fragment = spec.graph.fragments[0]
            hashed = compile_fragment_graph(
                connection, "SELECT range FROM range(3)", query_id="callback-hash",
                options=FragmentCompileOptions(2, (0,)),
            )
            edge = next(edge for edge in hashed.exchanges if edge.distribution is Distribution.HASH)
            schema = hashed.fragments[0].outputs[0].schema
            operations = {
                "compile_fragment_graph": lambda: compile_fragment_graph(selected, "SELECT 7", query_id="nested"),
                "prepare_ray_query": lambda: prepare(selected),
                "native_plan_capabilities": lambda: native_plan_capabilities(selected),
                "prepare_worker_plan": lambda: prepare_worker_plan(selected, spec),
                "inspect_submitted_fragment": lambda: native.inspect_submitted_fragment(
                    selected, fragment.native_plan, spec.connection_snapshot,
                    spec.source_snapshots[0].payload, spec.requires_replay,
                ),
                "inspect_fragment": lambda: native.inspect_fragment(selected, fragment.native_plan),
                "validate_hash": lambda: native.validate_hash(selected, schema, edge.partitioning),
                "execute_fragment": lambda: native._execute_fragment_for_test(selected, fragment.native_plan, {}, {}),
                "hash_rows": lambda: native._hash_rows_for_test(selected, schema, edge.partitioning, [(1,), (2,)], 2),
            }
            invoke = operations[entry]
            connection.register_filesystem(Filesystem(skip_instance_cache=True))
            armed = True
            # Isolate a regression so a lock wait cannot hang the pytest process.
            faulthandler.dump_traceback_later(8, exit=True)
            assert connection.execute(
                "SELECT x FROM read_parquet('fragmentcallback://data.parquet')"
            ).fetchall() == [(1,), (2,), (3,)]
            assert attempts == [entry]
            # The guard must leave both the outer query and later planning usable.
            invoke()
            assert connection.execute("SELECT 42").fetchall() == [(42,)]
            assert sibling.execute("SELECT 43").fetchall() == [(43,)]
            faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, entry, target],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "sql",
    [
        "select 42 as answer, NULL::BIGINT as missing, 'hello' as label, true as flag, 1.25::DOUBLE as score",
        "select 42 where false",
        "select * from range(0)",
        "select range + 2 as value from range(19) where range % 2 = 0",
        "select range from range(17, -8, -3) where range <> 2",
        "select x, lower(label) from (values (1, 'A'), (2, 'B'), (NULL, NULL)) t(x, label)",
    ],
)
@pytest.mark.parametrize("partitions", [1, 4])
def test_real_sql_survives_native_transport_and_partitioned_execution(connection, sql, partitions):
    expected = connection.execute(sql).fetchall()
    graph = compile_sql(connection, sql, partitions)
    transported = FragmentGraph.from_dict(
        json.loads(json.dumps(graph.to_dict())), expected_engine_identity=native.engine_identity()
    )
    with vane.connect(config={"threads": 1}) as worker:
        assert Counter(execute_graph(worker, transported)) == Counter(expected)
    assert all(fragment.outputs[0].schema for fragment in graph.fragments)


@pytest.mark.parametrize("setting", ["PRAGMA disable_optimizer", "SET disabled_optimizers='in_clause'"])
@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize(
    "sql",
    [
        "select range in (1, 3, 5, 7, 9) as present from range(10)",
        "select range from range(10) where range in (1, 3, 5, 7, 9)",
    ],
)
def test_compiler_respects_disabled_optimizer(connection, setting, partitions, sql):
    connection.execute(setting)
    expected = connection.execute(sql).fetchall()
    graph = compile_sql(connection, sql, partitions=partitions)
    with vane.connect(config={"threads": 1}) as worker:
        assert Counter(execute_graph(worker, graph)) == Counter(expected)


def test_large_range_is_assigned_once_across_native_scan_splits(connection):
    graph = compile_sql(connection, "select range from range(12003)", partitions=7)
    source = graph.fragments[0].sources[0]
    assert len(source.splits) == 7
    assert not source.requires_snapshot
    assert graph.fragments[0].required_capabilities == (source.capability,)
    assert [edge.distribution for edge in graph.exchanges] == [Distribution.GATHER]
    assert sorted(execute_graph(connection, graph)) == [(value,) for value in range(12003)]


def test_constants_are_not_replicated_to_every_partition(connection):
    graph = compile_sql(connection, "select 9", partitions=32)
    assert len(graph.fragments) == 1
    assert graph.fragments[0].partition_count == 1
    assert graph.fragments[0].sources == ()
    assert graph.exchanges == ()
    assert execute_graph(connection, graph) == [(9,)]


def test_native_plan_is_portable_after_planning_connection_closes(connection):
    with vane.connect(config={"threads": 1}) as planning:
        graph = compile_sql(planning, "select range * 2 from range(7)", partitions=1)
    program = """
import json
import sys
import vane
from vane._native import execution_plan as native
from vane.execution.compiler import validate_native_graph
from vane.execution.plan import FragmentGraph

graph = FragmentGraph.from_dict(json.load(sys.stdin), expected_engine_identity=native.engine_identity())
with vane.connect(config={"threads": 1}) as worker:
    validate_native_graph(worker, graph)
    fragment = graph.fragments[0]
    assignments = {source.source_id: [split.split_id for split in source.splits] for source in fragment.sources}
    rows = native._execute_fragment_for_test(worker, fragment.native_plan, {}, assignments)
    print(json.dumps(rows))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        input=json.dumps(graph.to_dict()),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == [[value * 2] for value in range(7)]


def test_hash_then_gather_runs_through_native_ports(connection):
    sql = "select range % 5 as key, range as value from range(103) where range % 3 <> 0"
    graph = compile_sql(connection, sql, partitions=4, hash_columns=(0,))
    assert [edge.distribution for edge in graph.exchanges] == [Distribution.HASH, Distribution.GATHER]
    assert [fragment.partition_count for fragment in graph.fragments] == [4, 4, 1]
    assert Counter(execute_graph(connection, graph)) == Counter(connection.execute(sql).fetchall())


@pytest.mark.parametrize("columns", [(0,), (0, 1), (1, 0)])
def test_hash_routes_null_and_multicolumn_keys_with_duckdb_rules(connection, columns):
    graph = compile_sql(connection, "select 0::BIGINT as key, ''::VARCHAR as label", partitions=7, hash_columns=columns)
    edge = graph.exchanges[0]
    rows = [(None, None), (None, "a"), (-1, "a"), (0, ""), (23, "a"), (23, "a"), (23, "b")]
    actual = native._hash_rows_for_test(connection, graph.fragments[0].outputs[0].schema, edge.partitioning, rows, 7)
    arguments = ["$1::BIGINT", "$2::VARCHAR"]
    expression = ", ".join(arguments[column] for column in columns)
    expected = []
    for key, label in rows:
        params = {str(column + 1): (key, label)[column] for column in columns}
        expected.append(connection.execute(f"select hash({expression}) % 7", params).fetchone()[0])
    assert actual == expected
    assert actual[4] == actual[5]


def test_parquet_sources_are_frozen_as_splits_but_require_content_snapshot(connection, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for index in range(3):
        pq.write_table(pa.table({"value": list(range(index * 9, (index + 1) * 9))}), tmp_path / f"part{index}.parquet")
    sql = f"select value * 2 as doubled from read_parquet('{tmp_path}/*.parquet') where value % 2 = 0"
    graph = compile_sql(connection, sql, partitions=5)
    source = graph.fragments[0].sources[0]
    assert source.requires_snapshot
    assert len(source.splits) == 3
    assert Counter(execute_graph(connection, graph)) == Counter(connection.execute(sql).fetchall())

    # A later directory entry must not become an extra split during worker load.
    pq.write_table(pa.table({"value": [1000]}), tmp_path / "late.parquet")
    assert sorted(execute_graph(connection, graph)) == [(value * 2,) for value in range(27) if value % 2 == 0]


def test_parquet_worker_preserves_union_schema_without_coordinator_readers(connection, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"value": [1, 2]}), tmp_path / "first.parquet")
    pq.write_table(pa.table({"label": ["A", "B"], "value": [3, 4]}), tmp_path / "second.parquet")
    sql = f"select value, lower(label) from read_parquet('{tmp_path}/*.parquet', union_by_name=true)"
    expected = connection.execute(sql).fetchall()
    with vane.connect(config={"threads": 1}) as planning:
        graph = compile_sql(planning, sql, partitions=3)
    with vane.connect(config={"threads": 1}) as worker:
        assert Counter(execute_graph(worker, graph)) == Counter(expected)


@pytest.mark.parametrize("scan", ["read_parquet", "parquet_scan"])
@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize(
    "projection,predicate,expected", [("file_index", "", [(0,), (1,), (2,)]), ("value", "where file_index = 1", [(1,)])]
)
def test_parquet_virtual_file_index_is_rejected_before_splitting(
    connection, tmp_path, scan, partitions, projection, predicate, expected
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for index in range(3):
        pq.write_table(pa.table({"value": [index]}), tmp_path / f"part{index}.parquet")
    sql = f"select {projection} from {scan}('{tmp_path}/*.parquet') {predicate}"
    assert Counter(connection.execute(sql).fetchall()) == Counter(expected)
    with pytest.raises(vane.NotImplementedException, match="virtual file_index"):
        compile_sql(connection, sql, partitions=partitions)


def test_parquet_physical_column_named_file_index_remains_supported(connection, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for index in range(3):
        pq.write_table(pa.table({"file_index": [index], "value": [index + 10]}), tmp_path / f"part{index}.parquet")
    sql = f"select file_index, value from read_parquet('{tmp_path}/*.parquet') where file_index = 1"
    assert connection.execute(sql).fetchall() == [(1, 11)]
    graph = compile_sql(connection, sql, partitions=3)
    with vane.connect(config={"threads": 1}) as worker:
        assert execute_graph(worker, graph) == [(1, 11)]


@pytest.mark.parametrize("predicate", ["value > 10", "value IS NULL", "false"])
@pytest.mark.parametrize("hash_columns", [(), (0,)])
def test_optimized_parquet_dependencies_do_not_create_scan_tasks(connection, tmp_path, predicate, hash_columns):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"value": [1, 2, 3]}), path)
    graph = compile_sql(
        connection,
        f"select value from read_parquet('{path}') where {predicate}",
        partitions=4,
        hash_columns=hash_columns,
    )
    fragment = graph.fragments[0]
    assert fragment.partition_count == 1
    assert not fragment.sources
    assert len(fragment.source_dependencies) == 1
    assert fragment.required_capabilities == (fragment.source_dependencies[0].capability,)
    transported = FragmentGraph.from_dict(
        json.loads(json.dumps(graph.to_dict())), expected_engine_identity=native.engine_identity()
    )
    with vane.connect(config={"threads": 1}) as worker:
        assert execute_graph(worker, transported) == []


def test_compiling_does_not_enter_old_runner_or_execute_a_query(connection, monkeypatch):
    from vane import _ray_cxx, runners

    def forbidden(*args, **kwargs):
        raise AssertionError("compiler entered an execution path")

    monkeypatch.setattr(runners, "get_or_create_runner", forbidden)
    monkeypatch.setattr(_ray_cxx, "validate_plan_serialization_for_submission", forbidden)
    first = compile_sql(connection, "select range * 3 from range(11)")
    monkeypatch.setenv("VANE_RUNNER", "not-a-runner")
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "not-an-exchange")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "999")
    second = compile_sql(connection, "select range * 3 from range(11)")
    assert first == second
    validate_native_graph(connection, second)


def test_native_local_query_does_not_enter_fragment_compiler(connection, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("local query entered distributed planning")

    monkeypatch.setattr(native, "compile", forbidden)
    assert connection.execute("select range + 1 from range(3)").fetchall() == [(1,), (2,), (3,)]


@pytest.mark.parametrize(
    "sql",
    [
        "create table must_not_exist as select 1",
        "select 1; select 2",
        "select random()",
        "select nextval('must_not_exist')",
        "select sum(range) from range(10)",
        "select * from range(10) order by range",
        "select * from range(10) limit 1",
        "select * from range(2) a join range(3) b on a.range = b.range",
        "select 1.2::DECIMAL(8, 2)",
        "select $1",
    ],
)
def test_unsupported_sql_fails_instead_of_falling_back(connection, sql):
    with pytest.raises(vane.NotImplementedException):
        compile_sql(connection, sql)


def test_python_udf_is_rejected_before_constant_folding_can_call_it(connection):
    calls = []

    def touch(value):
        calls.append(value)
        return value

    connection._create_vane_function(
        "compiler_touch", touch, [vane.sqltypes.INTEGER], vane.sqltypes.INTEGER, replace=False
    )
    for sql in ("select compiler_touch(1)", "select * from range(compiler_touch(4))"):
        with pytest.raises(vane.NotImplementedException, match="compiler_touch"):
            compile_sql(connection, sql)
    assert calls == []


def test_unbound_or_unknown_ports_cannot_execute_as_empty_input(connection):
    graph = compile_sql(connection, "select range from range(9)", partitions=2)
    root = graph.fragments[-1]
    with pytest.raises(vane.InvalidInputException, match="missing fragment input"):
        native._execute_fragment_for_test(connection, root.native_plan, {}, {})
    with pytest.raises(vane.InvalidInputException, match="unexpected fragment input"):
        native._execute_fragment_for_test(connection, root.native_plan, {"in": [], "unknown": []}, {})
    assert native._execute_fragment_for_test(connection, root.native_plan, {"in": []}, {}) == []


def test_source_assignments_are_explicit_and_validate_split_identity(connection):
    graph = compile_sql(connection, "select range from range(13)", partitions=3)
    fragment = graph.fragments[0]
    source = fragment.sources[0]
    with pytest.raises(vane.InvalidInputException, match="explicit assignment"):
        native._execute_fragment_for_test(connection, fragment.native_plan, {}, {})
    for assigned in (["unknown"], [source.splits[0].split_id] * 2):
        with pytest.raises(vane.InvalidInputException, match="unknown or duplicate"):
            native._execute_fragment_for_test(connection, fragment.native_plan, {}, {source.source_id: assigned})
    assert native._execute_fragment_for_test(connection, fragment.native_plan, {}, {source.source_id: []}) == []


@pytest.mark.parametrize("damage", ["trailing", "truncated", "foreign_engine"])
def test_native_loader_rejects_damaged_or_foreign_payload(connection, damage):
    fragment = compile_sql(connection, "select 1").fragments[0]
    payload = fragment.native_plan
    if damage == "trailing":
        payload += b"unexpected"
    elif damage == "truncated":
        payload = payload[:-1]
    else:
        identity = native.engine_identity().encode()
        assert identity in payload
        payload = payload.replace(identity, b"x" * len(identity))
    with pytest.raises(vane.SerializationException):
        native.inspect_fragment(connection, payload)


def test_native_validation_rejects_changed_python_graph_metadata(connection):
    graph = compile_sql(connection, "select range from range(13)", partitions=3)
    fragment = graph.fragments[0]
    source = fragment.sources[0]
    changed = replace(fragment, sources=(replace(source, requires_snapshot=True),))
    damaged = replace(graph, fragments=(changed, *graph.fragments[1:]))
    with pytest.raises(ValueError, match="disagrees"):
        validate_native_graph(connection, damaged)
    with pytest.raises(ValueError, match="engine identity"):
        validate_native_graph(connection, replace(graph, engine_identity="different"))


def test_hash_expression_is_validated_against_native_output_schema(connection):
    graph = compile_sql(connection, "select range from range(13)", hash_columns=(0,))
    edge = graph.exchanges[0]
    damaged = replace(
        graph, exchanges=(replace(edge, partitioning=edge.partitioning + b"trailing"), *graph.exchanges[1:])
    )
    with pytest.raises(vane.SerializationException):
        validate_native_graph(connection, damaged)
    with pytest.raises(vane.InvalidInputException, match="valid result-column"):
        compile_sql(connection, "select 1", hash_columns=(1,))


@pytest.mark.parametrize("count", [True, 0, -1, 1.5, "2", 2**31])
def test_compile_options_reject_invalid_partition_counts(count):
    with pytest.raises(ValueError, match="partition_count"):
        FragmentCompileOptions(count)


@pytest.mark.parametrize("columns", [[True], [-1], [0, 0], [1.5], "key"])
def test_compile_options_reject_invalid_hash_positions(columns):
    with pytest.raises(ValueError, match="hash_columns"):
        FragmentCompileOptions(hash_columns=columns)


def test_compile_options_snapshot_caller_owned_columns():
    columns = [0, 1]
    options = FragmentCompileOptions(4, columns)
    columns.clear()
    assert options.hash_columns == (0, 1)
