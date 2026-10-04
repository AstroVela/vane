# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Source copies precede optimization and remain the only replay inputs."""

import os

import pytest

import vane
from vane._native import execution_plan
from vane.execution.compiler import FragmentCompileOptions
from vane.execution.query_options import FteOptions, QueryExecutionOptions, RayExecution
from vane.execution.resource_demand import MemoryDemand, ResourceDemand
from vane.execution.submission import RayQuerySpec, prepare_worker_plan, stage_ray_query


def stage(connection, sql, directory, budget=1 << 20):
    return stage_ray_query(
        connection,
        sql,
        query_id="frozen-source",
        options=QueryExecutionOptions(RayExecution("fte", FteOptions("shared", 3, 0)), 10, 30, 30),
        resources=ResourceDemand(1, 8, MemoryDemand(2**26, 2**20, 2**20, 2**20), 8),
        compile_options=FragmentCompileOptions(1),
        source_directory=str(directory),
        source_budget=budget,
    )


def run(spec):
    with vane.connect(backend="local", config={"threads": 1}) as worker:
        prepare_worker_plan(worker, spec)
        assert len(spec.graph.fragments) == 1
        fragment = spec.graph.fragments[0]
        return execution_plan._execute_fragment_for_test(
            worker,
            fragment.native_plan,
            {},
            {source.source_id: [split.split_id for split in source.splits] for source in fragment.sources},
        )


@pytest.mark.parametrize("predicate", ["", " where value > 10"])
@pytest.mark.parametrize("change", ["rewrite", "delete", "new_member"])
def test_frozen_files_and_pruned_dependencies_ignore_original_changes(tmp_path, predicate, change):
    path = tmp_path / "original.parquet"
    with vane.connect(backend="local") as connection:
        connection.execute(f"copy (select range as value from range(1, 4)) to '{path}'")
        size = path.stat().st_size
        spec, used = stage(
            connection, f"select value from read_parquet('{tmp_path}/*.parquet')" + predicate, tmp_path / "snapshots"
        )
        assert used == size
        assert spec.requires_replay
        assert RayQuerySpec.from_dict(spec.to_dict(), expected_engine_identity=spec.graph.engine_identity) == spec
        if change == "delete":
            path.unlink()
        else:
            target = path if change == "rewrite" else tmp_path / "new.parquet"
            connection.execute(f"copy (select range as value from range(20, 24)) to '{target}' (overwrite true)")
    expected = [] if predicate else [(1,), (2,), (3,)]
    assert run(spec) == expected
    assert run(spec) == expected
    sources = [s for f in spec.graph.fragments for s in f.sources + f.source_dependencies]
    assert sources and all(s.codec == "vane.parquet-snapshot:1" and not s.requires_snapshot for s in sources)


@pytest.mark.parametrize("predicate", ["", " where value > 10"])
def test_frozen_content_digest_detects_damage_with_unchanged_stat(tmp_path, predicate):
    path = tmp_path / "original.parquet"
    directory = tmp_path / "snapshots"
    with vane.connect(backend="local") as connection:
        connection.execute(f"copy (select 1 as value) to '{path}'")
        spec, _ = stage(connection, f"select value from read_parquet('{path}')" + predicate, directory)
    frozen = next(directory.rglob("*.parquet"))
    stamp, original = frozen.stat(), frozen.read_bytes()
    frozen.write_bytes(original[:10] + bytes([original[10] ^ 1]) + original[11:])
    os.utime(frozen, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(Exception, match="snapshot changed"):
        run(spec)


def test_hive_paths_are_preserved_and_globs_are_not_reexpanded(tmp_path):
    with vane.connect(backend="local") as connection:
        for part in (1, 2):
            directory = tmp_path / f"part={part}"
            directory.mkdir()
            connection.execute(f"copy (select {part * 10} as value) to '{directory / 'input.parquet'}'")
        spec, _ = stage(
            connection,
            f"select part, value from read_parquet('{tmp_path}/part=*/*.parquet', "
            "hive_partitioning=true) where part = 2",
            tmp_path / "snapshots",
        )
    assert run(spec) == [(2, 20)]


def test_snapshot_storage_limit_is_enforced_before_writing_source_bytes(tmp_path):
    path = tmp_path / "source.parquet"
    with vane.connect(backend="local") as connection:
        connection.execute(f"copy (select range as value from range(100)) to '{path}'")
        with pytest.raises(Exception, match="storage capacity"):
            stage(connection, f"select * from read_parquet('{path}')", tmp_path / "snapshots", 40)
    assert sum(path.stat().st_size for path in (tmp_path / "snapshots").rglob("*.parquet")) <= 40


def test_virtual_filename_is_rejected_but_physical_column_is_preserved(tmp_path):
    path = tmp_path / "source.parquet"
    with vane.connect(backend="local") as connection:
        connection.execute(f"copy (select 1 as value) to '{path}'")
        with pytest.raises(Exception, match="virtual filename"):
            stage(connection, f"select filename from read_parquet('{path}')", tmp_path / "virtual")
        connection.execute(f"copy (select 'original' as filename) to '{path}' (overwrite true)")
        spec, _ = stage(connection, f"select filename from read_parquet('{path}')", tmp_path / "physical")
    assert run(spec) == [("original",)]


def test_staging_does_not_evaluate_implicit_side_effects(tmp_path):
    path = tmp_path / "source.parquet"
    with vane.connect(backend="local") as connection:
        connection.execute(f"copy (select 1 as value) to '{path}'")
        connection.execute("create sequence s")
        connection.execute("create macro current_user() as nextval('s')")
        with pytest.raises(Exception, match="built-in"):
            stage(
                connection,
                f"select * from read_parquet('{path}', binary_as_string=(current_user > 0))",
                tmp_path / "snapshots",
            )
        assert connection.execute("select nextval('s')").fetchone() == (1,)


def test_range_needs_no_file_staging(tmp_path):
    with vane.connect(backend="local") as connection:
        spec, used = stage(connection, "select range from range(3)", tmp_path / "snapshots")
    assert used == 0
    assert not (tmp_path / "snapshots").exists()
    assert run(spec) == [(0,), (1,), (2,)]


def test_snapshot_and_guard_support_unicode_paths(tmp_path):
    from vane._native import execution_runtime as native

    original = tmp_path / "原始文件.parquet"
    directory = tmp_path / "冻结存储"
    with vane.connect(backend="local") as connection:
        connection.execute(f"copy (select 42 as value) to '{original}'")
        spec, _ = stage(connection, f"select * from read_parquet('{original}')", directory)
    guard = native.StoreGuard.acquire(str(directory / "租约.lock"), False, True)
    assert guard is not None
    try:
        assert run(spec) == [(42,)]
    finally:
        guard.close()


def test_interrupted_staging_does_not_poison_the_planning_connection(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "large.parquet"
    # Staging happens before binding. A sparse input lets cancellation exercise
    # the copy without allocating a large test dataset or parsing its contents.
    with path.open("wb") as handle:
        handle.truncate(512 << 20)
    directory = tmp_path / "snapshots"
    with vane.connect(backend="local") as connection, ThreadPoolExecutor(1) as executor:
        future = executor.submit(stage, connection, f"select * from read_parquet('{path}')", directory, 1 << 30)
        deadline = time.monotonic() + 5
        while not any(p.stat().st_size > 0 for p in directory.rglob("*.parquet")):
            assert not future.done()
            assert time.monotonic() < deadline
            time.sleep(0.001)
        connection.interrupt()
        with pytest.raises(vane.InterruptException):
            future.result(timeout=5)
        assert connection.execute("select 42").fetchone() == (42,)
        spec, _ = stage(connection, "select range from range(3)", tmp_path / "second")
        assert run(spec) == [(0,), (1,), (2,)]


def test_literal_path_list_preserves_duplicate_files(tmp_path):
    paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    with vane.connect(backend="local") as connection:
        for i, path in enumerate(paths):
            connection.execute(f"copy (select {i} as value) to '{path}'")
        sql = f"select * from read_parquet(['{paths[0]}', '{paths[1]}', '{paths[0]}'])"
        expected = connection.execute(sql).fetchall()
        spec, used = stage(connection, sql, tmp_path / "snapshots")
        assert used == sum(path.stat().st_size for path in paths)
        assert sorted(run(spec)) == sorted(expected) == [(0,), (0,), (1,)]
        with pytest.raises(Exception, match="literal absolute paths"):
            stage(connection, f"select * from read_parquet([['{paths[0]}']])", tmp_path / "nested")


def test_repeated_globs_cannot_bypass_the_file_reference_limit(tmp_path):
    with vane.connect(backend="local") as connection:
        for name in ("a", "b"):
            connection.execute(f"copy (select 42 as value) to '{tmp_path / (name + '.parquet')}'")
        paths = ",".join([f"'{tmp_path}/*.parquet'"] * 2050)
        with pytest.raises(Exception, match="reference count exceeds limit"):
            stage(connection, f"select * from read_parquet([{paths}])", tmp_path / "snapshots")
